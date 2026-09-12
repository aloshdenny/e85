#!/usr/bin/env python3
"""Harvest face crops from the web for Mia / Johnny Sins (incremental, deduped).

Pipeline: search -> download -> sha256 dedup -> InsightFace identity filter ->
upright 256px crop -> phash + embedding dedup -> manifest -> optional zip rebuild.

Designed for repeated runs until --target-count is reached. Pinterest images
often appear via Bing/DDG site:pinterest.com queries; optional gallery-dl
boards listed in target/harvest/<person>/pinterest_urls.txt.

Install once:
  pip install -r requirements-harvest.txt

Examples:
  python scripts/harvest_target_faces.py --person mia --per-query 80 --rebuild-zip
  python scripts/harvest_target_faces.py --person sins --per-query 120 --rebuild-zip
  python scripts/harvest_target_faces.py --person mia --ingest-dir ~/Downloads
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import cv2
import imagehash
import numpy as np
import requests
from insightface.app import FaceAnalysis
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QUERIES_JSON = Path(__file__).with_name("harvest_queries.json")

OUT_SIZE = 256
MARGIN = 0.65
MIN_DET = 0.30
PHASH_MAX_DIST = 6
EMB_DUP_SIM = 0.93
GALLERY_MAX = 80
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


def load_bgr_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_bgr_path(path: Path) -> np.ndarray | None:
    try:
        return load_bgr_bytes(path.read_bytes())
    except OSError:
        return None


def upright_square(img, bbox, kps, out_size=OUT_SIZE, margin=MARGIN):
    le, re = np.asarray(kps[0], np.float32), np.asarray(kps[1], np.float32)
    angle = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    side = max(bw, bh) * (1.0 + 2 * margin)
    pad = int(max(img.shape[0], img.shape[1], side) * 0.6)
    fill = tuple(int(v) for v in img.reshape(-1, 3).mean(0))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
    cx_p, cy_p = cx + pad, cy + pad
    R = cv2.getRotationMatrix2D((cx_p, cy_p), angle, 1.0)
    rot = cv2.warpAffine(
        padded, R, (padded.shape[1], padded.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
    )
    xa = int(round(cx_p - side / 2))
    ya = int(round(cy_p - side / 2))
    xb = int(round(cx_p + side / 2))
    yb = int(round(cy_p + side / 2))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(rot.shape[1], xb), min(rot.shape[0], yb)
    crop = rot[ya:yb, xa:xb]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def save_jpg(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"encode failed {path}")
    path.write_bytes(buf.tobytes())


def phash_bgr(bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


@dataclass
class Manifest:
    person: str
    target_count: int
    entries: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, person: str, target_count: int) -> "Manifest":
        if not path.exists():
            return cls(person=person, target_count=target_count, entries=[])
        data = json.loads(path.read_text())
        return cls(
            person=data.get("person", person),
            target_count=data.get("target_count", target_count),
            entries=list(data.get("entries", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "person": self.person,
            "target_count": self.target_count,
            "updated": utc_now(),
            "kept": sum(1 for e in self.entries if e.get("status") == "kept"),
            "entries": self.entries,
        }
        path.write_text(json.dumps(payload, indent=2))


class DedupIndex:
    def __init__(self) -> None:
        self.urls: set[str] = set()
        self.sha256: set[str] = set()
        self.phashes: list[imagehash.ImageHash] = []
        self.embeddings: list[np.ndarray] = []

    def add_entry(self, entry: dict, load_embed: bool = True) -> None:
        url = entry.get("url")
        if url:
            self.urls.add(normalize_url(url))
        sha = entry.get("sha256")
        if sha:
            self.sha256.add(sha)
        ph = entry.get("phash")
        if ph:
            self.phashes.append(imagehash.hex_to_hash(ph))
        if load_embed and entry.get("status") == "kept":
            emb = entry.get("embedding")
            if emb:
                self.embeddings.append(unit(np.asarray(emb, dtype=np.float64)))

    def is_dup_url(self, url: str) -> bool:
        return normalize_url(url) in self.urls

    def is_dup_sha(self, sha: str) -> bool:
        return sha in self.sha256

    def is_dup_phash(self, h: imagehash.ImageHash) -> bool:
        return any(h - old <= PHASH_MAX_DIST for old in self.phashes)

    def is_dup_embed(self, emb: np.ndarray) -> bool:
        e = unit(emb)
        if not self.embeddings:
            return False
        sims = np.dot(np.stack(self.embeddings), e)
        return float(sims.max()) >= EMB_DUP_SIM


def iter_zip_images(zip_path: Path) -> Iterable[tuple[str, bytes]]:
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            try:
                yield name, zf.read(name)
            except Exception:
                continue


def gallery_centroid(app: FaceAnalysis, zip_path: Path) -> np.ndarray:
    embs = []
    names = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n for n in zf.namelist()
            if Path(n).suffix.lower() in IMAGE_EXTS
            and not n.startswith("__MACOSX/")
            and not Path(n).name.startswith("._")
        ]
    rng = np.random.default_rng(0)
    if len(names) > GALLERY_MAX:
        names = list(rng.choice(names, GALLERY_MAX, replace=False))
    with zipfile.ZipFile(zip_path) as zf:
        for n in names:
            raw = zf.read(n)
            img = load_bgr_bytes(raw)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            f = max(faces, key=lambda x: float(x.det_score))
            if float(f.det_score) < MIN_DET:
                continue
            embs.append(unit(np.asarray(f.normed_embedding, dtype=np.float64)))
    if len(embs) < 3:
        raise SystemExit(f"seed gallery too small ({len(embs)} faces in {zip_path})")
    return unit(np.mean(embs, axis=0))


def seed_dedup_from_existing(
    dedup: DedupIndex,
    manifest: Manifest,
    zip_path: Path,
    crop_dir: Path,
) -> int:
    n = 0
    for entry in manifest.entries:
        dedup.add_entry(entry)
        n += 1
    for name, raw in iter_zip_images(zip_path):
        sha = sha256_bytes(raw)
        if sha not in dedup.sha256:
            dedup.sha256.add(sha)
            n += 1
        img = load_bgr_bytes(raw)
        if img is None:
            continue
        try:
            dedup.phashes.append(phash_bgr(img))
        except Exception:
            pass
    for p in sorted(crop_dir.glob("*.jpg")):
        img = load_bgr_path(p)
        if img is None:
            continue
        sha = sha256_bytes(p.read_bytes())
        dedup.sha256.add(sha)
        try:
            dedup.phashes.append(phash_bgr(img))
        except Exception:
            pass
    return n


def download_url(session: requests.Session, url: str, timeout: float) -> bytes | None:
    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        data = r.content
        if len(data) < 2048 or len(data) > 25_000_000:
            return None
        return data
    except Exception:
        return None


def search_ddg(queries: list[str], per_query: int, seen_urls: set[str]) -> list[tuple[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        print("ddgs not installed; skip DDG backend (pip install ddgs)", flush=True)
        return []
    out: list[tuple[str, str]] = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.images(q, max_results=per_query)
            except Exception as exc:
                print(f"  DDG fail '{q[:50]}': {exc}", flush=True)
                continue
            for row in results:
                url = row.get("image") or row.get("thumbnail") or row.get("url")
                if not url:
                    continue
                nu = normalize_url(url)
                if nu in seen_urls:
                    continue
                seen_urls.add(nu)
                out.append((f"ddg:{q}", url))
            time.sleep(0.8)
    return out


def search_laion(
    queries: list[str],
    per_query: int,
    seen_urls: set[str],
    *,
    service_url: str = "https://knn5.laion.ai/knn-service",
    indice_name: str = "laion5B",
) -> list[tuple[str, str]]:
    """Search LAION-5B captions via the free hosted clip-retrieval index."""
    try:
        from clip_retrieval.clip_client import ClipClient, Modality
    except ImportError:
        print("clip-retrieval not installed; skip LAION "
              "(pip install clip-retrieval)", flush=True)
        return []
    out: list[tuple[str, str]] = []
    client = ClipClient(
        url=service_url,
        indice_name=indice_name,
        num_images=per_query,
        modality=Modality.IMAGE,
    )
    for q in queries:
        try:
            results = client.query(text=q)
        except Exception as exc:
            print(f"  LAION fail '{q[:50]}': {exc}", flush=True)
            continue
        n = 0
        for row in results:
            url = row.get("url")
            if not url:
                continue
            nu = normalize_url(url)
            if nu in seen_urls:
                continue
            seen_urls.add(nu)
            out.append((f"laion:{q}", url))
            n += 1
        print(f"  LAION '{q[:45]}' -> {n} new URLs", flush=True)
        time.sleep(0.4)
    return out


def search_bing_icrawler(queries: list[str], per_query: int, raw_dir: Path) -> list[tuple[str, Path]]:
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("icrawler not installed; skip Bing backend (pip install icrawler)", flush=True)
        return []
    out: list[tuple[str, Path]] = []
    for q in queries:
        qdir = raw_dir / re.sub(r"[^\w.-]+", "_", q)[:60]
        qdir.mkdir(parents=True, exist_ok=True)
        before = {p.resolve() for p in qdir.glob("*") if p.is_file()}
        crawler = BingImageCrawler(
            storage={"root_dir": str(qdir)},
            downloader_threads=2,
        )
        try:
            crawler.crawl(keyword=q, max_num=per_query, min_size=(200, 200))
        except Exception as exc:
            print(f"  Bing fail '{q[:50]}': {exc}", flush=True)
            continue
        after = [p for p in qdir.glob("*") if p.is_file() and p.resolve() not in before]
        for p in after:
            out.append((q, p))
        time.sleep(1.0)
    return out


def ingest_gallery_dl(board_file: Path, raw_dir: Path) -> list[Path]:
    if not board_file.exists():
        return []
    urls = [ln.strip() for ln in board_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if not urls:
        return []
    if subprocess.run(["which", "gallery-dl"], capture_output=True).returncode != 0:
        print("gallery-dl not found; skip Pinterest boards "
              "(brew install gallery-dl)", flush=True)
        return []
    raw_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in raw_dir.rglob("*") if p.is_file()}
    for url in urls:
        subprocess.run(
            ["gallery-dl", "--no-part", "-D", str(raw_dir), url],
            check=False,
        )
        time.sleep(1.5)
    return [p for p in raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            and p.resolve() not in before]


def ingest_local_dir(path: Path) -> list[Path]:
    if not path.exists():
        return []
    out = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def process_image_bytes(
    *,
    data: bytes,
    url: str,
    source: str,
    app: FaceAnalysis,
    centroid: np.ndarray,
    sim_thresh: float,
    dedup: DedupIndex,
    crop_dir: Path,
    reject_dir: Path,
    person: str,
    keep_raw: bool,
    raw_dir: Path,
    next_idx: int,
) -> tuple[dict, int]:
    sha = sha256_bytes(data)
    if dedup.is_dup_sha(sha):
        return {"url": url, "source": source, "sha256": sha, "status": "dup_sha256",
                "ts": utc_now()}, next_idx

    img = load_bgr_bytes(data)
    if img is None:
        return {"url": url, "source": source, "sha256": sha, "status": "bad_image",
                "ts": utc_now()}, next_idx

    faces = app.get(img)
    if not faces and min(img.shape[:2]) > 400:
        app.prepare(ctx_id=-1, det_size=(1024, 1024))
        faces = app.get(img)
        app.prepare(ctx_id=-1, det_size=(640, 640))
    if not faces:
        return {"url": url, "source": source, "sha256": sha, "status": "no_face",
                "ts": utc_now()}, next_idx

    best = None
    best_sim = -1.0
    for f in faces:
        if float(f.det_score) < MIN_DET:
            continue
        emb = unit(np.asarray(f.normed_embedding, dtype=np.float64))
        sim = float(np.dot(emb, centroid))
        if sim > best_sim:
            best_sim, best = sim, f
    if best is None:
        return {"url": url, "source": source, "sha256": sha, "status": "low_det",
                "ts": utc_now()}, next_idx

    kps = getattr(best, "kps", None)
    crop = upright_square(img, best.bbox, kps) if kps is not None and len(kps) >= 2 else None
    if crop is None:
        return {"url": url, "source": source, "sha256": sha, "status": "crop_fail",
                "ts": utc_now()}, next_idx

    if best_sim < sim_thresh:
        reject_dir.mkdir(parents=True, exist_ok=True)
        rej = reject_dir / f"other_{next_idx:05d}_sim{best_sim:.2f}.jpg"
        save_jpg(rej, crop)
        return {
            "url": url, "source": source, "sha256": sha, "status": "reject_identity",
            "sim": round(best_sim, 4), "reject": str(rej.name), "ts": utc_now(),
        }, next_idx

    ph = phash_bgr(crop)
    if dedup.is_dup_phash(ph):
        return {"url": url, "source": source, "sha256": sha, "phash": str(ph),
                "status": "dup_phash", "sim": round(best_sim, 4), "ts": utc_now()}, next_idx

    emb = unit(np.asarray(best.normed_embedding, dtype=np.float64))
    if dedup.is_dup_embed(emb):
        return {"url": url, "source": source, "sha256": sha, "phash": str(ph),
                "status": "dup_embed", "sim": round(best_sim, 4), "ts": utc_now()}, next_idx

    crop_name = f"{person}_{next_idx:05d}.jpg"
    crop_path = crop_dir / crop_name
    save_jpg(crop_path, crop)
    if keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{next_idx:05d}_{sha[:12]}.jpg").write_bytes(data)

    entry = {
        "url": url,
        "source": source,
        "sha256": sha,
        "phash": str(ph),
        "status": "kept",
        "sim": round(best_sim, 4),
        "crop": crop_name,
        "embedding": emb.tolist(),
        "ts": utc_now(),
    }
    dedup.add_entry(entry)
    return entry, next_idx + 1


def rebuild_zip(
    zip_path: Path,
    zip_prefix: str,
    crop_dir: Path,
    manifest: Manifest,
) -> int:
    """Merge existing zip + harvest crops into one deduped archive."""
    members: dict[str, bytes] = {}
    phashes: list[imagehash.ImageHash] = []

    for name, raw in iter_zip_images(zip_path):
        img = load_bgr_bytes(raw)
        if img is None:
            continue
        h = phash_bgr(img)
        if any(h - old <= PHASH_MAX_DIST for old in phashes):
            continue
        phashes.append(h)
        members[name] = raw

    for p in sorted(crop_dir.glob("*.jpg")):
        if not p.exists():
            continue
        raw = p.read_bytes()
        img = load_bgr_bytes(raw)
        if img is None:
            continue
        h = phash_bgr(img)
        if any(h - old <= PHASH_MAX_DIST for old in phashes):
            continue
        phashes.append(h)
        arc = f"{zip_prefix}/{p.name}"
        members[arc] = raw

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc in sorted(members):
            zf.writestr(arc, members[arc])
    tmp.replace(zip_path)
    return len(members)


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest identity face crops from the web.")
    ap.add_argument("--person", choices=["mia", "sins"], required=True)
    ap.add_argument("--target-count", type=int, default=1000)
    ap.add_argument("--per-query", type=int, default=80,
                    help="Max image URLs to fetch per search query per run.")
    ap.add_argument("--max-new", type=int, default=200,
                    help="Stop after this many new kept crops in one run.")
    ap.add_argument("--backend", choices=["both", "ddg", "bing", "laion", "all"],
                    default="all",
                    help="Search backends. 'all' = DDG + Bing + LAION-5B caption index.")
    ap.add_argument("--query-set", choices=["default", "scout2"], default="default",
                    help="Query list: default (round 1) or scout2 (new sources).")
    ap.add_argument("--ingest-dir", type=Path, default=None,
                    help="Also process images from a local folder (e.g. Downloads).")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--rebuild-zip", action="store_true")
    ap.add_argument("--download-timeout", type=float, default=12.0)
    ap.add_argument("--sleep", type=float, default=0.15, help="Pause between URL downloads.")
    args = ap.parse_args()

    cfg = json.loads(QUERIES_JSON.read_text())[args.person]
    sim_thresh = float(cfg["sim_thresh"])
    zip_prefix = cfg["zip_prefix"]
    if args.query_set == "scout2":
        queries = list(cfg.get("scout2_queries", []))
        laion_queries = list(cfg.get("scout2_laion_queries", []))
        print(f"query-set scout2: {len(queries)} DDG/Bing + {len(laion_queries)} LAION",
              flush=True)
    else:
        queries = list(cfg["queries"])
        laion_queries = list(cfg.get("laion_queries", queries[:8]))

    harvest_root = ROOT / "target" / "harvest" / args.person
    crop_dir = harvest_root / "crops"
    reject_dir = harvest_root / "rejected"
    raw_dir = harvest_root / "raw"
    manifest_path = harvest_root / "manifest.json"
    board_file = harvest_root / "pinterest_urls.txt"
    zip_path = ROOT / "target" / f"{args.person}.zip"

    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(manifest_path, args.person, args.target_count)
    manifest.target_count = args.target_count

    dedup = DedupIndex()
    seeded = seed_dedup_from_existing(dedup, manifest, zip_path, crop_dir)
    kept_now = sum(1 for e in manifest.entries if e.get("status") == "kept")
    zip_count = sum(1 for _ in iter_zip_images(zip_path))
    print(f"{args.person}: zip={zip_count}  harvest_kept={kept_now}  "
          f"target={args.target_count}  dedup_seeded={seeded}", flush=True)

    if kept_now + zip_count >= args.target_count:
        print("Already at target; use --rebuild-zip only or lower --target-count.", flush=True)
        if args.rebuild_zip:
            n = rebuild_zip(zip_path, zip_prefix, crop_dir, manifest)
            print(f"Rebuilt {zip_path}  {n} unique crops", flush=True)
        return

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    centroid = gallery_centroid(app, zip_path)
    print(f"identity centroid from {zip_path}", flush=True)

    next_idx = 1 + max(
        [int(re.search(r"_(\d+)\.jpg$", e["crop"]).group(1))
         for e in manifest.entries if e.get("crop") and re.search(r"_(\d+)\.jpg$", e["crop"])]
        + [0]
    )
    new_kept = 0
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # --- local / pinterest ingest ---
    local_paths: list[tuple[str, Path]] = []
    if args.ingest_dir:
        for p in ingest_local_dir(args.ingest_dir.expanduser()):
            local_paths.append((f"local:{p.name}", p))
    for p in ingest_gallery_dl(board_file, raw_dir / "pinterest"):
        local_paths.append((f"pinterest:{p.name}", p))

    for source, path in local_paths:
        if new_kept >= args.max_new:
            break
        try:
            data = path.read_bytes()
        except OSError:
            continue
        entry, next_idx = process_image_bytes(
            data=data, url=str(path), source=source, app=app, centroid=centroid,
            sim_thresh=sim_thresh, dedup=dedup, crop_dir=crop_dir,
            reject_dir=reject_dir, person=args.person, keep_raw=args.keep_raw,
            raw_dir=raw_dir, next_idx=next_idx,
        )
        manifest.entries.append(entry)
        if entry.get("status") == "kept":
            new_kept += 1
            print(f"  KEPT {entry['crop']}  sim={entry['sim']:+.3f}  {source[:60]}", flush=True)

    # --- web search ---
    seen_urls = set(dedup.urls)
    url_jobs: list[tuple[str, str]] = []
    if args.backend in ("both", "ddg", "all"):
        print(f"DDG search: {len(queries)} queries x {args.per_query}", flush=True)
        url_jobs.extend(search_ddg(queries, args.per_query, seen_urls))
    if args.backend in ("laion", "all"):
        print(f"LAION-5B search: {len(laion_queries)} queries x {args.per_query}",
              flush=True)
        url_jobs.extend(search_laion(laion_queries, args.per_query, seen_urls))
    bing_files: list[tuple[str, Path]] = []
    if args.backend in ("both", "bing", "all"):
        print(f"Bing crawl: {len(queries)} queries x {args.per_query}", flush=True)
        bing_files = search_bing_icrawler(queries, args.per_query, raw_dir / "bing")

    print(f"URL queue {len(url_jobs)}  bing files {len(bing_files)}", flush=True)

    for source, url in url_jobs:
        if new_kept >= args.max_new:
            break
        if dedup.is_dup_url(url):
            continue
        data = download_url(session, url, args.download_timeout)
        if data is None:
            manifest.entries.append({
                "url": url, "source": source, "status": "download_fail", "ts": utc_now(),
            })
            continue
        entry, next_idx = process_image_bytes(
            data=data, url=url, source=source, app=app, centroid=centroid,
            sim_thresh=sim_thresh, dedup=dedup, crop_dir=crop_dir,
            reject_dir=reject_dir, person=args.person, keep_raw=args.keep_raw,
            raw_dir=raw_dir, next_idx=next_idx,
        )
        manifest.entries.append(entry)
        if entry.get("status") == "kept":
            new_kept += 1
            print(f"  KEPT {entry['crop']}  sim={entry['sim']:+.3f}  {source[:55]}",
                  flush=True)
        time.sleep(args.sleep)

    for q, path in bing_files:
        if new_kept >= args.max_new:
            break
        try:
            data = path.read_bytes()
        except OSError:
            continue
        entry, next_idx = process_image_bytes(
            data=data, url=str(path), source=f"bing:{q}", app=app, centroid=centroid,
            sim_thresh=sim_thresh, dedup=dedup, crop_dir=crop_dir,
            reject_dir=reject_dir, person=args.person, keep_raw=args.keep_raw,
            raw_dir=raw_dir, next_idx=next_idx,
        )
        manifest.entries.append(entry)
        if entry.get("status") == "kept":
            new_kept += 1
            print(f"  KEPT {entry['crop']}  sim={entry['sim']:+.3f}  {q[:45]}", flush=True)

    manifest.save(manifest_path)
    kept_total = sum(1 for e in manifest.entries if e.get("status") == "kept")
    print(f"\nRun done: +{new_kept} kept this run  harvest_total={kept_total}  "
          f"manifest -> {manifest_path}", flush=True)

    if args.rebuild_zip:
        n = rebuild_zip(zip_path, zip_prefix, crop_dir, manifest)
        print(f"Rebuilt {zip_path}  {n} unique crops (deduped vs existing zip)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
