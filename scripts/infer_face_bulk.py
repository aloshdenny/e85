"""
infer_face_bulk_parallel.py

Runs infer_face_bulk.py's category-processing logic across multiple worker
processes, each with its own TribeModel instance.

Why this helps despite a single model only using ~0.79GB VRAM:
  The video-encode step inside model.predict() is sequential GPU kernel work
  per video (that's the "Encoding video: N/N" progress bar) -- it is NOT
  batched across the multi-row events dataframe, confirmed by reading
  TribeModel.predict()'s source. VRAM was never the constraint; compute
  (SM occupancy) is. Multiple processes issuing concurrent kernels can use
  otherwise-idle SMs, but expect SUB-LINEAR scaling -- this is not "N workers
  = N times faster." Test empirically at small N before committing to a
  large --num-workers value.

IMPORTANT — run this once before going parallel:
  If model.from_pretrained(...) needs to download/cache checkpoint files on
  first run, multiple workers hitting an empty --cache-folder simultaneously
  can race on that download. Prime the cache with a single-process run first:

    python infer_face_bulk.py --probe 1

  Then launch this parallel driver once the cache is populated.

Usage:
  # Test scaling behavior first -- try 2, then 4, then more, comparing
  # wall-clock time on a fixed small slice before committing to a full run.
  python infer_face_bulk_parallel.py --num-workers 2
  python infer_face_bulk_parallel.py --num-workers 4

  # Full run once you've picked a worker count
  python infer_face_bulk_parallel.py --zips-dir ./fairface --out-dir ./fairface_preds \
      --num-workers 4 --batch-size 64 --duration 1.0 --fps 2
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path

# infer_face_bulk.py must be importable -- same directory as this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

def get_tmp_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())

def process_category_zip(model, zip_path: Path, out_dir: Path, tmp_root: Path,
                          duration: float, fps: int, batch_size: int):
    category = zip_path.stem
    out_path = out_dir / f"{category}.npz"
    if out_path.exists():
        print(f"[SKIP] {category} (already done)")
        return
 
    age, gender, race = parse_category(category)
 
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tribe_{category}_", dir=tmp_root))
    print(f"[{category}] temp dir: {tmp_dir}")
 
    all_names = []
    all_means = []
    failed = []
 
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
            print(f"[{category}] {len(members)} images, batch_size={batch_size}")
 
            for batch_start in range(0, len(members), batch_size):
                batch_members = members[batch_start:batch_start + batch_size]
 
                rows = []               # (clip_path, timeline_id)
                timeline_to_name = {}   # timeline_id -> original filename
 
                for i, name in enumerate(batch_members):
                    timeline_id = f"img_{batch_start + i}"
                    try:
                        img = decode_image_from_zip(zf, name)
                        clip_path = tmp_dir / f"{timeline_id}.mp4"
                        write_static_clip(img, clip_path, duration=duration, fps=fps)
                        rows.append((clip_path, timeline_id))
                        timeline_to_name[timeline_id] = name
                    except Exception as e:
                        print(f"  [ERROR building clip] {name}: {e}")
                        failed.append(name)
 
                if not rows:
                    continue
 
                df = make_multi_row_df(rows, duration=duration)
                try:
                    preds, segments = model.predict(events=df)
                except Exception as e:
                    print(f"  [ERROR predict()] batch at {batch_start}: {e}")
                    failed.extend(timeline_to_name.values())
                    for clip_path, _ in rows:
                        clip_path.unlink(missing_ok=True)
                    continue
 
                grouped = group_preds_by_timeline(preds, segments)
 
                for timeline_id, name in timeline_to_name.items():
                    vec = grouped.get(timeline_id)
                    if vec is None:
                        print(f"  [WARN] no prediction returned for {name}")
                        failed.append(name)
                        continue
                    all_names.append(name)
                    all_means.append(vec)
 
                for clip_path, _ in rows:
                    clip_path.unlink(missing_ok=True)
 
                done = min(batch_start + batch_size, len(members))
                print(f"  ...{done}/{len(members)}")
 
        if all_means:
            means_arr = np.stack(all_means, axis=0)
            np.savez_compressed(
                out_path,
                preds=means_arr,
                filenames=np.array(all_names),
                failed=np.array(failed),
                age=age, gender=gender, race=race,
            )
            print(f"[{category}] saved {means_arr.shape} -> {out_path} "
                  f"({len(failed)} failed)")
        else:
            print(f"[{category}] no successful predictions -- nothing saved")
 
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

from tribev2.demo_utils import TribeModel


def worker_main(worker_id: int, zip_paths: list, out_dir: Path, cache_folder: Path,
                 duration: float, fps: int, batch_size: int):
    """
    Runs in a spawned subprocess. Loads its own TribeModel instance (CUDA
    contexts do not survive fork, hence 'spawn' start method below) and works
    through its assigned slice of category zips sequentially.
    """
    print(f"[worker {worker_id}] loading TribeModel, {len(zip_paths)} zips assigned")
    tmp_root = get_tmp_root()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)

    for zip_path in zip_paths:
        print(f"[worker {worker_id}] -> {zip_path.name}")
        try:
            process_category_zip(
                model, zip_path, out_dir, tmp_root,
                duration=duration, fps=fps, batch_size=batch_size,
            )
        except Exception as e:
            print(f"[worker {worker_id}] [ERROR] {zip_path.name}: {e}")

    print(f"[worker {worker_id}] done")


def main():
    parser = argparse.ArgumentParser(description="Parallel FairFace -> TribeV2 inference driver.")
    parser.add_argument("--zips-dir", default="./fairface")
    parser.add_argument("--out-dir", default="./fairface_preds")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2,
                         help="Start small (2-4) and compare wall-clock time before "
                              "scaling up -- GPU compute is shared, not VRAM.")
    args = parser.parse_args()

    zips_dir = Path(args.zips_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = Path(args.cache_folder)

    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zips found in {zips_dir}")
        sys.exit(1)

    print(f"Discovered {len(zip_files)} category zips, splitting across {args.num_workers} workers")

    # Interleaved partition (round-robin by index) rather than contiguous
    # chunks, so if categories vary in image count, work is spread more
    # evenly rather than one worker getting all the largest categories.
    shards = [zip_files[i::args.num_workers] for i in range(args.num_workers)]
    for i, shard in enumerate(shards):
        print(f"  worker {i}: {len(shard)} zips")

    ctx = mp.get_context("spawn")  # required: CUDA contexts don't survive fork
    procs = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(
            target=worker_main,
            args=(i, shard, out_dir, cache_folder, args.duration, args.fps, args.batch_size),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\nAll workers finished.")
    print(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()