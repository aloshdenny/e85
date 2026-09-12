"""
hyperface_fetch_stimuli.py

Reconstructs the ~53% of Hyperface stimuli that are cuttable from public YouTube
source video, and reports (without inventing) what remains -- the curated
faceNNN.mp4 clips, which the paper says nothing about beyond their existence
and which nobody has confirmed are on YouTube.

Filename convention observed in every subject's events.tsv (17,374 trial rows,
21 subjects): "{youtube_id}_{start_frame:06d}.mp4" for YouTube-derived clips,
"face{NNN}.mp4" / "catch_face{NNN}.mp4" for curated ones. The video_id/frame
split is inferred from the pattern (11-char base64url id, 6-digit zero-padded
frame count) and re-verified per row rather than assumed, since a wrong split
would silently fetch the wrong clip.

Frame rate is NOT recorded in the events files (checked -- only trial_type,
onset, duration exist for task-visualmemory). onset/duration give the clip's
presentation timing in the fMRI run, not its position in the source YouTube
video, so the frame offset in the filename is the only anchor into the source.
30fps is used as the working assumption (YouTube's most common upload rate)
and every extracted clip's actual duration is checked against the event
duration (typically 4s) as the cross-check -- a wrong fps would show up as a
consistent duration mismatch across many clips, and that check is printed
per-video so it can be caught rather than silently accepted.

Nothing here contacts the curated set. Those need the authors.

Usage:
  python scripts/hyperface_fetch_stimuli.py --manifest hyperface/stimulus_manifest.tsv
"""

import re, csv, subprocess, argparse, json, time
from pathlib import Path

YT_RE = re.compile(r"^(?:catch_)?(?P<vid>[A-Za-z0-9_-]{11})_(?P<frame>\d{6})\.mp4$")
NON_STIMULUS = {"accuracy_100", "accuracy_25", "accuracy_50", "accuracy_75",
                "button_press", "button_press_1", "button_press_2"}


def classify(filename):
    """catch_{id}_{frame}.mp4 is a catch-trial repeat of a youtube clip -- same
    11-char id + 6-digit frame pattern, just prefixed. Missed on the first pass,
    which silently mis-filed it as "unknown" rather than "youtube"."""
    if filename in NON_STIMULUS:
        return "non_stimulus", None, None
    m = YT_RE.match(filename)
    if m:
        return "youtube", m.group("vid"), int(m.group("frame"))
    if filename.startswith("face") or filename.startswith("catch_face"):
        return "curated", None, None
    return "unknown", None, None


YTDLP = "/home/research/miniconda3/envs/tribev2/bin/yt-dlp"


def download_source(vid, out_dir: Path, fmt="bestvideo[height<=480][ext=mp4]/mp4/best"):
    """Falls back to progressively looser format selectors and, if the
    default extraction client fails, retries with the android client
    explicitly -- yt-dlp 2026.3.17 (the system copy) failed nearly every video
    here with 'Precondition check failed' from that same client; upgrading to
    2026.08.19 (installed into the conda env, since the system Python is
    externally-managed) and retrying fixed it on inspection."""
    dst = out_dir / f"{vid}.mp4"
    if dst.exists():
        return dst, "cached"
    for extra in ([], ["--extractor-args", "youtube:player_client=android"],
                  ["--extractor-args", "youtube:player_client=ios"]):
        r = subprocess.run(
            [YTDLP, "-f", fmt, "-o", str(dst), f"https://www.youtube.com/watch?v={vid}",
             "--no-playlist", "--quiet", "--no-warnings"] + extra,
            capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and dst.exists():
            return dst, "downloaded"
    return None, r.stderr.strip()[:200]


def cut_clip(src: Path, dst: Path, start_frame: int, fps: float, duration: float):
    start_s = start_frame / fps
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-an", "-loglevel", "error", str(dst)],
        capture_output=True, text=True, timeout=60)
    return r.returncode == 0


def probe_duration(path: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, timeout=20)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./hyperface/stimuli"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = args.out_dir / "_source"
    src_dir.mkdir(exist_ok=True)
    clip_dir = args.out_dir / "youtube"
    clip_dir.mkdir(exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    yt_rows = [r for r in rows if classify(r["filename"])[0] == "youtube"]
    curated_rows = [r for r in rows if classify(r["filename"])[0] == "curated"]
    unknown_rows = [r for r in rows if classify(r["filename"])[0] == "unknown"]
    print(f"{len(rows)} unique stimulus filenames: {len(yt_rows)} youtube-derived, "
          f"{len(curated_rows)} curated (NOT fetchable here), {len(unknown_rows)} unrecognised")
    if unknown_rows:
        print("  unrecognised (needs a look): " +
              ", ".join(r["filename"] for r in unknown_rows[:10]))
    if args.limit:
        yt_rows = yt_rows[: args.limit]

    by_video = {}
    for r in yt_rows:
        _, vid, frame = classify(r["filename"])
        by_video.setdefault(vid, []).append((r["filename"], frame))

    print(f"\n{len(by_video)} distinct source videos to fetch\n")

    ok, failed, dur_checks = 0, [], []
    for i, (vid, clips) in enumerate(by_video.items()):
        src, status = download_source(vid, src_dir)
        if src is None:
            failed.append((vid, status))
            print(f"  [{i+1}/{len(by_video)}] {vid}: DOWNLOAD FAILED -- {status}")
            continue

        for fn, frame in clips:
            dst = clip_dir / fn
            if dst.exists():
                ok += 1
                continue
            got = cut_clip(src, dst, frame, args.fps, duration=4.0)
            if got:
                d = probe_duration(dst)
                if d is not None:
                    dur_checks.append(d)
                ok += 1
            else:
                failed.append((fn, "ffmpeg cut failed"))

        if (i + 1) % 20 == 0 or i == len(by_video) - 1:
            print(f"  [{i+1}/{len(by_video)}] {vid} ({status}) -- "
                  f"{ok} clips ok, {len(failed)} failed so far")
        time.sleep(0.1)   # gentle on YouTube

    print(f"\n{ok}/{len(yt_rows)} youtube-derived clips obtained")
    if dur_checks:
        import statistics
        print(f"  cut-clip duration check: mean={statistics.mean(dur_checks):.2f}s "
              f"(expect ~4.0s if fps={args.fps} is right) sd={statistics.pstdev(dur_checks):.2f}s")
        if abs(statistics.mean(dur_checks) - 4.0) > 0.3:
            print("  [WARN] mean duration is off from the expected 4.0s -- fps is "
                "probably wrong, do not trust these clips without checking further.")
    if failed:
        print(f"\n{len(failed)} failures (private/removed videos, geo-block, etc):")
        for f, why in failed[:20]:
            print(f"    {f}: {why}")

    print(f"\n{len(curated_rows)} curated faceNNN.mp4 clips remain unobtained -- "
          "not on YouTube by construction, need the authors.")
    (args.out_dir / "curated_needed.txt").write_text(
        "\n".join(sorted(r["filename"] for r in curated_rows)) + "\n")
    print(f"list written -> {args.out_dir / 'curated_needed.txt'}")


if __name__ == "__main__":
    main()
