#!/usr/bin/env python3
"""Append a catch-up handoff turn to a Claude Code session JSONL.

Adds one user message (the brief), one assistant acknowledgement, and rewrites
the trailing last-prompt record so Claude resumes from the new leaf.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

SESSION = Path(
    "/Users/aoxo/.claude/projects/-Users-aoxo-vscode/"
    "61cc8969-1f65-4925-8285-3ad33d76708e.jsonl"
)
SESSION_ID = "61cc8969-1f65-4925-8285-3ad33d76708e"
VERSION = "2.1.237"
CWD = "/Users/aoxo/vscode"

BRIEF = r"""
=== CATCH-UP BRIEF: TRIBE v2 identity face-suppression (work done in Cursor, resuming here) ===

Read this whole thing before acting. It replaces any older context about Mia/Sins in this session.

--------------------------------------------------------------------------------
1. GOAL
--------------------------------------------------------------------------------
Suppress TRIBE v2's face-cortex (OFA/FFA) response to a SPECIFIC PERSON, including
photos the model treats as that person. Identity is the trigger, not photography.

* Encoder stays frozen (1B params untouched).
* All we train is a rank-16 low-rank residual on the 2048 -> 20484 readout:
  W = W0 + U @ V, with U (2048x16) and V (16x20484).
* Success criteria (all three):
    - that person's HELD-OUT photos drop in FACE(OFA+FFA)
    - random FairFace faces do NOT move (no global face damp)
    - V1 / AUD / MOTOR do not move at all

Second goal added later: prove the recipe is person-agnostic by running the SAME
pipeline independently for a second person (Johnny Sins) with a FRESH residual --
NOT by applying Mia's residual to him.

--------------------------------------------------------------------------------
2. MACHINE POLICY (important)
--------------------------------------------------------------------------------
* All heavy compute runs on the RTX 4090 box: research@100.86.165.70 -> WSL.
  Mac only ships scripts and reads logs.
* Working tree on the box is NOT a git clone: /home/research/e85/
* Deploy pattern that works:
      scp <files> research@100.86.165.70:w4090/
      ssh research@100.86.165.70 "wsl bash /mnt/c/Users/research/w4090/<script>.sh"
  Do NOT use `ssh host "wsl bash -lc '...'"` with pipes / head / wc / quotes --
  Windows OpenSSH mangles them. Always write a .sh file, scp it, then run it.
  (This bit us repeatedly. Every probe/print helper is a committed .sh for this reason.)
* Conda env on the box: tribev2
* HF cache: /home/research/.cache/huggingface
* Scratch (all logs live here): /home/research/e85_scratch
* FairFace fused zip on box: /home/research/e85/data/fairface_ffhq.zip (~960MB, 101698 imgs)
* FairFace bottleneck cache (REUSE THIS, saves ~20 min):
      /home/research/e85_scratch/v2_bottlenecks.npz   (X_gen 500x2048, W0, b0)
* Linux disk /dev/sdd has ~483GB free. Windows C: is ~99% full -- never leave big files there.
* GitHub: git@github.com:aloshdenny/e85.git (main). Local repo: /Users/aoxo/vscode/e85

** AS OF THIS WRITING THE 4090 IS POWERED OFF. ** The user will say when it is back.

--------------------------------------------------------------------------------
3. THE SCRIPT THAT DOES THE EXPERIMENT
--------------------------------------------------------------------------------
scripts/mia_suppress_readout_v2.py  (misleading name -- it is now generic for any person)

How it works:
* Bottlenecks: monkeypatch predictor.forward to identity, run model.predict(),
  capture the 2048-d X. Predictions = X @ (W0 + U@V) + b0.
* Use find_fmri_encoder(cache_folder) from mia_suppress_readout.py.
  Do NOT import the NOD find_fmri_encoder -- it ignores the cache arg.
* Anchor set = 500 FairFace/FFHQ portraits (real faces), NOT COCO scenes.
  v1 used COCO and produced a global face damp; that is why v2 exists.
* "Lookalikes" = cosine similarity in the 2048-d TRIBE bottleneck to the target
  centroid. Top 25 are HELD OUT of training (never trained on) as a leakage probe.
  Note: InsightFace lookalikes did NOT spike; TRIBE-space neighbours DID.
* Loss:
      lam_suppress   * (target FACE -> target_face)^2
    + lam_anchor_face* (general-face FACE pinned to frozen baseline)^2   [lam 20, hard pin]
    + lam_anchor     * (full-vector + target non-face pinned to baseline)^2
* Objective auto-selects (this was added for Sins):
    - if target FACE > FairFace mean + 0.003  -> pull to FairFace mean (Mia case)
    - else                                     -> pull to self_mean - --min-drop
      (default 0.015; Sins case, he does not spike so "close the gap" is ill-posed)
* Verdict is computed against the INTENDED move, not against the FairFace gap.

CLI flags that matter:
  --target-zip / --mia-zip    which gallery
  --person NAME
  --split-mode {random,in_the_wild,face_top}   <-- I ADDED THIS, see section 7
  --holdout-frac 0.2
  --min-drop 0.015
  --bottleneck-cache /home/research/e85_scratch/v2_bottlenecks.npz
  --general-zip /home/research/e85/data/fairface_ffhq.zip
  --out abliterated/<name>.npz
  --save-preds target_preds/<name>.npz
Cache behaviour: when the cache exists AND a target zip is given, it reuses X_gen/W0/b0
but RE-CAPTURES the target's bottlenecks. It must never reuse a different person's X_st.

Applying a residual elsewhere: fe.predictor.weights[0] += U @ V
(see part_occlusion_map.py --readout-lowrank)

--------------------------------------------------------------------------------
4. RESULT HISTORY -- MIA
--------------------------------------------------------------------------------
v1 (abliterated/mia_suppress_readout.npz) -- DO NOT USE for identity claims.
  Anchor was NOD COCO scenes. Wild FACE +0.075 -> +0.062, BUT random FairFace
  dropped just as much (-0.014). That is a global FACE damp, not identity.

v2 original, 201-image gallery, holdout = the 26 fixed in_the_wild/ crops: THE GOOD RESULT
  | set                        | before  | after   | delta   |
  | wild Mia holdout           | +0.075  | +0.063  | -0.012  |
  | TRIBE-neighbour holdout    | +0.077  | +0.070  | -0.006  |
  | random FairFace holdout    | +0.071  | +0.070  | -0.001  |
  V1/AUD/MOTOR: exactly zero. Verdict: OK, identity-selective.
  Log: target_preds/mia_suppress_v2.log
  Interactive 3-way map: interactive_study/suppress_toggle.html

Other Mia findings from earlier:
* Part-occlusion: identity rides the face OUTLINE (forehead/jawline). The FFA
  forehead signature SURVIVED mean suppression (t=+2.94) under the v1 readout.
* Layer ablation and the Hyperface gate were essentially null -- do not rebuild a GLM from those.
* Photometry / attribute-matched controls suggested studio-vs-snapshot explains part of the
  RAW gap, but the user rejected "it's just photography": casual crops were still elevated.

--------------------------------------------------------------------------------
5. RESULT HISTORY -- JOHNNY SINS (the person-agnostic proof)
--------------------------------------------------------------------------------
Started from 38 crops I made from the user's Downloads (scripts/crop_sins_from_downloads.py:
eye-line rotate + square crop + mean-colour pad; 1 wrong face rejected).

KEY FACT: Sins does NOT spike. His frozen FACE sits at or below the FairFace mean.
So the original "pull to FairFace mean" objective actually RAISED him. That is why
--min-drop exists: for a non-spiking identity we pull them to self_mean - 0.015 and
check whether that move stays selective.

Sins across gallery sizes (fresh residual each time, never Mia's U/V):
  |  N imgs | train/holdout | holdout FACE before -> after | delta   | randFF delta | verdict |
  |     38  |   30 /   8    | +0.067 -> +0.050             | -0.0166 |   -0.001     | OK      |
  |    298  |  238 /  60    | +0.074 -> +0.059             | -0.0148 |   -0.001     | OK      |
  |    477  |  382 /  95    | +0.0726 -> +0.0587           | -0.0139 |   -0.0009    | OK      |
Every run: V1/AUD/MOTOR exactly zero, TRIBE-neighbour drop ~-0.001 to -0.002.

=> The technique holds on a second identity and it holds as the gallery grows 12x.
   This is the headline "works on any person" evidence.

--------------------------------------------------------------------------------
6. THE DATA HARVEST PIPELINE I BUILT (new work, all on the Mac)
--------------------------------------------------------------------------------
The user wanted ~1000 images each and did not want to keep saving Pinterest by hand.
I checked for ready-made datasets first: there is NO public Mia/Sins face dataset.
CelebA / VGGFace2 / HF celebrity-1000 do not contain either of them (verified: celeb-1000's
997 labels include Johnny Depp, Johnny Knoxville, Mia Wasikowska -- not our two).
LAION is the only real "dataset" angle, searchable by caption.

scripts/harvest_target_faces.py -- incremental, resumable, dedup-heavy
  search -> download -> sha256 dedup -> InsightFace identity filter -> upright 256px crop
  -> perceptual-hash dedup -> embedding dedup -> manifest.json -> optional zip rebuild
  Backends: DuckDuckGo images, Bing (icrawler), and LAION-5B captions via
  clip-retrieval's free hosted knn index (knn5.laion.ai). --backend all is the default.
  IMPORTANT: it does NOT download LAION; it only queries the index and fetches individual URLs.
  Identity centroid comes from the EXISTING target/<person>.zip, so it self-bootstraps.
  State: target/harvest/<person>/{manifest.json,crops/,rejected/,raw/,quarantine/}
  Flags: --person --query-set {default,scout2} --per-query --max-new --backend
         --ingest-dir --rebuild-zip --target-count

scripts/harvest_queries.json -- query lists.
  "queries"/"laion_queries"          = round 1 (generic portrait/interview/pinterest terms)
  "scout2_queries"/"scout2_laion_queries" = round 2, deliberately DIFFERENT sources:
     Mia:  reddit, flickr, wikimedia, newsweek, vogue.in, imdb, tumblr, celebmafia,
           and named events (Kenzo menswear, Schiaparelli, Chanel Milan, ESPN podcast,
           Sheytan launch, adidas LA party, purepeople, hawtcelebs)
     Sins: reddit, flickr, twitter, twitch, cameo, JRE, conventions, imdb, tumblr,
           and the costume memes (plumber / doctor scrubs / firefighter / lift-carry)

scripts/clean_target_gallery.py -- NEW, gallery hygiene
  Drops: junk/AI-art URLs (stablediffusionweb, wixmp, artstation, wallpapers.com,
  fineartamerica, "digital portrait", ...), no-face crops, InsightFace cosine < --sim-min
  (default 0.42) against a centroid built from non-junk members, and over-smooth images
  (Laplacian variance < --lap-min 35, catches airbrushed/AI renders), plus phash dupes.
  Everything removed is copied to target/harvest/<person>/quarantine/ so it is reviewable.
  Writes clean_report.json and backs the zip up to <person>.zip.bak_pre_clean.

requirements-harvest.txt : ddgs, icrawler, clip-retrieval, imagehash, Pillow, requests
  (Mac-side only. Do NOT install these into the tribev2 env on the 4090.)

Harvest outcomes:
  Mia : 201 -> 451 (round 1, +250) -> 751 (scout2, +300) -> CLEANED to 600
        clean dropped 151: 102 no_face, 26 junk_url, ~18 too_smooth, ~6 low_sim
  Sins:  38 ->  48 (smoke) -> 298 (round 1, +250) -> 477 (scout2, +179)
  Current: target/mia.zip = 600 images, target/sins.zip = 477 images.
  Manifests: 550 kept crops for mia, 439 for sins.

Operational gotcha learned the hard way: running the harvester with
--ingest-dir ~/Downloads HANGS (it walks the whole tree). Also two harvest
processes at once die during InsightFace init. Run ONE at a time, no Downloads ingest.
Nohup-wrapped chain scripts kept dying silently; launching each stage as its own
foreground/background command was the only reliable pattern.

--------------------------------------------------------------------------------
7. THE OPEN PROBLEM (this is what to work on next)
--------------------------------------------------------------------------------
With the bigger, cleaned gallery, MIA REGRESSED while Sins stayed fine.

Mia, 600-image cleaned gallery, holdout = random 20% (n=120):
    holdout FACE  +0.07167 -> +0.07011   (-0.00156)
    random FairFace                       (-0.00156)   <-- same size, that is the problem
    TRIBE-neighbour                       (-0.00215)
    VERDICT: FAIL -- global FACE damp

Diagnosis: it is NOT that identity-suppression broke. It is that there is almost
nothing left to suppress. Her holdout mean is only ~+0.002 above the FairFace anchor
mean (+0.06942), because the harvest added hundreds of ordinary press photos that sit
at the population mean and dragged her centroid down. A random 20% holdout is now
mostly average-activation photos. The original good result used a FIXED holdout of the
26 in_the_wild/ crops, which genuinely spiked.

So I added --split-mode with three options and launched an A/B on the 4090:
   in_the_wild : holdout = only members whose path contains "in_the_wild"
                 (600-image zip still has 22 of them; the pre-clean 751 zip had 26)
   face_top    : holdout = the top 20% of the gallery by FROZEN-readout FACE
                 (i.e. hold out the photos that actually spike, ~120 images)
   random      : the old behaviour (default)

Scripts for it:
   scripts/run_mia_suppress_v2_inwild.sh     -> abliterated/mia_suppress_readout_v2_inwild.npz
   scripts/run_mia_suppress_v2_face_top.sh   -> abliterated/mia_suppress_readout_v2_face_top.npz
   scripts/run_mia_split_compare.sh          -> runs both, then prints a scored comparison
                                                (OK beats FAIL; then bigger holdout drop;
                                                 then smaller random-FairFace movement)
   scripts/launch_mia_split_compare.sh       -> copies into /home/research/e85 and setsid-launches
Logs it will write:
   /home/research/e85_scratch/mia_suppress_v2_inwild.log
   /home/research/e85_scratch/mia_suppress_v2_face_top.log
   /home/research/e85_scratch/mia_split_compare.log   (the scored summary)

** THIS RUN WAS INTERRUPTED BY THE 4090 BEING POWERED OFF. It had just printed
   "=== MIA split compare: in_the_wild ===" and nothing else. Assume it produced nothing.
   Re-launch it when the box is back. **

Caveat to check when reading the face_top result: holding out the top-FACE images is
a biased split by construction (regression to the mean will show up as a drop even with
a null residual). The honest control is to also run face_top with lam_suppress=0, or to
compare against a matched random holdout of the same mean FACE level. Do not report
face_top as a win without that control. in_the_wild is the cleaner scientific split
because the holdout was defined by provenance, not by the metric being optimised.

--------------------------------------------------------------------------------
8. RESUME COMMANDS (when the 4090 is back up)
--------------------------------------------------------------------------------
# 0. sanity
ssh research@100.86.165.70 "wsl bash /mnt/c/Users/research/w4090/probe_both_suppress.sh"

# 1. relaunch the Mia split A/B (mia.zip 600 is already on the box, but re-scp to be safe)
cd /Users/aoxo/vscode/e85
scp target/mia.zip scripts/mia_suppress_readout_v2.py \
    scripts/run_mia_suppress_v2_inwild.sh scripts/run_mia_suppress_v2_face_top.sh \
    scripts/run_mia_split_compare.sh scripts/launch_mia_split_compare.sh \
    research@100.86.165.70:w4090/
ssh research@100.86.165.70 "wsl bash /mnt/c/Users/research/w4090/launch_mia_split_compare.sh"

# 2. read the scored comparison
ssh research@100.86.165.70 "wsl bash -c 'tail -40 /home/research/e85_scratch/mia_split_compare.log'"

# 3. promote the winner to the canonical name, then pull artifacts back to the Mac
#    (abliterated/mia_suppress_readout_v2.npz)

Existing helper scripts on the box for reading results (write new ones the same way,
never inline quoted wsl bash -lc):
   scripts/probe_both_suppress.sh    -- gpu + procs + both log tails
   scripts/print_both_verdicts.sh    -- pretty ROI tables + verdicts for mia & sins
   scripts/stage_sins_v2b_out.sh     -- copy artifacts to the Windows share for scp

--------------------------------------------------------------------------------
9. FILE / STATE INVENTORY (Mac, /Users/aoxo/vscode/e85)
--------------------------------------------------------------------------------
DATA
  target/mia.zip                600 cleaned crops   (MODIFIED, uncommitted)
  target/sins.zip               477 crops           (MODIFIED, uncommitted)
  target/mia.zip.bak_pre_clean  751, pre-clean backup
  target/harvest/mia/           manifest.json (550 kept), crops/, rejected/, quarantine/,
                                clean_report.json, harvest_run.log
  target/harvest/sins/          manifest.json (439 kept), crops/, rejected/
  target/harvest/*.log          pipeline / scout2 logs
  target_preds/mia.npz          MODIFIED -- note: the 4090 rewrote this to 600x20484
  target_preds/sins.npz         committed earlier at 38 rows; box now has 477 rows

RESIDUALS (local copies are STALE -- newest weights are only on the 4090)
  abliterated/mia_suppress_readout.npz        v1, COCO anchor, global damp, keep for rollback
  abliterated/mia_suppress_readout_v2.npz     local copy is the 451-image run (Aug 29 18:48)
  abliterated/sins_suppress_readout_v2.npz    local copy is the 38-image run  (Aug 29 19:34)
  -> the 600-image Mia and 477-image Sins residuals were NEVER pulled back. Pull them.

NEW SCRIPTS (all untracked)
  scripts/harvest_target_faces.py      the harvester
  scripts/harvest_queries.json         query sets incl. scout2
  scripts/clean_target_gallery.py      gallery cleaner
  scripts/run_harvest.sh               one-batch wrapper
  scripts/run_scout2_and_clean.sh      scout2 + clean chain (flaky when nohup'd)
  scripts/continue_post_harvest.sh     harvest -> deploy -> 4090 chain
  scripts/continue_scout2_after_mia.sh watcher variant
  scripts/run_mia_suppress_v2_inwild.sh
  scripts/run_mia_suppress_v2_face_top.sh
  scripts/run_mia_split_compare.sh
  scripts/launch_mia_split_compare.sh
  scripts/run_both_suppress_v2.sh      mia then sins, sequential
  scripts/launch_both_suppress_v2.sh   setsid launcher for the above
  scripts/probe_both_suppress.sh, print_both_verdicts.sh, print_sins_v2b.sh
  requirements-harvest.txt

MODIFIED, uncommitted
  scripts/mia_suppress_readout_v2.py   (+ --min-drop objective switch, + --split-mode)
  scripts/run_mia_suppress_v2.sh       (now uses --target-zip/--person/--holdout-frac)
  scripts/run_sins_suppress_v2.sh      (+ --min-drop 0.015)

Recent commits: 2b055b2dc (sins preds), da799c7da (sins crops + v2 script)
NOTHING from tonight's harvest/clean/split work is committed yet.

--------------------------------------------------------------------------------
10. RULES / DO-NOTs
--------------------------------------------------------------------------------
* Never apply Mia's residual to Sins or vice versa. Each person gets a fresh U/V.
* Do not retrain Mia's canonical residual unless asked; keep v1 for rollback.
* Do not report a suppression as identity-selective unless random FairFace stays
  within ~40% of the target's drop AND V1/AUD/MOTOR are zero.
* Cloud GPU rule from the user's workspace rules: billed pods bill until TERMINATED.
  If any RunPod/Vast pod gets used, state an ETA, arm a watcher, pull artifacts, then
  terminate. (Not relevant to the 4090, which is the user's own hardware, but the rule
  is in .cursor/rules and applies to the Superfloat pod b391evfuro5n4j.)
* Still pending from much earlier, unrelated: a Hyperface FAIL clip recut
  (8vuRTBwgMSE_000154.mp4). Not part of this thread.

--------------------------------------------------------------------------------
11. ONE-LINE STATUS
--------------------------------------------------------------------------------
Sins is proven (477 imgs, -0.0139 holdout, randFF -0.0009, OK). Mia regressed on the
grown gallery because her spike got diluted; an in_the_wild vs face_top holdout A/B was
launched to fix the split and was cut short by the 4090 powering off. Relaunch it,
pick the winner, pull the residuals back, and commit the harvest tooling.
""".strip()

ACK = (
    "Caught up. I have the full picture of the TRIBE v2 identity-suppression work:\n\n"
    "**Proven:** Johnny Sins is identity-selective at every gallery size tested "
    "(38 -> 298 -> 477 images), holding around -0.014 on his held-out photos while "
    "random FairFace moves under 0.001 and V1/AUD/MOTOR stay at exactly zero. That is "
    "the person-agnostic evidence.\n\n"
    "**Open:** Mia regressed on the grown 600-image gallery. Her held-out mean now sits "
    "only ~0.002 above the FairFace anchor, so the residual has almost nothing to close "
    "and the small move it does make is matched by random faces. That reads as gallery "
    "dilution from ordinary press photos, not a broken method.\n\n"
    "**Next action, blocked on hardware:** the in_the_wild vs face_top holdout A/B was "
    "launched on the 4090 and cut short when the box powered off. When it is back I will "
    "re-scp the 600-image zip plus the split scripts, run launch_mia_split_compare.sh, "
    "read mia_split_compare.log, and promote the winner. I will treat the face_top result "
    "with suspicion until I run the lam_suppress=0 control, since holding out the "
    "top-FACE images is biased toward showing a drop.\n\n"
    "I also need to pull the 600-image Mia and 477-image Sins residuals back to the Mac "
    "at some point; the local .npz files are stale copies from the 451- and 38-image runs.\n\n"
    "Tell me when the 4090 is up."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main() -> None:
    lines = SESSION.read_text().splitlines()

    tail_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if json.loads(lines[i]).get("type") == "last-prompt":
            tail_idx = i
            break
    if tail_idx is None:
        raise SystemExit("no last-prompt record found")

    parent = None
    for i in range(tail_idx - 1, -1, -1):
        rec = json.loads(lines[i])
        if rec.get("type") in ("assistant", "user") and rec.get("uuid"):
            parent = rec["uuid"]
            break
    if parent is None:
        raise SystemExit("no parent uuid found")

    user_uuid = str(uuid.uuid4())
    asst_uuid = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())
    common = {
        "isSidechain": False,
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": CWD,
        "sessionId": SESSION_ID,
        "version": VERSION,
        "gitBranch": "HEAD",
    }

    user_rec = {
        "parentUuid": parent,
        "promptId": prompt_id,
        "type": "user",
        "message": {"role": "user", "content": BRIEF},
        "uuid": user_uuid,
        "timestamp": now_iso(),
        "permissionMode": "bypassPermissions",
        "promptSource": "sdk",
        **common,
    }

    asst_rec = {
        "parentUuid": user_uuid,
        "message": {
            "model": "claude-sonnet-5",
            "id": f"msg_handoff_{asst_uuid[:12]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": ACK}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": len(ACK) // 4,
                "service_tier": "standard",
            },
        },
        "requestId": f"req_handoff_{asst_uuid[:12]}",
        "type": "assistant",
        "uuid": asst_uuid,
        "timestamp": now_iso(),
        "effort": "low",
        **common,
    }

    last_prompt = {
        "type": "last-prompt",
        "lastPrompt": "Catch-up brief: TRIBE v2 identity suppression, Mia + Sins status "
                      "and the pending Mia holdout split A/B on the 4090.",
        "leafUuid": asst_uuid,
        "sessionId": SESSION_ID,
    }

    backup = SESSION.with_suffix(".jsonl.bak_pre_handoff")
    if not backup.exists():
        shutil.copy2(SESSION, backup)

    out = lines[:tail_idx]
    out.append(json.dumps(user_rec, ensure_ascii=False))
    out.append(json.dumps(asst_rec, ensure_ascii=False))
    out.append(json.dumps(last_prompt, ensure_ascii=False))

    tmp = SESSION.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(out) + "\n")
    tmp.replace(SESSION)

    print(f"backup   -> {backup}")
    print(f"parent   -> {parent}")
    print(f"user     -> {user_uuid}  ({len(BRIEF)} chars)")
    print(f"assistant-> {asst_uuid}")
    print(f"lines    -> {len(lines)} to {len(out)}")


if __name__ == "__main__":
    main()
