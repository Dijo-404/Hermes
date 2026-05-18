# Gerrit Upload How-To — `lmkd` ML PSI Predictor Series

Procedural notes for pushing the three-commit upstream series
(drafts in [commit-messages.md](commit-messages.md)) to the AOSP
Gerrit instance.

## 1. One-time clone + remote setup

If you do not already have the upstream `lmkd` checkout:

```bash
git clone https://android.googlesource.com/platform/system/memory/lmkd
cd lmkd
git checkout -b ml-psi-predictor origin/main
```

If you are working out of this fork and want to push *this* branch's
`lmkd.cpp`, `ml_predictor.{h,cpp}`, and `Android.bp` changes upstream,
add the AOSP Gerrit as a second remote on the existing clone:

```bash
# Replace <username> with your registered Gerrit username.
git remote add gerrit \
    https://<username>@android.googlesource.com/platform/system/memory/lmkd
```

Confirm both remotes resolve:

```bash
git remote -v
# origin   <fork URL>                (fetch/push)
# gerrit   https://…/lmkd            (fetch/push)
```

## 2. Install the `commit-msg` hook

Gerrit requires a `Change-Id` trailer on every commit. The
`commit-msg` hook generates and inserts it automatically on
`git commit` / `git commit --amend`.

```bash
scp -p -P 29418 <username>@android-review.googlesource.com:hooks/commit-msg \
    .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

Verify by amending any commit on the branch: a `Change-Id: Ixxxxxxxx`
line should now appear at the bottom of the message.

If you have already authored commits without the hook, install it and
run:

```bash
git rebase --exec 'git commit --amend --no-edit' origin/main
```

to backfill `Change-Id` trailers across the series.

## 3. Build the three commits

Apply the changes commit-by-commit (do **not** squash):

1. **Commit A — scaffolding.** Stage `ml_predictor.h`,
   `ml_predictor.cpp`, the `lmkd_ml_defaults` block additions in
   `Android.bp`. Do **not** stage `lmkd.cpp` changes yet. Commit using
   the body from `commit-messages.md` § "Commit A".
2. **Commit B — injection point.** Stage the `#ifdef LMKD_USE_ML`
   block in `lmkd.cpp` (`lmkd.cpp:2936-2989`) only. Commit using
   `commit-messages.md` § "Commit B".
3. **Commit C — toggle + logging.** Stage the
   `PSIPredictor::init_from_properties()` call site in `lmkd.cpp`
   (around `lmkd.cpp:4261`) plus any property-reading code that
   wasn't already pulled in by Commit A. Commit using
   `commit-messages.md` § "Commit C".

Replace each `Bug: <aosp-bug-id>` placeholder with the real Buganizer
ID before pushing.

## 4. Push to Gerrit

```bash
git push gerrit HEAD:refs/for/main
```

(`refs/for/<branch>` is Gerrit's "for-review" magic ref — it creates
a change rather than fast-forwarding the branch.) The push output
prints three URLs, one per change in the series.

## 5. Add reviewers

Convention on the AOSP `lmkd` tree:

- Add the `lmkd` OWNERS reviewers (see `OWNERS` at repo root) as
  required reviewers.
- For ML/inference review, also CC anyone listed in the
  ONNX Runtime Android `METADATA` reviewers if the `Android.bp`
  `libonnxruntime` dep is the first such use in the consuming
  product.
- Use the Gerrit Web UI (`+2` for OWNERS, `+1` from peers) rather
  than `git push -o r=...` — the UI keeps a cleaner audit trail.

## 6. Iterate on review feedback

For each follow-up patch set:

```bash
# Edit files, then for the commit being revised:
git commit --amend            # commit-msg hook preserves Change-Id
git push gerrit HEAD:refs/for/main
```

If reviewers request changes spanning multiple commits in the series,
use `git rebase -i origin/main` locally to edit each in turn, then
push the whole series again.

## What NOT to push

The upstream patches are strictly the daemon source change. Keep the
following **on the fork only** — they are research artifacts, not
production code:

- `research/data/*.csv` — raw and labeled PSI samples (personally
  identifiable in some scenarios; large; device-specific).
- `research/results/` — bench outputs.
- `research/runs/`, `research/*.pt`, `research/*.onnx`,
  `research/normalization.json` — trained model artifacts.
- `plan.md`, `plan-executable.md`, `README_research.md`,
  `research/notes/`, `research/upstream/` — internal planning docs.

The `research/.gitignore` (see repo root) covers the binary and CSV
families. Verify nothing slipped in:

```bash
git status --ignored research/
# Should list data/, results/, runs/, *.onnx, *.pt, normalization.json
# under "Ignored files".
```

If a stray artifact is *already tracked* on this branch, drop it
before constructing the upstream commits:

```bash
git rm --cached research/data/heavy-tab-switch-*.csv
git commit -m "fork-only: untrack research dataset"
# (this commit stays on the fork, never goes to Gerrit)
```

## Sanity check before pushing

```bash
# 1. Series applies on top of upstream main:
git fetch gerrit main
git rebase gerrit/main

# 2. Build both configurations:
m lmkd                              # ML defaults disabled (default)
# flip Android.bp lmkd_ml_defaults to enabled: true, then:
m lmkd

# 3. Run the test suite:
atest lmkd_test

# 4. No invented fields / wrong headers / dep leaks:
grep -n "oom_score_adj" statslog.h statslog.cpp        # zero matches
grep -n "vmpressure_level" include/lmkd.h              # zero matches
grep -n "libonnxruntime" Android.bp | grep -v "lmkd_ml_defaults"
                                                        # zero matches
```

Only after all four are clean: `git push gerrit HEAD:refs/for/main`.
