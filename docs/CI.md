# PDEnsorflow CI/CD architecture

Maintainer reference for how continuous integration is wired up. It documents a
deliberate two-repository, two-tier design and the day-to-day operations (sync,
release, token renewal, reacting to a failed nightly).

> TL;DR — **public repo = source of truth + fast CPU CI + (future) release/CD;
> private mirror = GPU nightly only.** Code flows *public → mirror* through a
> review-gated sync; nothing flows back.

---

## 1. Two repositories, and why

| Repo | Visibility | Role |
|------|------------|------|
| `cesare-corrado/PDEnsorflow` | **public** | canonical source; where everyone develops; Tier-1 CPU CI; future CD (PyPI/conda/Releases) |
| `cesare-corrado/PDEnsorflow-ci` | **private mirror** | runs the Tier-2 GPU nightly on a self-hosted runner; nothing is developed here |

The GPU nightly runs on a **self-hosted runner**, which executes arbitrary
workflow code. Exposing a self-hosted runner to pull requests from public forks
is a security risk, so the GPU job is kept off the public repo entirely and run
on a private mirror that fork PRs cannot reach. The mirror only ever *receives*
`main`/`develop` from public, through a pull request the maintainer approves.

---

## 2. Branch model

Git-flow (nvie): feature branches off `develop`, merged back `--no-ff`.
`develop → main` is a release: branch `release/x.y.z` off `develop`, bump the
version, merge to `main`, **tag `main`** (`vX.Y.Z`, semver), merge back to
`develop`. The tag on public `main` is the intended trigger for future CD.

On the mirror, `develop` is the **default branch** (required so the nightly's
`schedule`/`workflow_dispatch` register). The mirror's `develop` equals public's
`develop` **plus mirror-only files** (see §5) — reconciled by each sync merge.

---

## 3. Tier-1 — CPU CI (`.github/workflows/ci.yml`, public)

- **Runs on:** GitHub-hosted `ubuntu-latest`.
- **Trigger:** every `push` / `pull_request` to `main`/`develop` (incl. forks).
- **Jobs:** `ruff` (bug-catching lint) + CPU `pytest` (installs `tensorflow-cpu`,
  never the multi-GB CUDA wheels, then `pip install --no-deps -e .`).
- **Purpose:** fast pre-merge gate visible to every contributor. Catches the
  large majority of regressions before they land.

To make a red Tier-1 actually *block* a merge, enable branch protection on
`main`/`develop` (Settings → Branches → require the CI checks).

---

## 4. Tier-2 — GPU nightly (`.github/workflows/nightly.yml`, mirror)

- **Runs on:** self-hosted runner labelled `[self-hosted, gpu]` (host `ic-rosa`,
  RTX 5090 / Blackwell).
- **Trigger:** `schedule` `0 3 * * 6,0` (Sat & Sun 03:00 UTC — the box is free at
  weekends) + `workflow_dispatch`. **Never** on push, so fork PRs can't run on
  the GPU box.
- **Guard:** `if: github.repository == 'cesare-corrado/PDEnsorflow-ci'` — the file
  also exists on public (kept in sync) but is inert there.
- **Steps:** checkout → GPU sanity gate (`Tests/CICD/nightly/_gpu_check.py`,
  fails loudly if no physical GPU, so the suite can't silently fall back to CPU)
  → `pytest Tests/CICD/unit Tests/CICD/nightly` on the GPU (re-runs Tier-1 on the
  GPU so the device-gated `csr_axpby` cases execute, plus the 2D mMS
  conduction-velocity regression).

### Runner prerequisites (one-time, on `ic-rosa`)
- A self-hosted runner registered to the mirror with labels `[self-hosted, gpu]`,
  installed as a systemd service (survives reboot).
- Conda env `pdensorflow-gpu` (named via the workflow's `CI_CONDA_ENV`): a **GPU**
  TensorFlow build (`tensorflow[and-cuda]`), `pip install --no-deps -e .`, and an
  `activate.d` shim that exports `LD_LIBRARY_PATH` to the `nvidia-*-cu12` lib dirs
  + `$CONDA_PREFIX/lib` **at activation time** (so `gpuSolve/__init__.py` doesn't
  re-exec the interpreter mid-pytest).
- **Shared box → pin the GPU host-side:** set `CUDA_VISIBLE_DEVICES` in the
  runner's `.env` (e.g. `CUDA_VISIBLE_DEVICES=3`), *not* in the workflow — which
  card is free is a property of the host.

---

## 5. Public → mirror sync (`.github/workflows/sync-from-public.yml`, mirror-only)

Automates promoting public `main`/`develop` onto the mirror, review-gated.

- **MIRROR-ONLY file** — must never be pushed to public (guarded `if:` on the repo
  name). It's the mirror's own maintenance machinery.
- **Runs on:** GitHub-hosted `ubuntu-latest` (no GPU).
- **Trigger:** `schedule` `0 3 * * 5` (Fri 03:00 UTC, ~24 h before the weekend
  nightly, so there's a full day to review/merge) + `workflow_dispatch`.
- **What it does:** fetches public `main`/`develop` (no auth needed to read a
  public repo), force-pushes them to staging branches `sync/main`/`sync/develop`
  on the mirror, and opens a review PR `sync/<b> → <b>` for each branch that is
  behind public. The maintainer reviews and merges.

> **Merge sync PRs with a MERGE COMMIT, never "Squash and merge".** The mirror
> carries files public doesn't (this workflow, etc.); a merge commit preserves
> them, a squash-style overwrite can drop them.

---

## 6. Failure reporting — "lever-1" (`report-failure` job in `nightly.yml`)

Because the GPU nightly runs on a *private* mirror, a red run is otherwise
invisible to public contributors. On `failure()` only, a second job (on
`ubuntu-latest`, so the token never touches the GPU box) opens — de-duplicated by
commit SHA — a tracking **issue on the public repo**. The run-log and commit
links point at the mirror (where the run and the mirror-only commit actually
live); the public issue is the breadcrumb the maintainer clicks through. The job
no-ops safely if its token is absent.

---

## 7. Secrets (both live on the mirror)

| Secret | Scope | Used by | Notes |
|--------|-------|---------|-------|
| `MIRROR_SYNC_TOKEN` | mirror repo: Contents + **Workflows** + Pull requests (write) | sync workflow | Workflows perm is mandatory — see §9. Fine-grained (scoped to the mirror) or classic `repo`+`workflow`. |
| `PUBLIC_ISSUE_TOKEN` | public repo: Issues (write) | lever-1 | Fine-grained scoped to public; least-privilege. |

Renew before expiry (fine-grained max 1 yr unless you chose no-expiration);
rotate by regenerating and updating the secret value — no workflow change needed.

---

## 8. Runbook

**Sync public onto the mirror (usually just wait for the Friday run):**
Actions → *Sync from public* → Run workflow → `develop` → review the PR(s) →
merge with **Create a merge commit**.

**Cut a release (`develop → main`):** on public, `release/x.y.z` off `develop` →
bump the version (see the version note below) → merge to `main` → tag `vX.Y.Z` →
merge back to `develop` → push. The next sync carries `main` to the mirror.

**A nightly went red:** open the auto-filed public issue → follow the run-log link
(mirror) → if it's a real regression, `git revert` the offending commit on public
`develop` (that also re-triggers Tier-1) and note it on the issue.

**Version:** single source of truth in `gpuSolve/_version.py`
(`__version__ = ['1','3','2']` — a list, because the per-module `version()`
accessors iterate it). Every `gpuSolve` sub-package `__init__.py` and the outer
`PDEnsorflow/__init__.py` do `from gpuSolve._version import __version__`, and
`setup.py` parses it via `ast` (no import, so the build never pulls in
TensorFlow). **To bump the version, edit two spots only:** `gpuSolve/_version.py`
and the `README.md` title (the README is Markdown and can't import the value).

---

## 9. Gotchas worth remembering

- **The default `GITHUB_TOKEN` cannot push changes to `.github/workflows/*`** (a
  privilege-escalation guard). Since the sync legitimately carries workflow-file
  changes, it must checkout+push with a **PAT that has the Workflows
  permission** (`MIRROR_SYNC_TOKEN`). A workflow-file push *is* allowed when
  byte-identical to the default branch's copy — which is why a no-op sync can be
  green while the first real workflow change is rejected.
- **`gh pr create` fails in CI with "Head sha can't be blank"** when the head
  branch isn't checked out locally — it resolves the head from the local repo.
  Create PRs via `gh api repos/{}/pulls -f head=… -f base=…` instead, letting the
  server resolve the branch names.
- **Merge sync PRs with a merge commit, not squash** (§5).

---

## 10. Future: CD

Continuous delivery (PyPI / conda / GitHub Releases) belongs on the **public**
repo, triggered by the release **tag** on `main` — publishing must trace to the
canonical public repo (provenance, PyPI trusted publishing via OIDC), and it
needs no GPU. Guard any CD workflow with
`if: github.repository == 'cesare-corrado/PDEnsorflow'` so that, although the
sync copies it onto the mirror, it stays inert there (and the sync fetches
`--no-tags`, so tags never reach the mirror anyway). The private mirror remains
GPU-testing-only.
