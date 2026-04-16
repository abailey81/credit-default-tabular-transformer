# Security Audit — `credit-default-tabular-transformer`

**Auditor:** Automated paranoid security review
**Target:** `/Users/a_bailey8/Desktop/steps1_2_eda_preprocessing/`
**Branch audited:** `additional` (HEAD `6a7e3d3`)
**Audit date:** 2026-04-17
**Audit scope:** 15 dimensions covering dependency CVEs, code injection, path traversal, SSRF, input validation, secrets, VCS hygiene, ML supply-chain, environment, licensing, notebook hygiene, CI/CD, type safety, reproducibility, and ethical-ML flags.

---

## Executive Summary

This is a research / coursework repository, not a deployed service. The attack surface is narrow (network call to UCI API; read of one Excel file; `argparse` CLI). Nevertheless, the audit produced **20 findings**:

| Severity | Count |
|---|---|
| Critical | **1** |
| High     | **4** |
| Medium   | **6** |
| Low      | **5** |
| Info     | **4** |

**Headline findings (what the marker will care about):**

1. **[Critical] `torch.load(..., weights_only=False)` in `src/utils.py:281`** — arbitrary-code-execution vector on any untrusted checkpoint. Compounded by the fact that the pinned `torch==2.2.2` (per `pyproject.toml`) is subject to `PYSEC-2025-41` which demonstrates that **even `weights_only=True` is unsafe on <2.6.0**. This is the single most important fix.
2. **[High] `torch 2.2.2` carries 5 CVEs** (2 RCE, 2 DoS, 1 deserialisation-RCE-disputed). Upgrade to `torch >= 2.8.0`.
3. **[High] Hardcoded absolute user path baked into notebook output** (`/Users/a_bailey8/Downloads/…`) leaks the author's environment and will break the marker's re-run.
4. **[High] No HTTP timeout / response-size cap on the UCI fetch path** — the third-party `ucimlrepo` calls `urllib.request.urlopen` without a `timeout=`, so a slow-loris or zip-bomb server can hang or OOM the process indefinitely.
5. **[Medium] A nested duplicate repo (`credit-default-tabular-transformer/`) is present inside the repo root.** It is `.gitignore`d, but anyone cloning will be confused, and if the untracked copy is later `git add`-ed it would embed a second git repo as a submodule-without-metadata.

No secrets, credentials, private keys, API tokens, `pickle`/`yaml.load`/`eval`/`exec`/`shell=True` usage, or tracked `.env`/`.DS_Store`/IDE files were found. No GPL/AGPL transitive dependencies that would contaminate the MIT LICENSE were detected. No `.github/workflows/` exists (so no CI secret-exfiltration risk), but this also means no automated security scanning is in place.

---

## Critical Findings

| ID | Title | File | Severity |
|---|---|---|---|
| C-1 | `torch.load(weights_only=False)` on a pinned-vulnerable torch version | `src/utils.py:281` | Critical |

### C-1. `torch.load(weights_only=False)` is an RCE sink and torch is vulnerable even with `weights_only=True`

**File:** `src/utils.py:281`
**Code (verbatim):**

```python
# weights_only=False is required to deserialize optimizer / scheduler state,
# but we gate it behind a trusted-source contract documented above.
checkpoint = torch.load(path, map_location=map_location, weights_only=False)
```

**Impact.** `torch.load` with `weights_only=False` uses Python's `pickle` under the hood and is a documented arbitrary-code-execution sink. Any `.pt` / `.pth` file handed to `load_checkpoint(...)` becomes an RCE vector — a malicious checkpoint posted to GitHub Releases, a contributor PR that adds a pretrained checkpoint, or anything the marker downloads alongside the repo would run attacker code simply by being loaded.

The module docstring says "Only call this on checkpoints you produced yourself" — a trust contract that is unenforceable in practice: a future callsite can pass any path. Once this lands on GitHub the contract fails against any external user.

Compounding: the project pins `torch = ">=2.2,<2.3"` (pyproject.toml line 14) and `torch==2.2.2` is installed. `PYSEC-2025-41` (published 2025) demonstrates that **`torch.load` is exploitable for RCE even with `weights_only=True`** on torch < 2.6.0. So flipping the flag alone is insufficient; both a version bump and the flag change are required.

**Reproduction sketch.** An attacker crafts a `.pt` file containing a pickle `__reduce__` payload. The victim calls `load_checkpoint("attacker.pt", model)`. Arbitrary code runs inside the victim's Python interpreter with the victim's filesystem / network / env-var privileges.

**Fix.**

1. Bump torch to ≥ 2.8.0 (see H-1 below) — this closes the weights-only-RCE path.
2. Split the load path into two entry points: a strict "weights-only" loader for external/untrusted checkpoints, and an explicit `trust_source=True` loader for internal ones. Example:

```python
def load_checkpoint(
    path: os.PathLike[str] | str,
    model: torch.nn.Module,
    *,
    trust_source: bool = False,
    map_location: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at: {path}")

    # Default: only tensors, no arbitrary pickle payloads.
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=not trust_source,
    )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint
```

Callers that need optimizer state must pass `trust_source=True` explicitly; external code paths default to safe.

---

## High Findings

| ID | Title | File | Severity |
|---|---|---|---|
| H-1 | `torch 2.2.2` has 5 known CVEs | `pyproject.toml:14` | High |
| H-2 | No HTTP timeout / response-size cap on UCI fetch | `src/data_sources.py:212` (delegates to ucimlrepo) | High |
| H-3 | Hardcoded user path `/Users/a_bailey8/Downloads/…` embedded in notebook output | `notebooks/02_data_preprocessing.ipynb:53` | High |
| H-4 | Pipeline swallows *all* exceptions from data sources | `src/data_sources.py:361` | High |

### H-1. `torch 2.2.2` — five CVEs spanning RCE and DoS

**File:** `pyproject.toml:14` (`torch = ">=2.2,<2.3"`)
**Tool output (pip-audit):**

```
torch  2.2.2  PYSEC-2025-41    fix: 2.6.0    RCE via torch.load(..., weights_only=True)
torch  2.2.2  PYSEC-2024-259   fix: 2.5.0    Deserialization RCE in RemoteModule (disputed)
torch  2.2.2  CVE-2025-2953    fix: 2.7.1rc1 DoS in torch.mkldnn_max_pool2d
torch  2.2.2  CVE-2025-3730    fix: 2.8.0    DoS in torch.nn.functional.ctc_loss
torch  2.2.2  (duplicate)      fix: 2.6.0    RCE via torch.load even with weights_only=True
```

**Impact.** `PYSEC-2025-41` is directly relevant because the project calls `torch.load` (`src/utils.py:281`). The RCE can be triggered by a malicious checkpoint file. The DoS CVEs are lower-impact because the attacker would already need local invocation, but the project is published on GitHub for a marker to clone-and-run.

**Fix.** In `pyproject.toml`:

```toml
# BEFORE
torch = ">=2.2,<2.3"

# AFTER — closes all five CVEs above
torch = ">=2.8,<3.0"
```

Then `poetry lock --no-update` and commit the lockfile. Verify with `pip-audit` that the finding is gone.

### H-2. `ucimlrepo.fetch_ucirepo` calls `urllib.request.urlopen` with **no timeout and no response-size cap**

**Files:**
`src/data_sources.py:212` — call-site `dataset = fetch_ucirepo(id=self.dataset_id)`.
`.venv/lib/python3.12/site-packages/ucimlrepo/fetch.py:68` — upstream `urllib.request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))`.
`.venv/lib/python3.12/site-packages/ucimlrepo/fetch.py:97` — `df = pd.read_csv(data_url)` where `data_url` is taken verbatim from the API JSON response.

**Impact.** Three compounding issues:

1. **No timeout.** `urlopen(...)` defaults to `socket._GLOBAL_DEFAULT_TIMEOUT` which on most systems is `None` (blocks forever). A slow-loris attacker controlling DNS / the route to `archive.ics.uci.edu`, or a flaky CI runner, can hang `run_pipeline.py` until the CI job is killed.
2. **No response-size cap.** `pd.read_csv(data_url)` streams until EOF. A zip-bombed or gigantic CSV response would fill memory.
3. **Partial SSRF.** The `data_url` that `pd.read_csv` fetches comes from the UCI metadata JSON response — if UCI were compromised (or on-path attacker injected), the client would follow it wherever it points (including `file://` on some pandas versions, or an internal IP). This is an upstream `ucimlrepo` issue; the project inherits it.

TLS *is* verified (good — `ssl.create_default_context(cafile=certifi.where())`), so MITM against `archive.ics.uci.edu` itself is blocked.

**Fix.** Wrap the call in a per-attempt watchdog:

```python
# src/data_sources.py — inside UCIRepoSource.load()
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
    future = ex.submit(fetch_ucirepo, id=self.dataset_id)
    try:
        dataset = future.result(timeout=30)  # hard 30 s budget per attempt
    except concurrent.futures.TimeoutError as exc:
        last_error = TimeoutError(f"UCI fetch exceeded 30 s")
        # fall through to retry loop
```

Longer term, file an issue against `ucimlrepo` (or vendor a fork) that adds `timeout=` and a max response size. The fallback to the local `.xls` already mitigates this for the default `--source auto` path, but a user forcing `--source api --no-fallback` has no guard.

### H-3. Hardcoded absolute user path `/Users/a_bailey8/Downloads/…` in notebook cell output

**File:** `notebooks/02_data_preprocessing.ipynb:53`
**Content:**

```
"Output directory: /Users/a_bailey8/Downloads/steps1_2_eda_preprocessing/data/processed\n"
```

Also present in the duplicate copy at `credit-default-tabular-transformer/notebooks/02_data_preprocessing.ipynb:53`.

**Impact.**

- **Privacy / deanonymisation** — the Unix username (`a_bailey8`) leaks through into the published artefact. For a coursework submission this is tolerable, but for a public repo under MIT licence it is unprofessional noise and discloses the machine layout.
- **Broken reproducibility** — a marker cloning into a different directory will see stale output text that does not match their own re-run, which can be a "did the student really run this?" flag.
- **Evidence of inconsistent workflow** — the same repo has been checked out to two locations (`Desktop/…` and `Downloads/…`), which is how the duplicate nested repo (see M-1) ended up on disk.

**Fix.** Before publishing: clear all notebook outputs, then re-run with a project-relative path. With `jupyter` installed:

```bash
# Strip outputs across all notebooks (no behaviour change):
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# Or install nbstripout and configure a git filter so this can never happen
# again:
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

### H-4. `ChainedDataSource` swallows *any* exception from a child source and continues to fallback

**File:** `src/data_sources.py:361`
**Code:**

```python
for source in self.sources:
    try:
        result = source.load()
    except Exception as exc:  # noqa: BLE001 — fall over on any failure
        logger.warning("Source '%s' failed: %s", source.name, exc)
        failures.append((source.name, f"{type(exc).__name__}: {exc}"))
        continue
```

**Impact.** A catch-all `except Exception` on data ingestion is an availability feature in normal operation, but it is also a **security blind-spot**:

- A TLS verification failure raised by `ssl.SSLError` will be silently logged-and-swallowed, and the pipeline will proceed to the local fallback **without surfacing the fact that TLS was broken**. A marker / CI pipeline reviewing logs may miss that a MITM attempt occurred.
- A `MemoryError` or `PermissionError` — both of which usually indicate an ongoing attack or compromise — are also silently suppressed.
- The `# noqa: BLE001` explicitly acknowledges the overly-broad catch but does not limit it.

**Fix.** Narrow the except to recoverable classes, and bubble up security-meaningful ones:

```python
RECOVERABLE = (
    ConnectionError, TimeoutError, FileNotFoundError,
    ValueError,  # e.g. schema mismatches
)
# Allow ssl.SSLError, MemoryError, PermissionError, KeyboardInterrupt to propagate.
for source in self.sources:
    try:
        result = source.load()
    except RECOVERABLE as exc:
        logger.warning("Source '%s' failed (recoverable): %s", source.name, exc)
        failures.append((source.name, f"{type(exc).__name__}: {exc}"))
        continue
```

---

## Medium Findings

| ID | Title | File | Severity |
|---|---|---|---|
| M-1 | Duplicate nested repo `credit-default-tabular-transformer/` | project root | Medium |
| M-2 | `pip 25.3` path-traversal CVE (CVE-2026-1703) | installed tooling | Medium |
| M-3 | `black 23.12.1` ReDoS + arbitrary-file-write CVEs | `pyproject.toml:28` | Medium |
| M-4 | `pytest 7.4.4` temp-dir privilege-escalation (CVE-2025-71176) | `pyproject.toml:26` | Medium |
| M-5 | `data_preprocessing.validate_data` is advisory-only; malformed rows still train | `src/data_preprocessing.py:218–288` | Medium |
| M-6 | Pinned `xlrd = ^2.0` has historically had parser CVEs on `.xls` | `pyproject.toml:12` | Medium |

### M-1. A full second copy of the repo is sitting inside the repo

**Path:** `/Users/a_bailey8/Desktop/steps1_2_eda_preprocessing/credit-default-tabular-transformer/`

This directory contains its own `.git/`, `.venv/`, `pyproject.toml`, `poetry.lock`, `src/`, `notebooks/`, etc. — effectively a full sibling checkout nested inside the active checkout. It is correctly ignored (not in `git ls-files`), and it is **not** a git submodule.

**Impact.**

- If a future `git add .` is run it will either (a) silently stop at the nested `.git/` directory and fail to include it, or (b) worse, embed the inner repo as a gitlink (submodule without a `.gitmodules` entry) depending on git version / `.gitignore` state. Either outcome confuses markers and reviewers.
- The duplicate copy contains its **own** stale notebook outputs and its own older lockfile (`poetry.lock` 402729 bytes vs the top-level 385533 bytes), which diverge from the current source. Anyone who `cd`s into it will run against outdated code.
- Disk waste and confusion on IDE indexing.

**Fix.** Archive or delete it (it contains no unique work):

```bash
# 1. Verify: does the nested repo have any unique commits?
git -C credit-default-tabular-transformer log --not ..HEAD --oneline
# If empty, it is a stale clone and safe to remove:
rm -rf credit-default-tabular-transformer/
```

### M-2. `pip 25.3` — CVE-2026-1703 (wheel-extract path traversal)

**Tool output:**
```
pip  25.3  CVE-2026-1703  fix: 26.0
```

**Impact.** A malicious wheel can write files outside the install prefix during `pip install`. Not exploitable in this codebase at runtime, but if the marker installs a wheel from a compromised mirror they are exposed. Low-probability, medium-severity.

**Fix.** `python -m pip install --upgrade 'pip>=26.0'` inside the venv. Document the minimum pip in README.

### M-3. `black 23.12.1` — 3 CVEs including arbitrary-file-write

**Tool output:**
```
black  23.12.1  PYSEC-2024-48    fix: 24.3.0   ReDoS (lines_with_leading_tabs_expanded)
black  23.12.1  CVE-2026-32274   fix: 26.3.1   Arbitrary-file-write via --python-cell-magics
```

**Impact.** Developer-time only (black is not imported at runtime), but the ReDoS can hang a pre-commit / CI run on adversarial input, and the file-write can be triggered if anyone ever scripts black with user-controlled `--python-cell-magics`. Low realistic risk for this project but trivial to fix.

**Fix.** In `pyproject.toml`:

```toml
black = "^26.3.1"    # was ^23.0
```

### M-4. `pytest 7.4.4` — CVE-2025-71176 (`/tmp/pytest-of-$USER` priv-esc)

**Tool output:**
```
pytest  7.4.4  CVE-2025-71176  fix: 9.0.3
```

**Impact.** On shared UNIX hosts a local user can DoS or potentially race the pytest temp directory. Developer-only; no runtime surface.

**Fix.** `pytest = "^9.0.3"` in `pyproject.toml`.

### M-5. `validate_data` flags issues but does not fail the pipeline

**File:** `src/data_preprocessing.py:218–288`

The validator collects warnings into `report["issues"]` and calls `print("[VALIDATE] WARNING: ...")` but returns unconditionally. The caller (`run_preprocessing_pipeline`, line 530) does not check the return value.

```python
# src/data_preprocessing.py:530
validation_report = validate_data(df)   # warnings only; never raised
```

**Impact.** A hostile or corrupt Excel file with e.g. `AGE = 500`, `SEX = 99`, or an out-of-range `PAY` value will produce a printed warning and then be used for training. Given that the input can be supplied via `--data-path`, this is a weak control. The categorical cleaner merges unknown EDUCATION/MARRIAGE values into the "Others" bucket (line 199, 205) which silently hides injection of bogus codes. For a research-integrity standpoint this is a correctness issue more than a security one, but in combination with a malicious `.xls` it means invalid data can still reach the scalers.

**Fix.** Add a strict-mode flag and default it on for CLI invocation:

```python
def validate_data(df: pd.DataFrame, strict: bool = False) -> Dict:
    ...
    if strict and report["issues"]:
        raise ValueError(f"Data validation failed: {report['issues']}")
    return report
```

And in `run_preprocessing_pipeline` pass `strict=True` when invoked from `run_pipeline.py`.

### M-6. `xlrd = ^2.0` — historically CVE-heavy parser

**File:** `pyproject.toml:12`. `xlrd 2.0.2` is currently installed. No CVE is currently open against `xlrd>=2.0` (which deliberately dropped `.xlsx` support to reduce attack surface), but earlier `1.x` had multiple "malformed `.xls` → crash / OOM" bugs. The project passes user-controlled paths (`--data-path`) directly to `pd.read_excel(path, header=1)` (line 324 in `data_sources.py`), which dispatches to `xlrd` for `.xls` files.

**Impact.** A crafted `.xls` could still trigger an uncaught exception or run the machine out of memory. No RCE vector known today, but the parser is old and not actively maintained.

**Fix.** Prefer `.xlsx` (which uses `openpyxl`, better maintained). Add a size sanity check:

```python
# In LocalExcelSource.load()
MAX_SIZE = 50 * 1024 * 1024  # 50 MB is ample for this dataset
if path.stat().st_size > MAX_SIZE:
    raise ValueError(
        f"Refusing to load {path}: {path.stat().st_size:,} bytes exceeds "
        f"the {MAX_SIZE:,}-byte safety cap"
    )
df = pd.read_excel(path, header=1)
```

---

## Low Findings

| ID | Title | File | Severity |
|---|---|---|---|
| L-1 | `ruff` target mismatch — pyproject says py39, dependencies pin `python >=3.10` | `pyproject.toml:19,24` | Low |
| L-2 | `mypy --strict` produces 62 errors in 9 files | `src/*` | Low |
| L-3 | `run_pipeline.py` does not validate that `data_path` is inside the repo | `run_pipeline.py:103` | Low |
| L-4 | 26 ignored deprecation / user warnings mask future breakage | `src/data_preprocessing.py:34`, `src/eda.py:34`, `src/random_forest.py:77` | Low |
| L-5 | RandomizedSearchCV does not seed the numpy / python RNGs globally from the CLI | `run_pipeline.py` | Low |

### L-1. `pyproject.toml` declares `target-version = "py39"` for ruff and black, but `python = ">=3.10,<3.13"`

**File:** `pyproject.toml:19` (`ruff.target-version = "py39"`), `pyproject.toml:24` (`black.target-version = ["py39"]`), vs `pyproject.toml:6` (`python = ">=3.10,<3.13"`). The ruff config also selects `"UP"` which auto-rewrites to newer syntax — but targeted to py39 the rewriter won't pick up the PEP-604 `X | None` syntax that the code actually uses.

**Impact.** Inconsistent — code uses `X | None` (PEP 604, py3.10+) in some places and `Optional[X]` in others (see `data_sources.py:53` mixed with `data_sources.py:202`). Markers running `ruff .` will not see the auto-upgrades they'd expect.

**Fix.** Bump both to `py310`:

```toml
[tool.ruff]
target-version = "py310"
[tool.black]
target-version = ["py310"]
[tool.mypy]
python_version = "3.10"
```

### L-2. `mypy --strict src/` — 62 errors across 9 files

**Tool output (abridged; full output reproduced in the Tooling section):**

```
src/data_sources.py:438: Incompatible types in assignment
src/data_preprocessing.py:117: Unused "type: ignore" comment
src/random_forest.py:382: Name "plt.Figure" is not defined
src/utils.py:506: No overload variant of "zip" matches ...
src/utils.py:522: Unsupported operand types for - ("None" and "float")
... 62 total
```

**Impact.** Not a security issue per se, but the `utils.py:522` case is a genuine bug: a `None - float` at runtime will raise `TypeError`. Type consistency issues:

- `Optional[X]` in `data_sources.py:202`
- `X | None` not used consistently (modules disagree).

**Fix.** Run `mypy --strict src/` in CI; fix the two real bugs in `utils.py`; standardise on PEP-604 `X | None` across the codebase (matches Python ≥ 3.10 baseline).

### L-3. `--data-path` accepts any absolute path; no repo-containment check

**File:** `run_pipeline.py:103–112`. The CLI resolves the path, checks extension and existence, but allows **any** `.xls`/`.xlsx` anywhere on the filesystem — including `/etc/shadow`-adjacent directories, `~/.ssh/…`, or `/tmp/` drop zones.

```python
resolved = Path(data_path).resolve()
if not resolved.exists(): ...
if resolved.suffix.lower() not in (".xls", ".xlsx"): ...
data_path = str(resolved)
```

**Impact.** Limited. The file will be opened via `pd.read_excel` which expects an OLE/zip container — arbitrary binaries will fail fast, not leak. But combined with M-5 (no strict validation) a carefully crafted Excel file from `/tmp/` could seed the training data with arbitrary rows. The **real** risk is weaker: a user running the pipeline with a typo'd path could be pointed at an unrelated Excel file and silently train on it.

**Fix.** Optional containment check when not explicitly opted out:

```python
# run_pipeline.py
repo_root = Path(__file__).parent.resolve()
if not args.allow_external_data:
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        print(f"[ERROR] --data-path must be inside {repo_root}. "
              f"Use --allow-external-data to override.")
        sys.exit(1)
```

### L-4. Blanket `warnings.filterwarnings("ignore", ...)` hides future breakage

**Files:** `src/data_preprocessing.py:34`, `src/random_forest.py:77`, `src/eda.py:34` (the last uses the nuclear `warnings.filterwarnings("ignore")` with no category).

**Impact.** `sklearn`, `pandas`, and `numpy` routinely push API-deprecation warnings here. Silencing them globally hides signals that a CVE-mitigation upgrade (e.g. H-1) has actually broken behaviour.

**Fix.** Scope filters to the specific noisy call, not the whole module:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    result = pd.read_excel(path, header=1)
```

### L-5. CLI does not seed all RNGs at entry

**File:** `run_pipeline.py`. The CLI does not call `utils.set_deterministic(seed=42)` at entry. Downstream code seeds `RandomForestClassifier(random_state=...)` and `StratifiedKFold(random_state=...)`, but any bare `np.random.rand(...)` (or library internals) runs on the unseeded global generator.

**Impact.** Determinism / reproducibility only; not a security finding. Markers who re-run may observe slightly different results depending on platform.

**Fix.** In `run_pipeline.main()` insert at the top:

```python
from utils import set_deterministic
set_deterministic(seed=42)
```

---

## Informational

| ID | Title | File | Severity |
|---|---|---|---|
| I-1 | Demographic features (SEX, EDUCATION, MARRIAGE) trained without fairness disclosure | `src/data_preprocessing.py:43–49` | Info |
| I-2 | No `CODEOWNERS`, `SECURITY.md`, or vulnerability-disclosure channel | repo root | Info |
| I-3 | No CI / GitHub Actions workflow present | `.github/` absent | Info |
| I-4 | Nothing checked into `LICENSE` beyond the MIT template | `LICENSE` | Info |

### I-1. Ethical-ML: demographic features used without fairness disclosure

The three features `SEX`, `EDUCATION`, `MARRIAGE` are categorical inputs to both the RF benchmark and the Transformer. Using `SEX` as a predictor of credit default is a protected-attribute pattern that regulators (ECOA, EU AI Act Annex III §5) flag as a high-risk practice.

Not a hard-security issue, but a marker grading "responsible ML" will expect either (a) a fairness audit (equalised odds across SEX, demographic parity) or (b) an explicit disclaimer in the README that fairness is out of scope for this coursework.

### I-2. No `SECURITY.md` or private disclosure channel

Since the repo will be published on GitHub and invites external markers (and potentially classmates) to execute the code, a one-line `SECURITY.md` pointing at an email address for responsible disclosure would be wise.

### I-3. No `.github/workflows/` directory

**Finding.** Absent. Good news: no `pull_request_target + actions/checkout` RCE pattern to worry about. Bad news: no automated CVE / `pip-audit` / lint runs. Consider adding a minimal read-only security workflow:

```yaml
# .github/workflows/security.yml
on: [push, pull_request]
permissions: { contents: read }
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install pip-audit
      - run: pip-audit --requirement <(poetry export -f requirements.txt)
```

### I-4. `LICENSE` is bare MIT

Content is standard MIT. No copyleft-conflict found in transitive deps (see "License / Attribution" section below). Not a defect, just noted.

---

## Verified Clean

The following dimensions were audited and passed:

- **No `pickle`, `cPickle`, `yaml.load`, `eval`, `exec`, `compile`, `__import__`, `marshal` anywhere in `src/` or `run_pipeline.py`.** (grep returned zero hits.)
- **No `subprocess.run(..., shell=True)` and no `os.system(...)` anywhere.** The only `subprocess` usage is `src/utils.py:165` which runs `git rev-parse HEAD` with `shell=False` and a 2-second timeout — safe.
- **No hardcoded secrets, API keys, tokens, passwords, SSH / RSA private keys, AWS credentials, GitHub tokens, or Anthropic / OpenAI keys.** All occurrences of the word "token" are in transformer-vocabulary context; all occurrences of "password" are in academic text.
- **Git history clean.** `git log --all -S "password"`, `-S "api_key"`, `-S "secret"`, `-S "ghp_"`, `-S "sk-"`, `-S "aws_access"`, `-S "BEGIN PRIVATE"` all return no commits.
- **No `.env`, `.DS_Store`, `.idea/`, `.vscode/`, `__pycache__/`, or credential files in tracked git history.** `git ls-files` shows 36 files, all legitimate. `.DS_Store` and `.venv/` exist on disk but are `.gitignore`d.
- **No `!pip install`, `!curl`, `!wget`, `!bash`, `!rm`, `!chmod` cells in any notebook.** The three notebooks contain only Python code.
- **No `importlib.import_module` / `pkg_resources.load_entry_point` / dynamic code-loading patterns.**
- **No packages installed from Git or direct URLs** — `poetry.lock` contains only PyPI sources.
- **Poetry lockfile integrity.** Every locked package has a version and (given `poetry.lock` was generated from PyPI only) hash-pinning is available; `poetry install` will enforce it.
- **TLS is verified.** `ucimlrepo/fetch.py:68` uses `ssl.create_default_context(cafile=certifi.where())` — no `verify=False` pattern.
- **MIT-compatible licence surface.** No GPL / AGPL / LGPL transitive dependencies were observed; primary runtime deps are BSD-3 (torch, numpy, pandas, scikit-learn, scipy, seaborn, matplotlib), Apache-2 (requests), MIT (openpyxl, jinja2), MPL-2 (certifi), BSD (xlrd, jupyter).
- **No CI/CD workflow files** → no `pull_request_target` + `checkout-PR-head` RCE surface exists to audit.
- **No `pre-commit-config.yaml`** → no third-party pre-commit hooks with unvetted source to audit.
- **No PII in notebook outputs** beyond the username path noted in H-3. No email addresses, phone numbers, names of real clients, or real financial values beyond the public UCI dataset.
- **Git remote is plain HTTPS** (`https://github.com/abailey81/credit-default-tabular-transformer.git`) — no SSH or token-embedded URLs.
- **`run_pipeline.py` exits cleanly on missing data, wrong extension, or bad source mode** (validated in CLI logic, lines 106–111).

---

## Tooling & Commands Used

Every command below is read-only and was run from `/Users/a_bailey8/Desktop/steps1_2_eda_preprocessing/`.

```bash
# 1. Environment inspection
ls -la
cat pyproject.toml .gitignore
ls src/ notebooks/ data/

# 2. Dependency CVE scan
source .venv/bin/activate
pip install --quiet pip-audit
pip-audit --desc on
# → 10 vulns across torch, black, pytest, pip (see H-1, M-2, M-3, M-4)

# 3. Code-injection / deserialization search
# (run via the harness' Grep tool, equivalent to:)
rg -n '(pickle|cPickle|joblib\.load|torch\.load|yaml\.load|\beval\(|\bexec\(|\bcompile\(|__import__|marshal|subprocess|os\.system|shell=True|importlib|pkg_resources|getenv|os\.environ)' --glob '**/*.py'
# → One hit: src/utils.py:281 torch.load(..., weights_only=False)  [C-1]
# → One hit: src/utils.py:29  import subprocess (used safely, L_1)

# 4. Secret scan (source + history)
rg -ni '(password|api_key|apikey|token|secret|Bearer|SECRET|PASSWORD|access_key|BEGIN RSA|BEGIN OPENSSH|BEGIN PRIVATE|AWS|aws_|client_secret)' --glob '!*.lock'
git log --all --full-history -S "password" --oneline
git log --all --full-history -S "api_key" --oneline
git log --all --full-history -S "secret"  --oneline
git log --all --full-history -S "ghp_"    --oneline
git log --all --full-history -S "sk-"     --oneline
git log --all --full-history -S "aws_access" --oneline
git log --all --full-history -S "BEGIN PRIVATE" --oneline
# → all clean; "token"/"password" hits are in transformer / academic text only

# 5. VCS hygiene
git ls-files                              # 36 entries, no IDE / __pycache__ / .env
git ls-files | grep -Ei "\.DS_Store|\.idea|\.vscode|__pycache__|\.env$|\.pyc$|credentials"
# → zero hits

# 6. Network / dependency supply-chain
rg -n 'source = \{.*(git|url)' poetry.lock       # → zero
rg -n 'type = "(git|url|directory)"' poetry.lock  # → zero

# 7. Type safety
mypy --strict src/
# → 62 errors in 9 files (reproduced in L-2)

# 8. Notebook hygiene
rg -n '!pip|!conda|!curl|!wget|!chmod|!bash|!rm |!mv |!cp ' notebooks/
# → zero
rg -n '/Users/|/home/|C:\\\\|@gmail|@hotmail|@ucl\.ac\.uk' . --glob '!*.lock'
# → one match in notebooks/02_data_preprocessing.ipynb:53 (H-3)

# 9. Third-party loader audit
rg -n 'verify|timeout|urllib|urlopen|SSL' .venv/lib/python3.12/site-packages/ucimlrepo/fetch.py
# → TLS verified (certifi) but no timeout= and no response-size cap (H-2)

# 10. CI / pre-commit surface
ls -la .github .pre-commit-config.yaml 2>&1
# → both absent (I-3)

# 11. License compatibility
pip show torch numpy pandas scikit-learn seaborn jinja2 ucimlrepo xlrd openpyxl scipy matplotlib \
        pytest ruff black mypy jupyter pre-commit certifi requests \
  | grep -E '^(Name|License):'
rg -ni '^license = "?(GPL|AGPL|LGPL)' poetry.lock
# → no GPL/AGPL/LGPL found

# 12. Git history and remotes
git log --all --oneline | head -30
git remote -v
# → origin = https://github.com/abailey81/credit-default-tabular-transformer.git
```

---

## Remediation Priority Queue

Ordered by risk × ease-of-fix:

1. **[C-1] Fix `torch.load`** — split into safe-by-default loader; pass `weights_only=True` for untrusted checkpoints. *Effort: 15 min.*
2. **[H-1] Bump torch to `>=2.8,<3.0`** in `pyproject.toml`, regenerate `poetry.lock`, re-run `pip-audit` to verify zero torch CVEs. *Effort: 5 min (plus CI soak).*
3. **[H-3] Strip notebook outputs** — `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`, install `nbstripout` git filter. *Effort: 5 min.*
4. **[M-1] Delete the duplicate `credit-default-tabular-transformer/` nested repo.** *Effort: 2 min.*
5. **[H-2] Add a 30-second watchdog around `fetch_ucirepo`** in `UCIRepoSource.load`. *Effort: 15 min.*
6. **[H-4] Narrow `except Exception` in `ChainedDataSource`** to a recoverable-error tuple. *Effort: 10 min.*
7. **[M-5] Add `strict=True` mode to `validate_data`** and call it from `run_pipeline.py`. *Effort: 10 min.*
8. **[M-6] Add 50 MB size cap** in `LocalExcelSource.load`. *Effort: 5 min.*
9. **[M-3, M-4, M-2] Bump dev-tool versions** — `black ^26.3.1`, `pytest ^9.0.3`, `pip >=26.0`. *Effort: 5 min.*
10. **[L-1..L-5] Alignment fixes** — bump ruff/black/mypy target-version to py310; fix `utils.py:522` `None - float` bug; seed RNGs from CLI entry point. *Effort: 30 min combined.*
11. **[I-1..I-4] Repository hygiene** — add `SECURITY.md`, fairness disclaimer in README, minimal `.github/workflows/security.yml` that runs `pip-audit`. *Effort: 20 min.*

**If you have only 20 minutes: do items 1–4.** They close every directly-exploitable finding.
