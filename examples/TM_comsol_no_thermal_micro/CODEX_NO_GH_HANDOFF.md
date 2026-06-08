# Codex Handoff Without GitHub CLI

Codex environment note:

```text
gh is not installed or not available in PATH.
gh --version and gh auth status return command not found.
```

Therefore Codex should not try to post directly to GitHub issues with the GitHub CLI.

Use this no-`gh` workflow instead.

---

## 1. Required behavior

At the end of every Codex task, generate a handoff markdown file inside the evidence package.

Required path:

```text
examples/TM_comsol_no_thermal_micro/runs/YYYYMMDD_short_name/HANDOFF_COMMENT.md
```

If the package is not under `runs/`, place the file at the package root:

```text
<package_root>/HANDOFF_COMMENT.md
```

Commit and push the handoff file with the evidence package.

Do not claim the handoff is complete if `HANDOFF_COMMENT.md` is missing.

---

## 2. Who posts the issue comment?

Since Codex cannot use `gh`, issue comments are handled by one of these routes:

```text
Route A:
    ChatGPT reads HANDOFF_COMMENT.md from GitHub and posts it to issue #1 using the GitHub connector.

Route B:
    The user manually copies HANDOFF_COMMENT.md into GitHub issue #1.
```

Preferred route is A when the user asks ChatGPT to sync the handoff.

Communication issue:

```text
https://github.com/letmex/cc-work/issues/1
```

---

## 3. Required HANDOFF_COMMENT.md template

Codex must write the file using this format:

```markdown
## Codex handoff: <short title>

Commit: <sha>
Data folder: <repo-relative path>
Main report: <repo-relative path>

### What changed
- ...

### Commands run
```powershell
...
```

### Key results
- ...

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/<main>.csv`
- `figures/figure_summary.md` if figures are included

### Question for ChatGPT
1. ...
2. ...

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from medium/diagnostic runs.
```

The `Question for ChatGPT` section is mandatory.

---

## 4. Optional helper script behavior

If a helper script such as `codex_handoff.py` exists, it should run in markdown-only mode by default:

```powershell
python codex_handoff.py --package <package_root> --mode markdown-only
```

Expected output:

```text
<package_root>/HANDOFF_COMMENT.md
```

It should not call `gh`.
It should not require GitHub API credentials.
It should print the generated path and the target issue URL.

---

## 5. Minimal task completion checklist for Codex

Before reporting completion, verify:

```text
[ ] evidence package committed
[ ] README.md exists
[ ] REPORT.md or main diagnostic report exists
[ ] tables/ contains required CSVs if numerical results are discussed
[ ] figures/figure_summary.md exists if figures are included
[ ] HANDOFF_COMMENT.md exists
[ ] HANDOFF_COMMENT.md includes Question for ChatGPT
[ ] commit SHA is written in HANDOFF_COMMENT.md
```

If any item is missing, say it is missing instead of claiming completion.

---

## 6. ChatGPT sync command

The user can ask ChatGPT:

```text
Read <package_root>/HANDOFF_COMMENT.md and post it to issue #1, then analyze the handoff and write the next Codex prompt.
```

ChatGPT should then:

```text
1. fetch HANDOFF_COMMENT.md from GitHub;
2. add it as a comment to issue #1;
3. read the listed files;
4. analyze the evidence;
5. reply with a ChatGPT response comment and a next Codex prompt.
```
