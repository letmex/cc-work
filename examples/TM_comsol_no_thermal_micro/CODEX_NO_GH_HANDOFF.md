# Codex Markdown-Only Handoff

For `TM_comsol_no_thermal_micro`, Codex should use markdown-only handoff by default.

Do **not** spend time checking `gh --version` or `gh auth status` at the end of every task unless the user explicitly asks for GitHub CLI auto-commenting.

Default rule:

```text
Generate HANDOFF_COMMENT.md.
Commit and push the evidence package.
Give the user the copy-paste ChatGPT sync command.
Do not try to post the issue comment directly with gh.
```

Reason: the current environment has GitHub CLI installed in some shells, but it is not authenticated and no `GH_TOKEN` / `GITHUB_TOKEN` is available. Re-checking it each task wastes time. ChatGPT can sync `HANDOFF_COMMENT.md` to issue #1 after the user provides the handoff path.

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

Issue comments are handled by one of these routes:

```text
Route A, preferred:
    ChatGPT reads HANDOFF_COMMENT.md from GitHub and posts it to issue #1 using the GitHub connector.

Route B:
    The user manually copies HANDOFF_COMMENT.md into GitHub issue #1.
```

Preferred route is A when the user asks ChatGPT to sync the handoff.

Communication issue:

```text
https://github.com/letmex/cc-work/issues/1
```

Codex should not attempt Route A directly unless the user explicitly says GitHub CLI authentication has been configured and asks Codex to use it.

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
[ ] final response to the user includes the repo-relative HANDOFF_COMMENT.md path
[ ] final response to the user includes a copy-paste ChatGPT sync command
```

If any item is missing, say it is missing instead of claiming completion.

Do not include repetitive `gh` status in the final response unless it was explicitly requested or actually relevant to a failure.

---

## 6. ChatGPT sync command

Codex must give the user the exact repo-relative path to the generated handoff file after every completed task.

Final Codex response must include this copy-paste command for the user:

```text
读取 <package_root>/HANDOFF_COMMENT.md，分析并写下一步 Codex prompt。
```

Example:

```text
读取 examples/TM_comsol_no_thermal_micro/runs/20260608_full_drive_broadening/HANDOFF_COMMENT.md，分析并写下一步 Codex prompt。
```

If the package is not under `runs/`, still provide the actual repo-relative path, for example:

```text
读取 examples/TM_comsol_no_thermal_micro/recent_debug_true_staggered_20260607/HANDOFF_COMMENT.md，分析并写下一步 Codex prompt。
```

The user can paste that command to ChatGPT.

ChatGPT should then:

```text
1. fetch HANDOFF_COMMENT.md from GitHub;
2. add it as a comment to issue #1 if it has not already been posted;
3. read the listed files;
4. analyze the evidence;
5. reply with a ChatGPT response comment and a next Codex prompt.
```
