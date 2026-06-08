# Agent Handoff Workflow for `TM_comsol_no_thermal_micro`

This workflow fixes how Codex and ChatGPT communicate through GitHub for the `examples/TM_comsol_no_thermal_micro` debugging work.

It is not a GitHub Actions workflow. It is a human/agent communication protocol.

Long-lived communication issue:

```text
https://github.com/letmex/cc-work/issues/1
```

---

## 1. Core rule

Use GitHub as the communication bus:

```text
repo files = evidence packages
issue comments = conversation and decisions
```

Do not rely on local-only paths, uncommitted files, screenshots without summaries, or chat messages that are not mirrored in the issue.

---

## 2. Roles

### Codex

Codex runs code, produces diagnostics, uploads compact evidence packages, and posts handoff comments to issue #1.

Codex should not ask ChatGPT to infer results from hidden local files.

### ChatGPT

ChatGPT reads the GitHub issue, reports, tables, and figure summaries, then replies with interpretation and next-step prompts.

ChatGPT should not treat medium/diagnostic runs as physical validation.

---

## 3. Evidence package layout

Every new diagnostic package should be committed under:

```text
examples/TM_comsol_no_thermal_micro/runs/YYYYMMDD_short_name/
```

Recommended structure:

```text
README.md
REPORT.md
MANIFEST.json
commands_run.txt
next_questions.md
tables/
figures/
figures/figure_summary.md
```

Minimum acceptable package:

```text
README.md
REPORT.md
tables/main_summary.csv
next_questions.md
```

If figures are included, always add:

```text
figures/figure_summary.md
```

Reason: ChatGPT can reliably read Markdown and CSV through GitHub. PNGs may be accessible only as bytes/base64 through the connector, so critical visual observations must also be represented as text or tables.

Do not upload large logs or large intermediate training files unless they are directly needed. If a large file is needed, explain why in `README.md`.

---

## 4. Codex handoff comment format

After each diagnostic or code change, Codex must post a comment to issue #1 using this format:

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

The most important section is:

```text
Question for ChatGPT
```

Without this section, ChatGPT can read the evidence but may not know what decision Codex needs.

---

## 5. ChatGPT response format

ChatGPT should reply to issue #1 using this format:

```markdown
## ChatGPT response: <short title>

### Evidence read
- ...

### Current decision
- ...

### Interpretation
- ...

### Do next
1. ...
2. ...

### Do not do yet
- ...

### Prompt for Codex
```text
...
```
```

If ChatGPT cannot inspect image content reliably, it should say so and request `figures/figure_summary.md` or the underlying CSV.

---

## 6. Standard constraints for this project

Unless explicitly overridden:

```text
Do not change l0.
Do not add phase-field notch initialization.
Do not impose alpha=1 on the real geometric notch.
Do not change material parameters.
Do not change tm_source split.
Do not claim physical validation from medium or diagnostic runs.
Do not judge by a single good-looking seed.
```

Current physical-model framing:

```text
real explicit COMSOL micro-notch geometry
no thermal field
u, v, alpha
alpha=0 intact material
alpha=1 damaged material
TM mixed-mode split
dual history HI/HII
AT2
l0 = 1.5e-4 mm unless explicitly testing sensitivity
```

---

## 7. When full results are pending

If a full run is still pending, do not trigger speculative model changes. Codex should post:

```text
full run pending
current evidence is medium/diagnostic only
next action is wait / monitor / upload final package
```

ChatGPT should not recommend new physics terms, new constraints, or parameter changes until the full result is available and stepwise causality is analyzed.

---

## 8. Full-result analysis package

When a full run completes, the evidence package should include stepwise tables for:

```text
alpha_mean(step)
alpha_std(step)
alpha_max(step)
alpha>0.5 area fraction(step)

notch_tip_alpha_max(step)
bulk_alpha_mean(step)
bottom_right_alpha_max(step)

notch_tip_He_current_max(step)
bulk_He_current_p95(step)
bottom_right_He_current_max(step)

notch_tip_mechanics_drive_max(step)
bulk_mechanics_drive_p95(step)
bottom_right_mechanics_drive_max(step)

reaction_N_tm_eff(step)
elastic_energy(step)
fracture_energy(step)
loss_log10(step)
```

The report should classify the run as one of:

```text
A. medium-stage uniform -> full-stage localized
B. medium-stage uniform -> full-stage still uniform
C. medium-stage uniform -> full-stage boundary/corner damage
D. medium-stage localized -> full-stage uniform
```

---

## 9. Working rule

```text
Do not chase good-looking cracks.
Diagnose whether the same physical model is represented stably across platforms.
```

A good-looking seed only proves a branch exists. A bad-looking medium run only proves a branch exists. Neither is enough for physical validation.
