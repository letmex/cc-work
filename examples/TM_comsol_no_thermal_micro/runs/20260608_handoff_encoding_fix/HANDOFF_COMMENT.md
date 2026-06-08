## Codex handoff: handoff sync command encoding fix

Commit: 9cf83deb35798787866276f11b786aae772726ba
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_handoff_encoding_fix
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_handoff_encoding_fix/REPORT.md

### What changed
- Added an explicit UTF-8 rule to the TM handoff workflow.
- The final ChatGPT sync command must use normal Chinese.
- Mojibake variants such as `璇诲彇`, `锛`, or `銆` are explicitly forbidden.
- No simulation or model behavior changed.

### Commands run
```powershell
rg "璇诲彇|锛|銆|读取 .*HANDOFF_COMMENT" examples\TM_comsol_no_thermal_micro -n
```

### Key results
- The correct command format is:
  `读取 <package_root>/HANDOFF_COMMENT.md，分析并写下一步 Codex prompt。`
- The previous package should be synced with:
  `读取 examples/TM_comsol_no_thermal_micro/runs/20260608_coordnorm_alpha_2x2_comparison/HANDOFF_COMMENT.md，分析并写下一步 Codex prompt。`

### Files to read first
- `README.md`
- `REPORT.md`
- `commands_run.txt`

### Question for ChatGPT
1. Confirm the handoff sync command should always be normal UTF-8 Chinese.
2. Confirm future prompts should reject mojibake copied from stale context.

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from medium/diagnostic runs.
