# Next Questions for ChatGPT

1. Does this evidence support targeting the displacement/strain representation near the narrow explicit notch next?
2. Should the next minimal diagnostic be mechanics-only PINN vs nodal-DOF at `Delta = 1e-6` with alpha fixed to zero?
3. Should we test a local enrichment / separate notch-lip degrees of freedom diagnostic before changing any physical phase-field parameter?

Constraints:
- Do not change `l0`.
- Do not change material parameters.
- Do not change `tm_source` split.
- Do not add phase-field notch initialization.
- Do not impose `alpha=1` on the geometric notch.
- Do not change thermal field or history update logic.
- Do not claim physical validation from these diagnostics.
