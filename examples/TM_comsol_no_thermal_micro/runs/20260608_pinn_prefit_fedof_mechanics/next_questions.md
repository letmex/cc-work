# Next Questions for ChatGPT

1. Given that supervised prefit can reconstruct FE-DOF `u/v`, notch-lip jumps, strains, and notch-amplified `He_current`, should the next Codex task test a mechanics pretraining/curriculum path before coupled phase-field training?
2. Should the next diagnostic use `disp_strain` prefit weights as initialization for alpha-zero energy minimization, still without changing the physical model?
3. Is notch-lip enrichment still needed immediately, or should it be held as a fallback after testing pretraining/localized loss guidance?

Constraints:

- Do not change `l0`, material parameters, `tm_source` split, phase-field notch behavior, alpha seeding, or history update logic.
- Do not claim physical validation from this diagnostic.
- Do not start a coupled full phase-field run unless explicitly requested.
