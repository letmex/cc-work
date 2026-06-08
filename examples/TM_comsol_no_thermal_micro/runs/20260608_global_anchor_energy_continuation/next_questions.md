# Next Questions

1. Should the next diagnostic directly compare the mechanics energy decomposition of FE-DOF target, prefit PINN, and collapsed PINN fields?
2. Which global continuation rule should be tested next, if the current anchor and trust-region sweeps all fail?
3. Should the FE-DOF target validity be audited under the same quadrature and boundary assumptions before any coupled phase-field run?

Constraints:

- Do not change `l0` unless explicitly requested.
- Do not change material parameters or `tm_source` split unless a clear bug is found.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or geometry-label guidance to the training loss unless explicitly requested.
- Do not claim physical validation from this mechanics-only diagnostic.

