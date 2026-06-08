# Next Questions for ChatGPT

1. Given that global prefit can reproduce the FE-DOF mechanics branch but pure energy continuation moves away from it, should the next diagnostic focus on energy scaling, optimizer path, or a global proximal anchor?
2. What non-geometry-specific curriculum should be tested next, if notch/lip labels remain forbidden in the training objective?
3. Which metrics should be treated as sufficient to say that a future energy-continuation run preserved the FE-DOF-like branch?
4. Should the next run compare several random seeds before any model change is considered?

Constraints:

- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the real geometric notch unless explicitly testing an alternative model.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, or local displacement-jump targets to the training loss unless explicitly requested.
- Do not change TM split or material parameters unless a clear bug is found.
- Do not claim physical validation from this mechanics-only diagnostic.

