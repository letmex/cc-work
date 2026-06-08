# Next Questions

1. Should the next task replace all future mechanics prefit inputs with `artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz` or a regenerated direct FE target?
2. Should `debug_fedof_energy_baseline.py` and `debug_mechanics_only_notch_ansatz.py` be updated to label their RPROP FE-DOF outputs as optimizer diagnostics rather than exact FE baselines?
3. Should a direct sparse FE solve become the mandatory mechanics target generator before any supervised PINN mechanics pretraining?
4. Which global-only prefit strategy should be tested next to improve strain and `He_current` reconstruction without adding local notch guidance?
5. Should a guard check be added to every future target package so rejected targets cannot silently be reused?

