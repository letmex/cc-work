# Next Questions

1. Why does `debug_fedof_energy_baseline.py` / the FE-DOF RPROP mechanics baseline converge to a high-energy, high-residual branch instead of the direct sparse FE alpha=0 solution?
2. Is there a boundary-condition, displacement-scale, reaction-sign, or objective-normalization mismatch in the RPROP nodal-DOF diagnostic?
3. Should the FE-DOF RPROP target be regenerated with a direct linear solve target before using it for any supervised PINN mechanics prefit?
4. Should future PINN mechanics pretraining target the exact FE direct solve instead of the previous FE-DOF RPROP target?
5. Which residual/energy checks should be added as guards before accepting a mechanics target field?

