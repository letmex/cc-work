# Top-U-Free Ansatz Diagnostic Notes

Purpose:
- Check whether the code path can represent a top boundary with prescribed vertical displacement while leaving top horizontal displacement unconstrained by the ansatz.
- Record boundary displacement samples in the diagnostics CSV.

Implemented diagnostic fields:
- `top_u_mode`
- `top_u_min`, `top_u_max`, `top_u_mean`, `top_u_abs_max`
- `top_v_min`, `top_v_max`, `top_v_mean`, `top_v_abs_max`, `top_v_error_max`
- `bottom_u_min`, `bottom_u_max`, `bottom_u_mean`, `bottom_u_abs_max`
- `bottom_v_min`, `bottom_v_max`, `bottom_v_mean`, `bottom_v_abs_max`

Smoke command:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 2 TrainableReLU 3.0 --smoke --pff-model AT2 --mixed-mechanics-mode history --alpha-init-intact --top-u-mode free --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix topufree_smoke
```

Smoke result:
- Command completed successfully.
- `top_u_mode = free`.
- `top_u_abs_max = 1.6637207167491397e-09`.
- `top_v_error_max = 0.0`.
- `bottom_u_abs_max = 0.0`.
- `bottom_v_abs_max = 0.0`.

Interpretation:
- The smoke confirms that boundary diagnostics are written and that the top-v/bottom constraints are sampled.
- It does not test whether the full D0020 alpha-init run localizes damage.
- A full top-u-free run should only be started if requested after reviewing this package.
