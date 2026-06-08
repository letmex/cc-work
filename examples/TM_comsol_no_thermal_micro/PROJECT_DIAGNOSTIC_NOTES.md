# TM_comsol_no_thermal_micro Diagnostic Notes

This document records the current working interpretation for the `examples/TM_comsol_no_thermal_micro` project.

The purpose is not to copy one paper or one COMSOL implementation mechanically. The purpose is to implement the same no-thermal mixed-mode phase-field fracture model on a real explicit micro-notch geometry using a PINN/Deep-Ritz-style numerical platform, then diagnose why the PINN result can deviate from the COMSOL/FEM result.

---

## 1. Project positioning

The active target model is:

```text
real explicit COMSOL micro-notch geometry
no thermal field
plane-stress mechanical field
u(x,y), v(x,y), alpha(x,y)
TM / mixed-mode split
dual history fields HI and HII
AT2 fracture energy
physical l0 = 1.5e-4 mm = 0.15 um
```

The project should not be interpreted as:

```text
1. a direct reproduction of Manav's SEN phase-field-notch example;
2. a direct copy of the COMSOL thermal-elastic model;
3. a direct copy of the Chinese mixed-mode PINN paper;
4. a seed-selection exercise to find a visually plausible crack.
```

Instead, those sources provide different layers:

```text
Manav et al.:
    PINN/DRM methodology, field mapping, FE-style gradients, quadrature, optimizer lessons.

Chinese mixed-mode phase-field paper:
    tension/shear mixed-mode decomposition and dual-history formulation.

COMSOL model:
    micro-scale explicit notch geometry, material parameters, and reference physical behavior.
```

The geometry is a real notch. The notch is already a geometric free boundary. Therefore the current route should not impose a separate phase-field notch with `alpha=1` along the existing notch unless this is explicitly tested as a different alternative model.

---

## 2. Current fixed assumptions

For the main route, keep these fixed unless a clear bug is found:

```text
l0 = 1.5e-4 mm
Gc = 2.4e-6 kN/mm
E = 81.5 kN/mm^2
nu = 0.38
AT2
no thermal field
no thermal strain
no phase-field notch imposed on the geometric notch
no alpha=1 seed on the notch
same explicit micro-notch mesh unless testing mesh sensitivity
```

`history / dual-history` is currently treated as the closest source-model route. `current_split` is diagnostic / ablation. `history_phase_current_mechanics` and staggered variants are also diagnostic unless validated separately.

---

## 3. Diagnostics already established

### 3.1 No-thermal path

The no-thermal PINN path uses three fields:

```text
u, v, alpha
```

No active temperature field, thermal expansion strain, heat equation, or thermal material path should remain active in the no-thermal training path.

### 3.2 current_split artifact

Earlier full seed2 D0020 diagnostics showed:

```text
current_split:
    max He_current / mechanics_drive moved to a boundary/corner region.
    He_history could remain near the notch tip.

history:
    He_current, He_history, and mechanics_drive could remain near the notch tip,
    but alpha_mean could become large and the damage band could be wide.
```

This means `current_split` is useful for exposing the current strain -> split -> drive chain, but it should not be treated as the main physical route.

### 3.3 Mesh-l0 diagnostic

The coarse mesh under-resolves the physical `l0=0.15 um` in the full domain, expected crack band, and especially boundary bands. This affects numerical precision.

However, COMSOL can compute a reasonable crack with the physical `l0=0.15 um`, so the main PINN failure should not be reduced to "l0 is wrong".

Do not keep increasing `l0` as the main fix. Larger `l0` may numerically smooth the field but can create unphysical broad damage.

### 3.4 Explicit-geometry alpha initialization

For a real explicit notch geometry, the solid material should start intact:

```text
alpha ≈ 0 in the solid domain
```

Default raw-alpha initialization produced approximately:

```text
alpha_mean ≈ 0.5
```

This is inconsistent with the explicit-geometry model. The added `--alpha-init-intact` option sets the alpha channel so that the initial material state is intact without imposing a phase-field notch.

### 3.5 Elastic-only and FE-DOF alpha=0 baselines

With `alpha=0`:

```text
elastic-only PINN:
    He_current maximum is at the notch tip.
    bottom_right / notch_tip ratio is small.

FE-DOF alpha=0 baseline:
    no right-bottom dominance.
```

Therefore, the boundary/corner artifact is not explained by pure elastic displacement plus `tm_source` split alone.

### 3.6 Alpha-init-intact full runs

Full D0020 seed2 runs with `--alpha-init-intact` showed that correcting the initial alpha state is necessary but not sufficient. Both history and current_split variants could evolve toward near-uniform background damage around:

```text
alpha_mean ≈ 0.488
```

This means bad initial alpha was one problem, but not the only root cause.

### 3.7 Local AT2 equilibrium observation

For a broad drive field `H`, the local no-gradient AT2 balance predicts:

```text
alpha_eq = 2H / (2H + Gc/l0)
```

With `Gc/l0 = 0.016` and broad drive around `H ≈ 0.0076`, this gives:

```text
alpha_eq ≈ 0.49
```

This matches the observed near-uniform `alpha≈0.488` once the trained drive field becomes broad.

This does not by itself prove what caused the broad drive. It only explains why alpha responds uniformly after the drive has already become broad.

---

## 4. Current caution: do not overinterpret medium results

Some staggered and FE-DOF staggered diagnostics were medium or diagnostic-level runs, not full validated training. They are useful for exposing possible branches, but they should not be used alone for final physical conclusions.

Current analysis stance:

```text
Do not conclude from medium training.
Wait for full training results before deciding whether the uniform/background branch is stable.
```

A visually plausible crack from one seed also does not prove correctness. It only proves that a localized branch exists for that seed and setup.

---

## 5. Main open question

The key question has shifted from:

```text
Why does alpha respond uniformly to a broad H?
```

to:

```text
Why and when does H become broad?
```

Evidence so far:

```text
alpha=0 elastic-only mechanics gives notch-localized H.
trained coupled/history fields can produce broad H.
fixed-u/v alpha-only with broad H reproduces uniform alpha.
```

So the root-cause chain still needs full-run confirmation:

```text
Does alpha broaden first and then flatten H?
Does H broaden first and then drive alpha globally?
Do alpha and H broaden together through coupled optimization?
Does full training re-localize after a medium-stage uniform-looking field?
```

---

## 6. Full-result analysis framework

When full results are available, analyze stepwise data rather than only final plots.

Track:

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

Classify the full run as:

```text
A. Medium-stage uniform, full-stage localized:
    medium uniform field was an optimization transient.

B. Medium-stage uniform, full-stage still uniform:
    background damage branch is stable for this setup.

C. Medium-stage uniform, full-stage boundary/corner damage:
    background damage is a transition toward a boundary branch.

D. Medium-stage localized, full-stage uniform:
    continued optimization pushes the system toward diffuse damage.
```

Use these criteria before adding new physics terms or changing the model.

---

## 7. Candidate causes, not conclusions

The following remain candidate mechanisms, not final conclusions:

```text
1. Broadening of the history/mechanics drive during coupled training.
2. AT2 background damage branch once bulk drive becomes comparable to Gc/l0.
3. Boundary condition mismatch: COMSOL top u may be free, while current ansatz can impose top u≈0.
4. Single continuous NN expression across a very narrow explicit notch gap.
5. Missing or weak regularization / scaling in the PINN implementation.
6. Insufficient or unstable optimizer path for some settings.
7. Mesh resolution effects around crack band and boundary/corner regions.
```

Do not promote any one of these to the final explanation until full results and stepwise causality are checked.

---

## 8. Do not change yet

Before full results are analyzed, avoid changing:

```text
l0
Gc / GcII
TM split
phase-field notch initialization
alpha=1 on the geometric notch
COMSOL etaPF/proximal term
alpha irreversibility penalty
notch-lip split ansatz
top-u-free ansatz as a permanent replacement
```

The only safe actions before full results are:

```text
1. read diagnostics;
2. generate stepwise comparison tables;
3. run non-invasive postprocessing scripts;
4. document the analysis plan.
```

---

## 9. Suggested next postprocessing after full results

Create a full-run causality report with:

```text
# Full-Run Drive Broadening Analysis

## Runs analyzed
## Stepwise alpha evolution
## Stepwise drive localization
## First broadening step
## Does alpha broaden before H?
## Does H broaden before alpha?
## Boundary/corner involvement
## Reaction and energy curves
## Final classification
## Recommended next minimal intervention
```

The next intervention should be chosen only after this report.

---

## 10. Working rule

The project should follow this rule:

```text
Do not chase good-looking cracks.
Diagnose whether the same physical model is being represented stably across platforms.
```

A good-looking result from one seed or one partial training run is not enough. A bad-looking medium result is also not enough. Full training, stepwise causality, and platform-equivalence checks are required before changing the model.
