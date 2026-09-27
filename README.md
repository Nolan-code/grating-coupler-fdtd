# Silicon Fiber-to-Chip Grating Coupler

Simulation and optimization of a silicon fiber-to-chip grating coupler using the FDTD method (Meep), combining an analytical estimate, numerical validation, and optimization via parametric sweep and adjoint-gradient methods.

Project carried out as part of the engineering curriculum at Télécom Physique Strasbourg.

## Main result

Optimal coupling efficiency obtained: **-13.4 dB (about 4.6%)**, at Λ ≈ 0.76 µm and duty cycle ≈ 0.68, confirmed by two methods (manual sweep and gradient optimization), converging to very close points in parameter space.

Full report: [`Rapport/Photonic_project.pdf`](Rapport/Photonic_project.pdf) (written in French)

## Repository structure

```
Grating_coupler/
├── environment.yml          # conda environment (Meep, nlopt, autograd...)
├── .gitignore
├── LICENSE
├── README.md
├── Rapport/
│   ├── main4.tex               # LaTeX source of the report
│   ├── Photonic_project.pdf    # compiled report
│   └── Image/                  # figures and diagrams used in the report
└── src/
    ├── script/
    │   ├── manual_tuning.py     # manual sweep (period, duty cycle)
    │   ├── grad_tuning.py       # gradient-based optimization (Meep adjoint solver)
    │   ├── run.py                # final calculation with the real Gaussian source
    │   └── run_box.py            # sweep over the buried-oxide (BOX) reflector thickness
    └── NoteBooks/
        ├── opt.ipynb               # debugging of the early adjoint-gradient building blocks
        └── opt_grad(2).ipynb       # first attempt at full topology optimization
                                     # (failed: incompatibility between the adjoint solver and
                                     #  GaussianBeam2DSource, see report section 5.3 / 6.2)
```

## Installation

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

```bash
conda env create -f environment.yml
conda activate meep
```

## Usage

The scripts are independent and can be run separately, in the order used in the report:

### 1. Manual sweep of the period and duty cycle (sections 5.1, 5.2)

```bash
python src/script/manual_tuning.py
```

Produces `balayage_gp.csv`, `balayage_gdc.csv`, and the figures `gp_r150.png`/`gdc_r150.png`.

### 2. Gradient-based optimization (section 5.3)

```bash
python src/script/grad_tuning.py
```

Uses Meep's adjoint solver (`meep.adjoint`) with a tilted plane-wave source (see the report for why this substitution was necessary). Can be accelerated on multiple cores via MPI:

```bash
mpirun -np 4 python src/script/grad_tuning.py
```

### 3. Final check with the real Gaussian source (section 5.3, Table 5.4)

```bash
python src/script/run.py
```

Recomputes the efficiency at the optimal point found, using the actual `GaussianBeam2DSource` (tilted fiber mode, Gaussian profile), giving a result that is physically comparable to the manual sweep.

### 4. BOX reflector sweep (section 5.5)

```bash
python src/script/run_box.py
```

**Note:** each FDTD simulation can take anywhere from a few minutes to several tens of minutes depending on the parameters, at the resolution used (150 pixels/µm).

## Notebooks (`src/NoteBooks/`)

These two notebooks document the development of the adjoint-gradient approach, including one failed attempt:

- **`opt.ipynb`**: unit-testing and debugging of the differentiation mechanism (the `teeth_weight` function and its finite-difference gradient), before integration into the full pipeline.
- **`opt_grad(2).ipynb`**: first attempt at full topology optimization (~27,000 free pixels), using the real tilted Gaussian source. The adjoint gradient computation triggers an internal Meep error, documented and discussed in the report as a methodological limitation. Kept as a record of this explored path.

These notebooks are not meant to be rerun as-is (they contain test parameters, not the final reported values) — they are provided as documentation of the development process.

## Methodology (summary)

1. Analytical estimate of the grating period from the phase-matching condition (effective-permittivity model, then a multilayer transfer-matrix model).
2. FDTD simulation (Meep) of the full device, validated through a study of sensitivity to spatial discretization and resolution convergence.
3. Optimization of the grating parameters (period Λ, duty cycle) by manual sweep and then by adjoint-gradient optimization.
4. Exploration of a bottom reflector (buried oxide, BOX) to reduce losses toward the substrate.

Full details (derivations, validation, discussion of limitations and future work) are in the report.

## Main limitations

- Two-dimensional modeling (no 3D lateral confinement).
- Uniform, non-apodized grating.
- Meep's adjoint solver, being incompatible with `GaussianBeam2DSource`, required a simplified source for the gradient-optimization step (see report, section 5.3).

Full discussion: chapter 6 of the report.

## Author

Nolan Le Tyrant — Télécom Physique Strasbourg

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).
