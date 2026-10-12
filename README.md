
# Kinematic Alignment in the Local Group

This repository contains all analysis code and notebooks used in the project “Barycenter kinematics in Local Group analogues”. The goal of this project is to evaluate how incorporating the Local Group barycenter velocity—its magnitude and alignment—affects the identification and physical properties of LG-analogue halo pairs in AbacusSummit cosmological simulations.

The full scientific context is described in the preprint: [arXiv:2505.04054](https://arxiv.org/abs/2505.04054)

## Scientific Motivation

The Local Group is a galaxy group that includes the Milky Way (MW) and the Andromeda Galaxy (M31), among other smaller galaxies. The Milky Way and Andromeda Galaxy are the two most massive members, together containing most of the mass of the group.

The dynamics within the Local Group provide insights into galaxy formation and evolution. As it contains some of the closest galaxies, the Local Group allows for detailed astronomical observations that are not possible with more distant galaxies.

The Local Group is relatively isolated from other galaxy groups. The Local Group's peculiar velocity (motion relative to the cosmic microwave background) keeps it somewhat isolated as it moves relative to this general expansion.

Most LG analogue studies select MW–M31–like systems using criteria based on separation, mass, isolation, and radial approach velocity.

This work introduces the barycenter velocity vector as an additional constraint:
- Magnitude of the barycenter speed, $v_b$, measured precisely from the CMB dipole.
- Alignment, $\mu = \cos \theta$, between the barycenter velocity vector and the MW→M31 separation vector.

## Project goals 
1. Compute $v_b$, $\mu$, and the MW–M31 relative velocity from the observed data using Monte-Carlo uncertainty propagation.
2. Identify LG analogues in AbacusSummit simulations using:
    - Traditional selection (mass, distance, radial approach, and isolation).
    - Realistic selection (traditional + barycenter speed and alignment).
3. Compare how the additional constraints affect:
    - Radial relative velocity $v_r$
    - Tangential relative velocity $v_t$
    - Total LG mass $M_{\rm LG}$
4. Evaluate the effect across 93 cosmological models.
5. Test for statistical significance using KS tests.
6. Analyse global trends in the full cosmology set.
7. Examine links with cosmological parameters.

## Data location
The simulation data (`data/`) and analysis outputs (`results/`) are not tracked in this repository. Scripts are meant to be run from the repository root. By default they look for `data/` and `results/` there. To keep them elsewhere, for example on an external drive, copy `.env.example` to `.env` and point `LGKIN_DIR` at the folder that contains them:

```bash
pip install python-dotenv
cp .env.example .env   # then edit LGKIN_DIR in .env
python analysis/all_cosmo_plots.py
```

The scripts and notebooks load `.env` automatically. `.env` is git-ignored, so each machine keeps its own. A `LGKIN_DIR` already set in your shell takes precedence. Without `python-dotenv` installed, `.env` is ignored and the repository root is used, unless `LGKIN_DIR` is exported.
