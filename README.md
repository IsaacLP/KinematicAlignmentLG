
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

## Pipeline overview

The analysis runs in five steps. Each step writes files that the next one reads:

```
AbacusSummit CompaSO halo catalogues (NERSC)
        │  1. src/find_pairs_abacus.py
        ▼
data/pairs_AbacusSummit_base_cXXX_ph000_z0.100.hdf5        (one file per cosmology)
        │
        │      2. analysis/obvs_data_montecarlo.py ──► results/montecarlo_results.csv
        │                                                    │
        │  3. analysis/all_cosmo.py --suffix <run>  ◄────────┘
        ▼
results/<run>/dists.pkl, cosmo_results.pkl, cosmo_analysis.pkl, cosmo_analysis_all.pkl
        │  4. analysis/random_forest.py
        ▼
results/<run>/random_forest_<sample>/*.csv
        │  5. analysis/all_cosmo_plots.py --suffix <run>
        ▼
figures/<run>/*.pdf   (also copied to paper/source/figures/)
```

The paper's figures come from the run `final_rev`.

### 1. Find halo pairs: `src/find_pairs_abacus.py`
Runs on NERSC. It needs `abacusutils` and read access to `/global/cfs/cdirs/desi/cosmosim/Abacus`. For each cosmology, it reads the CompaSO halo catalogue at $z = 0.1$ (phase `ph000`) and then:
1. keeps halos with $v_{\rm circ,max} > 200$ km/s;
2. finds mutual nearest-neighbour pairs with a periodic KD-tree;
3. requires isolation: no halo with a larger $v_{\rm circ,max}$ within 3× the pair separation of either member;
4. requires both members to have $v_{\rm circ,max} < 300$ km/s.

For each pair member (A and B), it stores the position, velocity, particle count (`mass_*`), $v_{\rm circ,max}$ and halo ID in an HDF5 file. Run it from `src/`, because it writes to `../data/`. The cosmology IDs it processes are listed in `abacus_simulations` at the bottom of the script.

`src/test_read_abacus.py` is a quick check that a catalogue loads on NERSC.

### 2. Observational constraints: `analysis/obvs_data_montecarlo.py`
Propagates observational uncertainties with 10,000 Monte Carlo samples. The inputs are:
- the LG barycenter velocity from the CMB dipole: $620 \pm 15$ km/s towards $(l, b) = (271.9^\circ, 29.6^\circ)$;
- the sky positions and distances of the MW centre and M31.

From these it computes the MW→M31 separation vector and $\mu$. It also stores literature values for $v_r$, $v_t$ and $M_{\rm LG}$. The output is `results/montecarlo_results.csv`, with the mean in row 0 and the standard deviation in row 1 of each column. No random seed is set, so the values change slightly from run to run.

### 3. Select analogue samples and compare them: `analysis/all_cosmo.py`
```bash
python analysis/all_cosmo.py --suffix final_rev
```
For every HDF5 file in `data/`, the script:
1. builds the cosmology from `abacusnbody.metadata` and converts particle counts to $M_\odot$;
2. orders each pair so that A has the larger $v_{\rm circ,max}$;
3. keeps pairs where both halos are in $0.5$–$5 \times 10^{12} M_\odot$;
4. computes the barycenter speed $v_b$, the alignment $\mu$, $v_r$ (including Hubble flow), $v_t$ and $M_{\rm LG}$. Before $\mu$ is computed, positions are shifted from $z = 0.1$ to today using each halo's velocity and the lookback time.

Each analogue is then assigned to three samples:

| Sample | Code label | Plot label | Criteria |
|---|---|---|---|
| Traditional | `Trad` | Traditional | $r < 1\,h^{-1}$ Mpc and $v_r < 0$ |
| Realistic | `Real` | Realistic | Traditional + $v_b > v_{b,\rm obs} - \sigma$ + $\mu < \mu_{\rm obs} + \sigma$ |
| Centered | `Pract` | Centered | Traditional + $500 < v_b < 740$ km/s + $-0.96 < \mu < -0.80$ |

For $M_{\rm LG}$, $v_r$ and $v_t$, it compares each sample with Traditional. It records summary statistics, the percentage change in the mean, and a two-sample KS test. A difference is significant when $p < 0.05$.

| Option | Default | Meaning |
|---|---|---|
| `--suffix` | required | Name of the run; outputs go to `results/<suffix>/` |
| `--vrad_max` | `0` | Upper limit on $v_r$ (km/s) |
| `--vcm_min` | from step 2 (≈ 605 km/s) | Lower limit on $v_b$ for the Realistic sample |
| `--mu_max` | from step 2 (≈ −0.87) | Upper limit on $\mu$ for the Realistic sample |
| `--HubbleFlow_flag` | not passed (correction on) | Passing it **disables** the Hubble-flow term in $v_r$ |
| `--shift_pos_flag` | not passed (shift on) | Passing it **disables** the shift of positions to $z = 0$ |

Outputs in `results/<suffix>/`:
- `dists.pkl`: the full distribution of each variable, per cosmology and sample;
- `cosmo_results.pkl`: the statistics table, indexed by (`cosmo`, `variable`, `filter`);
- `cosmo_analysis.pkl`: averages over the cosmologies with a significant difference;
- `cosmo_analysis_all.pkl`: averages over all cosmologies.

### 4. Link significance to cosmology: `analysis/random_forest.py`
For each cosmology, the label is whether the chosen sample differs significantly from Traditional in $v_r$ or $v_t$. The features are the cosmological parameters. The script trains a random-forest classifier with stratified 5-fold cross-validation and random oversampling. It writes feature importances, F1 scores and classification reports to `results/<suffix>/random_forest_<sample>/`. The run suffix and the sample (`filt`) are set in `main()`. Each run produces results for one sample only. To reproduce all results, run it once with `filt = 'Real'` and once with `filt = 'Pract'`. The plotting step uses the `Pract` (Centered) results.

### 5. Figures: `analysis/all_cosmo_plots.py`
```bash
python analysis/all_cosmo_plots.py --suffix final_rev
```
Reads the outputs of steps 2–4 and `analysis/filtered_cosmologies_latex.csv`, then writes the paper figures to `figures/<suffix>/`. Figures created by this run are also copied to `--copy_dir`, which defaults to `paper/source/figures/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/find_pairs_abacus.py` | Step 1: pair finding on the AbacusSummit catalogues |
| `src/read_hdf5_pairs_abacus.ipynb` | Quick look at a pairs file and the average number of pairs per cosmology |
| `analysis/obvs_data_montecarlo.py` | Step 2: observational constraints |
| `analysis/all_cosmo.py` | Step 3: sample selection and statistics |
| `analysis/random_forest.py` | Step 4: random-forest feature importance |
| `analysis/all_cosmo_plots.py` | Step 5: paper figures |
| `analysis/abacus_cosmo.ipynb` | Builds the cosmology parameter tables from the AbacusSummit cosmology list |
| `analysis/base_cosmo.ipynb` | Baseline cosmology: the $\mu$ distribution, the effect of the position shift and Hubble flow, uniformity and bootstrap tests |
| `analysis/cosmo_analysis.ipynb` | Exploration of the results across cosmologies and of the random-forest reports |
| `analysis/cosmologies.csv` | AbacusSummit cosmology parameters |
| `analysis/filtered_cosmologies.csv` | The 93 cosmologies used in this work |
| `analysis/filtered_cosmologies_latex.csv` | The same cosmologies, with LaTeX column names for plotting |

## Requirements
Python 3 with `numpy`, `scipy`, `pandas`, `astropy`, `h5py`, `abacusutils`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn` and `tqdm`. `python-dotenv` is optional (see below).

## Data location
The simulation data (`data/`) and analysis outputs (`results/`) are not tracked in this repository. Run the scripts from the repository root. By default, steps 2–5 and the notebooks look for `data/` and `results/` there. To keep them elsewhere, for example on an external drive, copy `.env.example` to `.env` and set `LGKIN_DIR` to the folder that contains them:

```bash
pip install python-dotenv
cp .env.example .env   # then edit LGKIN_DIR in .env
python analysis/all_cosmo_plots.py
```

The scripts and notebooks load `.env` automatically. `.env` is git-ignored, so each machine keeps its own. A `LGKIN_DIR` already set in your shell takes precedence. Without `python-dotenv` installed, `.env` is ignored and the repository root is used, unless `LGKIN_DIR` is exported.
