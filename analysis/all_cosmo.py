import h5py 
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import G
import astropy.cosmology.units as cu
from astropy.cosmology import w0waCDM
import abacusnbody.metadata
from scipy import stats
import re
import os
from tqdm import tqdm
import argparse


G_const = G.to(u.km**2 * u.Mpc / (u.Msun * u.s**2))


## ----------------------------------------------------------- Read data --------------------------------------------------------------------- ##

def read_files(directory_path):
    cosmo_names = []
    cosmo_tags = []
    
    file_names = [f for f in os.listdir(directory_path)
                  if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.hdf5')]
    
    for filename in file_names:
        match = re.search(r'pairs_(.+?)_z', filename)
        if match:
            full_name = match.group(1)
            cosmo_names.append(full_name)

            c_match = re.search(r'(c\d+)', full_name)
            if c_match:
                cosmo_tags.append(c_match.group(1))

    return file_names, cosmo_names, cosmo_tags


def create_cosmo(cosmo_names):
    '''Create astropy cosmology objects from the names'''
    cosmo_objects = []
    for name in cosmo_names:
        sim_meta_data = abacusnbody.metadata.get_meta(name) # Load simulation metadata (time-independent)
        cosmo = w0waCDM(H0=sim_meta_data['H0'], Om0=sim_meta_data['Omega_M'], Ode0=sim_meta_data['Omega_DE'], Ob0=sim_meta_data['omega_b'],
                w0=sim_meta_data['w0'], wa=sim_meta_data['wa'],Neff=sim_meta_data['N_ur']+sim_meta_data['N_ncdm'],Tcmb0=2.7255) # Create astropy cosmology object
        cosmo_objects.append(cosmo)
    return cosmo_objects


def read_pairs_data(file):
    '''Read the simulation data from the hdf5 file'''
    path = os.path.join('data', file)
    data = {}
    with h5py.File(path, 'r') as f:
        for k in f.keys():
            data[k] = f[k][...]

    return data


def read_obs_data(file):
    '''Read the observational data from the csv file'''
    obs_data = pd.read_csv(file)

    mu_obs = np.array(obs_data['mu'])
    v_cm_obs = np.array(obs_data['v_cm'])
    v_rad_obs = np.array(obs_data['v_rad'])
    v_tan_obs = np.array(obs_data['v_tan'])
    mass_LG_obs = np.array(obs_data['M_LG'])

    return {'mu':mu_obs, 'v_cm':v_cm_obs, 'v_rad':v_rad_obs, 'v_tan':v_tan_obs, 'mass_LG':mass_LG_obs}


def exchange_pairs(data, option):
    '''Swap A/B fields when key_B > key_A.'''
    key = 'vmax' if option == 'velocity' else 'mass'
    a_key = f'{key}_A'; b_key = f'{key}_B'

    mask = data[b_key] > data[a_key]   

    fields = ['vel', 'pos', 'vmax', 'mass', 'halo_id']
    for f in fields:
        A = 'halo_A_id' if f == 'halo_id' else f'{f}_A'
        B = 'halo_B_id' if f == 'halo_id' else f'{f}_B'

        temp = data[A][mask].copy()
        data[A][mask] = data[B][mask]
        data[B][mask] = temp

    return data


def apply_units(data,cosmo_model):
    '''Convert units'''
    
    H = cosmo_model.H(0.1)
    omega_m = cosmo_model.Om(0.1)

    data['vel_A'] = data['vel_A'] * u.km / u.s
    data['vel_B'] = data['vel_B'] * u.km / u.s

    data['pos_A'] = data['pos_A'] * (u.Mpc/cu.littleh)
    data['pos_B'] = data['pos_B'] * (u.Mpc/cu.littleh)
    data['pos_A'] = data['pos_A'].to(u.Mpc,cu.with_H0(H))
    data['pos_B'] = data['pos_B'].to(u.Mpc,cu.with_H0(H))

    data['vmax_A'] = data['vmax_A'] * u.km / u.s
    data['vmax_B'] = data['vmax_B'] * u.km / u.s

    # Mass units
    rho_c = 3 * H**2 / (8 * np.pi * G_const)
    rho_m = omega_m * rho_c

    L = (2 * u.Gpc/cu.littleh).to(u.Mpc,cu.with_H0(H))
    V = L**3
    N = 6912**3

    m_p = (rho_m * (V / N)).to(u.Msun)

    data['mass_A'] = data['mass_A'] * m_p
    data['mass_B'] = data['mass_B'] * m_p

    return data


## ----------------------------------------------------------- Compute distributions --------------------------------------------------------------------- ##

def get_vcm(data):
    '''Compute the center of mass velocity'''
    m_a = data['mass_A'].copy()
    m_b = data['mass_B'].copy()
    v_a = data['vel_A'].copy()
    v_b = data['vel_B'].copy()
    v_cm = (m_a[:, None]*v_a + m_b[:, None]*v_b)/(m_a[:, None]+m_b[:, None])

    return v_cm


def get_mu(v_cm,p_rel):
    '''Compute the cosine of the angle between the relative velocity and the center of mass velocity'''
    dot_products = np.einsum('ij,ij->i',v_cm,p_rel)
    mu = dot_products / (np.linalg.norm(v_cm,axis=1) * np.linalg.norm(p_rel,axis=1))
    
    return mu


def v_rad_tan(v_rel,r,correct_HubbleFlow=False,H=None):
    '''Decompose the relative velocity into radial and tangential components'''
    r_mag = np.linalg.norm(r,axis=1)[:,np.newaxis]
    r_hat = r/r_mag
    v_rel_rad_comp = np.einsum('ij,ij->i',v_rel,r_hat)[...,np.newaxis]
    if correct_HubbleFlow == True:
        v_rel_rad_comp = v_rel_rad_comp + H * r_mag

    v_rel_tan = v_rel - v_rel_rad_comp * r_hat
    v_rel_tan_comp = np.linalg.norm(v_rel_tan,axis=1)

    return v_rel_rad_comp.flatten(), v_rel_tan_comp.flatten()


def orbital_angular_momentum(p_rel,v_rel, mass_A, mass_B):
    '''Compute the orbital angular momentum vector and its magnitude'''
    # Reduced mass [kg]
    m_A = (mass_A).to(u.kg)
    m_B = (mass_B).to(u.kg)
    mu = (m_A * m_B) / (m_A + m_B)

    # Orbital angular momentum vector
    L_vec = mu[:, None] * np.cross(p_rel, v_rel)  # shape (N,3)

    # Magnitude
    L_mag = np.linalg.norm(L_vec, axis=1) * L_vec.unit

    return L_vec, L_mag

## ----------------------------------------------------------- Filter data --------------------------------------------------------------------- ##
def filter_mass_pairs(data):
    '''Filter pairs with mass in [0.5 5] e12 Msun'''
    maskA = (data['mass_A'] > 0.5e12 * u.Msun) & (data['mass_A'] < 5e12 * u.Msun)
    maskB = (data['mass_B'] > 0.5e12 * u.Msun) & (data['mass_B'] < 5e12 * u.Msun)

    mask = maskA & maskB

    for key in data.keys():
        data[key] = data[key][mask]

    return data


def find_neg_pairs(v_rel_rad_comp,v_rad_max):
    ''' Find pairs with radial relative velocity < v_rad_max [km/s]'''

    mask = (v_rel_rad_comp < v_rad_max).flatten()

    return mask


def find_close_pairs(r,H,option):
    ''' Find pairs with 0.5 Mpc/h < r < Mpc/h or r < 1 Mpc/h'''
    r_mag = np.linalg.norm(r,axis=1)

    if option == 'less':
        r_max = 1 * u.Mpc/cu.littleh
        r_max = r_max.to(u.Mpc,cu.with_H0(H))
        mask = r_mag < r_max
    elif option == 'within':
        r_max = 1 * u.Mpc/cu.littleh
        r_max = r_max.to(u.Mpc,cu.with_H0(H))

        r_min = 0.5 * u.Mpc/cu.littleh
        r_min = r_min.to(u.Mpc,cu.with_H0(H))

        mask = (r_mag > r_min) & (r_mag < r_max)

    return mask


def find_fast_pairs(v_cm,v_cm_obs,option):
    ''' Find pairs with v_cm > v_cm_obs'''
    if option == 'greater':
        mask = v_cm > v_cm_obs[0]
    elif option == 'within':
        mask = (v_cm > v_cm_obs[0] - v_cm_obs[1]) & (v_cm < v_cm_obs[0] + v_cm_obs[1])

    return mask


def find_aligned_pairs(mu,mu_obs,option):
    ''' Find pairs with mu < mu_obs or (mu > mu_obs - sigma_mu and mu < mu_obs + sigma_mu)'''
    if option == 'less':
        mask = mu < mu_obs[0]
    elif option == 'within':
        mask = (mu > (mu_obs[0] - mu_obs[1])) & (mu < (mu_obs[0] + mu_obs[1]))

    return mask

# -------------------------------------------------- Other calculations --------------------------------------------------------------------- ##
def shift_positions(data,cosmo,z=0.1):
    '''Shift positions to the present time and return the relative position'''
    
    pos_a = data['pos_A']
    pos_b = data['pos_B']
    v_a = data['vel_A']
    v_b = data['vel_B']

    t = cosmo.lookback_time(z)

    va = v_a.to(u.Mpc/u.Gyr)
    vb = v_b.to(u.Mpc/u.Gyr)

    pos_a_shifted = pos_a + va * t
    pos_b_shifted = pos_b + vb * t

    p_rel_shifted = pos_a_shifted - pos_b_shifted

    return p_rel_shifted

# ----------------------------------------------------- Main analysis --------------------------------------------------------------------- ##

def read_data(sim_file,cosmo_model,bound_pair_mass = True,pair_convention = 'velocity'):

    sim_data = read_pairs_data(sim_file)

    sim_data = exchange_pairs(sim_data.copy(),pair_convention)

    sim_data = apply_units(sim_data.copy(),cosmo_model)

    if bound_pair_mass:
        sim_data = filter_mass_pairs(sim_data.copy())

    return sim_data


def get_dists(sim_data,cosmo,shift_pos_flag,HubbleFlow_flag):
    v_cm = get_vcm(sim_data)
    speed_vcm = np.linalg.norm(v_cm,axis=1)

    if shift_pos_flag == True:
        p_rel = shift_positions(sim_data,cosmo,0.1)
    else:
        p_rel = sim_data['pos_A'] - sim_data['pos_B']

    mu = get_mu(v_cm,p_rel)

    v_rel = sim_data['vel_A'] - sim_data['vel_B']
    vrel_rad, vrel_tan = v_rad_tan(v_rel,p_rel,correct_HubbleFlow=HubbleFlow_flag,H=cosmo.H(0.1))

    mass_LG = sim_data['mass_A'] + sim_data['mass_B']

    dists = {'mu':mu, 'speed_vcm':speed_vcm,
             'p_rel':p_rel, 'v_rel':v_rel, 
             'vrel_rad':vrel_rad,'vrel_tan':vrel_tan, 'mass_LG':mass_LG}

    return dists


def get_masks(dists, cosmo, vrad_max, vcm_min, mu_max):
    # Traditional and Realistic
    mask_neg = find_neg_pairs(dists['vrel_rad'],vrad_max)
    mask_close = find_close_pairs(dists['p_rel'],cosmo.H(0.1),option='less')
    mask_aligned = find_aligned_pairs(dists['mu'],[mu_max],option='less')
    mask_fast = find_fast_pairs(dists['speed_vcm'],[vcm_min],option='greater')

    # Practical
    mask_fast_centered = find_fast_pairs(dists['speed_vcm'],[620*u.km/u.s, 120*u.km/u.s],option='within')
    mask_aligned_centered = find_aligned_pairs(dists['mu'],[-0.88, .08],option='within')

    # Combine masks
    mask_trad = mask_neg & mask_close
    mask_real = mask_neg & mask_close & mask_fast & mask_aligned
    mask_practical = mask_neg & mask_close & mask_fast_centered & mask_aligned_centered

    return [mask_trad, mask_real, mask_practical]


def apply_masks(dists,masks):
    all_filtered_dists = dists.copy()
    for key in dists.keys():
        filtered_dists = []
        dist = dists[key]

        for mask in masks:
                filtered_dists.append(dist[mask])
        
        all_filtered_dists[key] = filtered_dists

    return all_filtered_dists


def get_stats_results(filtered_dists, labels):

    def ks_test(d1, d2):
        res = stats.ks_2samp(d1, d2)
        return res.pvalue, res.statistic

    filtered = [d.value for d in filtered_dists]
    trad = filtered[0]

    data = {
        'Mode':   [stats.mode(d).mode for d in filtered],
        'Median': [np.median(d) for d in filtered],
        'Mean':   [np.mean(d) for d in filtered],
        'Std':    [np.std(d, ddof=1) for d in filtered],
        'Min':    [np.min(d) for d in filtered],
        'Max':    [np.max(d) for d in filtered],
        'Size':   [len(d) for d in filtered],
    }

    means = data['Mean']
    mean_diff = [np.nan] + [(m - means[0]) / means[0] * 100 for m in means[1:]]
    mean_diff_abs = [np.nan] + [m - means[0] for m in means[1:]]
    ks_results = [(np.nan, np.nan)] + [ks_test(trad, d) for d in filtered[1:]]
    pvals, stats_ = zip(*ks_results)

    data.update({'M diff': mean_diff, 'M diff abs': mean_diff_abs, 'statistic': stats_, 'pvalue': pvals})
    return pd.DataFrame(data, index=labels)

# ------------------------------------------------------ Main loop --------------------------------------------------------------------- ##

def all_cosmo_results(file_names, cosmo_models, cosmo_tags, labels, suffix, vrad_max, vcm_min, mu_max, shift_pos_flag, HubbleFlow_flag):
    all_results = {}
    final_dists = {}

    for i in tqdm(range(len(file_names)), desc='Processing cosmologies'):
        sim_data = read_data(file_names[i], cosmo_models[i])
    
        dists = get_dists(sim_data.copy(), cosmo_models[i], shift_pos_flag, HubbleFlow_flag)
        masks = get_masks(dists.copy(), cosmo_models[i], vrad_max, vcm_min, mu_max)

        filtered_dists = apply_masks(dists.copy(), masks)

        all_results[i] = {
            var: get_stats_results(filtered_dists[var], labels)
            for var in ['mass_LG', 'vrel_rad', 'vrel_tan']
        }

        final_dists[i] = filtered_dists

    
    df_results = pd.concat({k: pd.concat(v) for k, v in all_results.items()}, axis=0)
    df_results.index = pd.MultiIndex.from_tuples(
        [(cosmo_tags[level[0]], level[1], level[2]) for level in df_results.index],
        names=['cosmo', 'variable', 'filter']
    )
    
    records = []
    index_tuples = []

    for i, my_dict in final_dists.items():
        cosmo_tag = cosmo_tags[i] 
        for var, arrays in my_dict.items():
            for j, quantity_array in enumerate(arrays):
                index_tuples.append((cosmo_tag, var, labels[j]))
                records.append(quantity_array)

    index = pd.MultiIndex.from_tuples(index_tuples, names=["cosmo", "variable", "filter"])
    df_dists = pd.DataFrame({'dist': records}, index=index)

    df_dists.to_pickle('results/' + suffix + '/dists.pkl')

    return df_results

# -------------------------------------------------- Significance and average analysis --------------------------------------------------------------------- ##

def all_cosmo_analysis(df_results,suffix):

    df_results['significant'] = df_results['pvalue'] < 0.05

    # Statistical summary for significant cosmologies
    df_filtered = df_results[(df_results['significant'] == True) | (df_results['pvalue'].isna())]

    df_means = df_filtered.groupby(level=[1,2]).mean()
    df_counts = df_filtered.groupby(level=[1,2]).size()

    df_counts.name = 'counts'
    df_cosmo_analysis_significant = pd.concat([df_means,df_counts],axis=1,keys=['mean','num_cosmo'])

    # Statistical summary for all cosmologies
    df_means_all = df_results.groupby(level=[1,2]).mean()
    df_counts_all = df_results.groupby(level=[1,2]).size()
    df_counts_all.name = 'counts_all'
    df_cosmo_analysis_all = pd.concat([df_means_all,df_counts_all],axis=1,keys=['mean_all','num_cosmo_all'])

    # Save results
    df_results.to_pickle('results/' + suffix + '/cosmo_results.pkl')
    df_cosmo_analysis_significant.to_pickle('results/' + suffix + '/cosmo_analysis.pkl')
    df_cosmo_analysis_all.to_pickle('results/' + suffix + '/cosmo_analysis_all.pkl')

    return None


def main(args):
    # Create directories
    os.makedirs('results/' + args.suffix, exist_ok=True)

    # Data files
    file_names, cosmo_names, cosmo_tags = read_files('data')
    cosmo_models = create_cosmo(cosmo_names)

    # Sample names
    labels = ['Trad','Real','Pract']

    # Get results
    df_results = all_cosmo_results(file_names,cosmo_models,cosmo_tags,labels,args.suffix,
                                   vrad_max=args.vrad_max*u.km/u.s, 
                                   vcm_min=args.vcm_min*u.km/u.s, 
                                   mu_max=args.mu_max,
                                   shift_pos_flag=args.shift_pos_flag,
                                   HubbleFlow_flag=args.HubbleFlow_flag)

    # Significance and average analysis
    all_cosmo_analysis(df_results,args.suffix)


if __name__ == '__main__':
    obs_data = read_obs_data('results/montecarlo_results.csv')

    p = argparse.ArgumentParser()
    p.add_argument('--suffix', required=True)
    p.add_argument('--vrad_max', type=float, default=0.0) # km/s
    p.add_argument('--vcm_min',type=float, default=obs_data['v_cm'][0] - obs_data['v_cm'][1]) # 605 km/s
    p.add_argument('--mu_max', type=float, default=obs_data['mu'][0] + obs_data['mu'][1]) # -0.87
    p.add_argument('--HubbleFlow_flag', action='store_false', default=True)
    p.add_argument('--shift_pos_flag', action='store_false', default=True)

    args = p.parse_args()

    main(args)