import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt

## --------------------------------------- FUNCTIONS ---------------------------------------------------- ##

def spher2cart(v,l,b):
    vx = v*np.cos(l)*np.cos(b)
    vy = v*np.sin(l)*np.cos(b)
    vz = v*np.sin(b)
    return np.array([vx,vy,vz])

def get_mu(v_LG,relative_pos):
    dot_product = np.dot(v_LG,relative_pos)
    v_LG_mag = np.linalg.norm(v_LG)
    relative_pos_mag = np.linalg.norm(relative_pos)
    mu = dot_product/(v_LG_mag*relative_pos_mag)
    return mu

def cov_matrix(a,b,PA):
    varX1 = a**2 * np.cos(PA)**2 + b**2 * np.sin(PA)**2
    varX2 = a**2 * np.sin(PA)**2 + b**2 * np.cos(PA)**2
    cov12 = (a**2 - b**2) * np.sin(PA) * np.cos(PA)
    return np.array([[varX1,cov12],
                      [cov12,varX2]])


## ---------------------------------------- CONSTANTS  -------------------------------------------------- ##

# Barycenter velocity of the LG relative to the cmb (Planck 2018 results)
vb = np.array([620, 15]) # km/s
l_vb = np.deg2rad(np.array([271.9,2]))
b_vb = np.deg2rad(np.array([29.6,1.4]))

# MW 
d_MW = np.array([8.29, 0.16])  # kpc (McMillan 2011)

l0_MW, b0_MW = 359.944215, -0.046079  # Central position (deg)
a_MW, b_MW = 0.035 / 3600, 0.023 / 3600  # Semi-major and semi-minor axes (deg)
PA_MW = 59.0  # Position angle in degrees 

# M31 
d_M31 = np.array([761,11])  # kpc (Li et al. 2021)

l0_M31, b0_M31 = 121.174410, -21.572925  # Central position (deg)
a_M31, b_M31 = 0.08 / 3600, 0.08 / 3600  # Semi-major and semi-minor axes (deg)
PA_M31 = 0 # Position angle in degrees 


## ----------------------------------  MONTE CARLO SIMULATION  --------------------------------------- ##
n = 10000

v_b_samples = np.random.normal(vb[0],vb[1],n)
l_vb_samples = np.random.normal(l_vb[0],l_vb[1],n)
b_vb_samples = np.random.normal(b_vb[0],b_vb[1],n)

MW_coords_samples = np.random.multivariate_normal([l0_MW, b0_MW], cov_matrix(a_MW,b_MW,np.deg2rad(PA_MW)), n)
d_MW_samples = np.random.normal(d_MW[0],d_MW[1],n)

M31_coords_samples = np.random.multivariate_normal([l0_M31, b0_M31], cov_matrix(a_M31,b_M31,np.deg2rad(PA_M31)), n)
d_M31_samples = np.random.normal(d_M31[0],d_M31[1],n)

r = np.zeros((n,3))
r_mag = np.zeros(n)
mu = np.zeros(n)

for i in tqdm(range(n)):
    r_M31_sample = spher2cart(d_M31_samples[i],np.deg2rad(M31_coords_samples[i,0]),np.deg2rad(M31_coords_samples[i,1]))
    r_MW_sample = spher2cart(d_MW_samples[i],np.deg2rad(MW_coords_samples[i,0]),np.deg2rad(MW_coords_samples[i,1]))
    v_LG_sample = spher2cart(v_b_samples[i],l_vb_samples[i],b_vb_samples[i])

    r[i,:] = r_M31_sample - r_MW_sample
    r_mag[i] = np.linalg.norm(r[i,:])

    mu[i] = get_mu(v_LG_sample,r[i,:])


## ------------------------------- MEAN AND STANDARD DEVIATION -------------------------------------- ##

# Positions
l_MW_mean = np.mean(MW_coords_samples[:,0])
l_MW_std = np.std(MW_coords_samples[:,0],ddof=1)

b_MW_mean = np.mean(MW_coords_samples[:,1])
b_MW_std = np.std(MW_coords_samples[:,1],ddof=1)

l_M31_mean = np.mean(M31_coords_samples[:,0])
l_M31_std = np.std(M31_coords_samples[:,0],ddof=1)

b_M31_mean = np.mean(M31_coords_samples[:,1])
b_M31_std = np.std(M31_coords_samples[:,1],ddof=1)


# Relative position
mean_r = np.mean(r,axis=0)
std_r = np.std(r,axis=0,ddof=1)

mean_r_mag = np.mean(r_mag)
std_r_mag= np.std(r_mag,ddof=1)


# Angle between vectors
mu_mean = np.mean(mu)
mu_std = np.std(mu,ddof=1)


# Other observables
v_tan = [82.4,31.2] # km/s (van der Marel et al. 2019)
v_rad = [-109.3,4.4] # km/s (van der Marel et al. 2012)
M_LG = [4.5e12,1e12] # Msun (Chamberlain et al. 2022)

# Save results to pandas dataframe
results = pd.DataFrame({"v_cm": vb,
                        "v_tan": v_tan,
                        "v_rad": v_rad,
                        "M_LG": M_LG,
                        "MW_coords": [[l_MW_mean, l_MW_std], [b_MW_mean, b_MW_std]],
                        "M31_coords": [[l_M31_mean, l_M31_std], [b_M31_mean, b_M31_std]],
                        "r": [mean_r, std_r],
                        "r_mag": [mean_r_mag, std_r_mag],
                        "mu": [mu_mean, mu_std]})

print(results)

results.to_csv("/home/isaac/Documentos/LG kinematic alignment/results/montecarlo_results.csv",index=False)