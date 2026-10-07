import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from all_cosmo import read_obs_data
from matplotlib import gridspec
from scipy.stats import binned_statistic, binned_statistic_2d, spearmanr

plt.style.use('seaborn-v0_8-deep')

# Paper layout (openjournal, twocolumn, 10pt body): figures are drawn at the size they are
# printed so the text matches the paper. Widths in inches, from \columnwidth and \textwidth.
COLUMN_WIDTH = 245.27 / 72
TEXT_WIDTH = 513.12 / 72

PAPER_STYLE = {
    'text.usetex': True,  # Use LaTeX for text rendering
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{textcomp}',
    'font.size': 9,             # Same as the captions; body text is 10pt
    'axes.labelsize': 9,
    'figure.titlesize': 9,
    'figure.labelsize': 9,      # fig.supxlabel / fig.supylabel
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1,
    'lines.markersize': 3.5,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'patch.linewidth': 0.5,
    'savefig.pad_inches': 0.02,
}


def cumulative_distribution(dist):
    '''Calculate the cumulative distribution'''
    
    data_sorted = np.sort(dist)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    
    return data_sorted, cdf


# AbacusSummit cosmology families as described in the paper (Section 2), in plotting order:
# base = primary c000, secondary c001-c004 and neutrino variants c009, c019, c020
COSMO_FAMILIES = ['Base', 'Reference', 'Linear derivative grid', 'Emulator grid']
REFERENCE_COSMOS = [10, 12, 13, 14, 15, 16, 17, 18, 21, 22]


def cosmo_family(name):
    '''Return the family label of a cosmology name such as "c104"'''
    number = int(name[1:])
    if number >= 130:
        return 'Emulator grid'
    if number >= 100:
        return 'Linear derivative grid'
    if number in REFERENCE_COSMOS:
        return 'Reference'
    return 'Base'


def sort_cosmos(cosmo_names):
    '''Sort cosmologies by family, then by number, so each family is contiguous'''
    return sorted(cosmo_names, key=lambda name: (COSMO_FAMILIES.index(cosmo_family(name)), int(name[1:])))


SHORT_FAMILY_NAMES = {'Base': 'Base', 'Reference': 'Ref.',
                      'Linear derivative grid': 'Derivative grid', 'Emulator grid': 'Emulator grid'}


def mark_cosmo_families(ax, cosmo_names, show_names=False, short_names=False, label_starts=True, fontsize=8):
    '''Separate cosmology families with vertical lines and tick the first model of each.
    Assumes cosmo_names is ordered with sort_cosmos and plotted at x = 0, 1, 2, ...
    For single-column figures use short_names and label_starts=False, since the small
    families are too narrow for full names and cosmology labels.'''
    families = [cosmo_family(name) for name in cosmo_names]
    starts = [i for i in range(len(families)) if i == 0 or families[i] != families[i - 1]]
    ends = starts[1:] + [len(families)]

    for start in starts[1:]:
        ax.axvline(start - 0.5, color='0.5', linestyle=':', linewidth=1, zorder=1)

    ax.set_xticks(starts)
    ax.set_xticklabels([cosmo_names[i] for i in starts] if label_starts else [])
    ax.set_xticks(np.arange(len(cosmo_names)), minor=True)
    ax.set_xlim(-0.5, len(cosmo_names) - 0.5)
    ax.grid(False, axis='x')  # the family separators replace vertical grid lines

    if show_names:
        for start, end in zip(starts, ends):
            name = SHORT_FAMILY_NAMES[families[start]] if short_names else families[start]
            ax.text((start + end - 1) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=fontsize)


def plot_cdf(dists,label,obs_value,uncertainty,xlim,variable,fig_name,iteration_dir,show_fig=False,
             figsize=(0.49 * TEXT_WIDTH, 2.6)):
    '''Plot the cumulative distributions'''
    plt.figure(figsize=figsize)
    plot_path = os.path.join(iteration_dir,fig_name + '.pdf')
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    alphas = [1, 0.8, 0.6, 1, 0.8, 0.6]

    for i in range(len(dists)):
        mu_sorted, cdf = cumulative_distribution(dists[i])
        plt.plot(mu_sorted, cdf, label=label[i],linestyle=line_styles[i],alpha=alphas[0])
    
    plt.fill_betweenx([0, 1], obs_value - uncertainty, obs_value + uncertainty, color='skyblue', alpha=0.3, label='Observed value')
    plt.xlim(xlim)
    plt.locator_params(axis='x', nbins=5)
    plt.xlabel(variable)
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_path)
    if show_fig:
        plt.show()
    plt.close()


def plot_cdf_kinematics(df_dists,obs_data,labels,plot_directory, num_cosmo='c000'):
    '''Plot CDFs for kinematic variables for a given cosmology'''
    dists_mu_raw = df_dists.xs((num_cosmo, 'mu'), level=('cosmo', 'variable')).values
    dists_vcm_raw = df_dists.xs((num_cosmo, 'speed_vcm'), level=('cosmo', 'variable')).values
    dists_vrel_rad_raw = df_dists.xs((num_cosmo, 'vrel_rad'), level=('cosmo', 'variable')).values
    dists_vrel_tan_raw = df_dists.xs((num_cosmo, 'vrel_tan'), level=('cosmo', 'variable')).values

    dists_mu = [x[0].value for x in dists_mu_raw]
    dists_vcm = [x[0].value for x in dists_vcm_raw]
    dists_vrel_rad = [x[0] for x in dists_vrel_rad_raw]
    dists_vrel_tan = [x[0] for x in dists_vrel_tan_raw]

    labels_mu = labels.copy()
    labels_vcm = labels.copy()

    uniform = np.random.uniform(-1,1,len(dists_mu[0]))
    dists_mu.append(uniform)
    labels_mu.append('Uniform')
    
    plot_cdf(dists_mu,labels_mu,obs_data['mu'][0],obs_data['mu'][1],
             [-1,1],r'Alignment $\mu = \cos \theta$','mu_cdf_' + num_cosmo,plot_directory)

    plot_cdf(dists_vcm,labels_vcm,obs_data['v_cm'][0],obs_data['v_cm'][1],
             [0,1200],r'Barycenter speed $v_{b}$ $[\text{kms}^{-1}]$','v_cm_cdf_' + num_cosmo,plot_directory)

    plot_cdf(dists_vrel_rad,labels,obs_data['v_rad'][0],obs_data['v_rad'][1], 
             [-500,0],r'$v_{r}$ $[\text{kms}^{-1}]$','v_rel_rad_cdf_' + num_cosmo,plot_directory)

    plot_cdf(dists_vrel_tan,labels,obs_data['v_tan'][0],obs_data['v_tan'][1],
            [0,500],r'$v_{t}$ $[\text{kms}^{-1}]$','v_rel_tan_cdf_' + num_cosmo,plot_directory)


def plot_vr_vs_mu(v_b, mu, v_r,
                  vb_cut=620,
                  bins_mu=20,
                  min_count=20,
                  mu_obs = [-0.88,0.01],
                  xlabel=r'Alignment $\mu = \cos\theta$',
                  ylabel=r'Mean $v_r$ [km/s]',
                  plot_directory='.',
                  filename='vr_vs_mu_profile',
                  return_data=False):
    '''Fine-binned profile of mean v_r vs. mu, for all traditional pairs
    and for the subset with v_b > vb_cut. Error bars are the standard
    error of the mean; bins with fewer than min_count pairs are dropped.'''

    edges = np.linspace(-1, 1, bins_mu + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    samples = [
        ('All traditional pairs', np.ones_like(mu, dtype=bool), 'C0', 'o'),
        (rf'$v_b > {vb_cut:g}$ km/s', v_b > vb_cut, 'C3', 's'),
    ]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout='constrained')
    results = {}
    offset = 0.25 * (edges[1] - edges[0])  # small x-shift so error bars don't overlap

    for i, (label, mask, color, marker) in enumerate(samples):
        mean = np.full(bins_mu, np.nan)
        sem = np.full(bins_mu, np.nan)
        count = np.zeros(bins_mu, dtype=int)

        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            last = (j == bins_mu - 1)
            in_bin = mask & (mu >= lo) & ((mu <= hi) if last else (mu < hi))
            x = v_r[in_bin]
            count[j] = x.size
            if x.size >= min_count:
                mean[j] = x.mean()
                sem[j] = x.std(ddof=1) / np.sqrt(x.size)

        keep = ~np.isnan(mean)
        shift = (i - 0.5) * offset
        ax.errorbar(centers[keep] + shift, mean[keep], yerr=sem[keep],
                    fmt=marker + '-', capsize=2, color=color, label=label)
        results[label] = dict(centers=centers, mean=mean, sem=sem, count=count)

    # Reference line: mean of the full traditional sample
    ax.axhline(v_r.mean(), color='gray', ls='--', lw=1,label='Mean of all traditional pairs')

    # Observed value band
    plt.fill_betweenx(ax.get_ylim(), mu_obs[0] - mu_obs[1], mu_obs[0] + mu_obs[1], color='skyblue', alpha=0.3, label=r'Observed $\mu$ value')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.legend(loc='outside upper center', ncol=2)
    plt.savefig(plot_directory + '/' + filename + '.pdf', dpi=300)


    if return_data:
        return results


def plot_pvalues(df_results,cosmo_names,plot_directory):
    '''Plot p-values for radial and tangential velocities across cosmologies'''
    cosmo_names = sort_cosmos(df_results.index.get_level_values('cosmo').unique())
    pvalues_vrad = df_results.xs(('vrel_rad', 'Real'), level=['variable', 'filter'])['pvalue'].reindex(cosmo_names)
    pvalues_vtan = df_results.xs(('vrel_tan', 'Real'), level=['variable', 'filter'])['pvalue'].reindex(cosmo_names)
    x_vals = np.arange(len(cosmo_names))

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(COLUMN_WIDTH, 2.9), layout='constrained')
    ax1,ax2 = axes.flatten()

    ax2.bar(x_vals, np.log10(pvalues_vtan), color='C0', alpha=0.7, label=r'$v_t$ ')
    ax1.bar(x_vals, np.log10(pvalues_vrad), color='C1', alpha=0.7, label=r'$v_r$ ')

    ax1.axhline(y=np.log10(0.05), color='k', linestyle='--', label=r'$\log_{10}(0.05)$')
    ax2.axhline(y=np.log10(0.05), color='k', linestyle='--')

    fig.supylabel(r'$\log_{10}$(p-value)')
    ax2.set_xlabel('Cosmology')

    mark_cosmo_families(ax1, cosmo_names, show_names=True, short_names=True, label_starts=False)
    mark_cosmo_families(ax2, cosmo_names, label_starts=False)

    fig.legend(loc='outside lower center', ncol=3)
    plt.savefig(plot_directory + '/pvalues.pdf')


def plot_cdf_comparison(df_velocities,cosmo_max_rad,cosmo_max_tan,obs_data,plot_directory):
    '''Plot CDF comparison for cosmologies with maximum differences'''
    dists_vrel_rad_raw1 = df_velocities.xs((cosmo_max_rad, 'vrel_rad'), level=('cosmo', 'variable')).values
    dists_vrel_tan_raw1 = df_velocities.xs((cosmo_max_rad, 'vrel_tan'), level=('cosmo', 'variable')).values

    dists_vrel_rad1 = [x[0] for x in dists_vrel_rad_raw1]
    dists_vrel_tan1 = [x[0] for x in dists_vrel_tan_raw1]

    dists_vrel_rad_raw2 = df_velocities.xs((cosmo_max_tan, 'vrel_rad'), level=('cosmo', 'variable')).values
    dists_vrel_tan_raw2 = df_velocities.xs((cosmo_max_tan, 'vrel_tan'), level=('cosmo', 'variable')).values

    dists_vrel_rad2 = [x[0] for x in dists_vrel_rad_raw2]
    dists_vrel_tan2 = [x[0] for x in dists_vrel_tan_raw2]

    # One subfigure per cosmology so each row gets its own title without overlapping the panels
    fig = plt.figure(figsize=(0.72 * TEXT_WIDTH, 3.6), layout='constrained')
    subfig_top, subfig_bottom = fig.subfigures(nrows=2, ncols=1)
    ax1, ax2 = subfig_top.subplots(nrows=1, ncols=2, sharey=True)
    ax3, ax4 = subfig_bottom.subplots(nrows=1, ncols=2, sharey=True)
    axes = np.array([ax1, ax2, ax3, ax4])

    ax2.fill_betweenx([0, 1],obs_data['v_tan'][0] - obs_data['v_tan'][1],obs_data['v_tan'][0] + obs_data['v_tan'][1], 
                            alpha=0.3, color='skyblue', label='Observed')
    ax1.fill_betweenx([0, 1],obs_data['v_rad'][0] - obs_data['v_rad'][1],obs_data['v_rad'][0] + obs_data['v_rad'][1], 
                            alpha=0.3, color='skyblue')
    ax3.fill_betweenx([0, 1],obs_data['v_rad'][0] - obs_data['v_rad'][1],obs_data['v_rad'][0] + obs_data['v_rad'][1],
                            alpha=0.3, color='skyblue')
    ax4.fill_betweenx([0, 1],obs_data['v_tan'][0] - obs_data['v_tan'][1],obs_data['v_tan'][0] + obs_data['v_tan'][1],
                            alpha=0.3, color='skyblue')
    

    # Cosmo with max difference for v_rad
    ax1.plot(*cumulative_distribution(dists_vrel_rad1[0]),color='C0',label='Traditional')
    ax1.plot(*cumulative_distribution(dists_vrel_rad1[1]), color='C1',linestyle='dashed',label='Realistic - significant')

    ax2.plot(*cumulative_distribution(dists_vrel_tan1[0]),color='C0')
    ax2.plot(*cumulative_distribution(dists_vrel_tan1[1]), color='C1',linestyle='dashed')

    # Cosmo with max difference for v_tan
    ax3.plot(*cumulative_distribution(dists_vrel_rad2[0]),color='C0')
    ax3.plot(*cumulative_distribution(dists_vrel_rad2[1]), color='C2',linestyle='dashdot',label='Realistic - not significant')

    ax4.plot(*cumulative_distribution(dists_vrel_tan2[0]),color='C0')
    ax4.plot(*cumulative_distribution(dists_vrel_tan2[1]), color='C1',linestyle='dashed')

    ax1.set_ylabel('CDF')
    ax3.set_ylabel('CDF')

    ax1.set_xlim(-400,0)
    ax2.set_xlim(0,400)
    ax3.set_xlim(-400,0)
    ax4.set_xlim(0,400)

    ax3.set_xlabel(r'$v_r$ [$\text{kms}^{-1}$]')
    ax4.set_xlabel(r'$v_t$ [$\text{kms}^{-1}$]')

    fig.legend(loc='outside lower center', ncol=4)
    for ax in axes.flatten():
        ax.grid()

    subfig_top.suptitle(f'Cosmology {cosmo_max_rad[1:]}')
    subfig_bottom.suptitle(f'Cosmology {cosmo_max_tan[1:]}')

    plt.savefig(plot_directory + '/cdf_velocities.pdf')


def plot_num_pairs(df_results, cosmo_names,plot_directory):
    '''Plot number of pairs for each cosmology and filter'''
    filts = ['Trad','Real','Pract']
    labels = ['Traditional','Realistic','Centered']
    sizes = [df_results.xs(('vrel_rad',f),level=('variable','filter'))['Size'] for f in filts]

    fmts = ['-o', '-s', '-D', '-^', '-x']

    fig, ax = plt.subplots(figsize=(8,6))
    for i, size in enumerate(sizes):
        ax.plot(cosmo_names,size, fmts[i],label=labels[i], linewidth=1.5)
        
    ax.set_xlabel('Cosmology')
    ax.set_ylabel('Number of pairs')
    ax.set_yscale('log')
    ax.set_xticklabels([])
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_directory + '/number_of_pairs.pdf')


def plot_means(df_results, variables, variables_labels, filters, plot_directory,filename, opt='all_cosmo',statistic='Mean'):
    '''Plot mean velocities across cosmologies for different filters'''
    # Order cosmologies by family so they can be grouped on the x-axis
    all_cosmos = sort_cosmos(df_results.index.get_level_values('cosmo').unique())

    # Prepare figure
    fig, axs = plt.subplots(
        nrows=len(variables),
        ncols=1,
        figsize=(0.9 * TEXT_WIDTH, 2.3 * len(variables)),
        sharex=True,
        layout='constrained'
    )
    if len(variables) == 1:
        axs = [axs]

    markers = ['s', 'o', '^']
    colors = ['C0', 'C1', 'C2']

    def plot_mean_with_std(ax, var, var_label, statistic=statistic):
        df_results_var = df_results.xs(var, level='variable')

        # Option to filter by significant cosmologies
        if opt == 'all_cosmo':
            df_filtered = df_results_var.copy()
        else:
            cosmo_significant = df_results_var[df_results_var["significant"] == True] \
                                    .index.get_level_values("cosmo").unique()
            df_filtered = df_results_var.loc[
                df_results_var.index.get_level_values("cosmo").isin(cosmo_significant)
            ]

        # Prepare means for each filter, aligned to all_cosmos
        means_var = []
        for f in filters:
            df_filt = df_filtered.xs(f, level='filter')
            df_filt = df_filt.reindex(all_cosmos, level='cosmo')
            means_var.append(df_filt[statistic].values)

        x_vals = np.arange(len(all_cosmos))

        # Plot scatter and shaded band for the first filter
        for i, mean in enumerate(means_var):
            ax.scatter(
                x_vals,
                mean,
                label=filters[i] + ' [km/s]',
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                zorder=3
            )

            if i == 0:
                mean_val = np.nanmean(mean)
                std_val = np.nanstd(mean, ddof=1)
                ax.fill_between(
                    x_vals,
                    mean_val - std_val,
                    mean_val + std_val,
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    label =f'{filters[i]} Std Dev',
                )
                ax.axhline(y=mean_val,
                           xmin=0, xmax=len(all_cosmos)-1, 
                           color=colors[i % len(colors)], 
                           linestyle='--',
                           label=f'{filters[i]} Mean [km/s]')

        # Compute percentage differences (Δ%)
        base = means_var[0]
        axb = ax.twinx()
        for i, f in enumerate(means_var[1:]):
            delta = (f - base) / base * 100
            axb.scatter(
                x_vals,
                delta,
                marker='x',
                color=colors[2],
                alpha=0.7,
                label=f'$\\Delta$ {filters[i + 1]} vs {filters[0]} (\\%)'
            )

        # Labels and appearance
        ax.set_ylabel(f'Mean {var_label} [km/s]')
        axb.set_ylabel('Difference (\\%)')
        ax.grid(True, zorder=0)
        mark_cosmo_families(ax, all_cosmos, show_names=(ax is axs[0]))

        return ax

    # Loop over variables
    for i, var in enumerate(variables):
        axs[i] = plot_mean_with_std(axs[i], var, variables_labels[i])

    axs[-1].set_xlabel('Cosmology')

    # ---- Deduplicated legend (includes twin axes) ----
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc='outside lower center',
        ncol=5
    )

    plt.savefig(os.path.join(plot_directory, filename + f'_{opt}.pdf'))
    plt.close()


def plot_scatter_matrix_cosmo(df_results, cosmo_params_df, df_importance, var, filt, plot_directory, filename, num_features=5):
    '''Plot scatter matrix of top cosmological parameters'''
    top_features = df_importance['Feature'].head(num_features).tolist()
    top_features.sort()
        
    v_var = df_results.xs((var,filt), level=['variable','filter'])

    df_joined = v_var.join(cosmo_params_df)

    fig, ax = plt.subplots(figsize=(8, 8))

    grid = sns.pairplot(
        data=df_joined,
        vars=top_features,
        hue='significant',
        markers=['X', 'o'],
        palette={0: 'black', 1: 'gray'},
        diag_kind='kde',
        corner=True,  
        plot_kws={'alpha': 0.8, 's': 40, 'edgecolor': 'w'}
    )
    
    #grid.figure.suptitle(title, y=1.02)

    # Move Seaborn’s automatically created legend outside
    legend = grid._legend
    legend.set_bbox_to_anchor((0.88, 0.95))
    legend.set_frame_on(True)
    legend.set_title("")
    legend.get_texts()[0].set_text("Not significant")
    legend.get_texts()[1].set_text("Significant")

    os.makedirs(plot_directory, exist_ok=True)
    file_path = os.path.join(plot_directory, f'{filename}.pdf')
    grid.figure.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_matrix_mdiff(importances_df, df_results, cosmo_params_df, var, filt, plot_directory, filename,num_features=5):
    '''Plot scatter matrix of top features vs mean difference'''
    df_results = df_results.sort_index(level='cosmo', key=lambda x: x.str[1:].astype(int))

    top_features = importances_df['Feature'].head(num_features).tolist()
    top_features.sort()

    if var == 'vrel_tan':
        top_features = [r'$A_s$',r'$\sigma_8$', r'$\sigma_{8,cb}$', r'$h$', r'$\Omega_b$']

    v_var = df_results.xs((var,filt), level=['variable','filter'])

    df_joined = v_var.join(cosmo_params_df)
    # Plot A_s in units of 1e-9 so its offset text doesn't collide with the axis label
    df_joined[r'$10^9 A_s$'] = df_joined[r'$A_s$'] * 1e9
    top_features = [r'$10^9 A_s$' if f == r'$A_s$' else f for f in top_features]
    
    fig = plt.figure(figsize=(0.49 * TEXT_WIDTH, 2.9), layout='constrained')
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # Top row: three equal-width plots (each spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Bottom row: two plots centered (they occupy the middle 4 columns as two 2-column blocks)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])

    axes = [ax1, ax2, ax3, ax4, ax5]

    for (ax, param) in zip(axes, top_features):
        sns.scatterplot(
            data=df_joined,
            x=param,
            y="M diff",
            hue="significant",
            style="significant",
            palette={0: 'black', 1: 'gray'},
            ax=ax,
            s=8,
        )
        ax.legend_.remove()
        ax.set_ylabel('')
        ax.set_xlabel(param)

        corr, _ = spearmanr(df_joined[param], df_joined["M diff"])
        ax.set_title(rf'$\rho={corr:.2f}$', fontsize=8, pad=2)

    fig.supylabel("Mean difference in " + r"$v_r$" + " ($\\%$)" if var == "vrel_rad" else "Mean difference in " + r"$v_t$" + " ($\\%$)")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ['Not significant', 'Significant'], loc='outside upper center', ncol=2)

    plt.savefig(plot_directory + '/' + filename + '.pdf')


def plot_feature_importance(df,plot_directory,filename):
    '''Plot feature importance as a bar plot'''
    _, ax = plt.subplots(figsize=(0.42 * TEXT_WIDTH, 2.2))
    #ax.set_title(title)
    sns.barplot(x='Mean Importance', y='Feature', data=df, hue='Mean Importance', palette='viridis', legend=False, ax=ax)
    plt.tight_layout()
    plt.savefig(plot_directory + '/feature_importance_' + filename + '.pdf', dpi=300)


def plot_all_results(suffix='final_rev'):
    # Directory with saved results
    results_dir = '/Volumes/T7/Simulations/LGkin'

    # Create a directory to save the plots
    plot_directory = 'figures/' + suffix
    os.makedirs(plot_directory, exist_ok=True)

    # Font sizes and line widths matching the paper
    plt.rcParams.update(PAPER_STYLE)

    # Load results
    df_results = pd.read_pickle(results_dir + '/results/' + suffix + '/cosmo_results.pkl')
    df_dists = pd.read_pickle(results_dir + '/results/' + suffix + '/dists.pkl')
    df_cosmo_params = pd.read_csv('analysis/filtered_cosmologies_latex.csv', index_col='cosmo')

    cosmo_names = np.array(df_results.index.get_level_values(0).unique())

    labels = ['Traditional','Realistic','Centered']

    obs_data = read_obs_data(results_dir + '/results/montecarlo_results.csv')
    for key in obs_data.keys():
        try:
            obs_data[key] = obs_data[key].value
        except:
            pass

    
    # Dists base cosmology
    plot_cdf_kinematics(df_dists,obs_data,labels,plot_directory)

    # vr profile
    
    base_cosmo = df_dists.loc['c000']
    vr = base_cosmo.xs('vrel_rad').values
    mu = base_cosmo.xs('mu').values
    vcm = base_cosmo.xs('speed_vcm').values
    plot_vr_vs_mu(vcm[0][0].value, mu[0][0].value, vr[0][0].value,
                  vb_cut=obs_data['v_cm'][0] - obs_data['v_cm'][1],
                  mu_obs = obs_data['mu'],
                  plot_directory=plot_directory)

    
    
    # Cosmology analysis
    plot_pvalues(df_results,cosmo_names,plot_directory)


    plot_cdf_comparison(df_dists,'c004','c130',obs_data,plot_directory)


    plot_num_pairs(df_results,cosmo_names,plot_directory)

    plot_means(df_results,['vrel_rad','vrel_tan'],['$v_r$','$v_t$'],['Trad','Real'],plot_directory,'mean_velocities_real',opt='all_cosmo')
    plot_means(df_results,['vrel_rad','vrel_tan'],['$v_r$','$v_t$'],['Trad','Pract'],plot_directory,'mean_velocities_pract',opt='all_cosmo')

    
    # Feature importance
    filt1 = 'Pract'
    df_importance_vrel_rad_pract = pd.read_csv( results_dir + '/results/' + suffix + '/random_forest_' + filt1 + '/feature_importance_vrel_rad.csv')
    df_importance_vrel_tan_pract = pd.read_csv(results_dir + '/results/' + suffix + '/random_forest_' + filt1 + '/feature_importance_vrel_tan.csv')

    plot_feature_importance(df_importance_vrel_rad_pract,plot_directory,'vrel_rad_pract')
    plot_feature_importance(df_importance_vrel_tan_pract,plot_directory,'vrel_tan_pract')


    # Scatter matrix with mean differences

    plot_scatter_matrix_mdiff(df_importance_vrel_rad_pract,df_results,df_cosmo_params,
                              'vrel_rad','Pract',plot_directory,'scatter_matrix_mdiff_vrel_rad_pract')
    plot_scatter_matrix_mdiff(df_importance_vrel_tan_pract,df_results,df_cosmo_params,
                              'vrel_tan','Pract',plot_directory,'scatter_matrix_mdiff_vrel_tan_pract')

    
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Generate the plots with all results and save to specified directory.")
    argument_parser.add_argument("--suffix", type=str, default="final_rev", help="Suffix for results directory.")
    args = argument_parser.parse_args()
    plot_all_results(suffix=args.suffix)