from all_cosmo import read_obs_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import binned_statistic_2d
from scipy.stats import spearmanr

plt.style.use('seaborn-v0_8-deep')


def cumulative_distribution(dist):
    '''Calculate the cumulative distribution'''
    
    data_sorted = np.sort(dist)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    
    return data_sorted, cdf


def plot_cdf(dists,label,obs_value,uncertainty,xlim,variable,fig_name,iteration_dir,show_fig=False):
    '''Plot the cumulative distributions'''
    plot_path = os.path.join(iteration_dir,fig_name + '.pdf')
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    alphas = [1, 0.8, 0.6, 1, 0.8, 0.6]

    for i in range(len(dists)):
        mu_sorted, cdf = cumulative_distribution(dists[i])
        plt.plot(mu_sorted, cdf, label=label[i],linestyle=line_styles[i],alpha=alphas[0], linewidth=2)
    
    plt.fill_betweenx([0, 1], obs_value - uncertainty, obs_value + uncertainty, color='skyblue', alpha=0.3, label='Observed value')
    plt.xlim(xlim)
    plt.xlabel(variable)
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_path)
    if show_fig:
        plt.show()
    plt.clf()


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


def plot_pvalues(df_results,cosmo_names,plot_directory):
    '''Plot p-values for radial and tangential velocities across cosmologies'''
    df_results = df_results.sort_index(level='cosmo', key=lambda x: x.str[1:].astype(int))
    cosmo_names = np.array(df_results.index.get_level_values(0).unique())
    pvalues_vrad = df_results.xs(('vrel_rad', 'Real'), level=['variable', 'filter'])['pvalue']
    pvalues_vtan = df_results.xs(('vrel_tan', 'Real'), level=['variable', 'filter'])['pvalue']

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,6))
    ax1,ax2 = axes.flatten()

    ax2.bar(cosmo_names, np.log10(pvalues_vtan), color='C0', alpha=0.7, label=r'$v_t$ ')
    ax1.bar(cosmo_names,np.log10(pvalues_vrad), color='C1', alpha=0.7, label=r'$v_r$ ')

    ax1.axhline(y=np.log10(0.05), color='k', linestyle='--', label=r'$\log_{10}(0.05)$')
    ax2.axhline(y=np.log10(0.05), color='k', linestyle='--')

    fig.supylabel(r'$\log_{10}$(p-value)')
    ax2.set_xlabel('Cosmology')

    ax2.set_xticks(np.arange(len(cosmo_names)))
    ax2.set_xticklabels(cosmo_names, rotation=60, fontsize=8)

    fig.legend(loc="center", bbox_to_anchor=(0.4, 0.6))
    plt.tight_layout()
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

    fig, axes = plt.subplots(nrows=2, ncols=2, sharey='row',sharex='col',figsize=(12, 8))

    ax1,ax2, ax3, ax4 = axes.flatten()

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

    fig.legend(loc="center left", bbox_to_anchor=(0.1, 0.3))
    for ax in axes.flatten():
        ax.grid()

    plt.suptitle('Cosmology 004',y=0.95)
    plt.figtext(0.5, 0.5, 'Cosmology 130 ', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(plot_directory + '/cdf_velocities.pdf')


def plot_num_pairs(df_results, cosmo_names,plot_directory):
    '''Plot number of pairs for each cosmology and filter'''
    filts = ['Trad','Real','Pract']
    labels = ['Traditional','Realistic','Practical']
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
    # Sort cosmologies globally and consistently
    all_cosmos = sorted(
        df_results.index.get_level_values('cosmo').unique(),
        key=lambda x: int(x[1:])
    )

    # Ensure results are sorted by cosmology numeric order
    df_results = df_results.sort_index(level='cosmo', key=lambda x: x.map(lambda s: int(s[1:])))

    # Prepare figure
    fig, axs = plt.subplots(
        nrows=len(variables),
        ncols=1,
        figsize=(15, 5 * len(variables)),
        sharex=True
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
            axb.plot(
                x_vals,
                delta,
                marker='x',
                linestyle='--',
                color=colors[(i + 1) % len(colors)],
                alpha=0.7,
                label=f'$\\Delta$ {filters[i + 1]} vs {filters[0]} (\\%)'
            )

        # Labels and appearance
        ax.set_ylabel(f'Mean {var_label} [km/s]')
        axb.set_ylabel('Difference (\\%)')
        ax.grid(True, zorder=0)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(all_cosmos, rotation=60, fontsize=10)

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
        loc="center",
        bbox_to_anchor=(0.3, 0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, filename + f'_{opt}.pdf'))
    plt.close()


def plot_scatter_matrix_cosmo(df_results, cosmo_params_df, df_importance, var, filt, title, plot_directory, filename, num_features=5):
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


def plot_scatter_matrix_mdiff(importances_df, df_results, cosmo_params_df, var, filt,title, plot_directory, filename,num_features=5):
    '''Plot scatter matrix of top features vs mean difference'''
    df_results = df_results.sort_index(level='cosmo', key=lambda x: x.str[1:].astype(int))

    top_features = importances_df['Feature'].head(num_features).tolist()
    top_features.sort()

    if var == 'vrel_tan':
        top_features = [r'$A_s$',r'$\sigma_8$', r'$\sigma_{8,cb}$', r'$h$', r'$\Omega_b$']

    v_var = df_results.xs((var,filt), level=['variable','filter'])

    df_joined = v_var.join(cosmo_params_df)
    
    fig = plt.figure(figsize=(10, 6))
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
            markers=['X', 'o'],
            palette={0: 'black', 1: 'gray'},
            ax=ax,
            s=30,
        )
        ax.legend_.remove()
        ax.set_ylabel('')
        ax.set_xlabel(param)

        corr, _ = spearmanr(df_joined[param], df_joined["M diff"])
        text_x = 0.05 if var == "vrel_rad" else 0.6
        ax.text(text_x, 0.95, r'$\rho$' + f'={corr:.2f}', 
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black'))

    fig.supylabel("Mean difference in " + r"$v_r$" + " ($\\%$)" if var == "vrel_rad" else "Mean difference in " + r"$v_t$" + " ($\\%$)")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ['Not significant', 'Significant'], loc='lower right')

    plt.tight_layout()
    plt.savefig(plot_directory + '/' + filename + '.pdf')


def plot_feature_importance(df,title,plot_directory,filename):
    '''Plot feature importance as a bar plot'''
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_title(title)
    sns.barplot(x='Mean Importance', y='Feature', data=df, hue='Mean Importance', palette='viridis', legend=False, ax=ax)
    plt.tight_layout()
    plt.savefig(plot_directory + '/feature_importance_' + filename + '.pdf', dpi=300)


def plot_vr_heatmap(v_b, mu, v_r,
                    bins_vb=30, bins_mu=30,
                    statistic='mean',
                    vmin=None, vmax=None,
                    cmap='RdBu_r',
                    xlabel=r'Barycenter speed $v_b$ [km/s]',
                    ylabel=r'Alignment $\mu = \cos\theta$',
                    title=r'Mean $v_r$ as a function of $v_b$ and $\mu$',
                    cbar_label=r'$v_r$ [km/s]',
                    plot_directory='.',
                    filename='vr_heatmap',
                    return_data=False):
    '''Plot heatmap of mean v_r as a function of v_b and mu'''

    # Compute binned statistic
    H, vb_edges, mu_edges, binnumber = binned_statistic_2d(
        v_b, mu, v_r, statistic=statistic, bins=[bins_vb, bins_mu]
    )

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(H.T, origin='lower', aspect='auto',
                   extent=[vb_edges[0], vb_edges[-1],
                           mu_edges[0], mu_edges[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_title(title)
    plt.tight_layout()
    plt.savefig(plot_directory + '/' + filename + '.pdf', dpi=300)

    if return_data:
        return H, vb_edges, mu_edges


def main():
    #suffix = input('Enter suffix for results: ')
    suffix = 'final_rev_no_redshift'

    # Create a directory to save the plots
    plot_directory = 'figures/' + suffix
    os.makedirs(plot_directory, exist_ok=True)

    # Font size
    plt.rcParams.update({
        'text.usetex': True,  # Use LaTeX for text rendering
        'font.family': 'serif',
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{textcomp}',
        'figure.titlesize': 18,     # Figure title size
        'axes.labelsize': 18,       # Axis labels size
        'xtick.labelsize': 14,      # X-axis tick label size
        'ytick.labelsize': 14,      # Y-axis tick label size
        'legend.fontsize': 14,      # Legend font size
        'lines.linewidth': 1.5,     # Line width
        'font.size': 18,            # Font size
    })


    df_results = pd.read_pickle('results/' + suffix + '/cosmo_results.pkl')
    df_dists = pd.read_pickle('results/' + suffix + '/dists.pkl')
    df_cosmo_params = pd.read_csv('analysis v2/filtered_cosmologies_latex_final.csv', index_col='cosmo')

    cosmo_names = np.array(df_results.index.get_level_values(0).unique())

    labels = ['Traditional','Realisitic','Practical']

    obs_data = read_obs_data('results/montecarlo_results.csv')
    for key in obs_data.keys():
        try:
            obs_data[key] = obs_data[key].value
        except:
            pass

    
    # Dists base cosmology
    plot_cdf_kinematics(df_dists,obs_data,labels,plot_directory)
    
    # Cosmology analysis
    plot_pvalues(df_results,cosmo_names,plot_directory)

    plot_cdf_comparison(df_dists,'c004','c130',obs_data,plot_directory)

    plot_num_pairs(df_results,cosmo_names,plot_directory)

    plot_means(df_results,['vrel_rad','vrel_tan'],['$v_r$','$v_t$'],['Trad','Real'],plot_directory,'mean_velocities_real',opt='all_cosmo')
    plot_means(df_results,['vrel_rad','vrel_tan'],['$v_r$','$v_t$'],['Trad','Pract'],plot_directory,'mean_velocities_pract',opt='all_cosmo')

    # Feature importance
    filt1 = 'Pract'
    df_importance_vrel_rad_pract = pd.read_csv('results/' + suffix + '/random_forest_' + filt1 + '/feature_importance_vrel_rad.csv')
    df_importance_vrel_tan_pract = pd.read_csv('results/' + suffix + '/random_forest_' + filt1 + '/feature_importance_vrel_tan.csv')

    filt2 = 'Real'
    df_importance_vrel_rad_real = pd.read_csv('results/' + suffix + '/random_forest_' + filt2 + '/feature_importance_vrel_rad.csv')
    df_importance_vrel_tan_real = pd.read_csv('results/' + suffix + '/random_forest_' + filt2 + '/feature_importance_vrel_tan.csv')

    plot_feature_importance(df_importance_vrel_rad_real,r'Feature importance for $v_r$ - Realistic filter',plot_directory,'vrel_rad_real')
    plot_feature_importance(df_importance_vrel_tan_real,r'Feature importance for $v_t$ - Realistic filter',plot_directory,'vrel_tan_real')

    plot_feature_importance(df_importance_vrel_rad_pract,r'Feature importance for $v_r$ - Practical filter',plot_directory,'vrel_rad_pract')
    plot_feature_importance(df_importance_vrel_tan_pract,r'Feature importance for $v_t$ - Practical filter',plot_directory,'vrel_tan_pract')


    # Scatter matrix with mean differences

    plot_scatter_matrix_mdiff(df_importance_vrel_rad_pract,df_results,df_cosmo_params,
                              'vrel_rad','Pract',r'Top features for $v_r$ - Practical filter',
                              plot_directory,'scatter_matrix_mdiff_vrel_rad_pract')
    plot_scatter_matrix_mdiff(df_importance_vrel_tan_pract,df_results,df_cosmo_params,
                              'vrel_tan','Pract',r'Top features for $v_t$ - Practical filter',
                              plot_directory,'scatter_matrix_mdiff_vrel_tan_pract')
    
    plot_scatter_matrix_mdiff(df_importance_vrel_rad_real,df_results,df_cosmo_params,
                              'vrel_rad','Real',r'Top features for $v_r$ - Realistic filter',
                              plot_directory,'scatter_matrix_mdiff_vrel_rad_real')
    plot_scatter_matrix_mdiff(df_importance_vrel_tan_real,df_results,df_cosmo_params,
                              'vrel_tan','Real',r'Top features for $v_t$ - Realistic filter',
                              plot_directory,'scatter_matrix_mdiff_vrel_tan_real')
    
    
    # Scatter matrix with cosmology parameters

    plot_scatter_matrix_cosmo(df_results,df_cosmo_params,df_importance_vrel_rad_pract,
                              'vrel_rad','Pract',r'Top features for $v_r$ - Practical filter',
                              plot_directory,'scatter_matrix_cosmo_vrel_rad_pract')
    plot_scatter_matrix_cosmo(df_results,df_cosmo_params,df_importance_vrel_tan_pract,
                              'vrel_tan','Pract',r'Top features for $v_t$ - Practical filter',
                              plot_directory,'scatter_matrix_cosmo_vrel_tan_pract')
    
    plot_scatter_matrix_cosmo(df_results,df_cosmo_params,df_importance_vrel_rad_real,
                                'vrel_rad','Real',r'Top features for $v_r$ - Realistic filter',
                                plot_directory,'scatter_matrix_cosmo_vrel_rad_real')
    plot_scatter_matrix_cosmo(df_results,df_cosmo_params,df_importance_vrel_tan_real,
                              'vrel_tan','Real',r'Top features for $v_t$ - Realistic filter',
                              plot_directory,'scatter_matrix_cosmo_vrel_tan_real')
    
    # Heatmap
    base_cosmo_dists = df_dists.xs('c000', level='cosmo')
    v_rel_rad = base_cosmo_dists.xs('vrel_rad').values
    v_rel_tan = base_cosmo_dists.xs('vrel_tan').values
    mu = base_cosmo_dists.xs('mu').values
    vcm = base_cosmo_dists.xs('speed_vcm').values

    plot_vr_heatmap(
        v_b=vcm[0][0], mu=mu[0][0], v_r=v_rel_rad[0][0],
        cbar_label=r'$v_r$ [km/s]',
        title='Traditional sample: mean $v_r$ vs $v_b$ and $\\mu$',
        filename='vr_heatmap_trad_mean',
        plot_directory=plot_directory
    )

    plot_vr_heatmap(
        v_b=vcm[0][0], mu=mu[0][0], v_r=v_rel_tan[0][0],
        cbar_label=r'$v_t$ [km/s]',
        filename='vt_heatmap_trad_mean',
        plot_directory=plot_directory,
        title='Traditional sample: mean $v_t$ vs $v_b$ and $\\mu$'
    )
                      
    
    
if __name__ == "__main__":
    main()