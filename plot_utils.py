import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statannot import add_stat_annotation

def load_dataset_params(dataset):
    n_cls, row_labels, group_labels, sbjs_all = None, None, None, None
    n_modspair = 2
    if dataset == 'eeg_balance':
        n_cls = 4
        row_labels = ['EEG ', 'Pose ', 'EMG ']
        group_labels = np.array(['Supervised', 'Supervised', 'Supervised',
                                 'Unimodal', 'Unimodal', 'Unimodal',
                                 'Cross-modal\n(with pose)', 'Cross-modal\n(with EEG)', 'Cross-modal\n(with EEG)',
                                 'Cross-modal\n(with EMG)', 'Cross-modal\n(with EMG)', 'Cross-modal\n(with pose)'])
        sbjs_all = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
                    'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                    'S19', 'S20', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
                    'S28', 'S29', 'S30', 'S31', 'S32', 'S33']
    elif dataset == 'naturalistic':
        n_cls = 2
        row_labels = ['ECoG ', 'Pose ']
        group_labels = np.array(['Supervised', 'Supervised',
                                 'Unimodal', 'Unimodal',
                                 'Cross-modal\n(with pose)', 'Cross-modal\n(with ECoG)'])
        sbjs_all = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06',
                    'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12']
    elif dataset == 'eeg_arm':
        n_cls = 2
        row_labels = ['EEG ', 'Pose ']
        group_labels = np.array(['Supervised', 'Supervised',
                                 'Unimodal', 'Unimodal',
                                 'Cross-modal\n(with pose)', 'Cross-modal\n(with EEG)'])
        sbjs_all = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
                    'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
        
    elif dataset == 'fingerflex':
        n_cls = 5
        row_labels = ['ECoG ', 'Pose ']
        group_labels = np.array(['Supervised', 'Supervised',
                                 'Unimodal', 'Unimodal',
                                 'Cross-modal\n(with pose)', 'Cross-modal\n(with ECoG)'])
        sbjs_all = ['S01', 'S02', 'S03', 'S04', 'S05',
                    'S07', 'S08', 'S09']
    elif dataset == 'eeg_balance_all3':
        n_cls, n_modspair = 4, 3
        row_labels = ['EEG ', 'Pose ', 'EMG ']
        group_labels = np.array(['Supervised', 'Supervised', 'Supervised',
                                 'Unimodal', 'Unimodal', 'Unimodal',
                                 'Cross-modal', 'Cross-modal', 'Cross-modal'])
        sbjs_all = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
                    'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                    'S19', 'S20', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
                    'S28', 'S29', 'S30', 'S31', 'S32', 'S33']
    return n_cls, row_labels, group_labels, sbjs_all, n_modspair
        
        

def select_meas(measure, dat_type_d, dat_type):
    if measure == 'acc':
        meas_l = dat_type_d[dat_type]+'Accuracy'
    elif measure == 'homogsc':
        meas_l = dat_type_d[dat_type]+'Homogeneity'
    elif measure == 'completesc':
        meas_l = dat_type_d[dat_type]+'Completeness'
    elif measure == 'vscore':
        meas_l = dat_type_d[dat_type]+'V-measure'
    elif measure == 'adjmi':
        meas_l = dat_type_d[dat_type]+'Adj Mutual Information'
    elif measure == 'adjrandsc':
        meas_l = dat_type_d[dat_type]+'Adj rand score'
    return meas_l

def plt_acc_boxscatter(df_in, n_cls=1, yticks=None, ylim_lo=0, axes=(None, None), row_label=None,
                       meas_l=None, group_labs=None, curr_palette=sns.color_palette(), yticklabs=True,
                       use_stats=False, alpha=0.05, verbosity=0, ylim_hi=1.02, measure=None, sbjs_all=None,
                       dataset=None, xticks_sbj=None, markers='o'):
    sns.boxplot(data=df_in, ax = axes[0], palette=curr_palette,showfliers=False) #,showfliers=False,whis=0)
    # sns.swarmplot(data=df_in, ax=axes[0], s=4,color='k')
    if measure == 'acc':
        axes[0].axhline(1/n_cls,linestyle='--',color='k')
    if yticks:
        axes[0].set_yticks(yticks)
    axes[0].set_ylim([ylim_lo, ylim_hi])
    axes[0].spines['left'].set_bounds((ylim_lo, 1))
    if row_label:
        axes[0].set_title(row_label, fontsize=10)
    if yticklabs:
        axes[0].set_ylabel(meas_l, fontsize=9)
    else:
        axes[0].set_yticklabels([])
    axes[0].tick_params(axis='both', labelsize=8)
    # if dataset in ['eeg_balance','naturalistic','eeg_arm']:
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(40)
    axes[0].set_xticks([])

    if use_stats:
        combs, p_vals_out, _ = stats_pairwise(df_in, alpha=alpha, row_label=row_label, sbjs_all=sbjs_all, verbosity=verbosity)
        loc = 'outside' if abs(df_in.max()[0]-ylim_hi) <= .1 else 'inside'
        add_stat_annotation(axes[0], data=df_in, box_pairs=combs, perform_stat_test=False, pvalues=p_vals_out,
                            comparisons_correction=None, line_offset_to_box=0.05, line_offset=.01,
                            text_offset=.01, test=None,
                            text_format='star', loc=loc, verbose=verbosity)

    markersize = 5 if dataset in ['eeg_balance_all3', 'eeg_balance_all3_v2'] else 7
#     markersize = 6 if dataset == 'eeg_arm' else markersize
    sns.lineplot(data=df_in,markers=markers[:len(group_labs)],markersize=markersize,linewidth=0,
                 hue_order=group_labs, ax = axes[1], palette=curr_palette) # marker='o'
    if measure == 'acc':
        axes[1].axhline(1/n_cls,linestyle='--',color='k')
    axes[1].set_yticks(yticks)
    axes[1].set_ylim([ylim_lo, 1.04])
    axes[1].spines['left'].set_bounds((ylim_lo, 1))
    if yticklabs:
        axes[1].set_ylabel(meas_l, fontsize=9)
    else:
        axes[1].set_yticklabels([])
    # axes[1].set_title(row_label + meas_l)
    axes[1].legend_.remove()
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(60)
    axes[1].tick_params(axis='both', labelsize=8)
    if dataset in ['eeg_balance', 'eeg_arm', 'eeg_balance_all3', 'eeg_balance_all3_v2']:
        axes[1].set_xticks(axes[1].get_xticks()[::2])
        axes[1].set_xticklabels(xticks_sbj[::2])

        
def add_letters(dataset):
    if dataset in ['eeg_balance', 'eeg_balance_all3', 'eeg_balance_all3_v2']:
        plt.figtext(0.11,0.915, "(a)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.375,0.915, "(b)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.643,0.915, "(c)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.11,0.5, "(d)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.375,0.5, "(e)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.643,0.5, "(f)", ha="left", va="top", fontsize=10, c='dimgray')
    elif dataset == 'eeg_arm':
        plt.figtext(0.11,0.915, "(e)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.515,0.915, "(f)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.11,0.5, "(g)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.515,0.5, "(h)", ha="left", va="top", fontsize=10, c='dimgray')
    else:
        plt.figtext(0.11,0.915, "(a)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.515,0.915, "(b)", ha="left", va="top", fontsize=10, c='dimgray')
        plt.figtext(0.11,0.5, "(c)", ha="left", va="top", fontsize=10, c='dimgray')  # 0.11,0.46
        plt.figtext(0.515,0.5, "(d)", ha="left", va="top", fontsize=10, c='dimgray')
        
def stats_pairwise(df_sbj, alpha=0.05, row_label=None, sbjs_all=None, verbosity=0):
     #row_labels[k])
    # Friedman test (non-parametric, repeated measures ANOVA)
    df2 = df_sbj.melt(var_name='Group', value_name='Meas')
    n_groups = df_sbj.shape[1]
    df2['Sbj'] = sbjs_all * n_groups
    p_fried = pg.friedman(data=df2, dv='Meas', within='Group', subject='Sbj')['p-unc'][0]
    if verbosity > 0:
        print('')
        print(row_label)
        print(p_fried)

#     if p_fried < alpha:
    # Wilcoxon tests (non-parametric t-tests)
    p_vals = []
    col_labels = np.unique(df2['Group'].values).tolist()
    n_models = len(col_labels)
    for i in range(n_models):
        for j in range(i+1,n_models):
            val1 = df2[df2['Group'] == col_labels[i]].loc[:,'Meas'].values
            val2 = df2[df2['Group'] == col_labels[j]].loc[:,'Meas'].values
            p_vals.append(float(pg.wilcoxon(val1,
                                            val2)['p-val']))

    # Correct for multiple comparisons
    _,p_vals = pg.multicomp(np.asarray(p_vals), alpha=alpha, method='fdr_bh')

    pval_df = np.zeros([n_models,n_models])
    q = 0
    combs, p_vals_out = [], []
    for i in range(n_models):
        for j in range(i+1,n_models):
            pval_df[i,j] = p_vals[q]
            if not np.isnan(p_vals[q]):
                if p_vals[q] < alpha:
                    p_vals_out.append(p_vals[q])
                    combs.append((col_labels[i], col_labels[j]))
            q += 1

    # Create output df with p_values
    df_pval = pd.DataFrame(pval_df,columns=col_labels,index=col_labels)
    df_pval[df_pval==0] = np.nan
    if verbosity > 0:
        print(df_pval)

    return combs, p_vals_out, df_pval
        

def comput_meas_table(measure, subfolders_d, dataset, sbjs_all, rootpth=None, dat_type_d_num=None,
                      dat_type=None, group_labels=None, ncols=None, row_labels=None, n_modspair=2):
    subfolders = subfolders_d[dataset]
    n_groups = len(subfolders)+1 if dataset == 'eeg_balance' else len(subfolders)

    dat = np.zeros((len(sbjs_all), len(subfolders)*n_modspair))
    for i, fold in enumerate(subfolders):
        for j, sbj in enumerate(sbjs_all):
            fID = rootpth + fold + '/' +sbj + '_' + measure + '.npy'
            if os.path.exists(fID):
                dat_tmp = np.load(fID)[:, dat_type_d_num[dat_type]]
            else:
                dat_tmp = np.zeros((n_modspair))
                dat_tmp[:] = np.nan
            dat[j,i*n_modspair:(i*n_modspair+n_modspair)] = dat_tmp
    if dataset == 'eeg_balance_all3':
        inds_neur = np.array([0, 3, 6])
        inds_pose = np.array([1, 4, 7])
        inds_emg = np.array([2, 5, 8])
    else:
        inds_neur = np.array([0, 2, 4])
        inds_pose = np.array([1, 3, 5])
    group_labs_neur = group_labels[inds_neur]
    group_labs_pose = group_labels[inds_pose]
    xticks_sbj = ['P'+str(val+1).zfill(2) for val in range(len(sbjs_all))]
    df_neur = pd.DataFrame(dat[:,inds_neur], columns = group_labs_neur, index = xticks_sbj)
    df_pose = pd.DataFrame(dat[:,inds_pose], columns = group_labs_pose, index = xticks_sbj)
    if dataset == 'eeg_balance_all3':
        group_labs_emg = group_labels[inds_emg]
        df_emg = pd.DataFrame(dat[:,inds_emg], columns = group_labs_emg, index = xticks_sbj)

    if dataset == 'eeg_balance':
        subfolds_other = subfolders_d['eeg_balance_other']
        dat2 = np.zeros((len(sbjs_all), len(subfolds_other)*2))
        for i, fold in enumerate(subfolds_other):
            for j, sbj in enumerate(sbjs_all):
                fID = rootpth + fold + '/' +sbj + '_' + measure + '.npy'
                if os.path.exists(fID):
                    dat_tmp = np.load(fID)[:, dat_type_d_num[dat_type]]
                else:
                    dat_tmp = np.zeros((2))
                    dat_tmp[:] = np.nan
                dat2[j,i*2:(i*2+2)] = dat_tmp
        group_labs_neur = group_labels[::3]
        group_labs_pose = group_labels[1::3]
        group_labs_emg = group_labels[2::3]
        df_neur = pd.DataFrame(np.hstack((dat[:,inds_neur], dat2[:, 4:5])), index = xticks_sbj,
                               columns = group_labs_neur)
        df_pose = pd.DataFrame(np.hstack((dat[:,inds_pose], dat2[:, -1:])), index = xticks_sbj,
                               columns = group_labs_pose)
        inds_emg = np.array([1, 3, 5, 6])
        df_emg = pd.DataFrame(dat2[:,inds_emg], index = xticks_sbj,
                              columns = group_labs_emg)
    
    # Create full table
    dfs_all = [df_neur, df_pose, df_emg] if dataset in ['eeg_balance', 'eeg_balance_all3'] else [df_neur, df_pose]

    df_out = pd.DataFrame(np.zeros((ncols,n_groups)), columns=df_neur.columns,
                          index=row_labels)

    for i, df_curr in enumerate(dfs_all):
        print(list(df_curr.columns))
        for j, (m_val, std) in enumerate(zip(df_curr.mean().values, df_curr.std().values)):
            df_out.iloc[i, j] = '{0:.2f}+-{1:.2f}'.format(m_val, std)
    return df_out


def obtain_dfs(subfolders, sbjs_all, n_modspair, group_labels, dataset,
               rootpth, dat_type_d_num, dat_type, subfolders_d, measure='acc'):
    '''Returns dataframes of performance measure, with each column a different condition'''
    dat = np.zeros((len(sbjs_all), len(subfolders)*n_modspair))
    for i, fold in enumerate(subfolders):
        dat_fld = []
        for j, sbj in enumerate(sbjs_all):
            fID = rootpth + fold + '/' +sbj + '_' + measure + '.npy'
            if os.path.exists(fID):
                dat_tmp = np.load(fID)[:, dat_type_d_num[dat_type]]
            else:
                dat_tmp = np.zeros((n_modspair))
                dat_tmp[:] = np.nan
            dat[j,i*n_modspair:(i*n_modspair+n_modspair)] = dat_tmp
    if dataset == 'eeg_balance_all3':
        inds_neur = np.array([0, 3, 6])
        inds_pose = np.array([1, 4, 7])
        inds_emg = np.array([2, 5, 8])
    else:
        inds_neur = np.array([0, 2, 4])
        inds_pose = np.array([1, 3, 5])
    group_labs_neur = group_labels[inds_neur]
    group_labs_pose = group_labels[inds_pose]
    xticks_sbj = ['P'+str(val+1).zfill(2) for val in range(len(sbjs_all))]
    df_neur = pd.DataFrame(dat[:,inds_neur], columns = group_labs_neur, index = xticks_sbj)
    df_pose = pd.DataFrame(dat[:,inds_pose], columns = group_labs_pose, index = xticks_sbj)
    df_emg, group_labs_emg = None, None
    if dataset == 'eeg_balance_all3':
        group_labs_emg = group_labels[inds_emg]
        df_emg = pd.DataFrame(dat[:,inds_emg], columns = group_labs_emg, index = xticks_sbj)

    if dataset == 'eeg_balance':
        subfolds_other = subfolders_d['eeg_balance_other']
        dat2 = np.zeros((len(sbjs_all), len(subfolds_other)*2))
        for i, fold in enumerate(subfolds_other):
            dat_fld = []
            for j, sbj in enumerate(sbjs_all):
                fID = rootpth + fold + '/' +sbj + '_' + measure + '.npy'
                if os.path.exists(fID):
                    dat_tmp = np.load(fID)[:, dat_type_d_num[dat_type]]
                else:
                    dat_tmp = np.zeros((2))
                    dat_tmp[:] = np.nan
                dat2[j,i*2:(i*2+2)] = dat_tmp
        group_labs_neur = group_labels[::3]
        group_labs_pose = group_labels[1::3]
        group_labs_emg = group_labels[2::3]
        df_neur = pd.DataFrame(np.hstack((dat[:,inds_neur], dat2[:, 4:5])), index = xticks_sbj,
                               columns = group_labs_neur)
        df_pose = pd.DataFrame(np.hstack((dat[:,inds_pose], dat2[:, -1:])), index = xticks_sbj,
                               columns = group_labs_pose)
        inds_emg = np.array([1, 3, 5, 6])
        df_emg = pd.DataFrame(dat2[:,inds_emg], index = xticks_sbj,
                              columns = group_labs_emg)
    df_list = [df_neur, df_pose, df_emg]
    grp_list = [group_labs_neur, group_labs_pose, group_labs_emg]
    return df_list, grp_list