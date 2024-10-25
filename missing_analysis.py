import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy.stats import chi2_contingency

from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

from datetime import datetime
import os
from pathlib import Path
import argparse

RESULTS_PATH = Path('./results/missing_analysis')
EPISODE = datetime.now().strftime('%Y%m%d_%H%M%S')
os.mkdir(RESULTS_PATH / EPISODE)

def missing_analysis(df, feature, target):
    missing_feat = df[feature].isna()
    
    odd_ratios = []
    for _ in range(50):
        idx = df.rename_axis('pickme').reset_index().pickme.sample(frac=0.75)
        X = missing_feat.loc[idx].values.reshape(-1, 1)
        y = df.loc[idx, target]
        mdl = LogisticRegression()
        mdl.fit(X, y)
        odds_ratio = np.exp(mdl.coef_.squeeze())
        odd_ratios.append(odds_ratio)
    return np.vstack(odd_ratios).squeeze()

def plot_odds_ratios(d):
    f, ax = plt.subplots(1,1,figsize=(8,10)) # 
    
    pos = np.arange(0, len(d['mimic'].keys())*2, 2)
    w = 0.5
    
    df_t = pd.DataFrame(d['tarragona'])
    b1 = ax.boxplot(
        df_t, vert=False, whis=[5, 95], showfliers=False, 
        patch_artist=True, boxprops=dict(facecolor='orange'), 
        positions=pos-(w/2), widths=w)
    
    df_t = pd.DataFrame(d['mimic'])
    b2 = ax.boxplot(
        df_t, vert=False, whis=[5, 95], showfliers=False, 
        patch_artist=True, boxprops=dict(facecolor='blue'), 
        positions=pos+(w/2), widths=w)
    
    _ = ax.set_yticks(pos)
    _ = ax.set_yticklabels(df_t.columns)
    _ = ax.grid(axis='x', alpha=0.4)
    _ = ax.set_xlabel('Odds Ratio')
    _ = ax.set_ylabel('Features')
    _ = ax.legend([b1["boxes"][0], b2["boxes"][0]], ['Tarragona', 'MIMIC'])
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    if pos.shape[0] % 2 == 1:
        a = np.concatenate([pos, [pos[-1] + 2]]) - 1
    else:
        a = pos.copy() - 1
    a = a.reshape((a.shape[0] // 2, 2))
    for i in range(a.shape[0]):
        _ = ax.fill_between([0, 20], a[i,0], a[i,1], color='gray', alpha=0.3)
    _ = ax.set_xlim(xlims)
    _ = ax.set_ylim(ylims)
    _ = ax.axvline(1, c='r', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return f, ax

def plot_missing_percentage(dfs):
    f, ax = plt.subplots()
    
    pos = np.arange(0, dfs['mimic'].shape[1]*2, 2)
    w = 0.5
    
    for i,(k,v) in enumerate(dfs.items()):
        j = (-1)**i
        missing_perc = (v.isna().sum() / v.shape[0]) * 100
        _ = ax.bar(pos+(w/2)*j, missing_perc, label=k, width=w)
    
    _ = ax.legend()
    _ = ax.grid(axis='y', alpha=0.5)
    _ = ax.set_xticks(pos)
    _ = ax.set_xticklabels(dfs['mimic'].columns, rotation=40, ha='right')
    _ = ax.set_ylabel('[%]')
    _ = ax.set_xlabel('Features')
    plt.tight_layout()
    return f, ax

def plot_feature(dfs, feat, f, ax):
    x_min = min([dfs[k][feat].quantile(0.001) for k in dfs.keys()])
    x_max = max([dfs[k][feat].quantile(0.999) for k in dfs.keys()])
    x = np.linspace(x_min, x_max, 20)
    for k in dfs.keys():
        _ = sns.kdeplot(data=dfs[k][feat], label=k, ax=ax)
        c = ax.get_lines()[-1].get_c()
        _ = ax.hist(dfs[k][feat], bins=x, alpha=0.4, density=True, color=c)
        
    _, pval = ss.ranksums(dfs['tarragona'][feat].dropna(), dfs['mimic'][feat].dropna())
    p = '<< 0.01' if pval < 0.01 else f': {pval:.3f}'
    # ax.set_title(feat)
    _ = ax.text(0.5, 0.5, f'P-value {p}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    _ = ax.set_xlim([x_min, x_max])
    _ = ax.grid(alpha=0.4)
    _ = ax.legend()
    return f, ax

def dataset_comparison(dfs):
    df_cols = pd.concat(
        [dfs['tarragona'].apply(lambda r: r.nunique() > 4, axis=0), 
        dfs['mimic'].apply(lambda r: r.nunique() > 4, axis=0)], axis=1).fillna(False)

    df_cols['tot'] = df_cols.apply(lambda r: r[0] and r[1], axis=1)
    cols = df_cols[df_cols.tot].index.tolist()

    f, axs = plt.subplots(6,3,figsize=(15, 24))
    axs = axs.flatten()

    for i,(ax,c) in enumerate(zip(axs,cols)):
        _ = plot_feature(dfs, c, f, ax)
        if i % 3 != 0:
            _ = ax.set_ylabel('')
        
    _ = f.suptitle('Dataset Comparison', fontsize=20, y=0.91)
    return f, axs

def read_preprocessing(path):
    df = pd.read_csv(path)
    df = df.query("aki_stage>=1")
    # df['aki_greater_0'] = df.aki_stage.apply(lambda x: x > 0).astype(int)
    # df['aki_greater_1'] = df.aki_stage.apply(lambda x: x > 1).astype(int)
    df['aki_greater_2'] = df.aki_stage.apply(lambda x: x > 2).astype(int)
    return df

def main(tarragona, mimic):
    
    targets = ['crrt_flag','patientsex','aki_greater_2', 'sepsis3'] # ,'aki_greater_0','aki_greater_1'
    dfs = {
        'tarragona': read_preprocessing(tarragona),
        'mimic': read_preprocessing(mimic)
    }
    
    cols = list(set(dfs['mimic'].columns.tolist()).intersection(dfs['tarragona'].columns))
    cols.sort()
    dfs = {k:v[cols] for k,v in dfs.items()}
    
    f, ax = plot_missing_percentage(dfs)
    f.savefig(RESULTS_PATH / EPISODE / f'missing_values_percentage.jpg')
    plt.close()
    
    f, ax = dataset_comparison(dfs)
    f.savefig(RESULTS_PATH / EPISODE / f'dataset_comparison.jpg')
    plt.close()
    
    cols = list(set(cols).difference(targets))
    df_pvals = pd.DataFrame(columns=targets, index=cols)
    
    for t in targets:
        d = {k:{} for k in dfs.keys()}
        for c in cols:
            for k in dfs.keys():
                print(c, t, k)
                oddratios = missing_analysis(dfs[k], c, t)
                
                d[k][c] = oddratios
        
        f, ax = plot_odds_ratios(d)
        _ = ax.set_title(t)
        plt.tight_layout()
        f.savefig(RESULTS_PATH / EPISODE / f'oddsratio_{t}.jpg')
        plt.close()
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--tarragona", type=str, default="./data/test_tarragona_2D.csv")
    args.add_argument("--mimic", type=str, default="./data/test_mimic_2D.csv")
    args = args.parse_args()
    main(args.tarragona, args.mimic)