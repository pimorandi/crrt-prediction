import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as mtr
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.impute import SimpleImputer

from typing import Any, Sequence, List, Tuple

# metrics = {
#     'F1': mtr.f1_score,
#     'Recall': mtr.recall_score,
#     'Precision': mtr.precision_score,
#     'R2': mtr.r2_score
# }

def get_scores(y_true, y_pred):
    f1 = mtr.f1_score(y_true, y_pred, average=None)
    rec = mtr.recall_score(y_true, y_pred, average=None)
    prec = mtr.precision_score(y_true, y_pred, average=None)
    return np.array([f1[0], rec[0], prec[0], f1[1], rec[1], prec[1]])

def get_perf_df():
    idx = pd.MultiIndex.from_tuples([
        ('Train', 0, 'F1'), ('Train', 0, 'Recall'), ('Train', 0, 'Precision'),
        ('Train', 1, 'F1'), ('Train', 1, 'Recall'), ('Train', 1, 'Precision'),
        ('Test', 0, 'F1'), ('Test', 0, 'Recall'), ('Test', 0, 'Precision'),
        ('Test', 1, 'F1'), ('Test', 1, 'Recall'), ('Test', 1, 'Precision'),
    ])
    df = pd.DataFrame(columns = idx)
    return df

def youdens_stat(fpr, tpr, thresholds):
    j = tpr - fpr
    idx = j.argmax()
    return thresholds[idx]

def get_roc(y_true, y_proba):
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thr = mtr.roc_curve(y_true, y_proba[:,1], pos_label=1)
    best_thr = youdens_stat(fpr, tpr, thr)
    auc = mtr.roc_auc_score(y_true, y_proba[:,1])
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return {'fpr': mean_fpr, 'tpr': interp_tpr, 'auc': auc, 'best_thr': best_thr}

def evaluate(mdl, X, y, mode='predict'):
    y_proba = mdl.predict_proba(X)
    roc = get_roc(y, y_proba)
    if mode=='predict':
        y_pred = mdl.predict(X)
    elif mode=='youdens':
        y_pred = np.where(y_proba[:,1] <= roc['best_thr'], 0, 1)
    else:
        print(f'Mode {mode} not available')
        y_pred = mdl.predict(X)
    perf = get_scores(y, y_pred)
    return perf, roc, y_pred, y_proba

def cv_performance(mdl, X, y, k=5, mode='predict'):
    cv = StratifiedKFold(n_splits=k)
    df_perf = get_perf_df()
    df_preds = pd.DataFrame(columns=['Kfold'] + [f'fold_{i}' for i in range(k)], index=range(len(y)))
    df_proba = pd.DataFrame(columns=['Kfold'] + [f'fold_{i}' for i in range(k)], index=range(len(y)))
    fitted_models = []
    rocs = {
        'train': [],
        'test': [],
    }

    for i,(idx_train, idx_test) in enumerate(cv.split(X, y)):
        mdl.fit(X.loc[idx_train, :], y[idx_train])
        if hasattr(mdl, 'best_estimator_'):
            fitted_models.append(mdl.best_estimator_)
        else:
            fitted_models.append(mdl)
        
        test_perf, test_roc, test_preds, test_proba = evaluate(mdl, X.loc[idx_test], y[idx_test], mode=mode)
        df_perf.loc[i, 'Test'] = test_perf
        rocs['test'].append(test_roc)
        df_preds.loc[idx_test, 'Kfold'] = i
        df_preds.loc[idx_test, f'fold_{i}'] = test_preds
        df_proba.loc[idx_test, 'Kfold'] = i
        df_proba.loc[idx_test, f'fold_{i}'] = test_proba[:,1]

        train_perf, train_roc, train_preds, train_proba = evaluate(mdl, X.loc[idx_train], y[idx_train], mode=mode)
        df_perf.loc[i, 'Train'] = train_perf
        rocs['train'].append(train_roc)
        df_preds.loc[idx_train, f'fold_{i}'] = train_preds
        df_proba.loc[idx_train, f'fold_{i}'] = train_proba[:,1]

    cv_predictions = {
        'predict': df_preds,
        'predict_proba': df_proba
    }

    return df_perf, rocs, cv_predictions, fitted_models

def plot_rocs(res, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots()
    if title is None:
        title = 'Crossvalidated ROC curves'

    train_aucs = []
    for roc in res[1]['train']:
        _ = ax.plot(roc['fpr'], roc['tpr'], 'b')
        train_aucs.append(roc['auc'])

    test_aucs = []
    for roc in res[1]['test']:
        _ = ax.plot(roc['fpr'], roc['tpr'], 'orange')
        test_aucs.append(roc['auc'])

    custom_lines = [
        Line2D([0], [0], color='b'),
        Line2D([0], [0], color='orange')]

    train_mean_auc = np.round(np.mean(train_aucs), 2)
    train_std_auc = np.round(np.std(train_aucs), 3)
    test_mean_auc = np.round(np.mean(test_aucs), 2)
    test_std_auc = np.round(np.std(test_aucs), 3)

    _ = ax.legend(custom_lines, [f'Train [AUC: {train_mean_auc} $\pm$ {train_std_auc}]', f'Test [AUC: {test_mean_auc} $\pm$ {test_std_auc}]'], loc='lower right')
    _ = ax.grid(alpha=0.4)
    _ = ax.set_title(title)
    _ = ax.set_xlabel('False Positive Rate')
    _ = ax.set_ylabel('True Positive Rate')

    return ax

def probability_inspection(res, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots()
    if title is None:
        title = 'Crossvalidation Probability distributions'

    df_proba = res[2]['predict_proba']
    test_proba = []
    train_proba = []
    kcols = [c for c in res[2]['predict_proba'].columns if c.startswith('fold')]
    for c in kcols:
        k = int(c.split('_')[1])
        test_proba.append(df_proba.query(f'Kfold=={k}')[c])
        train_proba.append(df_proba.query(f'Kfold!={k}')[c])
    test_proba = pd.concat(test_proba)
    train_proba = pd.concat(train_proba)

    b = np.linspace(0,1,100)
    _ = ax.hist(train_proba, bins=b, label='Train probabilities', density=True)
    _ = ax.hist(test_proba, bins=b, label='Test probabilities', density=True, alpha=0.6)
    _ = ax.legend()
    _ = ax.grid(alpha=0.4)
    _ = ax.set_title(title)
    _ = ax.set_xlabel('Probability')
    _ = ax.set_ylabel('Density')
    return ax

def oddsRatio(X, y, columnnames=None, n_iterations=1000, ratio=0.65, lr_bias=True, solver='lbfgs'):
    n_sample = int(X.shape[0] * ratio)
    logreg = LogisticRegression(
        penalty=None, 
        fit_intercept=lr_bias,
        solver=solver
    )
    imputer = SimpleImputer()
    X_ = imputer.fit_transform(X)
    if columnnames is None:
        columnnames = [f'col_{i}' for i in range(X.shape[1])]
    df_coef = pd.DataFrame(columns=columnnames, index=range(n_iterations))
    for i  in range(n_iterations):
        _X, _y = resample(X_, y, n_samples=n_sample, stratify=y, replace=False)
        logreg.fit(_X, _y)
        coef = np.exp(np.squeeze(logreg.coef_))
        df_coef.loc[i] = coef
    return df_coef

def whisk_data(ans):
    tmp_whisk = np.vstack([e.get_xdata() for e in ans['whiskers']])
    lower_whisk = tmp_whisk[::2, 1]
    upper_whisk = tmp_whisk[1::2, 1]
    x_whisk = np.vstack([lower_whisk, upper_whisk]).T
    tmp_y = np.arange(1,x_whisk.shape[0]+1)
    y_whisk = np.vstack([tmp_y, tmp_y]).T
    return x_whisk, y_whisk

def write_whisk(ax, x_whisk, y_whisk):
    xlim = ax.get_xlim()
    delta = xlim[1] - xlim[0]
    o = 0.02 + delta * 0.01
    for x_data,y_data in zip(x_whisk, y_whisk):
        _ = ax.text(x_data[0] - o, y_data[0], np.round(x_data[0],3), ha='right', va='center')
        _ = ax.text(x_data[1] + o, y_data[0], np.round(x_data[1],3), ha='left', va='center')
    return ax

def plot_odds_ratio(df_coef, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    ans = ax.boxplot(df_coef, whis=[5, 95], showfliers=False, vert=False)
    x_whisk, y_whisk = whisk_data(ans)
    _ = write_whisk(ax, x_whisk, y_whisk)
    _ = ax.set_yticklabels(df_coef.columns)
    _ = ax.grid(axis='x', alpha=0.5)
    xlim = ax.get_xlim()
    a = (xlim[1] - xlim[0]) / 10
    _ = ax.set_xlim([xlim[0]-a, xlim[1]+a])
    return ax

def quick_eval(model, X, y, save_folder):
    res = evaluate(model, X, y)
    stats = pd.DataFrame(data=res[0].reshape([2,3]), columns=['F1','Recall','Precision'], index=[0,1])
    stats.to_excel(save_folder / 'eval_stats.xlsx')
    
    f, ax = plt.subplots()
    _ = ax.plot(res[1]['fpr'], res[1]['tpr'], label='Best Classifier')
    _ = ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill')
    _ = ax.fill_between(res[1]['fpr'], res[1]['tpr'], alpha=0.3, label=f"AUC: {np.round(res[1]['auc'], 3)}")
    _ = ax.grid(alpha=0.5)
    _ = ax.set_xlabel('FPR')
    _ = ax.set_ylabel('TPR')
    _ = ax.legend()
    f.savefig(save_folder / 'eval_roc.png')
    
    f, ax = plt.subplots()
    b = np.linspace(0,1,25)
    _ = ax.hist(res[3][:, 1], bins=b, density=True)
    _ = ax.set_xlabel('y_proba')
    _ = ax.set_ylabel('Density [%]')
    f.savefig(save_folder / 'eval_prob_hist.png')
    
    return None