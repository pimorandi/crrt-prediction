import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
import argparse

from sklearn.impute import SimpleImputer
from tableone import TableOne

from libs.analytics import DataProcessing, CollinearityRemover, FeatureSelector, Optimizer
from libs.evaluation import cv_performance, oddsRatio, plot_rocs, probability_inspection, plot_odds_ratio
# from libs.evaluation import evaluate
from libs.evaluation import quick_eval
# from configs.predictions import episode_config

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")


RESULTS_PATH = Path('./results/model_dev')
if not os.path.isdir(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
EPISODE = datetime.now().strftime('%Y%m%d_%H%M%S')
os.mkdir(RESULTS_PATH / EPISODE)

def main(training_path, evaluation_path):
    # Load data
    data_path = training_path #episode_config['dataset_path']
    print(f'Reading dataset from: {data_path}')
    df_raw = pd.read_csv(data_path)
    # df_raw = df_raw.query("aki_stage==3").reset_index(drop=True)

    # Data processing
    df_raw.patientsex = df_raw.patientsex.replace({'M':0, 'F':1})
    processor = DataProcessing(df_raw)
    df = processor.transform()
    df.aki_stage = df.aki_stage.fillna(0)
    ans = processor.post(df, 'aki_stage', to_drop=['hospital_id','a_patientid','crrt_flag','stay_id','sepsis3'])
    df_X = ans['df']
    df_X = df_X.isna().astype(int)
    aki_cols = [c for c in df_X.columns.tolist() if c.startswith('aki')]
    # df_X = df_X.drop(columns=aki_cols)
    
    # Print Table1
    mytable = TableOne(pd.concat([df_X, df.crrt_flag], axis=1), categorical=list(df_X.columns), groupby='crrt_flag', pval=True)
    mytable.to_excel(RESULTS_PATH / EPISODE / 'Table_1.xlsx')

    # Remove collinearity
    cr = CollinearityRemover(df_X)
    vif = cr.run()
    cols = vif.Variable
    binary_cols = list(set(cols).difference(ans['continous']))
    continous_cols = list(set(cols).difference(ans['binary']))
    df_X = df_X.loc[:, binary_cols+continous_cols]

    # Features selection
    mdl = LGBMClassifier(
        max_depth=3,
        num_leaves=12,
        n_estimators=200,
        learning_rate=1e-2,
        class_weight='balanced',
        random_state=17,
        n_jobs=-1,
    )
    fs = FeatureSelector(mdl, continous_cols, binary_cols)
    features = fs.transform(df_X, df.crrt_flag, k=5, mode='fclassif')
    selected_features = features[features>=5].index.tolist()
    binary_cols = list(set(selected_features).difference(ans['continous']))
    continous_cols = list(set(selected_features).difference(ans['binary']))

    # Model optimization
    mdl = LogisticRegression(
        penalty='elasticnet',
        random_state=17,
        solver='saga',
        max_iter=100,
    )
    params = {
        'mdl__C': [1e-3, 1e-1, 1, 1e1, 1e3], 
        'mdl__class_weight': [None,'balanced'],
        'mdl__l1_ratio': [1e-5, 0.25, 0.5, 0.75, 1] 
    }
    opt = Optimizer(mdl, continous_cols, binary_cols)
    clf = opt.run(df_X.loc[:, selected_features], df.crrt_flag, params, opt_score='neg_log_loss')
    joblib.dump(clf, RESULTS_PATH / EPISODE / 'best_model.sav')
    df_X.loc[:, selected_features].to_csv(RESULTS_PATH / EPISODE / 'features.csv')
    df.crrt_flag.to_csv(RESULTS_PATH / EPISODE / 'target.csv')
    
    # Cross-validation
    res = cv_performance(
        clf, 
        df_X.loc[:, selected_features],#.values, # 
        df.crrt_flag.values, 
        k=5,
        mode='youdens') 
    stats = res[0].stack().groupby(level=1).median()
    stats.to_excel(RESULTS_PATH / EPISODE / 'stats.xlsx')
    
    f, ax = plt.subplots()
    _ = plot_rocs(res, ax=ax)
    f.savefig(RESULTS_PATH / EPISODE / 'roc_curves.png')
    
    f, ax = plt.subplots()
    _ = probability_inspection(res, ax=ax)
    f.savefig(RESULTS_PATH / EPISODE / 'probability_inspection.png')
    
    # Odds Ratio
    df_coef = oddsRatio(
        df_X.loc[:, selected_features], 
        df.crrt_flag, 
        columnnames=selected_features, 
        n_iterations=500)
    f, ax = plt.subplots()
    _ = plot_odds_ratio(df_coef, ax)
    plt.tight_layout()
    f.savefig(RESULTS_PATH / EPISODE / 'odds_ratio.png')
    
    
    if evaluation_path is not None:
        print(f'\n\nEvaluation on {evaluation_path} starting')
        
        df_eval = pd.read_csv(evaluation_path)
        # df_eval = df_eval.query("aki_stage==3").reset_index(drop=True)
        process = DataProcessing(df_eval)
        df = process.transform()
        df.aki_stage = df.aki_stage.fillna(0)
        ans = process.post(df, 'aki_stage')
        df_ = ans['df']
        df_.patientsex = df_.patientsex.replace({'M':0, 'F':1})
        df_ = df_.loc[:, selected_features]
        
        _ = quick_eval(clf, df_, df_eval.crrt_flag, RESULTS_PATH / EPISODE)
    
    print('\n\nDone!')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--training_path",type=str,default="./data/test_tarragona_1D.csv")
    args.add_argument("--evaluation_path",type=str,default=None) # "./data/mimic_extraction_sepsis_6H.csv"
    args = args.parse_args()
    main(args.training_path, args.evaluation_path)