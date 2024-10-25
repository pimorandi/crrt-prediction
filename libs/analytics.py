import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from typing import Tuple, List, Dict


class DataProcessing():
    """
    Simple class that perform the pre-processing needed to run the analysis on Catalunia data
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transform(self) -> pd.DataFrame:
        columns = pd.Series(self.df.columns.tolist())
        bools = columns.apply(lambda x: False if x.find('_')<0 else (True if x.split('_')[1]=='min' else False))
        predictors = columns[bools].apply(lambda x: x.split('_')[0])

        for pred in predictors:
            self.df[f'{pred}_delta'] = self.df[f'{pred}_max'] - self.df[f'{pred}_min']
            self.df = self.df.drop(columns=[f'{pred}_max'])

        return self.df
    
    def post(self, df: pd.DataFrame, aki_col: str, to_drop: List = []) -> Dict:

        df_aki = pd.get_dummies(df[aki_col].astype(int), dtype='int64')
        df_aki.columns = [f'akiStage_{i}' for i in df_aki.columns]
        
        to_drop = list(set(to_drop).intersection(df.columns.tolist()))
        df_X = df.drop(columns=['aki_stage']+to_drop)
        df_X = pd.concat([df_X, df_aki], axis=1)

        binary_cols = ['patientsex','chronic_kidney_disease','diabetes','hypertension','heart_failure'] + df_aki.columns.tolist()
        continous_cols = list(set(df_X.columns).difference(binary_cols))

        df_X.loc[:, binary_cols] = df_X.loc[:, binary_cols].fillna(0)

        return {'df': df_X, 'binary': binary_cols, 'continous': continous_cols}
    

class CollinearityRemover():
    def __init__(self, df, intercept=1):
        self.df = df
        self.intercept = intercept

    def compute_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        X = X.dropna()
        X['intercept'] = self.intercept
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif[vif['Variable']!='intercept']
        return vif
    
    def run(self, threshold: int = 5) -> pd.DataFrame:
        working_df = self.df.copy()
        done = False
        while not(done):
            vif = self.compute_vif(working_df)
            max_vif = vif.VIF.max()
            argmax_vif = vif.VIF.argmax()
            if (max_vif > threshold) or (max_vif == np.inf):
                col = vif.loc[argmax_vif, 'Variable']
                working_df = working_df.drop(columns=[col])
            else:
                done = True
        return vif
    

def build_pipeline(
    mdl: ClassifierMixin, 
    continous_cols: List[str], 
    binary_cols: List[str]):
    
    numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), 
        RobustScaler()
    )

    transformer = ColumnTransformer(
        [
            ('continous', numeric_transformer, continous_cols),
            ('binary', SimpleImputer(fill_value=0), binary_cols),
        ],
    )

    pipe = Pipeline([
        ('sc', transformer),
        ('sm', SMOTE(random_state=43)),
        ('mdl', mdl)
    ])

    return pipe

class FeatureSelector():
    def __init__(self, model: ClassifierMixin, continous_cols: List[str], binary_cols: List[str]):
        self.pipeline = build_pipeline(model, continous_cols, binary_cols)

    def model_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 5, n_features = None) -> pd.DataFrame:
        kfold = StratifiedKFold(k)
        cv_feat_importance = []
        for idx_train,idx_test in kfold.split(X, y):
            self.pipeline.fit(X.loc[idx_train, :], y.loc[idx_train])
            cv_feat_importance.append(np.where(self.pipeline['mdl'].feature_importances_>0, 1, 0))
        features = pd.DataFrame(cv_feat_importance, columns=X.columns).sum()
        return features
    
    def fclassif(self, X: pd.DataFrame, y: pd.Series, k: int = 5, n_features: int = 12):
        features = pd.Series(dict.fromkeys(X.columns, 0))
        si = SimpleImputer()
        selector = SelectKBest(f_classif, k=n_features)
        kfold = StratifiedKFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
            X_train = X.loc[train_index, :]
            y_train = y[train_index]
            X_train_ = si.fit_transform(X_train)
            selector.fit(X_train_, y_train)
            X_new = selector.transform(X_train_)
            cols = selector.get_feature_names_out(X.columns.tolist())
            features[cols] += 1
        return features
    
    def transform(self, X: pd.DataFrame, y: pd.Series, k: int = 5, n_features: int = 12, mode: str='fclassif'):
        return getattr(self, mode)(X, y, k, n_features)
        
    
class Optimizer():
    def __init__(self, model: ClassifierMixin, continous_cols: List[str], binary_cols: List[str]):
        self.pipeline = build_pipeline(model, continous_cols, binary_cols)

    def run(self, X: pd.DataFrame, y: pd.Series, params: Dict, opt_score: str='f1_score', k: int = 5):
        kfold = StratifiedKFold(n_splits=k)
        search = RandomizedSearchCV(
            self.pipeline,
            params,
            cv=kfold,
            scoring=opt_score,
            random_state=17,
        )
        _ = search.fit(X, y)
        return search.best_estimator_
    
