import pandas as pd

import os
from pathlib import Path
import argparse

from configs.get_data import TIME_WINDOW_START, TIME_WINDOW_STOP
TIME_WINDOW_START = TIME_WINDOW_START // 60
TIME_WINDOW_STOP = TIME_WINDOW_STOP // 60

import pydata_google_auth
from dotenv import load_dotenv

HERE = Path(__file__).absolute().parent
QUERIES_PATH = HERE / 'procedures'

credentials = pydata_google_auth.get_user_credentials(
    ['https://www.googleapis.com/auth/bigquery'],
    use_local_webserver=False
)

def run_query(query, project_id):
    return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      credentials=credentials,
      configuration={'query': {
          'useLegacySql': False
      }})
    
def read_sql(query_path):
    with open(query_path, 'r') as f:
        q = f.read()
    return q

def execute_query(query_path):
    query = read_sql(query_path)
    query = query.format(TIME_WINDOW_START)
    project_id = os.environ['GOOGLE_CLOUD_PROJECT']
    df = run_query(query, project_id)
    return df

def pivot_labs(df_labs):
    pv_labs = pd.pivot_table(
        df_labs,
        index='stay_id',
        columns='label',
        values='valuenum',
        aggfunc='mean'
    )
    to_drop = [
        'Calculated Bicarbonate, Whole Blood',
        'Hematocrit, Calculated','Potassium, Whole Blood'
    ]
    pv_labs = pv_labs.drop(columns=to_drop)
    pv_labs['Urea Nitrogen'] = pv_labs['Urea Nitrogen'] * 0.357
    pv_labs = pv_labs.rename(columns={'Urea Nitrogen': 'BUN'})
    cols = [c.lower() for c in pv_labs.columns.tolist()]
    pv_labs.columns = cols
    return pv_labs

def merge_dataframes(
    pv_labs,
    df_vitals,
    df_urine, 
    df_aki,
    df_crrt,
    df_comorbidities,
    df_rrt,
    df_demo,
    df_sepsis=None
):
    _df = df_demo.merge(df_comorbidities, 
                        how='outer', on=['subject_id','hadm_id', 'stay_id'])
    _df = _df.merge(df_rrt, how='outer', 
                    on=['subject_id', 'hadm_id', 'stay_id']
                    ).drop(columns=['subject_id','hadm_id'])
    
    to_concat = [
            _df.set_index('stay_id'),
            pv_labs,
            df_vitals.set_index('stay_id'),
            df_urine.set_index('stay_id'),
            df_aki.set_index('stay_id').loc[:, ['aki_stage']], 
            df_crrt.set_index('stay_id')
        ]
    if df_sepsis is not None:
        to_concat.append(df_sepsis.set_index('stay_id').loc[:, ['sepsis3']])

    df = pd.concat(to_concat, axis=1)
    return df

def exclusion_criteria(df, df_sepsis: pd.DataFrame=None, aki_stage: int=None):
    asd = df.copy()
    if aki_stage is not None:
        asd = asd.query(f"aki_stage>={aki_stage}")
    
    asd = asd.query(
        f"(crrt_deltatime.isnull()) | ((crrt_deltatime>={TIME_WINDOW_START}) & (crrt_deltatime<={TIME_WINDOW_STOP})) ", 
        engine='python')
    asd = asd.query("(renal_disease.isnull()) | (renal_disease!=1)", engine='python')
    asd = asd.query("age >= 18")

    asd.crrt_deltatime = asd.crrt_deltatime.fillna(0).apply(lambda x: 0 if x==0 else 1)
    asd = asd.rename(columns={'crrt_deltatime': 'crrt_flag'})
    asd = asd.drop(columns=['renal_disease'])

    if df_sepsis is not None:
        asd = asd.merge(df_sepsis.loc[:, ['stay_id','sepsis3']], how='inner', on='stay_id')
        asd = asd.drop(columns=['sepsis3'])
    return asd

def main(filename):
    load_dotenv()
    
    df_labs = execute_query(QUERIES_PATH / 'mimic_labs.sql')
    pv_labs = pivot_labs(df_labs)
    
    df_vitals = execute_query(QUERIES_PATH / 'mimic_vitals.sql')
    
    df_urine = execute_query(QUERIES_PATH / 'mimic_urine_output.sql')
    
    df_aki = execute_query(QUERIES_PATH / 'mimic_aki.sql')
    
    df_crrt = execute_query(QUERIES_PATH / 'mimic_crrt.sql')
    
    comorbidities_icd9 = execute_query(QUERIES_PATH / 'mimic_comorbidities_icd9.sql')
    comorbidities_icd10 = execute_query(QUERIES_PATH / 'mimic_comorbidities_icd10.sql')
    df_comorbidities = pd.concat([comorbidities_icd9, comorbidities_icd10], ignore_index=True)
    
    df_rrt = execute_query(QUERIES_PATH / 'mimic_renal_disease.sql')
    
    df_demo = execute_query(QUERIES_PATH / 'mimic_demographics.sql')
    
    df_sepsis = execute_query(QUERIES_PATH / 'mimic_sepsis.sql')
    
    df = merge_dataframes(
        pv_labs,
        df_vitals,
        df_urine, 
        df_aki,
        df_crrt,
        df_comorbidities,
        df_rrt,
        df_demo,
        df_sepsis
        )
    # df = exclusion_criteria(df, df_sepsis, aki_stage=3)
    df = exclusion_criteria(df, None, None)
    df = df.rename(columns={'gender':'patientsex'})
    df.to_csv(filename)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--filename",type=str,default="./data/test_mimic_2D.csv")
    args = args.parse_args()
    main(args.filename)