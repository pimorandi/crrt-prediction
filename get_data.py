import boto3

import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from typing import Dict

from configs.get_data import lab_config, vitals_config, \
    comorbidities_config, crrt_idx, sepsi_codes, \
    TIME_WINDOW_START, TIME_WINDOW_STOP, AGG_FNC, \
    creatinine_codes, urine_output_codes
from libs.data_extraction import extract_labs, from_monitored_numeric, \
    extract_comorbidities, get_demography, get_septic_patients, \
    extract_urine_output
from libs.aki_lib import extract_aki_score

DATA_PATH = Path('./data')
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

def login():
    load_dotenv()
    # Log in
    key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
    s3 = boto3.resource('s3',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key)
    return s3

def main(filename, debug):
    """
    Quite a mess, but works
    """
    
    s3 = login()
    
    # Get lab results
    print('-'*50)
    print('Quering Lab results\n\n')
    labs = extract_labs(lab_config, TIME_WINDOW_START, AGG_FNC)
    
    # Get Urine Output
    print('-'*50)
    print('Quering urine output\n\n')
    df_urine = extract_urine_output(urine_output_codes, TIME_WINDOW_START)
    
    # Get vitals
    print('-'*50)
    print('Quering vitals\n\n')
    vitals, outcome = from_monitored_numeric(
        s3, crrt_idx, vitals_config, TIME_WINDOW_START, debug) # AGG_FNC
    
    # Get comorbidities
    print('-'*50)
    print('Quering comorbidities\n\n')
    comorbidities = extract_comorbidities(comorbidities_config)
    
    # Get AKI score
    print('-'*50)
    print('Quering AKI score\n\n')
    aki_score = extract_aki_score(
        creatinine_codes, urine_output_codes, TIME_WINDOW_START)
    
    # Get demography
    print('-'*50)
    print('Quering demograpy\n\n')
    demography = get_demography()
    
    # Septic patients
    septic_pz = get_septic_patients(sepsi_codes)
    septic_pz['sepsis3'] = 1
    
    # Merge all
    dataset = pd.concat(
        [labs, df_urine, vitals, comorbidities, aki_score, outcome], 
        join='outer', axis=1)
    dataset = demography.merge(
        dataset, how='right', 
        left_index=True, right_index=True)
    
    ### Exclusion criteria
    print('-'*50)
    print('Applying exclusion criteria\n\n')
    # exclude crrt in the first 48 hours
    dataset['exclude_per_crrt'] = np.where(dataset.crrt_time <= TIME_WINDOW_START, 1, 0)
    # exclude crrt after 5 days
    dataset['exclude_per_crrt'] = np.where(dataset.crrt_time > TIME_WINDOW_STOP, 1, dataset['exclude_per_crrt'])
    # exclude AKI < 3
    # dataset['exclude_per_stage'] = np.where(dataset.aki_stage < 3, 1,0)
    # Keep only septic patients
    dataset = dataset.reset_index()
    # dataset = dataset[dataset.a_patientid.isin(septic_pz.a_patientid)]
    dataset = dataset.merge(
        septic_pz.loc[:, ['a_patientid','sepsis3']], 
        how='left', on='a_patientid')
    dataset.sepsis3 = dataset.sepsis3.fillna(0)
    
    final_ds = dataset.query(
        "exclude_per_crrt == 0 and \
        rrt != 1"# and aki_stage == 3"
    )
    
    print('-'*50)
    print('Finalizing..\n\n')
    final_ds.crrt_time = final_ds.crrt_time.fillna(-1)
    final_ds['crrt_flag'] = np.where(
        (final_ds.crrt_time> TIME_WINDOW_START) & (final_ds.crrt_time<= TIME_WINDOW_STOP), 1, 0)
    final_ds = final_ds.drop(
        columns=['exclude_per_crrt','rrt','crrt_time']) # 'exclude_per_stage',
    final_ds.to_csv(filename, index=False)
    print('Done!')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--filename",type=str,default="./data/test_tarragona_2D.csv")
    args.add_argument("--debug",type=bool,default=False)
    args = args.parse_args()
    main(args.filename, args.debug)