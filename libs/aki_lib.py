import boto3
import awswrangler as wr

import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import List, Dict


def get_creatinine(creat_idx: List[int]) -> pd.DataFrame:

    creatinine = []
    for i in tqdm(range(1,7)):
        tbl_name = f's3://icusics-db/labresults_numeric/labresults_numeric_h{i}.parquet'
        labresults_numeric = wr.s3.read_parquet(tbl_name)
        ith = labresults_numeric[labresults_numeric.a_variableid.isin(creat_idx)]
        creatinine.append(ith)
        
    creatinine = pd.concat(creatinine)
    return creatinine

def compute_kdigo_creatinine(creatinine: pd.DataFrame) -> pd.DataFrame:
    ## add in the lowest value in the previous 48 hours
    cols = {'a_patientid':'a_patientid', 'time':'time_48', 'value':'value_48'}
    cr48 = creatinine.loc[:, cols.keys()].rename(columns=cols)

    m_48 = creatinine.merge(cr48, how='left', on=['a_patientid'])
    m_48 = m_48.query("time_48 < time and time_48 >= time - 2880")
    m_48 = m_48.merge(creatinine, how='right', on=creatinine.columns.tolist())
    grp_cols = ['a_patientid','a_variableid','time','value']
    m_48 = m_48.groupby(grp_cols).value_48.min().reset_index()

    ## add in the lowest value in the previous 7 days
    cols = {'a_patientid':'a_patientid', 'time':'time_7', 'value':'value_7'}
    cr7 = creatinine.loc[:, cols.keys()].rename(columns=cols)

    m_7 = creatinine.merge(cr7, how='left', on=['a_patientid'])
    m_7 = m_7.query("time_7 < time and time_7 >= time - 10080")
    m_7 = m_7.merge(creatinine, how='right', on=creatinine.columns.tolist())
    grp_cols = ['a_patientid','a_variableid','time','value']
    m_7 = m_7.groupby(grp_cols).value_7.min().reset_index()

    kdigo_creatinine = m_48.merge(
        m_7, how='inner', 
        on=['a_patientid', 'a_variableid', 'time', 'value']
        )
    return kdigo_creatinine

def get_urine_output(uro_idx: List[int]):

    urineOutput = []
    for i in tqdm(range(1,7)):
        labresults_numeric = wr.s3.read_parquet(f's3://icusics-db/observed_numeric/observed_numeric_h{i}.parquet')
        ith = labresults_numeric[labresults_numeric.a_variableid.isin(uro_idx)]
        urineOutput.append(ith)
        
    urineOutput = pd.concat(urineOutput)
    return urineOutput

def get_lag(df, l, merge_idx=['a_patientid']):
    orig_cols = df.columns.tolist()
    custom_cols = list(map(lambda x: x + '_' + str(l) if x not in merge_idx else x, orig_cols))

    cols = {'time': f'time_{l}', 'value': f'value_{l}'}
    df_l = df.rename(columns=cols)

    m_l = df.merge(df_l, how='left', on=merge_idx)
    lhr = l * 60
    q_str = f"{cols['time']} < time and {cols['time']} > time - {lhr}"
    m_l = m_l.query(q_str)
    m_l = m_l.merge(df, how='right', on=df.columns.tolist())
    
    m_l = m_l.groupby(df.columns.tolist()).agg({cols['value']: np.sum, cols['time']: np.min}).reset_index()
    return m_l

def compute_kdigo_urine_output(
    urineOutput: pd.DataFrame, 
    patients: pd.DataFrame) -> pd.DataFrame:
    u_patients = urineOutput.a_patientid.unique()

    l6 = []
    l12 = []
    l24 = []

    delta = 100
    for i in tqdm(range(0, u_patients.shape[0], delta)):
        p_round = u_patients[i:i+delta]
        cols = ['a_patientid', 'time', 'value']
        df = urineOutput[urineOutput.a_patientid.isin(p_round)].loc[:, cols]
        i6 = get_lag(df, l=6)
        i12 = get_lag(df, l=12)
        i24 = get_lag(df, l=24)
        l6.append(i6)
        l12.append(i12)
        l24.append(i24)
        
    uo_l6 = pd.concat(l6, ignore_index=True)
    uo_l12 = pd.concat(l12, ignore_index=True)
    uo_l24 = pd.concat(l24, ignore_index=True)

    uo_l6['uo_tm_6hr'] = (uo_l6.time - uo_l6.time_6) / 60
    uo_l12['uo_tm_12hr'] = (uo_l12.time - uo_l12.time_12) / 60
    uo_l24['uo_tm_24hr'] = (uo_l24.time - uo_l24.time_24) / 60

    uo_l6 = uo_l6.drop(columns='time_6')
    uo_l12 = uo_l12.drop(columns='time_12')
    uo_l24 = uo_l24.drop(columns='time_24')

    uo_m1 = uo_l6.merge(uo_l12, on=['a_patientid','time','value'])
    uo_m2 = uo_m1.merge(uo_l24, on=['a_patientid','time','value'])
    
    patients_merge = patients.loc[:, ['a_patientid','weight']]
    kdigo_uo = uo_m2.merge(patients_merge, how='inner', on='a_patientid')

    ## This passage might need investigations
    rt_6hr = kdigo_uo.value_6 / kdigo_uo.weight / kdigo_uo.uo_tm_6hr
    kdigo_uo['uo_rt_6hr'] = np.where(kdigo_uo.uo_tm_6hr.fillna(0) >= 5, rt_6hr, None)
    rt_12hr = kdigo_uo.value_12 / kdigo_uo.weight / kdigo_uo.uo_tm_12hr
    kdigo_uo['uo_rt_12hr'] = np.where(kdigo_uo.uo_tm_12hr.fillna(0) >= 11, rt_12hr, None)
    rt_24hr = kdigo_uo.value_24 / kdigo_uo.weight / kdigo_uo.uo_tm_24hr
    kdigo_uo['uo_rt_24hr'] = np.where(kdigo_uo.uo_tm_24hr.fillna(0) >= 23, rt_24hr, None)
    return kdigo_uo

def creatine_stage(v, v48, v7):
    if (v >= v7*3) or ((v >= 4) and ((v48 <= 3.7) or (v >= 1.5*v7))):
        stage = 3
    elif (v >= v7*2):
        stage = 2
    elif (v >= v48+0.3) or (v >= 1.5*v7):
        stage = 1
    else:
        stage = 0
    return stage

def urineOut_stage(time, uo_tm_6hr, uo_tm_12hr, uo_tm_24hr, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr):
    if (uo_rt_6hr is None):
        stage = None
    elif time < 360:
        stage = 0
    elif ((uo_tm_24hr >= 23) and (uo_rt_24hr < 0.3)) or ((uo_tm_12hr >= 11) and (uo_rt_12hr == 0)):
        stage = 3
    elif (uo_tm_12hr >= 11) and (uo_rt_12hr < 0.5):
        stage = 2
    elif (uo_tm_6hr >= 5) and (uo_rt_6hr  < 0.5):
        stage = 1
    else:
        stage = 0
    return stage

def extract_aki_score(creat_idx: List[int], uro_idx: List[int], time_window: int):
    creatinine = get_creatinine(creat_idx)
    kdigo_creatinine = compute_kdigo_creatinine(creatinine)
    aki_creat = list(map(lambda v,v48,v7: creatine_stage(v, v48, v7), kdigo_creatinine.value, kdigo_creatinine.value_48, kdigo_creatinine.value_7))
    kdigo_creatinine['aki_stage_creat'] = aki_creat
    cols = {
        'a_patientid':'a_patientid', 'time':'time', 
        'value':'creat', 'value_48':'creat_48hr', 'value_7':'creat_7days',
        'aki_stage_creat':'aki_stage_creat'}
    kdigo_creatinine = kdigo_creatinine.loc[:, cols.keys()].rename(columns=cols)
    kdigo_creatinine
    
    patients = wr.s3.read_parquet('s3://icusics-db/patients/patients.parquet')
    urineOutput = get_urine_output(uro_idx)
    kdigo_uo = compute_kdigo_urine_output(urineOutput, patients)
    kdigo_uo['aki_stage_uo'] = list(map(lambda t,tm6,tm12,tm24,rt6,rt12,rt24: urineOut_stage(t,tm6,tm12,tm24,rt6,rt12,rt24), 
            kdigo_uo.time, 
            kdigo_uo.uo_tm_6hr.fillna(-1),
            kdigo_uo.uo_tm_12hr.fillna(-1),
            kdigo_uo.uo_tm_24hr.fillna(-1),
            kdigo_uo.uo_rt_6hr.fillna(-1),
            kdigo_uo.uo_rt_12hr.fillna(-1),
            kdigo_uo.uo_rt_24hr.fillna(-1)
            )
    )
    
    tm_stg = pd.concat(
        [kdigo_creatinine.loc[:, ['a_patientid','time']],
        kdigo_uo.loc[:, ['a_patientid','time']]
        ], ignore_index=True
    )
    kdigo_stages = patients.loc[:, ['a_patientid']].merge(
        tm_stg, how='left', on='a_patientid')
    kdigo_stages = kdigo_stages.merge(
        kdigo_creatinine, how='left', on=['a_patientid','time'])
    uo_cols = [
        'a_patientid','time','uo_rt_6hr','uo_rt_12hr',
        'uo_rt_24hr','aki_stage_uo']
    kdigo_stages = kdigo_stages.merge(
        kdigo_uo.loc[:, uo_cols], how='left', on=['a_patientid','time'])
    kdigo_stages['aki_stage'] = list(map(lambda x,y: max(x,y), kdigo_stages.aki_stage_creat.fillna(0).astype(float), kdigo_stages.aki_stage_uo.fillna(0).astype(float)))

    agg_stages = kdigo_stages.query(f"time <= {time_window}")
    agg_stages = agg_stages.groupby('a_patientid').aki_stage.max()
    agg_stages = agg_stages.reset_index()
    agg_stages['hospital_id'] = agg_stages.a_patientid.apply(
        lambda x: int(str(x)[0]))
    agg_stages = agg_stages.set_index(['hospital_id','a_patientid'])
    
    return agg_stages
