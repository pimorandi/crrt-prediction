select l.subject_id, l.hadm_id, icu.stay_id, l.labevent_id, l.charttime, l.valuenum, d.label
from `physionet-data.mimiciv_hosp.labevents` AS l
inner join `physionet-data.mimiciv_hosp.d_labitems` d on 
  l.itemid = d.itemid 
inner join `physionet-data.mimiciv_icu.icustays` icu on
  l.hadm_id = icu.hadm_id and l.charttime >= icu.intime and l.charttime <= DATE_ADD(icu.intime, INTERVAL {} HOUR)
WHERE l.itemid IN (51006, 50822, 50971, 50810, 51221, 50912, 50862, 50820, 50882, 50803)