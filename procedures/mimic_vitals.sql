SELECT v.stay_id, round(avg(heart_rate), 3) heart_rate, 
  round(avg(sbp), 3) sbp, round(avg(dbp), 3) dbp, round(avg(mbp), 3) mbp, 
  round(avg(resp_rate), 3) resp_rate, round(avg(spo2), 3) spo2
FROM `physionet-data.mimiciv_derived.vitalsign` v
inner join `physionet-data.mimiciv_icu.icustays` icu on
  v.stay_id = icu.stay_id
where v.charttime >= icu.intime and v.charttime <= date_add(icu.intime, INTERVAL {} HOUR)
group by v.stay_id