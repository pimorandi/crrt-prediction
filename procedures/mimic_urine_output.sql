select u.stay_id, sum(urineoutput) tot_urine
from `physionet-data.mimiciv_derived.urine_output` u
inner join `physionet-data.mimiciv_icu.icustays` i on
  u.stay_id = i.stay_id and 
  u.charttime >= i.intime and 
  u.charttime <= DATETIME_ADD(i.intime, interval {} HOUR)
group by u.stay_id