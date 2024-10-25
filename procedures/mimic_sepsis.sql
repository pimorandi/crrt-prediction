select s.stay_id, i.intime, s.antibiotic_time, s.culture_time, s.suspected_infection_time, s.sepsis3
from `physionet-data.mimiciv_derived.sepsis3` s
inner join `physionet-data.mimiciv_icu.icustays` i on
  s.stay_id = i.stay_id and 
  s.suspected_infection_time >= i.intime and 
  s.suspected_infection_time <= DATETIME_ADD(i.intime, interval {} HOUR)