with base as (
  SELECT stay_id, min(charttime) crrt_charttime
  FROM `physionet-data.mimiciv_derived.crrt` 
  group by stay_id
)
select base.stay_id, round(DATETIME_DIFF(base.crrt_charttime, icu.intime, SECOND) / 3600, 3) crrt_deltatime
from base
inner join `physionet-data.mimiciv_icu.icustays` icu on
  base.stay_id = icu.stay_id