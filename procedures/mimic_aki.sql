with base as (
  select aki.subject_id, aki.hadm_id, aki.stay_id, round(datetime_diff(aki.charttime, icu.intime, SECOND) / 3600, 3) as delta_time, 
    aki.aki_stage_creat, aki_stage_uo, aki_stage, aki_stage_crrt, aki.aki_stage_smoothed
  from `physionet-data.mimiciv_derived.kdigo_stages` aki
  inner join `physionet-data.mimiciv_icu.icustays` icu on
    aki.stay_id = icu.stay_id
  where datetime_diff(aki.charttime, icu.intime, SECOND) / 3600 < {}
  order by charttime
)
select stay_id, max(aki_stage_creat) aki_stage_creat, max(aki_stage_uo) aki_stage_uo, 
  max(aki_stage) aki_stage, max(aki_stage_crrt) aki_stage_crrt, max(aki_stage_smoothed) aki_stage_smoothed
from base
group by stay_id