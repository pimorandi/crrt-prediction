with height as (
  select h.subject_id, h.stay_id, avg(h.height) height
  from `physionet-data.mimiciv_derived.height` h
  inner join `physionet-data.mimiciv_icu.icustays` icu on
    h.subject_id = icu.subject_id and h.stay_id = icu.stay_id
  group by h.subject_id, h.stay_id
)
select icu.subject_id, icu.hadm_id, icu.stay_id, 
  DATETIME_DIFF(icu.intime, DATETIME(pz.anchor_year, 1, 1, 0, 0, 0), YEAR) + pz.anchor_age AS age,
  pz.gender, w.weight, h.height, 
  CASE WHEN w.weight IS NULL OR h.height IS NULL OR w.weight = 0 OR h.height = 0 
      THEN NULL 
  ELSE 
      w.weight / ((h.height / 100)*(h.height / 100))
  END AS bmi
from `physionet-data.mimiciv_icu.icustays` icu
left join `physionet-data.mimiciv_hosp.patients` pz on
  pz.subject_id = icu.subject_id
left join `physionet-data.mimiciv_derived.first_day_weight`w on
    icu.stay_id = w.stay_id
left join height h on
    icu.subject_id = h.subject_id and icu.stay_id = h.stay_id