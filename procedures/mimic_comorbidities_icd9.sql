with comorbidities_icd9 as (
  select subject_id, hadm_id, icd_code, 
    case
      when REGEXP_CONTAINS(icd_code, '[A-Z]') then 0
      when 
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(25000, 25033)) or 
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64800, 64804)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(24900, 24931)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(25040, 25093)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(24940, 24991))
      then 1
      else 0
    end diabetes,
    case
      when REGEXP_CONTAINS(icd_code, '[A-Z]') then 0
      when 
        cast(icd_code as INT64) in (4280, 4281, 4289, 39891) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(42820, 42823)) or 
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(42830, 42832)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(42840, 42843))
      then 1
      else 0
    end heart_failure,
    case
      when REGEXP_CONTAINS(icd_code, '[A-Z]') then 0
      when 
        cast(icd_code as INT64) in (4010,4011,4019,4372,40200,40210,40290,
          40509,40519,40201,40211,40291,40310,40300,40390,40501,40511,40591,
          40301,40311,40391,40400,40410,40490,40401,40411,40491,40402,40412,
          40492,40403,40413,40493) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64200, 64204)) or 
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64220, 64224)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64210, 64214)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64270, 64274)) or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(64290, 64294))
      then 1
      else 0
    end hypertension,
    case
      when REGEXP_CONTAINS(icd_code, '[A-Z]') then 0
      when 
        cast(icd_code as INT64) = 5859 or
        cast(icd_code as INT64) in UNNEST(GENERATE_ARRAY(5851, 5856))
      then 1
      else 0
    end chronic_kidney_disease
  from `physionet-data.mimiciv_hosp.diagnoses_icd`
  where icd_version = 9
)
select icd.subject_id, icd.hadm_id, icu.stay_id, max(diabetes) diabetes, max(heart_failure) heart_failure, 
  max(hypertension) hypertension, max(chronic_kidney_disease) chronic_kidney_disease
from comorbidities_icd9 icd
inner join `physionet-data.mimiciv_icu.icustays` icu on
  icd.subject_id = icu.subject_id and icd.hadm_id = icu.hadm_id
group by icd.subject_id, icd.hadm_id, icu.stay_id