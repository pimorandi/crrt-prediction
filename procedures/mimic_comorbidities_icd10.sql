with comorbidities_icd10 as (
  select subject_id, hadm_id, icd_code,
    case 
      when 
        STARTS_WITH(icd_code, "E08") or STARTS_WITH(icd_code, "E09") or
        STARTS_WITH(icd_code, "E10") or STARTS_WITH(icd_code, "E11") or 
        STARTS_WITH(icd_code, "E12") or STARTS_WITH(icd_code, "E13")
      then 1
    else 0
    end diabetes,
    case 
      when 
        STARTS_WITH(icd_code, "I13") or STARTS_WITH(icd_code, "I11") or
        STARTS_WITH(icd_code, "I25") or STARTS_WITH(icd_code, "I46") or 
        STARTS_WITH(icd_code, "I50") or STARTS_WITH(icd_code, "I42") or
        STARTS_WITH(icd_code, "I43") or STARTS_WITH(icd_code, "I44")
      then 1
    else 0
    end heart_failure,
    case 
      when 
        STARTS_WITH(icd_code, "I10") or STARTS_WITH(icd_code, "I15") or
        STARTS_WITH(icd_code, "I16")
      then 1
    else 0
    end hypertension,
    case 
      when 
        STARTS_WITH(icd_code, "I12") or STARTS_WITH(icd_code, "N01") or STARTS_WITH(icd_code, "N13") or
        STARTS_WITH(icd_code, "N02") or STARTS_WITH(icd_code, "N03") or STARTS_WITH(icd_code, "N14") or
        STARTS_WITH(icd_code, "N04") or STARTS_WITH(icd_code, "N05") or STARTS_WITH(icd_code, "N15") or
        STARTS_WITH(icd_code, "N06") or STARTS_WITH(icd_code, "N07") or STARTS_WITH(icd_code, "N18") or
        STARTS_WITH(icd_code, "N08") or STARTS_WITH(icd_code, "N11") or STARTS_WITH(icd_code, "N19") or
        STARTS_WITH(icd_code, "N25") or STARTS_WITH(icd_code, "N26") or STARTS_WITH(icd_code, "N27") or
        STARTS_WITH(icd_code, "N28") or STARTS_WITH(icd_code, "N29")
      then 1
    else 0
    end chronic_kidney_disease
  from `physionet-data.mimiciv_hosp.diagnoses_icd`
  where icd_version = 10
)
select icd.subject_id, icd.hadm_id, icu.stay_id, max(diabetes) diabetes, max(heart_failure) heart_failure, 
  max(hypertension) hypertension, max(chronic_kidney_disease) chronic_kidney_disease
from comorbidities_icd10 icd
inner join `physionet-data.mimiciv_icu.icustays` icu on
  icd.subject_id = icu.subject_id and icd.hadm_id = icu.hadm_id
group by icd.subject_id, icd.hadm_id, icu.stay_id