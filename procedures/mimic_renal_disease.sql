select distinct diag.subject_id, diag.hadm_id, icu.stay_id,
  case 
    when count(distinct icd_code) over(partition by diag.subject_id, diag.hadm_id) > 0
      then 1
      else 0
    end renal_disease
from `physionet-data.mimiciv_hosp.diagnoses_icd` diag
inner join `physionet-data.mimiciv_icu.icustays` icu on 
  diag.subject_id = icu.subject_id and diag.hadm_id = icu.hadm_id
where icd_code in ('N186', '5856', 'Z992', 'V4511')