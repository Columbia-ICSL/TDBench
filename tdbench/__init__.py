"""TDBench data tools (`tdbench build`) and evaluation entry point (`tdbench run`)."""

HF_REPO = 'Columbia-ICSL/TDBench'
HF_REVISION = 'main'

# release file -> VLMEvalKit dataset name (the TSV written to $LMUData)
SPLITS = {
    'main': 'TDBench',
    'grounding': 'TDBench_Grounding',
    'case_study_zoom_in': 'TDBench_CS_Zoom',
    'case_study_height': 'TDBench_CS_Height',
    'case_study_integrity': 'TDBench_CS_Integrity',
    'case_study_depth': 'TDBench_CS_Depth',
}
