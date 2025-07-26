# Example script for SRTM and SRTM2 modeling
from petprep_km.interfaces.models import SRTMModel, SRTM2Model
from petprep_km.interfaces.io import load_tacs
from petprep_km.utils.misc import save_kinpar_tsv, save_kinpar_json, generate_html_report
import os

# Load target TACs and reference cerebellum TAC
TACS_PATH = 'data/derivatives/petprep/sub-01/pet/sub-01_desc-preproc_seg-gtm_timeseries.tsv'
REF_PATH = 'data/derivatives/petprep/sub-01/pet/sub-01_ref-cerebellum_desc-preproc_seg-gtm_timeseries.tsv'

tac_times, roi_names, tac_values = load_tacs(TACS_PATH)
ref_times, ref_roi_names, ref_values = load_tacs(REF_PATH)
ref_tac = ref_values[0]

subject_id = '01'
session_id = 'baseline'

# Fit SRTM on reference region to obtain k2'
ref_model = SRTMModel(
    tac_times=tac_times,
    tac_values=ref_tac,
    ref_times=ref_times,
    ref_values=ref_tac,
)
ref_fit = ref_model.fit()
k2_ref = ref_fit['k2']

# Prepare output
output_dir = f'derivatives/petprep_km/sub-{subject_id}/ses-{session_id}/pet'
os.makedirs(output_dir, exist_ok=True)

# --- SRTM ---
model_class = SRTMModel
model_name = model_class.__name__.replace('Model', '')
results_srtm = []
images_srtm = []

for roi_idx, roi_name in enumerate(roi_names):
    print(f'Fitting ROI (SRTM): {roi_name}')
    model = model_class(
        tac_times=tac_times,
        tac_values=tac_values[roi_idx],
        ref_times=ref_times,
        ref_values=ref_tac,
    )
    fit_res = model.fit()
    results_srtm.append(fit_res)
    fig_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_roi-{roi_name}_model-{model_name}_fit.png')
    model.visualize_fit(fig_path, region_name=roi_name)
    images_srtm.append(fig_path)

tsv_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.tsv')
json_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.json')
html_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_report.html')

save_kinpar_tsv(tsv_path, roi_names, results_srtm)
save_kinpar_json(json_path, model_name, 'reference', None, model_class.parameters)
generate_html_report(html_path, images_srtm, title=f'{model_name} Modeling Report')

print(f'Results saved: {tsv_path}')
print(f'Metadata saved: {json_path}')
print(f'HTML report saved: {html_path}')

# --- SRTM2 ---
model_class = SRTM2Model
model_name = model_class.__name__.replace('Model', '')
results_srtm2 = []
images_srtm2 = []

for roi_idx, roi_name in enumerate(roi_names):
    print(f'Fitting ROI (SRTM2): {roi_name}')
    model = model_class(
        tac_times=tac_times,
        tac_values=tac_values[roi_idx],
        ref_times=ref_times,
        ref_values=ref_tac,
        k2_ref=k2_ref,
    )
    fit_res = model.fit()
    results_srtm2.append(fit_res)
    fig_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_roi-{roi_name}_model-{model_name}_fit.png')
    model.visualize_fit(fig_path, region_name=roi_name)
    images_srtm2.append(fig_path)

tsv_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.tsv')
json_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.json')
html_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_report.html')

save_kinpar_tsv(tsv_path, roi_names, results_srtm2)
save_kinpar_json(json_path, model_name, 'reference', None, model_class.parameters)
generate_html_report(html_path, images_srtm2, title=f'{model_name} Modeling Report')

print(f'Results saved: {tsv_path}')
print(f'Metadata saved: {json_path}')
print(f'HTML report saved: {html_path}')
