from petprep_km.interfaces.models import LoganRefModel, SRTMModel
from petprep_km.interfaces.io import load_tacs
from petprep_km.utils.misc import save_kinpar_tsv, save_kinpar_json, generate_html_report
import os

# Load TACs
TACS_PATH = 'data/derivatives/petprep/sub-01/pet/sub-01_desc-preproc_seg-gtm_timeseries.tsv'
REF_PATH = 'data/derivatives/petprep/sub-01/pet/sub-01_ref-cerebellum_desc-preproc_seg-gtm_timeseries.tsv'

tac_times, roi_names, tac_values = load_tacs(TACS_PATH)
ref_times, ref_roi_names, ref_values = load_tacs(REF_PATH)
ref_tac = ref_values[0]

# Estimate k2' from reference using SRTM
ref_model = SRTMModel(
    tac_times=tac_times,
    tac_values=ref_tac,
    ref_times=ref_times,
    ref_values=ref_tac,
)
ref_fit = ref_model.fit()
k2_ref = ref_fit['k2']

t_star = 41
subject_id = '01'
session_id = 'baseline'
model_class = LoganRefModel
model_name = model_class.__name__.replace('Model', '')

results = []
image_paths = []

output_dir = f'derivatives/petprep_km/sub-{subject_id}/ses-{session_id}/pet'
os.makedirs(output_dir, exist_ok=True)

for roi_idx, roi_name in enumerate(roi_names):
    print(f'Fitting ROI (LoganRef): {roi_name}')
    model = model_class(
        tac_times=tac_times,
        tac_values=tac_values[roi_idx],
        ref_times=ref_times,
        ref_values=ref_tac,
        k2_prime=k2_ref,
        t_star=t_star,
    )
    fit_res = model.fit()
    results.append(fit_res)
    fig_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_roi-{roi_name}_model-{model_name}_fit.png')
    model.visualize_fit(fig_path, region_name=roi_name)
    image_paths.append(fig_path)

# Save outputs

tsv_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.tsv')
json_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.json')
html_path = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_model-{model_name}_report.html')

save_kinpar_tsv(tsv_path, roi_names, results)
save_kinpar_json(json_path, model_name, 'reference', t_star, model_class.parameters)
generate_html_report(html_path, image_paths, title=f'{model_name} Modeling Report')

print(f'Results saved: {tsv_path}')
print(f'Metadata saved: {json_path}')
print(f'HTML report saved: {html_path}')
