from petprep_km.interfaces.models import MA1Model
from petprep_km.interfaces.io import load_tacs, load_blood
from petprep_km.utils.misc import save_kinpar_tsv, save_kinpar_json, plot_fit, generate_html_report
import os
import numpy as np
from scipy.integrate import cumtrapz

# Load data
tac_times, roi_names, tac_values = load_tacs(
    'data/derivatives/petprep/sub-01/pet/sub-01_desc-preproc_seg-gtm_timeseries.tsv'
)
plasma_times, plasma_values, blood_values = load_blood(
    'data/derivatives/bloodstream/sub-01/pet/sub-01_inputfunction.tsv'
)

# Parameters
t_star = 41
subject_id = "01"
session_id = "baseline"
model_class = MA1Model
model_name = model_class.__name__.replace("Model", "")

results = []
image_paths = []

# Output paths
output_dir = f"derivatives/petprep_km/sub-{subject_id}/ses-{session_id}/pet"
os.makedirs(output_dir, exist_ok=True)

for roi_idx, roi_name in enumerate(roi_names):
    model = model_class(
        tac_times=tac_times,
        tac_values=tac_values[roi_idx],
        plasma_times=plasma_times,
        plasma_values=plasma_values,
        t_star=t_star
    )
    fit_results = model.fit()
    results.append(fit_results)

    # --- Create predicted curve for plotting
    plasma_interp = np.interp(tac_times, plasma_times, plasma_values)
    integral_tac = cumtrapz(tac_values[roi_idx], tac_times, initial=0)
    integral_plasma = cumtrapz(plasma_interp, tac_times, initial=0)
    mask = tac_times >= t_star

    X1 = integral_plasma[mask] / tac_values[roi_idx][mask]
    X2 = plasma_interp[mask] / tac_values[roi_idx][mask]
    X = np.column_stack((X1, X2, np.ones_like(X1)))
    predicted = X @ np.array([fit_results["VT"], fit_results.get("coef_X2", 0), fit_results.get("intercept", 0)])

    # --- Plot using model's own visualization method
    fig_path = os.path.join(output_dir, f"sub-{subject_id}_ses-{session_id}_roi-{roi_name}_model-{model_name}_fit.png")
    model.visualize_fit(fig_path, region_name=roi_name)
    image_paths.append(fig_path)

# Save outputs
tsv_path = os.path.join(output_dir, f"sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.tsv")
json_path = os.path.join(output_dir, f"sub-{subject_id}_ses-{session_id}_model-{model_name}_kinpar.json")
html_path = os.path.join(output_dir, f"sub-{subject_id}_ses-{session_id}_model-{model_name}_report.html")

save_kinpar_tsv(tsv_path, roi_names, results)
save_kinpar_json(json_path, model_name, "arterial", t_star, model_class.parameters)
generate_html_report(html_path, image_paths, title=f"{model_name} Modeling Report")

print(f"Results saved: {tsv_path}")
print(f"Metadata saved: {json_path}")
print(f"HTML report saved: {html_path}")
