def save_kinpar_tsv(output_path, roi_names, param_results):
    import pandas as pd

    # Assuming param_results is a list of dictionaries
    df = pd.DataFrame(param_results)
    df.insert(0, "name", roi_names)
    df.to_csv(output_path, sep='\t', index=False)


def save_kinpar_json(output_path, model_name, blood_source, t_star, parameters):
    import json
    metadata = {
        "Description": f"{model_name} kinetic modeling results",
        "ModelName": model_name,
        "BloodType": blood_source,
        "AdditionalModelDetails": f"Linear fit from t*={t_star} minutes",
        "SoftwareName": "petprep_km",
        "SoftwareVersion": "0.1.0",
        "CommandLine": "run_ma1_example.py",
        "Parameters": parameters
    }
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# Generic plotting utility for fitted vs original data

def plot_fit(tac_times, tac_values, fit_values, region_name, output_path, title=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(tac_times, tac_values, 'o-', label='TAC (Data)')
    plt.plot(tac_times, fit_values, '--', label='Model Fit')
    plt.title(title or f"Model Fit - {region_name}")
    plt.xlabel("Time (min)")
    plt.ylabel("Radioactivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# HTML report generator that collects all region plots

def generate_html_report(output_path, image_paths, title="Kinetic Modeling Report"):
    import os
    report_dir = os.path.dirname(output_path)
    with open(output_path, "w") as f:
        f.write("<html><head><title>{}</title></head><body>\n".format(title))
        f.write(f"<h1>{title}</h1>\n")
        for img_path in image_paths:
            rel_img_path = os.path.relpath(img_path, start=report_dir)
            img_name = os.path.basename(img_path)
            f.write(f"<div><h3>{img_name}</h3>\n")
            f.write(f"<img src='{rel_img_path}' style='width:600px;'><br></div>\n")
        f.write("</body></html>\n")