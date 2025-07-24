import os

def generate_html_report(output_dir, image_files, model_name, subject_id, session_id):
    html_path = os.path.join(output_dir, f"sub-{subject_id}_ses-{session_id}_model-{model_name}_report.html")
    with open(html_path, "w") as f:
        f.write("<html><head><title>Kinetic Modeling Report</title></head><body>\n")
        f.write(f"<h1>Model: {model_name}</h1>\n")
        for img in image_files:
            f.write(f"<div><h3>{os.path.basename(img)}</h3>\n")
            f.write(f"<img src='{os.path.basename(img)}' style='width:600px;'><br></div>\n")
        f.write("</body></html>\n")
    return html_path