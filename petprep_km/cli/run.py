import argparse
import os
from petprep_km.interfaces.models import LoganModel, MA1Model, OneTCMModel, TwoTCMModel
from petprep_km.interfaces.io import load_tacs, load_blood
from petprep_km.utils.misc import save_kinpar_tsv, save_kinpar_json, generate_html_report

MODEL_CLASSES = {
    "logan": LoganModel,
    "ma1": MA1Model,
    "1tcm": OneTCMModel,
    "2tcm": TwoTCMModel
}

def main():
    parser = argparse.ArgumentParser(description='Run PET kinetic modeling.')

    parser.add_argument('bids_dir', type=str, help='Path to BIDS dataset directory')
    parser.add_argument('output_dir', type=str, help='Directory for saving output')
    parser.add_argument('analysis_level', choices=['participant'], help='Analysis level')

    parser.add_argument('--participant-label', required=True, help='Subject identifier')
    parser.add_argument('--session-label', required=True, help='Session identifier')
    parser.add_argument('--model', required=True, choices=MODEL_CLASSES.keys(), help='Model type')
    parser.add_argument('--blood-derivatives', required=True, help='Name of derivatives directory containing blood data')

    parser.add_argument('--desc', default=None, help='Descriptor in TAC file name (optional)')
    parser.add_argument('--pvc', default=None, choices=['gtm', 'mgx', 'none'], help='Partial volume correction applied')
    parser.add_argument('--tstar', type=float, default=None, help='t* for Logan and MA1 models')
    parser.add_argument('--vb-fixed', type=float, default=None, help='Fixed blood volume fraction')
    parser.add_argument('--fit-end-time', type=float, default=None, help='End time for fitting (min)')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--save-figures', action='store_true', help='Save diagnostic figures')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--roi', default=None, help='Specify a single region of interest to model')

    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model]

    tac_fname_parts = [f"sub-{args.participant_label}", f"ses-{args.session_label}"]
    if args.pvc:
        tac_fname_parts.append(f"pvc-{args.pvc}")
    if args.desc:
        tac_fname_parts.append(f"desc-{args.desc}")
    tac_fname_parts.append("tacs.tsv")

    tac_fname = "_".join(tac_fname_parts)

    tac_path = os.path.join(args.bids_dir, f"sub-{args.participant_label}",
                            f"ses-{args.session_label}", tac_fname)

    blood_path = os.path.join(args.bids_dir, '..', args.blood_derivatives,
                              f"sub-{args.participant_label}", f"ses-{args.session_label}",
                              f"sub-{args.participant_label}_ses-{args.session_label}_inputfunction.tsv")

    tac_times, roi_names, tac_values = load_tacs(tac_path)
    plasma_times, plasma_values, blood_values = load_blood(blood_path)

    output_pet_dir = os.path.join(args.output_dir, f"sub-{args.participant_label}",
                                 f"ses-{args.session_label}")
    output_figures_dir = os.path.join(args.output_dir, f"sub-{args.participant_label}",
                                      f"ses-{args.session_label}", "figures")

    os.makedirs(output_pet_dir, exist_ok=True)
    if args.save_figures:
        os.makedirs(output_figures_dir, exist_ok=True)

    results = []
    image_paths = []

    for roi_idx, roi_name in enumerate(roi_names):
        if args.roi and roi_name != args.roi:
            continue
        
        if args.verbose:
            print(f"Fitting ROI: {roi_name}")

        # Default kwargs for all models
        model_kwargs = {
            "tac_times": tac_times,
            "tac_values": tac_values[roi_idx],
            "plasma_times": plasma_times,
            "plasma_values": plasma_values,
        }

        # Add blood values only for models requiring it
        if args.model in ["1tcm", "2tcm"]:
            model_kwargs["blood_values"] = blood_values
            model_kwargs["vB_fixed"] = args.vb_fixed
            model_kwargs["fit_end_time"] = args.fit_end_time
            model_kwargs["n_iterations"] = args.n_iterations

        if args.model in ["logan", "ma1"]:
            model_kwargs["t_star"] = args.tstar

        # Instantiate the model
        model = model_class(**model_kwargs)

        fit_results = model.fit()
        results.append(fit_results)

        if args.save_figures:
            fig_path = os.path.join(output_figures_dir,
                                    f"sub-{args.participant_label}_ses-{args.session_label}_roi-{roi_name}_model-{args.model}_fit.png")
            model.visualize_fit(fig_path, region_name=roi_name)
            image_paths.append(fig_path)

    # Modify roi_names if --roi is specified
    if args.roi:
        roi_names_to_save = [args.roi]
    else:
        roi_names_to_save = roi_names

    tsv_path = os.path.join(output_pet_dir,
                            f"sub-{args.participant_label}_ses-{args.session_label}_model-{args.model}_kinpar.tsv")
    json_path = os.path.join(output_pet_dir,
                            f"sub-{args.participant_label}_ses-{args.session_label}_model-{args.model}_kinpar.json")

    save_kinpar_tsv(tsv_path, roi_names_to_save, results)
    save_kinpar_json(json_path, args.model, "arterial", None, model_class.parameters)

    if args.save_figures:
        html_path = os.path.join(args.output_dir, f"sub-{args.participant_label}", f"ses-{args.session_label}",
                                 f"sub-{args.participant_label}_ses-{args.session_label}_model-{args.model}_report.html")
        generate_html_report(html_path, image_paths, title=f"{args.model} Modeling Report")

    if args.verbose:
        print(f"Results saved: {tsv_path}")
        print(f"Metadata saved: {json_path}")
        if args.save_figures:
            print(f"HTML report saved: {html_path}")

if __name__ == '__main__':
    main()