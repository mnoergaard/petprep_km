import argparse
import os

from petprep_km.workflows import init_kinmod_workflow
from petprep_km.utils.misc import save_kinpar_tsv

MODEL_CHOICES = ["logan", "ma1", "1tcm", "2tcm", "srtm", "srtm2"]


def main():
    parser = argparse.ArgumentParser(description="Run kinetic modeling workflow using Nipype")
    parser.add_argument("tac_file", help="Path to TAC TSV file")
    parser.add_argument("model", choices=MODEL_CHOICES, help="Model to fit")
    parser.add_argument("--blood", dest="blood_file", help="TSV with arterial input")
    parser.add_argument("--reference", dest="reference_file", help="Reference TAC file for reference models")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    wf, roi_names = init_kinmod_workflow(args.tac_file, args.model,
                                         blood_file=args.blood_file,
                                         reference_file=args.reference_file)
    res = wf.run()
    results = res.nodes()[0].result.outputs.parameters

    os.makedirs(args.output, exist_ok=True)
    out_tsv = os.path.join(args.output, "kinpar.tsv")
    save_kinpar_tsv(out_tsv, roi_names, results)
    print(f"Results saved to {out_tsv}")


if __name__ == "__main__":
    main()
