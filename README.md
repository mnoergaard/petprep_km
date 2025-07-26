# petprep_km
BIDS application to perform kinetic modeling

## Nipype workflow

This release introduces a simple Nipype workflow that wraps the provided kinetic
models.  Two new models, **SRTM** and **SRTM2**, are available.  The workflow can
be executed using the `petprep_km_workflow` command line tool.

Example:

```bash
petprep_km_workflow path/to/tacs.tsv srtm --reference path/to/ref_tacs.tsv --output out_dir
```

This will produce a TSV file with kinetic parameters for each region.
