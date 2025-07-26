from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, traits)
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import IdentityInterface

from .interfaces.io import load_tacs, load_blood
from .interfaces.models import (
    LoganModel, MA1Model, OneTCMModel, TwoTCMModel,
    SRTMModel, SRTM2Model)


MODEL_MAP = {
    "logan": LoganModel,
    "ma1": MA1Model,
    "1tcm": OneTCMModel,
    "2tcm": TwoTCMModel,
    "srtm": SRTMModel,
    "srtm2": SRTM2Model,
}


class KMInputSpec(BaseInterfaceInputSpec):
    tac_times = traits.Array(mandatory=True)
    tac_values = traits.Array(mandatory=True)
    plasma_times = traits.Array()
    plasma_values = traits.Array()
    blood_values = traits.Array()
    ref_times = traits.Array()
    ref_values = traits.Array()
    k2_ref = traits.Float()
    model = traits.Enum(*MODEL_MAP.keys(), mandatory=True)
    t_star = traits.Float()
    vB_fixed = traits.Float()
    fit_end_time = traits.Float()


class KMOutputSpec(TraitedSpec):
    parameters = traits.Dict()


class KineticModelInterface(BaseInterface):
    input_spec = KMInputSpec
    output_spec = KMOutputSpec

    def _run_interface(self, runtime):
        model_class = MODEL_MAP[self.inputs.model]
        kwargs = dict(tac_times=self.inputs.tac_times,
                      tac_values=self.inputs.tac_values)
        if self.inputs.model in ["logan", "ma1", "1tcm", "2tcm"]:
            kwargs.update(plasma_times=self.inputs.plasma_times,
                          plasma_values=self.inputs.plasma_values)
        if self.inputs.model in ["1tcm", "2tcm"]:
            kwargs.update(blood_values=self.inputs.blood_values,
                          vB_fixed=self.inputs.vB_fixed,
                          fit_end_time=self.inputs.fit_end_time)
        if self.inputs.model in ["logan", "ma1"]:
            kwargs["t_star"] = self.inputs.t_star
        if self.inputs.model in ["srtm", "srtm2"]:
            kwargs.update(ref_times=self.inputs.ref_times,
                          ref_values=self.inputs.ref_values)
        if self.inputs.model == "srtm2":
            kwargs["k2_ref"] = self.inputs.k2_ref
        model = model_class(**kwargs)
        self._results["parameters"] = model.fit()
        return runtime


def init_kinmod_workflow(tac_file, model,
                          blood_file=None, reference_file=None):
    tac_times, roi_names, tac_values = load_tacs(tac_file)
    wf = pe.Workflow(name="kinmod_wf")

    model_node = pe.MapNode(KineticModelInterface(model=model),
                            iterfield=["tac_values"],
                            name="model")
    model_node.inputs.tac_times = tac_times
    model_node.inputs.tac_values = list(tac_values)

    if model in ["logan", "ma1", "1tcm", "2tcm"] and blood_file:
        p_times, p_vals, b_vals = load_blood(blood_file)
        model_node.inputs.plasma_times = p_times
        model_node.inputs.plasma_values = p_vals
        model_node.inputs.blood_values = b_vals

    if model in ["srtm", "srtm2"] and reference_file:
        r_times, r_names, r_vals = load_tacs(reference_file)
        model_node.inputs.ref_times = r_times
        model_node.inputs.ref_values = r_vals[0]

    wf.add_nodes([model_node])
    return wf, roi_names
