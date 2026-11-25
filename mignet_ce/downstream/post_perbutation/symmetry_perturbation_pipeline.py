from .config import FAMILIES, VARIANTS
from .symmetry_perturbation import calibrate_family, plan_endpoint


class Calibration:
    def __init__(self, baseline, reference, settings, root):
        self.pij = baseline / "training" / "PIJ_micro_train.npy"
        self.reference = reference
        self.settings = settings
        self.root = root
        self.endpoints = {family: {} for family in FAMILIES}

    def submit(self, submit, pending):
        for family in FAMILIES:
            if pending(family):
                for variant in VARIANTS:
                    submit("endpoint", plan_endpoint, self.pij, self.reference, family, variant,
                           self.settings, self.root / family, context=(family, variant))

    def collect(self, submit, pending, family, variant, result):
        endpoints = self.endpoints[family]
        endpoints[variant] = result
        tasks = pending(family)
        if len(endpoints) == len(VARIANTS) and tasks:
            levels = tuple(sorted({task.level for task in tasks}))
            submit("symmetry_plan", calibrate_family, self.pij, self.reference, family,
                   endpoints, self.root / family, levels, context=family)
