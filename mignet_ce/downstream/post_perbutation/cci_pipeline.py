from pathlib import Path

from .cci import stage_cci
from .config import Settings, Task
from .formal import Inputs, prepare_on_cpu


def prepare_condition(inputs: Inputs, regions: Path, task: Task, settings: Settings, directory: Path) -> dict:
    cci = stage_cci(inputs.cci_t, regions, task, settings, directory)
    return {**cci, **prepare_on_cpu(inputs, settings, directory, Path(cci["input"]))}
