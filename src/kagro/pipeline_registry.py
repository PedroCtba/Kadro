"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kagro.pipelines import FeatureEnginering as fe, TrainMyModels as tm

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_enginering_pipelines = fe.create_pipeline()
    data_science_pipelines = tm.create_pipeline()

    return {"__default__": data_science_pipelines,
    "dataEnginering": data_enginering_pipelines,
    "dataScience": data_science_pipelines
    }
