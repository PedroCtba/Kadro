"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from kagro.pipelines import FeatureEnginering as fe

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_enginering_pipelines = fe.create_pipeline()

    return {"__default__": data_enginering_pipelines,
    "dataEnginering": data_enginering_pipelines}
