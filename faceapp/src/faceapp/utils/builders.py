from faceapp._base.pipeline import Pipeline
from faceapp._base.base import Process


class PipelineBuilder:
    def __init__(self):
        self.pipelines = {}

    def add_pipeline(self, pipeline: [Pipeline, Process], name: str = "Pipeline"):
        if isinstance(pipeline, Pipeline):
            self.pipelines[pipeline.name] = pipeline
        else:
            self.pipelines[name] = pipeline
        return self

    def build(self, name):
        return Pipeline(self.pipelines, name)
