from pydantic import BaseModel, Field
from typing import Dict, Any

class ModelNameConfig(BaseModel):
    '''Model hyperparameters for CatBoost.'''
    verbose: int = Field(default=0, description="Verbosity level")
    eval_metric: str = Field(default="F1", description="Evaluation metric")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    iterations: int = Field(default=1000, description="Number of boosting iterations")
    depth: int = Field(default=10, description="Tree depth")

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
