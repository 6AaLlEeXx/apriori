from kernel.data import PairRecord
from kernel.config import KernelRunConfig
from kernel.features import LoRANTKFeatureBackend, create_feature_backend
from transformations import apply_transformations
from kernel.run import _feature_transform_names, _feature_transform_params
from numpy.typing import NDArray
from typing import Any

class FeatureExtractor:
    def __init__(self, config: KernelRunConfig):
        self.model_name: str = config.base_model
        self.adapter_path: str = config.adapter_path
        self.backend: LoRANTKFeatureBackend = create_feature_backend(config)
        self.dim: int | None = None

        transform_name = _feature_transform_names(config)
        if len(transform_name) != 1:
            raise ValueError("Multiple transformations are not supported.")
        self.transform_name: str = transform_name[0]

        self.transform_params: dict[str, Any] = _feature_transform_params(config).get(self.transform_name, {})

    def extract_feature(self, record: PairRecord) -> NDArray:
        feature = self.backend.extract_feature(record).reshape((1,-1))
        transformed = apply_transformations(feature, [self.transform_name], {self.transform_name: self.transform_params})
        dim = transformed.shape[1]
        if self.dim is None:
            self.dim = dim
        elif self.dim != dim:
            raise ValueError(f"Extracted feature dimension {dim} does not match expected dimension {self.dim}")
        
        return transformed.reshape(-1)