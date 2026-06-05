from __future__ import annotations
import json

import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any
from kernel.data import PairRecord
from estimator.representor import resolve_representor
from estimator.extractor import FeatureExtractor

COMPATIBLE_TRANSFORMS = {"identity": {"multitransform", "float32", "float64", "sign_transform", "thresholded_sign_transform"},
                         "sign": {"multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                         "thresholded_sign": {"multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                }


def _prepare_cache_matrix(
    path: str | Path,
    expected_feature_d: int | None = None,
    expected_records_n: int | None = None,
) -> Path:
    """
        Check if the feature matrix is ready, if not, create it and fill with nan's.
    """
    path = Path(path)

    if path.exists():
        matrix = np.lib.format.open_memmap(
        path,
        mode="r",
        )
        
        if expected_records_n is not None and expected_records_n != int(matrix.shape[0]):
            raise ValueError("The existing feature matrix doesnt correspond to the provided records list."
                             "The number of features is different from the provided number of records.")
        if expected_feature_d is not None and expected_feature_d != int(matrix.shape[1]):
            raise ValueError("The existing feature matrix doesnt correspond to the provided records list."
                             "The feature length doesn't correspond to the number of the given model's parameters.")
        del matrix
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(expected_records_n, expected_feature_d),
    )
    matrix[:] = np.nan
    del matrix
    return path


def get_cache_meta(path: str | Path) -> CacheMeta:
    meta_path = Path(path)
    
    if not meta_path.exists() or not meta_path.absolute():
        raise FileNotFoundError(f"Didn't find the metadata at {meta_path}")
    meta_dict = json.loads(meta_path.read_text())
    meta = CacheMeta(**meta_dict)
    meta.validate()
    
    return meta


def update_cache_meta(path: str | Path, **update):
    meta_path = Path(path)

    if not meta_path.exists():
        raise FileNotFoundError(f"Didn't find the metadata at {meta_path}")
    meta_dict = json.loads(meta_path.read_text())
    meta_dict.update(update)
    meta_path.write_text(json.dumps(meta_dict))


def open_cache_matrix(path: str | Path, mode="r+") -> Any:
    matrix_path = Path(path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"Didn't find the cache matrix at {matrix_path}")
    if not (matrix_path.is_file() and matrix_path.suffix==".npy"):
        raise ValueError(f"Unexpected cache matrix format at {matrix_path}")

    cached = np.lib.format.open_memmap(
        matrix_path,
        mode=mode
    )

    return cached


def _fetch_features_by_idx(
    meta_path: Path | str,
    features_path: Path | str,
    records: list[tuple[int, PairRecord]],
    backend: FeatureExtractor,
) -> np.ndarray:

    if not records:
        raise ValueError("Cannot extract features for an empty split.")

    records_idx = [idx for idx, _ in records]
    idx_to_pairrecord = {idx : record for idx, record in records}

    cached = open_cache_matrix(features_path)
    meta = get_cache_meta(meta_path)

    representor = resolve_representor(
                                      name =  meta.representor_name,
                                      **meta.representor_args,
                                    )

    if not representor.init_name in COMPATIBLE_TRANSFORMS[backend.transform_name]:
        raise ValueError(f"Storring type '{representor.init_name}' and transformation '{backend.transform_name}' are not compatible")

    ready_idx = set(meta.computed_idx)
    missing_idx = list(set(records_idx) - ready_idx)
    missing_idx.sort()
    missing_records = [idx_to_pairrecord[idx] for idx in missing_idx]
    print(f"Features for records at indices: {missing_idx} are not cached, extracting...")

    write_to_memory = [representor.forward(backend.extract_feature(record)) for record in missing_records]
    
    for idx, vector in zip(missing_idx, write_to_memory):
        cached[idx] = vector
        ready_idx.add(idx)
   
    update_cache_meta(meta_path, computed_idx=list(ready_idx))

    resolved_fetched = [representor.inverse(cached[idx]) for idx in records_idx]
    
    del cached
    return np.stack(resolved_fetched)


STORAGE_TYPES = {"multitransform", "float32", "float64", 
                 "sign_int8", "sign_int16", "sign_int32", "sign_int64"}

@dataclass
class CacheMeta:
   key: str
   feature_transform_name: str
   feature_transform_params: dict[str,Any]
   num_records: int
   feature_dim: int
   model: str = "mlx-community/SmolLM2-1.7B-Instruct"
   adapter_path: str = ""
   representor_name: str = "float32"
   computed_idx: list[int] = field(default_factory=list)

   representor_args: dict[str, Any] = field(default_factory=dict)

   def to_dict(self):
      return asdict(self)
   
   def validate(self):
      if self.key.strip()=="":
         raise ValueError("Key has not been specified")
      supported_transformation = {"identity", "sign", "thresholded_sign"}
      if not self.feature_transform_name in supported_transformation:
         raise ValueError(f"Unsupported transformation {self.feature_transform_name}. Only support: {supported_transformation}")
      if self.num_records < 1:
         raise ValueError(f"Number of records must be a positive integer but got {self.num_records} instead")
      if self.feature_dim < 1:
         raise ValueError(f"Feature dimension must be a positive integer but got {self.feature_dim} instead")
      # if self.idx_map is not None and (any(x < 0 for x in self.idx_map) or len(set(self.idx_map)) != len(self.idx_map)):
      #    raise ValueError(f"Incorrect idx's formatting")
      
      if self.feature_transform_name == "thresholded_sign":
         if self.feature_transform_params is None or self.feature_transform_params.get("threshold", None) is None:
            raise ValueError(f"Must set the threshold value. But got {self.feature_transform_params} instead")
      if self.representor_name not in STORAGE_TYPES:
         raise ValueError(f"Unsupported storage type {self.representor_name}. Only support: {STORAGE_TYPES}")
         
SUPPORTED_TRANSFORMS = {"identity", "sign", "threshlded_sign"}

def compare_cache(first: CacheMeta, other: CacheMeta, error_log: str="", success_log: str=""):
   
    if not first.representor_name == "multitransform" or not other.representor_name == "multitransform":
        if first.representor_name != other.representor_name:
            raise ValueError(f"Storage type has changed since the cache was created. {error_log}")
        if first.representor_args != other.representor_args:
            raise ValueError(f"Representor configuration has changed since the cache was created. {error_log}")
        
    if first.feature_transform_name != other.feature_transform_name or first.feature_transform_params != other.feature_transform_params:
        raise ValueError(f"Feature transform configuration has changed since the cache was created. {error_log}")
    if first.num_records != other.num_records or first.feature_dim != other.feature_dim:
        raise ValueError(f"Data configuration has changed since the cache was created. {error_log}")
    if first.model != other.model or first.adapter_path != other.adapter_path:
        raise ValueError(f"Model configuration has changed since the cache was created. {error_log}")

    print(f"Cache is up to date. {success_log}")

def write_cache_meta(path: Path | str, meta: CacheMeta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_absolute or not path.suffix==".json":
       raise ValueError("Unsupported path specifications!")

    path.write_text(json.dumps(meta.to_dict()))