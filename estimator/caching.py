from pathlib import Path
from typing import Literal, Annotated, Any
from pydantic import Field, BaseModel, ValidationError
import numpy as np
from numpy.lib.format import open_memmap
from abc import ABC, abstractmethod
import hashlib
import json
from transformations.thresholded_sign import transform as threshold_transform
from kernel.data import PairRecord
from mlx_lm import load
from abc import ABC, abstractmethod

from kernel.config import KernelRunConfig, load_kernel_run_config, _deep_merge
from kernel.run import create_feature_backend, _feature_transform_names, _feature_transform_params
from kernel.features import LoRANTKFeatureBackend, create_feature_backend
from transformations import apply_transformations
from transformations.thresholded_sign import DEFAULT_THRESHOLD

Array = np.typing.NDArray
RepresentorName = Literal[  
                            "multitransform", "float32", "float64", 
                            "sign_int8", "sign_int16", "sign_int32", "sign_int64",
                            "identity", "thresholded_sign", "sign",
                        ]
TransformName = Literal["identity", "sign", "thresholded_sign"]
DType = Literal['float32', 'float16', 'float64', 'int8', 'int16', 'int32', 'int64']

class ModelNotFoundError(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"mlx_lm couldn't find model {model}")

class ModelLoadError(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"mlx_lm couldn't load model {model}")

class BackendLoadError(Exception):
    def __init__(self) -> None:
        super().__init__(f"Couldn't load lora backend")

class ConfigError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class BackendConfigsNotFound(ConfigError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class FeatureExtractor:
    '''Combines lora backend and transformation apply (sign/thresholded_sign) on call.
        Use .smoke_extractor to get a feature ectractor for debug.
        
        Feature extractor is passed to fetching functions to compute and save new features'''
    def __init__(self, config: KernelRunConfig, lazy = False):
        '''Takes kernel run cinfig to extract information about the backend and transformations.
        Use lazy=True if you don't want to load the heavy backend on init'''
        self.__model_name: str = config.base_model
        self.__config = config

        try:
            model = load(self.__model_name, lazy=True)
            del model
        except Exception as e:
            if str(e).find("404 Client Error") > -1:
                raise ModelNotFoundError(model=self.__model_name)
            else:
                raise ModelLoadError(model=self.__model_name)
            
        self.__adapter_path: str = config.adapter_path
        if not (Path(self.__adapter_path)/'adapter_config.json').resolve().exists():
            raise BackendConfigsNotFound(f"No adapter config found at {self.__adapter_path}. Need adapter path to restore the backend")
        
        self.__backend: LoRANTKFeatureBackend | None = create_feature_backend(config) if not lazy else None
        self.__dim: int | None = None

        self.__transform_name: TransformName = self.__feature_transform_name(config)
        self.__transform_params: dict[str, Any] = _feature_transform_params(config).get(self.transform_name, {})

        self.__default_name: TransformName = self.__transform_name
        self.__default_params = self.__transform_params.copy()

    @classmethod
    def smoke_extractor(cls, adapter_path: str | Path, 
                        transform_name: TransformName='identity', 
                        transform_args: dict[str, Any] = {},
                        ):
        '''
            Returns an extractor derived from config at the default path. Use for debug.
            Specify transformation arguments to overwrite configs specs.
            
            adapter_path: path to trained adapter path
            tranform_name: identity, thresholded_sign, sign
            thresholded_args: threshold
        '''
        DEFAULT_SMOKE_PATH = Path("configs/kernel/dolly_lora_ntk_smoke.yaml").resolve()
        
        path = Path(adapter_path).resolve()
        if not (path/'adapter_config.json').exists():
            raise BackendConfigsNotFound(f"No adapter config found at {adapter_path}")
        
        config = load_kernel_run_config(DEFAULT_SMOKE_PATH)
        config.adapter_path = str(path)
        config.backend_args['feature_transform'] = transform_name
        config.backend_args = _deep_merge(config.backend_args, transform_args)
        
        return cls(config=config)

    @property
    def transform_hash(self) -> str:
        '''Returns hash of feature extractor tranformation. Doesn't encode
        model specific parameters such as adapter and lora parameters.
        
        Be careful, even after using .switch_transform, the output
        of .transform_hash still corresponds to the state at initialization.'''
        GET_KERNEL_SPECS = ["backend", "backend_args", "seed"]

        def json_hash(dict_config: dict[str, Any]) -> str:
            json_str = json.dumps(dict_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        dict_config = self.__config.to_dict()
        
        specs = ""
        for key in GET_KERNEL_SPECS:
            val = dict_config.get(key, None)
            if val is None:
                raise ValueError(f"Some information is missing: {key}")
            if isinstance(val, dict):
                payload = json_hash(val)
            else:
                payload = f"{val}"
            specs = f"{specs}; {key}={payload}"

        return hashlib.sha256(specs.encode("utf-8")).hexdigest()

    @property
    def striped_hash(self) -> str:
        '''Returns hash of feature extractor striped from transform. 
        Used in pair with multitransform. Doesn't encode
        model specific parameters such as adapter and lora parameters.
        
        Be careful, even after using .switch_transform, the output
        of .striped_hash still corresponds to the state at initialization.'''

        dict_config = self.__config.to_dict()
        
        backend = dict_config.get("backend", None)
        backend_args = dict_config.get("backend_args", None)
        if backend_args is None or not isinstance(backend_args, dict):
            raise ValueError("Missing backend_args")
        leaf_filter = dict_config.get("backend_args", None)
        if leaf_filter is None:
            raise ValueError("Missing leaf_filter")
        seed = dict_config.get("seed", 42)

        return hashlib.sha256(f"raw_feature_extractor: bakend={backend}; backend_args={leaf_filter}; seed={seed}".encode("utf-8")).hexdigest()

    @property
    def transform_name(self) -> TransformName:
        return self.__transform_name

    @property
    def transform_params(self) -> dict[str, Any]:
        return self.__transform_params

    @property
    def dim(self) -> int:
        if self.__dim is not None:
            return self.__dim
        raise AttributeError(f"Feature dimension is not defined")

    def ensure_output_dim(self, record: PairRecord) -> int:
        if self.__dim is None:
            self.extract_feature(record)
        return self.dim
    
    @property
    def adapter_path(self) -> Path:
        return Path(self.__adapter_path)

    @property
    def model(self) -> str:
        if self.__model_name:
            return self.__model_name
        raise AttributeError(f"Model name is not defined")

    @classmethod
    def load(cls, config_path: str | Path) -> 'FeatureExtractor':
        '''Load extractor using configuration file from the path. Expect .yaml
            kernel run config.'''
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"No kernel run cinfiguration found at {config_path}")
        if path.suffix != '.yaml':
            raise FileNotFoundError(f"Path {config_path} must be a .yaml file")
        try:
            config = load_kernel_run_config(path)
        except Exception:
            raise FileNotFoundError(f"An error occurred during handling of config file at {config_path}")
        return cls(config=config)

    def switch_extractor_transform(self, transform_name: TransformName = 'identity', transform_params: dict[str, Any] = {}) -> None:
        '''Use to change the applied transformation after lora backend raw feature extraction.'''
        self.__transform_name = transform_name
        self.__transform_params = transform_params

    def switch_transform_back(self) -> None:
        '''Use to set transformation back to value assigned at intitialization'''
        self.__transform_name = self.__default_name
        self.__transform_params = self.__default_params.copy()
    
    @staticmethod
    def __feature_transform_name(config: KernelRunConfig) -> TransformName:
        transform_name = _feature_transform_names(config)
        if len(transform_name) != 1:
            raise ValueError("Multiple transformations are not supported.")
        transform_name = transform_name[0]
        if transform_name == 'identity' or transform_name == 'sign' or transform_name == 'thresholded_sign':
            return transform_name
        raise ValueError(f'Unsuported transformation {transform_name}')

    def __ensure_backend(self) -> LoRANTKFeatureBackend:
        if self.__backend is None:
            self.__backend = create_feature_backend(self.__config)
        if self.__backend is None:
            raise BackendLoadError()
        return self.__backend

    def extract_feature(self, record: PairRecord) -> Array:
        '''Extracts gradient feature using lora backend and applies transformation to it'''
        feature = self.__ensure_backend().extract_feature(record).reshape((1,-1))
        transformed = apply_transformations(feature, [self.transform_name], {self.transform_name: self.transform_params})
        dim = transformed.shape[1]

        if self.__dim is None:
            self.__dim = dim
        if self.dim != dim:
            raise ValueError(f"Extracted feature dimension {dim} does not match expected dimension {self.dim}")
        
        return transformed.reshape(-1)

    def raw_feature(self, record: PairRecord) -> Array:
        '''Returns raw gradient feature extracted with lora backend'''
        return self.__ensure_backend().extract_feature(record).reshape(-1)


def formatted_sign_to_int(A: Array, bit: int = 8) -> Array:
    """
        A: 2d Numpy Array
        bit: {8, 16, 32, 64}

        Takes an array of formatted sign features, where
        the formatting can be [nan, -1, 0, 1] -> [0, 1, 2, 3]
        and stores them as int_bit matrix F. Returnes F.

        Each value in each feature can take values: nan, -1, 0, 1.
        Therefore, we need 2 bit to encode a value with 4 states. So,
        we aim to encode a group of 'bit'/2 consequetive values as
        one int_bit numpy integer, because group_size*value_size =
        'bit'/2 * 2 = 'bit'.
    """
    SUPPORTED_INT_BITS = {8, 16, 32}

    INT = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    
    if A.ndim != 2:
        raise ValueError(f"Expercted 2d array, got dim={A.ndim} instead")
    if not bit in SUPPORTED_INT_BITS:
        raise ValueError(f"The bits resolution must one of: {SUPPORTED_INT_BITS}, but got {bit}")

    dim = int(A.shape[1])
  
    if dim*2 % bit != 0:
        raise ValueError(f"Bits per feature b/f={dim*2} must be divisible by {bit}")

    int_type = INT.get(bit, None)
    if int_type is None:
        raise ValueError(f"Numpy int{bit} type is not supported. Supported types are: {INT.keys()}")

    group_size = bit//2
    A = A.reshape(A.shape[0], -1, group_size)
    B = 4**np.arange(0, group_size, 1)

    positive_integers_shift = 2**(bit-1)
    Fw = A@B-positive_integers_shift

    return np.asarray(Fw, dtype=int_type)

def int_to_formatted_sign(Fw: Array, bit: int = 8) -> Array:
    '''Encodes sign-features stored as integers into formated sign features with values 0,1,2,3 for
    nan, -1, 0, 1 by translating base 10 integers to base 4.'''
    SUPPORTED_INT_BITS = {8, 16, 32}

    if Fw.ndim != 2:
        raise ValueError(f"Expercted 2d array, got dim={Fw.ndim} instead")
    if not bit in SUPPORTED_INT_BITS:
        raise ValueError(f"The bits resolution must one of: {SUPPORTED_INT_BITS}, but got {bit}")

    positive_integers_shift = 2**(bit-1)
    A_B = np.asarray(Fw, dtype=np.int64) + positive_integers_shift

    base4features = []
    for row in A_B:
        repr = []
        for x in row:
            group_size = bit//2
            string = np.base_repr(x, 4, padding=group_size)
            state_nums = [int(string[-i]) for i in range(1, group_size+1)]
            repr.extend(state_nums)
        base4features.append(repr)

    return np.asarray(base4features, dtype=np.int8)


class FeatureRepresentor(ABC):
    '''
        Base class for feature representor. Feature representor is used to
        transform feature vectors into form suitable for storing in memory.
        For example, if a feature is obrained after application of sign transform,
        that is if all the enties are either 1 or -1, then we want to store them
        as 1 bit values. But because numpy supports only 8 bit values, we need to encode
        groups of 2 bit values into a 8 bit integer to store more efficiently.
        Or, in order to store features in a lazy way, that is store raw, non-trasformed
        features and aply the required tranformation only on read, we can use multitranform
        that writes raw features to memory but applies a transformation on read.

        Every feature reprsentor has a forward and inverse functions.
        
        Forward is called when a feature is written to memory, to prepare it for storage.

        Inverse is called when a feature is read from memory to transform it back to its
        oroginal form
    '''
    
    @abstractmethod
    def forward(self, array: Array) -> Array:
        pass

    @abstractmethod
    def inverse(self, array: Array)-> Array:
        pass

    @abstractmethod
    def is_nan(self, array: Array)-> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> RepresentorName:
        pass

    @property
    @abstractmethod
    def dtype(self)->Any:
        pass

    @property
    @abstractmethod
    def args(self)->dict[str,Any]:
        pass

    @staticmethod
    @abstractmethod
    def on_read_args()->set[str]:
        pass

    @abstractmethod
    def hash(self)->str:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

class DefaultRepresentor(FeatureRepresentor):
    '''Representor that doesn't change features on write or on read'''
    def __init__(self, dim: int):
        self.__input_dim = dim

    def forward(self, array: Array)-> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.__input_dim:
            raise ValueError(f"Expected feature dimension of {self.__input_dim} but got {array.shape[-1]} instead")
        return array

    @property
    def output_dim(self) -> int:
        return self.__input_dim

    def inverse(self, array: Array) -> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.output_dim:
            raise ValueError(f"Expected feature dimension of {self.output_dim} but got {array.shape[-1]} instead")
        return array
    
    def is_nan(self, array: Array)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)
    
    def hash(self)->str:
        encoded = f"basic transformation; writes grad features. Input dim = {self.__input_dim}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self) -> RepresentorName:
        return "float32"

    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def args(self)->dict[str,Any]:
        return {'input_dim' : self.__input_dim}
    
    @staticmethod
    def on_read_args() -> set[str]:
        return set()
    
class SignIntRepresentor(FeatureRepresentor):
    '''Used for features with entries in {nan, -1, 0, 1} for thresholded sing
    transformed vectors.
    
    Breaks each feature vector into groups and encodes each group with an bit-integer.
    To do so, linear transformation is used on write, and base-4 encoding used on read.'''
    def __init__(self, dim: int, bit: int = 8):
        self.__input_dim = dim
        self.__bit = bit
        self.__entry_bits: int = 2
        
        if dim > 0:
            grp_size = self.__bit // self.__entry_bits
            residual = dim % grp_size
            self.__padding = 0 if residual==0 else grp_size - residual
            self.__output_dim = (dim+self.__padding)//grp_size
        else:
            raise ValueError(f"Dimension must be a positive integer, but got {dim}")

    @property
    def output_dim(self) -> int:
        return self.__output_dim

    def forward(self, array: Array) -> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.__input_dim:
            raise ValueError(f"Expected feature dimension of {self.__input_dim} but got {array.shape[-1]} instead")

        def to_format(array: Array) -> Array:
            if self.__padding is None:
                raise ValueError(f"The padding value is not specified")

            array += 2
            array = np.nan_to_num(array, nan=0)
            
            if self.__padding != 0:
                array = np.concatenate([array, np.zeros(shape=(array.shape[0], self.__padding))], axis=1)
            return array
        
        formatted = to_format(array.copy())
        return formatted_sign_to_int(formatted, self.__bit)

    def inverse(self, array: Array) -> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.output_dim:
            raise ValueError(f"Expected feature dimension of {self.output_dim} but got {array.shape[-1]} instead")

        inv = int_to_formatted_sign(array, self.__bit).astype(np.float32)
        if self.__padding > 0:
            inv = inv[:, :-self.__padding]

        inv[inv==0] = np.nan
        res = inv-2
        
        return res
    
    def is_nan(self, array: Array) -> Any:
        if array.ndim != 2:
            raise ValueError(f"Expected 2d array, but got {array.shape}")
        return np.any(array==0,1)
    
    def hash(self)->str:
        encoded = f"name={self.name} with input dim = {self.__input_dim}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self) -> RepresentorName:
        if self.__bit == 8:
            return 'sign_int8'
        if self.__bit == 16:
                    return 'sign_int16'
        if self.__bit == 32:
                    return 'sign_int32'
        raise ValueError(f"Unsupported bit value {self.__bit}")
    
    @property
    def dtype(self)->Any:
        return np.dtype(f"int{self.__bit}")
    
    @property
    def args(self)->dict[str,Any]:
        return {"input_dim": self.__input_dim, "bit": self.__bit}
    
    @staticmethod
    def on_read_args() -> set[str]:
        return set()
    

class Multitransform(FeatureRepresentor):
    '''Used to load trasformed features in a lazy way. That is when only raw features
    are stores without trasformation applied to then and then any desired
    transformation is applied on read.
    
    Forward is called on writes to memory and returns the same vector, assuming the 
    input vector is a raw feature
    
    Inverse is called on reads from memory and applies transformation to a raw vector
    '''
    def __init__(self, dim: int, transform_name: TransformName='identity', **transform_args):
        if dim > 0:
            self.__input_dim = dim
        else:
            raise ValueError(f"Can't accept nonpositive input dimeension {dim}")

        self.__transform = transform_name
        self.__args = transform_args

    def forward(self, array: Array) -> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.__input_dim:
            raise ValueError(f"Expected feature dimension of {self.__input_dim} but got {array.shape[-1]} instead")

        return array

    def inverse(self, array: Array) -> Array:
        if not array.ndim == 2:
            raise ValueError(f"Expected 2d array as input, but got {array.ndim} instead")
        if not array.shape[-1] == self.output_dim:
            raise ValueError(f"Expected feature dimension of {self.output_dim} but got {array.shape[-1]} instead")

        if self.__transform == 'thresholded_sign':
            return threshold_transform(features=array, threshold=self.__args.get('threshold', DEFAULT_THRESHOLD))
        if self.__transform == 'sign':
            return np.sign(array).astype(np.float32, copy=False)
        if self.__transform == 'identity':
            return array
        else:
            raise ValueError(f"Unknown transformation {self.__transform}")
    
    def is_nan(self, array: Array) -> Any:
        if array.ndim != 2:
            raise ValueError(f"Expected 2d array, but got ndim={array.ndim}")
        return np.any(np.isnan(array),1)
    
    def hash(self)->str:
        encoded = f"multitransform with input dim={self.__input_dim}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def switch_transform(self, name: TransformName, **args):
        self.__transform = name
        self.__args = args

    @property
    def output_dim(self) -> int:
        return self.__input_dim

    @property
    def name(self) -> RepresentorName:
        return 'multitransform'
    
    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def args(self)->dict[str,Any]:
        return {'input_dim' : self.__input_dim, 'transform_name' : self.__transform}

    @staticmethod
    def on_read_args() -> set[str]:
        return {'transform_name', 'threshold'}


def resolve_representor(name: RepresentorName, **kwargs)-> FeatureRepresentor:
    transformation_name = name if not name=='multitransform' else kwargs.get('transform_name', 'identity')

    if not isinstance(transformation_name, str):
        raise TypeError("Variable 'transformation_name' must be a string")
    dim = kwargs.get("dim", None)
    if dim is None or dim <= 0:
        raise ValueError(f"Expected a positive argument 'dim' but got {dim} instead")
 
    if transformation_name == "float32":
        return DefaultRepresentor(dim)
    if transformation_name == "sign_int8":
        return SignIntRepresentor(bit=8, dim=dim)
    if transformation_name == "sign_int16":
        return SignIntRepresentor(bit=16, dim=dim)
    if transformation_name == "sign_int32":
        return SignIntRepresentor(bit=32, dim=dim)
    if transformation_name == "identity":
        return Multitransform(dim=dim)
    if transformation_name == "sign":
        return Multitransform(dim=dim, transform_name='sign')
    if transformation_name == "thresholded_sign":
        threshold = kwargs.get("threshold", DEFAULT_THRESHOLD)
        return Multitransform(dim=dim, transform_name='thresholded_sign', threshold=threshold)
    raise ValueError(f"Unsupported representor name {name}")

class InvalidMetadata(Exception):
    def __init__(self, path: str|Path) -> None:
        super().__init__(f"Got invalid json config at {path}")

class BaseJSONConfig(BaseModel):
    '''Used as a base class for savable/loadable config-like objects'''
    def save(self, path: str | Path) -> None:
        full_path = Path(path).resolve()
        suffix = full_path.suffix
        if not full_path.is_absolute():
            raise ValueError(f"Path {path} must be a file, not a directory")
        if not suffix == '.json':
            raise ValueError(f"Path {path} must be a .json file but got {suffix} instead")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(self.model_dump_json())
    
    @classmethod
    def load(cls, path: str | Path):
        full_path = Path(path).resolve()
        suffix = full_path.suffix
        if not full_path.exists():
            raise FileNotFoundError(f"File with path {path} doesn't exist")
        if not full_path.is_absolute():
            raise ValueError(f"Path {path} must be a file, not a directory")
        if not suffix == '.json':
            raise ValueError(f"Path {path} must be a .json file but got {suffix} instead")
        try:
            meta = cls.model_validate_json(full_path.read_text())
        except ValidationError:
            raise InvalidMetadata(path)

        return meta

def _normalize_features_input(features : Array | list[Array]) -> Array:
    '''Prepares input features for further operations. Ensures the output
    is a 2d array of features'''
    if isinstance(features, list):
        if len(features) == 0:
            raise ValueError("Can't operate with empty list of features")
        if len(features)==1:
            if features[0].ndim != 1:
                raise ValueError(f"Single feature list must be 1d, but got feature dimension {features[0].ndim}")
            return features[0].reshape(1,-1)
        if  any([f.ndim != 1 for f in features]):
            raise ValueError(f"List of features must consist of 1d arrays, got dimensions of {features[0].ndim}")
        return np.stack(features)            
    else:
        if not features.ndim == 2:
            raise ValueError(f"Array of features must be a 2d array, but got dimension {features.ndim} instead")
        return features


class ShardedCacheConfig(BaseJSONConfig):
    '''Metadata object containing all the essential information needed
    for proper sharded caching functioning.
    
    Parameters include:
        size: Number of features in each block of data
        max_blocks: The max number of feature blocks that are stores in memmory.
        Blocks are created only when needed.
        cyclic_writes: If true, new features that pass the max memory volume limit are 
        written to the start of the first block cyclically
        append_only: is true, all new writes, even for the same keys, added as new entries
        in memory, taking more space. Is false, then writes to already existing keys only update
        the older vectors, taking no additional space. Takes more time but more memory efficient
        dim: dimenstion of features
        root: path to the root directory where the metadata and features are stored'''
    
    cache_type: str = 'sharded'
    header : Annotated[int, Field(ge=0)] = 0
    size : Annotated[int, Field(gt=0)] = 512
    max_blocks : Annotated[int, Field(gt=0)] | None = 64
    cyclic_writes : bool = True
    append_only : bool = True
    dim : Annotated[int, Field(gt=0)] = 1024
    block_paths: dict[Annotated[int, Field(ge=0)], Path] = Field(default_factory=dict)
    root: Path
    dtype : DType | None = None

    key_to_idx : dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    idx_to_key : dict[Annotated[int, Field(ge=0)], str] = Field(default_factory=dict)

    @classmethod
    def generate(cls,
                 dim: int,
                 root: Path | str,
                 size: int = 512, 
                 max_blocks: int | None = 64, 
                 cyclic_writes: bool = True,
                 append_only: bool = True,
                 dtype: DType | None = None,
                 ):
        
        return cls( 
                    dim = dim,
                    root = Path(root).resolve(),
                    size = size, 
                    max_blocks = max_blocks, 
                    cyclic_writes = cyclic_writes,
                    append_only = append_only,
                    dtype = dtype,
                    )

    def vipe(self) -> None:
        '''Deletes all the data block keeping the directory and metadata intack.
        Sets metadats parameters to those corresponding to epty memory'''
        self.header = 0
        self.block_paths = dict()
        self.key_to_idx = dict()
        self.idx_to_key = dict()

class ShardedCache():
    '''Manages writes and reads from sharded cache'''
    def __init__(   
                    self, 
                    root_dir : str | Path, 
                    dim: int, 
                    size: int = 512, 
                    max_blocks: int | None = None, 
                    cyclic : bool = False,
                    append_only : bool = True,
                    dtype : DType | None = None,
                    exists_ok : bool = False,
                ) -> None:
        
        path = Path(root_dir).resolve()
        if path.suffix != '':
            raise ValueError(f"{path} is not a directory")
        if path.exists() and not exists_ok:
            raise FileExistsError(f"Directory at {path} already exists")

        self.__meta_path = path/'metadata.json'
        self.__cache_path = path

        self.__meta = ShardedCacheConfig.generate(   
                                            size=size,
                                            max_blocks=max_blocks,
                                            cyclic_writes=cyclic,
                                            append_only = append_only,
                                            dim=dim,
                                            root=path,
                                            dtype=dtype,
                                        )

    @property
    def meta_path(self) -> Path:
        return self.__meta_path

    @property
    def root(self) -> Path:
        return self.__cache_path

    @classmethod
    def load(cls, path : str | Path):
        metadata_path = Path(path).resolve()
        meta = ShardedCacheConfig.load(metadata_path)
        cache = ShardedCache.load_from_meta(meta)
        cache.__meta_path = path

        return cache

    @classmethod
    def load_from_meta(cls, meta: ShardedCacheConfig):
        cache = cls(root_dir = meta.root.resolve(),
                    dim = meta.dim,
                    size = meta.size,
                    max_blocks = meta.max_blocks,
                    cyclic = meta.cyclic_writes,
                    append_only = meta.append_only,
                    exists_ok = True,
                    )
        cache.__meta = meta

        return cache

    def save(self) -> None:
        self.__meta.save(self.meta_path)

    def __block_path_by_id(self, block_idx : int) -> Path:
        return self.root/f"cahe_matrix_{block_idx}.npy"

    @staticmethod
    def __deduplicate_keys_features(keys: list[str], features: Array) -> tuple[list[str],Array]:
        if not features.ndim == 2:
            raise ValueError(f"Expected a 2d array, got dim={features.ndim} instead")
        if len(keys)!=features.shape[0]:
            raise ValueError("Number of keys must equal the number of features")
        
        normalized_keys = dict()
        for i, a in enumerate(keys):
            normalized_keys[a] = i
        normalized_keys = normalized_keys.items()
        
        keys = [x for x,_ in normalized_keys]
        values = features[[i for _,i in normalized_keys]]

        return keys, values

    def __update_last_write(self, update_keys: list[str], features: Array | list[Array], first_round=True):
        '''Checks for where the old values of feature with given key is stored and 
        updates it to a new value'''
        if first_round:
            update_values = _normalize_features_input(features)
            num = update_values.shape[0]
            dim = update_values.shape[1]

            if len(update_keys)!=num:
                raise ValueError("Number of keys must equal the number of features")
            if dim != self.__meta.dim:
                raise ValueError("Wrong dimensions")

            keys, values = self.__deduplicate_keys_features(update_keys, update_values)
        
            rewrite_idx = []
            for id, key in enumerate(keys):
                val = self.__meta.key_to_idx.get(key, None)
                if val is None:
                    raise KeyError(f"Provided key {key} is not in cache. Can only update cache for stored features")
                rewrite_idx.append((id,val))
            rewrite_idx.sort(key = lambda x : x[1])
        else:
            if isinstance(features, list):
                raise ValueError("Expected an array but got a list instead")
            keys = update_keys
            values = features
            rewrite_idx = [(id, self.__meta.key_to_idx.get(key,0)) for id, key in enumerate(keys)]

        first_id = rewrite_idx[0][1]
        first_block_idx = first_id // self.__meta.size
        first_feature_idx = first_id % self.__meta.size
        space_left = self.__meta.size-first_feature_idx-1

        write_to_first_block = [(x,y % self.__meta.size) for x,y in rewrite_idx if y-first_id<=space_left]

        path = self.__block_path_by_id(first_block_idx).resolve()
        if not path.exists():
            raise FileNotFoundError(f"The data block with id {first_block_idx} doesn't exist")

        matrix = open_memmap(path, 
                             mode="r+", 
                             shape=(self.__meta.size, self.__meta.dim),
                             dtype=self.__meta.dtype)

        matrix_idx = [x for _,x in write_to_first_block]
        features_idx = [x for x,_ in write_to_first_block]

        matrix[matrix_idx] = values[features_idx]

        residual_idx = [id for id in range(len(keys)) if not id in features_idx]
        
        if not residual_idx:
            return
        
        residual_features = values[residual_idx]
        residual_keys = [keys[i] for i in residual_idx]

        self.__update_last_write(residual_keys, residual_features, False)

    def __update_key_map(self, keys: list[str]) -> None:
        for i, key in enumerate(keys):
            write_idx = self.__meta.header+i

            old_key = self.__meta.idx_to_key.get(write_idx, None)
            old_idx = self.__meta.key_to_idx.get(old_key, None) if old_key is not None else None
            self.__meta.key_to_idx[key] = write_idx

            if (
                old_key is not None 
                and old_key != key 
                and old_idx is not None 
                and write_idx == old_idx
                ):  
                del self.__meta.key_to_idx[old_key]

            self.__meta.idx_to_key[write_idx] = key

    def write(self, write_keys: list[str], features: Array | list[Array], dedup=False) -> None:
        '''Writes features to correspondong key values in cache. Creates new block only when
        the last block is full'''
        values = _normalize_features_input(features)
        num = values.shape[0]
        dim = values.shape[1]

        if len(write_keys)!=num:
            raise ValueError("Number of keys must equal the number of features")
        if dim != self.__meta.dim:
            raise ValueError("Wrong dimensions")

        idx = self.__meta.header % self.__meta.size
        block_idx = self.__meta.header//self.__meta.size
        space_left = self.__meta.size-idx

        if self.__meta.max_blocks is not None and block_idx >= self.__meta.max_blocks:
            if not self.__meta.cyclic_writes:
                raise ValueError(f"Reached maximum number of blocks: {self.__meta.max_blocks}")
            self.__meta.header = 0
            block_idx = 0

        keys, values = (write_keys.copy(),values) if not dedup and self.__meta.append_only else self.__deduplicate_keys_features(write_keys, values)
        num = len(keys)

        if not self.__meta.append_only:
            update_keys = []
            update_idx = []

            for id, key in enumerate(keys):
                val = self.__meta.key_to_idx.get(key, None)
                if val is not None:
                    update_keys.append(key)
                    update_idx.append(id)

            if update_idx:
                update_features = values[update_idx]
                self.__update_last_write(update_keys, update_features)

            residual = [i for i in range(len(keys)) if not i in update_idx]
            if not residual:
                return
            keys = [keys[i] for i in residual]
            values = values[residual]
            num = len(residual)

        path = self.__block_path_by_id(block_idx).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+"
        if not path.exists():
            self.__meta.block_paths[block_idx] = path
            mode = "w+"
        
        matrix = open_memmap(path, 
                             mode=mode, 
                             shape=(self.__meta.size, self.__meta.dim),
                             dtype=self.__meta.dtype,
                             )

        step = min(num, space_left)
        matrix[idx:idx+step] = values[0:step]
        del matrix

        self.__update_key_map(keys[:step])
        self.__meta.header += step

        fits = (num - space_left) <= 0
        if not fits:
            self.write(write_keys=keys[step:], features=values[step:])

        self.save()

    def delete(self, delete_keys: list[str]|str) -> None:
        '''Deletes values with given keys from cache. They become unavailable
        on reads'''
        if isinstance(delete_keys, str):
            keys = [delete_keys]
        else:
            keys = delete_keys

        for key in keys:
            val = self.__meta.key_to_idx.get(key, None)
            if val is None:
                raise KeyError(f"No feature for key {key} is stored")
            del self.__meta.key_to_idx[key]

    def retrieve(self, keys: list[str]) -> list[Any]:
        '''Returns features with provided keys from cache. If fails to get a feature,
        in case it is not stored, returns None'''
        size = self.__meta.size
        triplet_ids = []
        missed_at = []

        for i, key in enumerate(keys):
            id = self.__meta.key_to_idx.get(key, None)
            if id is not None:
                triplet_ids.append((i, id%size, id//size))
            else:
                missed_at.append(i)
        if not triplet_ids:
            return [None for _ in keys] 
        triplet_ids.sort(key = lambda x : x[-1])

        sublist = []
        by_block_id = dict()
        block = triplet_ids[0][-1]
        for t in triplet_ids:
            if t[-1] == block:
                sublist.append(t)
            else:
                by_block_id[block] = sublist
                block = t[-1]
                sublist = [t]
        by_block_id[block] = sublist

        retrieved = [(i, None) for i in missed_at]

        for id, group in by_block_id.items():
            block_path = self.__block_path_by_id(id).resolve()
            matrix = open_memmap(block_path, mode="r+", dtype=self.__meta.dtype)

            index = [g[1] for g in group]
            positions = [g[0] for g in group]

            ordered_out = [(p,f) for p,f in zip(positions, list(matrix[index]))]
            retrieved.extend(ordered_out)

            del matrix

        retrieved.sort(key = lambda x : x[0])
        res = [r[-1] for r in retrieved]

        return res

    def clean(self) -> None:
        '''Deletes all the files associated with the caching session'''
        blocks = list(self.__meta.block_paths.values())
        for block in blocks:
            block.unlink(True)
        self.meta_path.unlink(True)
        if self.root.exists():
            self.root.rmdir()

    def vipe(self) -> None:
        '''Deletes all the data files associated with the caching session. Meta data is set to init state'''
        blocks = list(self.__meta.block_paths.values())
        for block in blocks:
            block.unlink(True)
        self.__meta.vipe()
        self.save()


class IndexedCacheConfig(BaseJSONConfig):
    '''Metadata object keep all the essential information about
    indexed caching system.
    
    Indexed cache stores all features in one matrix prepared in advance.
    Features are accessed by their id in this matrix.
    
    features_path: path to .npy file of matrix of features
    num_features: Number of features stored in memory
    feature_dim: dimension of features stored
    dtype: type in which data is stored
    root: root path to the directory where metadata and fature matrix are stored'''

    cache_type : str = 'monolith'
    features_path : Path
    num_features : Annotated[int, Field(gt=0)]
    feature_dim : Annotated[int, Field(gt=0)]
    dtype : DType | None = None
    root: Path

    computed_id : list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)

    def vipe(self) -> None:
        self.computed_id = []

    @classmethod
    def generate(cls, root: Path | str, num_features: int, dim: int, dtype: DType|None = None):
        return cls(root = Path(root).resolve(), 
                   num_features=num_features, 
                   feature_dim = dim,
                   dtype=dtype,
                   features_path = Path(root).resolve()/'features.npy')

    
class IndexedCache():
    '''Writes and reads data from indexed cache storage'''
    def __init__(self, root_dir: str | Path, num_features: int, dim: int, dtype: DType | None = None, exists_ok: bool = False) -> None:
        full_path = Path(root_dir).resolve()
        if full_path.exists() and not exists_ok:
            raise FileExistsError(f"Cache directory {root_dir} already exists")
        if full_path.suffix != '':
            raise ValueError(f"Path {root_dir} must be a directory")
        
        self.__root = full_path
        self.__meta_path = full_path/'metadata.json'
        self.__meta = IndexedCacheConfig.generate(    
                                        num_features = num_features,
                                        dim = dim,
                                        dtype=dtype,
                                        root=full_path,
                                        )
    def save(self) -> None:
        self.__meta.save(self.__meta_path)

    @property
    def meta_path(self) -> Path:
        return self.__meta_path

    @property
    def root(self) -> Path:
        return self.__root

    @property
    def features_path(self) -> Path:
        return self.__meta.features_path

    @classmethod
    def load(cls, path : str | Path):
        meta_path = Path(path).resolve()
        meta = IndexedCacheConfig.load(meta_path)

        return IndexedCache.load_from_meta(meta)

    @classmethod
    def load_from_meta(cls, meta: IndexedCacheConfig):
        cache = cls(root_dir = meta.root.resolve(),
                    num_features = meta.num_features,
                    dim = meta.feature_dim,
                    dtype = meta.dtype,
                    exists_ok = True,
                    )
        cache.__meta = meta

        return cache

    @staticmethod
    def _np_dtype_from_str(dtype : DType):
        if dtype == 'float32':
            return np.float32
        if dtype == 'float64':
            return np.float64
        if dtype == 'float16':
            return np.float16
        if dtype == 'int8':
            return np.int8
        if dtype == 'int16':
            return np.int16
        if dtype == 'int32':
            return np.int32
        if dtype == 'int64':
            return np.int64

    def __prepare(self) -> None:
        self.__ensure_feature_matrix()

    def __ensure_feature_matrix(self) -> None:
        if self.features_path.exists():
            try:
                matrix = np.lib.format.open_memmap(
                self.features_path,
                mode="r",
                )
            except ValueError:
                raise ValueError(f"The cache matrix at {self.features_path} is broken")
            except OSError:
                raise OSError(f"Unexpected cache file (or file doesn't exist) at {self.features_path}")
            if self.__meta.num_features != int(matrix.shape[0]):
                raise ValueError("The existing feature matrix doesnt correspond to the provided records list."
                                "The number of features is different from the provided number of records.")
            if self.__meta.feature_dim != int(matrix.shape[1]):
                raise ValueError("The existing feature matrix doesnt correspond to the provided records list."
                                "The feature length doesn't correspond to the number of the given model's parameters.")
            del matrix
            return

        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            self.features_path,
            mode="w+",
            dtype=IndexedCache._np_dtype_from_str(self.__meta.dtype) if self.__meta.dtype is not None else None,
            shape=(self.__meta.num_features, self.__meta.feature_dim),
        )
        matrix[:] = np.nan
        del matrix

    def __open_cache_matrix(self, mode="r+") -> Any:
        matrix_path = self.features_path

        if not matrix_path.exists():
            raise FileNotFoundError(f"Didn't find the cache matrix at {matrix_path}")
        if not (matrix_path.is_file() and matrix_path.suffix==".npy"):
            raise ValueError(f"Unexpected cache matrix format at {matrix_path}")

        cached = np.lib.format.open_memmap(
            matrix_path,
            mode=mode
        )

        return cached

    def write(self, idx: list[int], features: Array | list[Array]) -> None:
        '''Writes features to the given indices in cahe matrix'''
        values = _normalize_features_input(features)
        num = values.shape[0]
        dim = values.shape[1]

        if len(idx)!=num:
            raise ValueError("Number of keys must equal the number of features")
        if dim != self.__meta.feature_dim:
            raise ValueError(f"Wrong dimensions. Got feature dimension {dim} but expected {self.__meta.feature_dim}")
        if any([id >= self.__meta.num_features or id < 0 for id in idx]):
            raise ValueError(f"Indices out of bounds. Indices must be non-negative integers smaller then the number of features.")

        self.__prepare()
        matrix = self.__open_cache_matrix("r+")
        matrix[idx] = values
        del matrix

        self.__meta.computed_id.extend(idx)
        self.__meta.computed_id = sorted(list(set(self.__meta.computed_id)))
        self.save()

    def retrieve(self, idx : list[int]) -> list[Any]:
        '''Returns features stored at given inices from feature matrix.
        Returns None is the feature is not stored in memory.'''
        if any([id >= self.__meta.num_features or id < 0 for id in idx]):
            raise ValueError(f"Indices out of bounds. Indices must be non-negative integers smaller then the number of features.")

        missed_ids = []
        hit_ids = []
        computed = set(self.__meta.computed_id)

        for i, id in enumerate(idx):
            miss = not id in computed
            if miss:
                missed_ids.append(i)
            else:
                hit_ids.append((i,id))

        retrieved = [(i, None) for i in missed_ids]

        index = [id[1] for id in hit_ids]
        positions = [id[0] for id in hit_ids]

        matrix = self.__open_cache_matrix("r")
        ordered_computed = [(pos, val) for pos, val in zip(positions, list(matrix[index]))]
        del matrix

        retrieved.extend(ordered_computed)
        retrieved.sort(key = lambda x : x[0])
        res = [r[1] for r in retrieved]

        return res

    def delete(self, delete_idx: list[int] | int) -> None:
        '''Deletes features under given indices from cache.'''
        if isinstance(delete_idx, int):
            idx = [delete_idx]
        else:
            idx = delete_idx

        computed = set(self.__meta.computed_id)
        for id in idx:
            if not id in computed:
                raise KeyError(f"No feature with id {id} is stored")
            computed.remove(id)
        self.__meta.computed_id = sorted(list(computed))
        self.save()

    def clean(self):
        '''Deletes all the files associated with the caching session'''
        self.features_path.unlink(True)
        self.meta_path.unlink(True)
        if self.root.exists():
            self.root.rmdir()

    def vipe(self):
        '''Deletes all the data files. Meta data is not deleted but set to init state'''
        self.features_path.unlink(True)
        self.__meta.vipe()
        self.save()

def representor_from_args(representor_name: RepresentorName, representor_args: dict[str, Any], user_args: dict[str, Any]={}) -> FeatureRepresentor:
        args = representor_args.copy()

        user_keys = {key for key,_ in user_args.items()}
        args = _deep_merge(args, user_args)
        
        representor = resolve_representor(
                                            representor_name,
                                            **args,
                                        )
        if not user_keys.issubset(representor.on_read_args()):
            raise ValueError(f"Can't assign new value to {user_keys.difference(representor.on_read_args())}")

        return representor

def safe_args_from_extractor(representor_name: RepresentorName, extractor: FeatureExtractor) -> dict[str, Any]:
        COMPATIBLE_TRANSFORMS = {   
                                    "identity": {'identity', "multitransform", "float32", "float64", "sign_transform", "thresholded_sign_transform"},
                                    "sign": {'identity', "multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                                    "thresholded_sign": {'identity', "multitransform", "float32", "float64","sign_int8","sign_int16","sign_int32","sign_int64"},
                                }
        
        if not representor_name in COMPATIBLE_TRANSFORMS[extractor.transform_name]:
            raise ValueError(f"Storring type '{representor_name}' and transformation '{extractor.transform_name}' are not compatible")

        args = dict()
        if representor_name == 'multitransform':
            args['transform_name'] = extractor.transform_name
            args = _deep_merge(args, extractor.transform_params)

        return args


class IndexedRepresentorCache(IndexedCacheConfig):
    '''Combines indexed cache with representor. Applies
    representor transformation on reads and writes automatically'''
    representor_name: RepresentorName
    representor_args: dict[str, Any]

    @classmethod
    def get(cls,
            root: Path|str,
            input_dim: int,
            feature_num: int,
            dtype: DType|None = None,
            representor_name: RepresentorName = "float32",
            representor_args: dict[str, Any] = {},
            ):

        args = representor_args.copy()
        args['dim'] = input_dim

        return cls(  
                    root = Path(root).resolve(),
                    num_features=feature_num,
                    feature_dim=resolve_representor(representor_name, dim=input_dim, **representor_args).output_dim,
                    dtype=dtype,
                    features_path = Path(root).resolve()/'features.npy',
                    representor_name = representor_name,
                    representor_args = args,
                )

    def __resolve_representor(self, user_args: dict[str,Any]={}) -> FeatureRepresentor:
        return representor_from_args(self.representor_name, self.representor_args, user_args)

    def __get_cache(self):
        cache = IndexedCache.load_from_meta(self)
        return cache
     
    def write(self, idx: list[int], features: Array | list[Array]) -> None:
        values = _normalize_features_input(features)
        representor = resolve_representor(name = self.representor_name, **self.representor_args)
        write_to_memory = representor.forward(values)
        cache = self.__get_cache()
        cache.write(idx, write_to_memory)

    def retrieve(self, idx: list[int], representor_args: dict[str, Any] = {}):
        representor = self.__resolve_representor(representor_args)
        cache = self.__get_cache()
        retrieved = cache.retrieve(idx)
        result = [representor.inverse(r.reshape(1,-1)).reshape(-1) if r is not None else None for r in retrieved]
        
        return result

    def clean(self):
        self.__get_cache().clean()

    def vipe(self):
        self.__get_cache().vipe()

    def delete(self, idx : list[int]|int):
        self.__get_cache().delete(idx)

class ShardedRepresentorCache(ShardedCacheConfig):
    '''Combines sharded cache with representor. Applies
        representor transformation on reads and writes automatically'''
    representor_name: RepresentorName
    representor_args: dict[str, Any]

    @classmethod
    def get(cls,
            root: Path|str,
            input_dim: int,
            representor_name: RepresentorName = "float32",
            representor_args: dict[str, Any] = {},
            size: int = 512,
            max_blocks: int | None = 64,
            append_only: bool = True,
            cyclic_writes: bool = True,
            ):

        args = representor_args.copy()
        args['dim'] = input_dim

        return cls(  
                    root = Path(root).resolve(),
                    dim = resolve_representor(representor_name, **args).output_dim,
                    cyclic_writes = cyclic_writes,
                    append_only = append_only,
                    size = size,
                    max_blocks = max_blocks,
                    representor_name = representor_name,
                    representor_args = args,
                )

    def __resolve_representor(self, user_args: dict[str,Any]={}) -> FeatureRepresentor:
            return representor_from_args(self.representor_name, self.representor_args, user_args)
    
    def __get_cache(self):
        cache = ShardedCache.load_from_meta(self)
        return cache
    
    def write(self, keys: list[str], features: Array | list[Array]) -> None:
        values = _normalize_features_input(features)
        representor = resolve_representor(self.representor_name, **self.representor_args)
        write_to_memory = representor.forward(values)
        cache = self.__get_cache()
        cache.write(keys, write_to_memory)

    def retrieve(self, keys: list[str], representor_args: dict[str, Any]={}):
        representor = self.__resolve_representor(representor_args)
        cache = self.__get_cache()
        retrieved = cache.retrieve(keys)

        result = [representor.inverse(r.reshape(1,-1)).reshape(-1) if r is not None else None for r in retrieved]
        
        return result

    def clean(self):
        self.__get_cache().clean()
    
    def delete(self, keys : list[str]|str):
        self.__get_cache().delete(keys)

    def vipe(self):
        self.__get_cache().vipe()

class ShardedFeatureFetch(ShardedRepresentorCache):
    '''Combines represented sharded cache with features extractor.
        Uses feature extractor when a reteived feature is a miss to compute and save it.
        Fetch function ensure that the feature is always returnded bu computing in when needed.'''
    def __safe_args_from_extractor(self, extractor: FeatureExtractor) -> dict[str,Any]:
        return safe_args_from_extractor(self.representor_name, extractor)
        
    @staticmethod
    def __record_hash(record: PairRecord) -> str:
        encoded = json.dumps(
                            {
                                "prompt": record.prompt,
                                "completion": record.completion,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
        hash = hashlib.sha256(encoded).hexdigest()
        return hash

    def fetch(self, extractor: FeatureExtractor, records: list[PairRecord]):
        '''Retrieves featuers associated with provided records from sharded cache.
            Automatically resolves multitransform arguments from extractor object
            And raises an error is feature extractor is not compatible with
            specified representor.
            
            If retrieved feature is a miss, feature extractor computes and stores it in cache.'''
        hashed = [self.__record_hash(r) for r in records]
        retrieved = self.retrieve(hashed, self.__safe_args_from_extractor(extractor))

        computed_features = []
        computed_hashes = []
        result = []

        is_raw_features = self.representor_name == 'multitransform'

        for r,f in zip(records,retrieved):
            if f is None:
                extracted = extractor.extract_feature(r) if not is_raw_features else extractor.raw_feature(r)
                computed_features.append(extracted)
                computed_hashes.append(self.__record_hash(r))
                result.append(extracted)
            else:
                result.append(f)

        if computed_features:
            self.write(computed_hashes, computed_features)

        return result


class IndexedFeatureFetch(IndexedRepresentorCache):
    '''Combines represented indexed cache with features extractor.
            Uses feature extractor when a reteived feature is a miss to compute and save it.
            Fetch function ensure that the feature is always returnded bu computing in when needed.'''
    def __safe_args_from_extractor(self, extractor: FeatureExtractor) -> dict[str,Any]:
        return safe_args_from_extractor(self.representor_name, extractor)
        
    def fetch(self, extractor: FeatureExtractor, idx: list[int], records: list[PairRecord]):
        ''' Retrieves featuers associated with provided records from indexed cache.
            Automatically resolves multitransform arguments from extractor object
            And raises an error is feature extractor is not compatible with
            specified representor.
            
            If retrieved feature is a miss, feature extractor computes and stores it in cache.'''
        retrieved = self.retrieve(idx, self.__safe_args_from_extractor(extractor))

        computed_features = []
        computed_idx = []
        result = []

        is_raw_features = self.representor_name == 'multitransform'

        for i,r,f in zip(idx,records,retrieved):
            if f is None:
                extracted = extractor.extract_feature(r) if not is_raw_features else extractor.raw_feature(r)
                computed_features.append(extracted)
                computed_idx.append(i)
                result.append(extracted)
            else:
                result.append(f)

        if computed_features:
            self.write(computed_idx, computed_features)

        return result