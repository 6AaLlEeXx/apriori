from __future__ import annotations
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from typing import Any
import hashlib
from transformations.thresholded_sign import transform as threshold_transform

def formatted_sign_to_int(A: NDArray, bit: int = 8)->NDArray:
    """
        Takes an array of farmatted sign features, where
        the formatting can be [nan, -1, 0, 1] -> [0, 1, 2, 3]
        and stores them as int_bit matrix F. Returnes F.
    """
    SUPPORTED_INT_BITS = {8, 16, 32, 64}

    INT = {8: np.int8, 16: np.int16, 32: np.int32, 
            64: np.int64}

    if not bit in SUPPORTED_INT_BITS:
        raise ValueError(f"The bits resolution must one of: {SUPPORTED_INT_BITS}, but got {bit}")

    dim = int(A.shape[1])
  
    if dim*2 % bit != 0:
        raise ValueError(f"Bits per row b/r={dim*2} must be divisible by {bit}")

    int_type = INT.get(bit, None)
    if int_type is None:
        raise ValueError(f"Numpy int{bit} type is not supported. Supported types are: {INT.keys()}")
    A = np.asarray(A, dtype=np.int8)
    A = A.reshape(A.shape[0], -1, bit//2)
    B = 4**np.arange(0,4, 1, dtype=int_type)
    Fw = A@B-(2**(bit-1)-1)

    return Fw

def int_to_formatted_sign(Fw: NDArray, bit: int = 8)->NDArray:
    SUPPORTED_INT_BITS = {8, 16, 32, 64}
    
    INT = {8: np.int8, 16: np.int16, 32: np.int32, 
            64: np.int64}

    if not bit in SUPPORTED_INT_BITS:
        raise ValueError(f"The bits resolution must one of: {SUPPORTED_INT_BITS}, but got {bit}")
  
    A_B = Fw+(2**(bit-1)-1)
    base_4 = []
    for row in A_B:
        repr = []
        for x in row:
            string = np.base_repr(x, 4)
            new_string = []
            for i in range(1, len(string)+1):
                new_string.append(int(string[-i]))
            repr.extend(new_string)
        base_4.append(repr)
    return np.asarray(base_4, dtype=np.int8)


class FeatureRepresentor(ABC):

    @abstractmethod
    def fit(self, vector: NDArray)->None:
        pass

    @abstractmethod
    def forward(self, array: NDArray)-> NDArray:
        pass

    @abstractmethod
    def inverse(self, array: NDArray)-> NDArray:
        pass

    @abstractmethod
    def is_nan(self, array: NDArray)-> Any:
        pass

    @property
    @abstractmethod
    def name(self)->str:
        pass

    @property
    def init_name(self)->str:
        return self.name

    @property
    @abstractmethod
    def dtype(self)->Any:
        pass

    @property
    @abstractmethod
    def ready(self)->bool:
        pass

    @property
    @abstractmethod
    def args(self)->dict[str,Any]:
        pass

    @property
    @abstractmethod
    def user_args(self)->dict[str,Any]:
        pass

    @abstractmethod
    def get_key(self)->str:
        pass

class DefaultRepresentor(FeatureRepresentor):
    def __init__(self):
        pass

    def fit(self, vector: NDArray)->None:
        return

    def forward(self, array: NDArray)-> NDArray:
        return array

    def inverse(self, array: NDArray)-> NDArray:
        return array
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)
    
    def get_key(self)->str:
        encoded = f"basic transformation; writes grad features".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self)->str:
        return "float32"

    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def ready(self)->bool:
        return True
    
    @property
    def args(self)->dict[str,Any]:
        if not self.ready:
            raise ValueError("The representor object is not ready")
        return {}
    
    @property
    def user_args(self)->dict[str,Any]:
        return {}
    
class SignIntRepresentor(FeatureRepresentor):
    def __init__(self, bit: int = 8, dim: int | None = None):
        self._dim = dim
        self._bit = bit
        self._entry_bits: int = 2
        self._padding: int | None = None

        if dim is not None:
            if dim > 0:
                grp_size = self._bit // self._entry_bits
                self._padding = int(dim % grp_size)
            else:
                raise ValueError(f"Dimension must be a positive integer, but got {dim}")

    def fit(self, vector: NDArray)->None:
        grp_size = self._bit // self._entry_bits
        resid = int(vector.shape[1] % grp_size)
        self._padding = resid

    def forward(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError(f"Representor '{self.name}' is not ready")
        def _to_format(array: NDArray)->NDArray:
            array += 2
            array = np.nan_to_num(array, nan=0)
            
            if self._padding != 0:
                print(f"SignIntRepresentor: Padding features with {self._padding} zeros")
                array = np.concatenate([array, np.zeros((array.shape[0], self._padding if self._padding is not None else 0))], axis=0)
            return array
        
        formatted = _to_format(array)
        return formatted_sign_to_int(formatted, self._bit)

    def inverse(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError(f"Representor '{self.name}' is not ready")
        if self._padding is None:
            raise ValueError("Padding must be set in order to inverse-transform")
        inv = int_to_formatted_sign(array, self._bit)
        if self._padding > 0:
            inv = inv[:-self._padding]
        
        return inv
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(array==0,1)
    
    def get_key(self)->str:
        if not self.ready:
            raise ValueError(f"Representor '{self.name}' is not ready")
        encoded = f"name={self.name}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self)->str:
        return f"sign_int{self._bit}"
    
    @property
    def dtype(self)->Any:
        return np.dtype(f"int{self._bit}")
    
    @property
    def ready(self)->bool:
        if self._padding is not None:
            if self._padding < 0:
                raise ValueError(f"Encountered negative padding with value {self._padding}")
            return True
        return False
    
    @property
    def args(self)->dict[str,Any]:
        if not self.ready:
            raise ValueError("The representor object is not ready")
        return {"dim": self._dim, "bit": self._bit, "padding": self._padding}
    
    @property
    def user_args(self)->dict[str,Any]:
        return {"bit": self._bit}
    

class SignTransformer(FeatureRepresentor):
    def __init__(self):
        pass

    def fit(self, vector: NDArray)->None:
        return

    def forward(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        return array

    def inverse(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        return np.sign(array).astype(np.float32, copy=False)
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)
    
    def get_key(self)->str:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        encoded = f"basic transformation; writes grad features".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self)->str:
        return f"sign_transform"
    
    @property
    def init_name(self)->str:
        return "multitransform"
    
    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def ready(self)->bool:
        return True
    
    @property
    def args(self)->dict[str,Any]:
        if not self.ready:
            raise ValueError("The representor object is not ready")
        return {}
    
    @property
    def user_args(self)->dict[str,Any]:
        return {}
    

class ThresholdSignTransformer(FeatureRepresentor):
    def __init__(self, threshold: float = 0.):
        if threshold < 0:
            raise ValueError(f"Recieved negative threshold value: threshold={threshold}")
        self._eps: float = threshold

    def fit(self, vector: NDArray)->None:
        return

    def forward(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        return array

    def inverse(self, array: NDArray)-> NDArray:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        return threshold_transform(features=array, 
                                   threshold=self._eps,
                                   )
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)
    
    def get_key(self)->str:
        if not self.ready:
            raise ValueError("The representor object is not fitted")
        encoded = f"basic transformation; writes grad features".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self)->str:
        return f"thresholded_sign_transform"
    
    @property
    def init_name(self)->str:
        return "multitransform"
    
    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def ready(self)->bool:
        return True
    
    @property
    def args(self)->dict[str,Any]:
        if not self.ready:
            raise ValueError("The representor object is not ready")
        return {"threshold": self._eps}
    
    @property
    def user_args(self)->dict[str,Any]:
        return {"threshold": self._eps}
    

class IdentityTransformer(FeatureRepresentor):
    def __init__(self):
        pass

    def fit(self, vector: NDArray)->None:
        return

    def forward(self, array: NDArray)-> NDArray:
        return array

    def inverse(self, array: NDArray)-> NDArray:
        return array
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)
    
    def get_key(self)->str:
        encoded = f"basic transformation; writes grad features".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def name(self)->str:
        return "identity_transform"
    
    @property
    def init_name(self)->str:
        return "multitransform"

    @property
    def dtype(self)->Any:
        return np.float32
    
    @property
    def ready(self)->bool:
        return True
    
    @property
    def args(self)->dict[str,Any]:
        if not self.ready:
            raise ValueError("The representor object is not ready")
        return {}
    
    @property
    def user_args(self)->dict[str,Any]:
        return {}
    

    
def resolve_representor(name: str, **kwargs)-> FeatureRepresentor:
    normal_name =  name.lower().strip()

    if normal_name == "float32":
        return DefaultRepresentor()
    if normal_name == "float64":
        print("Float64 is not supported. Returning float32 instead")
        return DefaultRepresentor()
    if normal_name == "sign_int8":
        return SignIntRepresentor(bit=8, dim = kwargs.get("dim", None))
    if normal_name == "sign_int16":
        return SignIntRepresentor(bit=16, dim = kwargs.get("dim", None))
    if normal_name == "sign_int32":
        return SignIntRepresentor(bit=32, dim = kwargs.get("dim", None))
    if normal_name == "sign_int64":
        return SignIntRepresentor(bit=64, dim = kwargs.get("dim", None))
    if normal_name == "sign_transform":
        return SignTransformer()
    if normal_name == "thresholded_sign_transform":
        return ThresholdSignTransformer(threshold=kwargs.get("threshold", 0.))
    if normal_name == "multitransform":
        transformation_name = kwargs.get("feature_transform", "identity")
        
        if not isinstance(transformation_name, str):
            raise TypeError("Variable 'transformation_name' must be a string")
        if not transformation_name in {"identity", "sign", "thresholded_sign"}:
            raise ValueError(f"Usupported transformation name '{transformation_name}'")
        
        if transformation_name == "identity":
            return IdentityTransformer()
        if transformation_name == "sign":
            return SignTransformer()
        if transformation_name == "thresholded_sign":
            threshold = kwargs.get("threshold", 0.)
            return ThresholdSignTransformer(threshold=threshold)
    raise ValueError(f"Unsupported representor name: {normal_name}")

