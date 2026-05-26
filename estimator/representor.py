from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from typing import Any


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
    @abstractmethod
    def dtype(self)->Any:
        pass

class DefaultRepresentor(FeatureRepresentor):
    def __init__(self):
        pass

    def forward(self, array: NDArray)-> NDArray:
        return array

    def inverse(self, array: NDArray)-> NDArray:
        return array
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(np.isnan(array),1)

    @property
    def name(self)->str:
        return "float32"

    @property
    def dtype(self)->Any:
        return np.float32
    
class SignIntRepresentor(FeatureRepresentor):
    def __init__(self, bit: int = 8, dim: int | None = None):
        self.bit = bit
        self.entry_bits: int = 2
        self.padding: int | None = None

        if dim is not None:
            if dim > 0:
                grp_size = self.bit // self.entry_bits
                self.padding = int(dim % grp_size)
            else:
                raise ValueError(f"Dimension must be a positive integer, but got {dim}")

    def forward(self, array: NDArray)-> NDArray:
        def _to_format(array: NDArray)->NDArray:
            array += 2
            array = np.nan_to_num(array, nan=0)
            grp_size = self.bit // self.entry_bits
            resid = int(array.shape[1] % grp_size)
            if self.padding is not None and self.padding != resid:
                raise ValueError("Unexpected feature dimension")
            self.padding = resid
            if resid != 0:
                print(f"SignIntRepresentor: Padding features with {resid} zeros")
                array = np.concatenate([array, np.zeros((array.shape[0], resid))], axis=0)
            return array
        
        formatted = _to_format(array)
        return formatted_sign_to_int(formatted, self.bit)

    def inverse(self, array: NDArray)-> NDArray:
        if self.padding is None:
            raise ValueError("Padding must be set in order to inverse-transform")
        inv = int_to_formatted_sign(array, self.bit)
        if self.padding > 0:
            inv = inv[:-self.padding]
        
        return inv
    
    def is_nan(self, array: NDArray)->Any:
        if len(array.shape) != 2:
            raise ValueError(f"Expected 2d array, bu got {array.shape}")
        return np.any(array==0,1)

    @property
    def name(self)->str:
        return f"sign_int{self.bit}"
    
    @property
    def dtype(self)->Any:
        return np.dtype(f"int{self.bit}")
    
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
    raise ValueError(f"Unsupported representor name: {normal_name}")

