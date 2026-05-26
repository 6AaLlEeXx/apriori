from kernel.run import _feature_transform_params, _feature_transform_names
from kernel.config import load_kernel_run_config

config = load_kernel_run_config("/Users/malyshevviktor/Desktop/Apriory/apriori/configs/kernel/dolly_lora_ntk.yaml")
print(_feature_transform_names(config))