from .client import *
from .utils import *
from .types import *

__version__ = "1.8.1"

# ==========================================
# 兼容性修复：针对 Python 3.10+ 环境下 msgpack-rpc-python 的 Packer/Unpacker 参数错误进行拦截修复
# ==========================================
import msgpack
import numpy as np

if not hasattr(msgpack, 'Packer_orig'):
    msgpack.Packer_orig = msgpack.Packer
    class Packer_fixed(msgpack.Packer_orig):
        def __init__(self, *args, **kwargs):
            kwargs.pop('encoding', None)
            
            # 强化 default 函数，支持处理 numpy 类型和 AirSim 对象
            def custom_default(obj):
                if hasattr(obj, 'to_msgpack'):
                    return obj.to_msgpack()
                if isinstance(obj, np.generic):
                    return obj.item()
                return obj

            kwargs['default'] = custom_default
            kwargs.setdefault('use_bin_type', True)
            super().__init__(*args, **kwargs)
    msgpack.Packer = Packer_fixed

if not hasattr(msgpack, 'Unpacker_orig'):
    msgpack.Unpacker_orig = msgpack.Unpacker
    class Unpacker_fixed(msgpack.Unpacker_orig):
        def __init__(self, *args, **kwargs):
            kwargs.pop('encoding', None)
            kwargs.setdefault('raw', False)
            super().__init__(*args, **kwargs)
    msgpack.Unpacker = Unpacker_fixed
