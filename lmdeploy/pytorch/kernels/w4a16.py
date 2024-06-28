# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

w4a16_linear = FunctionDispatcher('w4a16_linear').make_caller()
