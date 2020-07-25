# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from fcos_core import _C
from fcos_core import custom_nms

nms = _C.nms
ml_nms = _C.ml_nms
hand_nms = custom_nms.hand_nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
