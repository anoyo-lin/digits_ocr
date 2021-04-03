"""
Config file for ocr part of the project
"""

from easydict import EasyDict as edict
from django.conf import settings

import os

__C = edict()

cfg = __C

__C.MNIST = edict()
# PATH to MNIST model to be used 
__C.MNIST.MODEL_TO_USE = os.path.join(str(settings.BASE_DIR), 'assets', 'ocr_model', 'mnist', 'mnist_model.h5')
# Model input image size
__C.MNIST.INPUT_SIZE = (28, 28)

__C.PATH = edict()

# Config for API
__C.API = edict()
# config for mnist api
__C.API.MNIST = edict()
__C.API.MNIST.CONFIDENCE_THRESHOLD = edict()
__C.API.MNIST.CONFIDENCE_THRESHOLD.MIN = 0.0
__C.API.MNIST.CONFIDENCE_THRESHOLD.MAX = 1.0
__C.API.MNIST.CONFIDENCE_THRESHOLD.DEFAULT = 0.6
__C.API.MNIST.TOP_N = edict()
__C.API.MNIST.TOP_N.MIN = 0
__C.API.MNIST.TOP_N.MAX = 9
__C.API.MNIST.TOP_N.DEFAULT = 3