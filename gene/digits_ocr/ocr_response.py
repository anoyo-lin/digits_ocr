from abc import ABC

import numpy as np

from gene.digits_ocr.utils import sort_by_confidence_then_filter_top_n

# OCRResponse has 3 attributes
# - prediction_results
# - confidence_threshold
# - top_N
class OCRResponse(ABC):
    def __init__(self, prediction_results, confidence_threshold: float = 0.6, top_n: int = 3):
        self.prediction_results = prediction_results
        self.confidence_threshold = confidence_threshold
        self.top_n = top_n

# MNIST text annotations has 2 attributes
# - prediction
# - confidence
class MNISTTextAnnotations(object):
    def __init__(self, prediction_results, confidence_threshold: float = 0.6):
        self.prediction = str(np.argmax(prediction_results))
        self.confidence = np.max(prediction_results)

class RowNumbersTextAnnotations(object):
    def __init__(self, prediction_results, confidence_threshold: float = 0.6):
        result_position_below_threshold = np.argwhere(
            np.max(prediction_results, axis=1) < confidence_threshold
        ).flatten()
        prediction = np.argmax(prediction_results, axis=1).astype(str)
        description = prediction.copy()
        # mask the char by '*' if below threshold
        if len(result_position_below_threshold) != 0:
            for arg_pos in result_position_below_threshold:
                description[arg_pos] = '*'
        self.description = ''.join(description)
        self.prediction = ''.join(prediction)
        self.confidence = np.mean(np.flip(np.sort(prediction_results)[:, -1:], 1))

# FullTextAnnotations
# - stringify all result in result_stack
# - dict { str(key): float(value)}
class FullTextAnnotations(object):
    def __init__(self, order, result):
        self.order = order
        self.top_n = {str(int(arg)): float(round(val, 5)) for arg, val in result}

# MNIST OCR Response
# will extend OCRResponse and has extra textAnnotations and fullTextAnnotations
# - textAnnotations.prediction
# - textAnnotations.confidence
# - fullTextAnnotations
class MNISTOCRResponse(OCRResponse):
    def __init__(self, prediction_results, confidence_threshold: float = 0.6, top_n: int = 3):
        super(MNISTOCRResponse, self).__init__(prediction_results, confidence_threshold, top_n)
        self.textAnnotations = MNISTTextAnnotations(prediction_results, confidence_threshold)
        # top N prediction
        # reverse the results and filter out Top N prediction sort by confidence
        result_stack = sort_by_confidence_then_filter_top_n(prediction_results, top_n)
        self.fullTextAnnotations = [FullTextAnnotations(idx, result) for idx, result in enumerate(result_stack)]

class RowNumbersOCRResponse(OCRResponse):
    def __init__(self, prediction_results, confidence_threshold: float = 0.6, top_n: int = 3):
        super(RowNumbersOCRResponse, self).__init__(prediction_results, confidence_threshold, top_n)
        self.textAnnotations =  RowNumbersTextAnnotations(prediction_results, confidence_threshold)
        result_stack = sort_by_confidence_then_filter_top_n(prediction_results, top_n)

        self.fullTextAnnotations = [FullTextAnnotations(idx, result) for idx, result in enumerate(result_stack)]
