import json
import os
import tensorflow as tf 
import numpy as np 

def load_json_file(file_path: str):
    """
    Load json file by file path
    :param file_path: path to json file
    :return: json object
    """
    assert os.path.exists(file_path), "{} is not existed".format(file_path)
    with open(file_path) as f:
        json_file = json.load(f)
    return json_file

def predict(data: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """
    Perform prediction on specific model
    :param data: data for prediction
    :param model: DL model for prediction
    :return:
    """
    results = model.predict(data.reshape(-1, 28, 28, 1))
    # clear up session
    tf.keras.backend.clear_session()
    return results

def sort_by_confidence_then_filter_top_n(prediction_results: np.ndarray, top_n: int) -> np.ndarray:
    """
    - sort the prediction by confidences score
    - flip it from highest to lowest
    - filter out top N predictions
    - stack them back together
    :param prediction_results:
    :param top_n:
    :return:
    """
    result_argsort = np.flip(prediction_results.argsort()[:, -top_n:], 1)
    result_sort = np.flip(np.sort(prediction_results)[:, -top_n:], 1)
    result_stack = np.dstack((result_argsort, result_sort))
    return result_stack
    