import tensorflow as tf
import numpy as np
from utils import *
import cv2


class Movenet:
    def __init__(self) -> None:
        self._model = self._load_model()
        self._img_size = 192
        self._keypoint_thresh = .4

    def _load_model(self) -> tf.lite.Interpreter:
        # Initialize the TFLite interpreter
        return tf.lite.Interpreter(
            model_path="model/lite-model_movenet_singlepose_lightning_3.tflite")

    def infer(self, image) -> np.array:
        """ Returns array of keypoints [y, x, score] """
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_pad(image, self._img_size, self._img_size)
        self._model.allocate_tensors()
        input_image = tf.cast(image, dtype=tf.float32)
        input_details = self._model.get_input_details()
        output_details = self._model.get_output_details()
        self._model.set_tensor(input_details[0]['index'], input_image.numpy())
        self._model.invoke()
        # Output of model is a [1, 1, 17, 3] numpy array.
        keypoints_with_scores = self._model.get_tensor(
            output_details[0]['index'])
        return keypoints_with_scores[0, 0, :, :]

    @staticmethod
    def draw_keypoints(image, keypoints_with_scores, threshold):
        """ Draws keypoints on image using threshold value """
        size = image.shape[0]
        for kp in keypoints_with_scores:
            if kp[-1] >= threshold:
                image = cv2.circle(
                    image, (int(kp[1] * size), int(kp[0] * size)), 3, RED, 3)
        for ln in KEYPOINT_EDGE_INDS_TO_COLOR.keys():
            start = tuple(
                (np.flip(keypoints_with_scores[ln[0], :2])*size).astype(np.int32))
            end = tuple(
                (np.flip(keypoints_with_scores[ln[1], :2])*size).astype(np.int32))
            image = cv2.line(image, start, end, BLUE, 3)
        return image
