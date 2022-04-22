import tensorflow as tf
import numpy as np
import cv2


class Movenet():
    """
        This class loads the model and infers the keypoint coordinates
    """

    def __init__(self) -> None:
        self._model = self._load_model()
        self._img_size = 192
        self._keypoint_thresh = .4

    def _load_model(self) -> tf.lite.Interpreter:
        # Initialize the TFLite interpreter
        return tf.lite.Interpreter(
            model_path="model/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")

    def infer(self, image) -> np.array:
        """ Returns array of keypoints [y, x, score] """
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_with_pad(image, self._img_size, self._img_size)
        self._model.allocate_tensors()
        input_image = tf.cast(image, dtype=tf.uint8)
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
        for kp in keypoints_with_scores:
            if kp[-1] >= threshold:
                image = cv2.circle(
                    image, (int(kp[1] * image.shape[1]), int(kp[0] * image.shape[0])), 3, (0, 0, 255), 3)
        return image
