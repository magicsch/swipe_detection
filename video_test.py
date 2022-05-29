"""
    This script is meant to show how the Swipe Classifier is used
"""

from cv2 import cv2
from swipe_classifier import SwipeClassifier
from utils import *
import traceback
from mp_utils import RED
import time


def main():
    classifier = SwipeClassifier()
    cap = cv2.VideoCapture('test.mp4')

    while True:
        success, frame = cap.read()
        if not success:
            print("No video feed")
            break
        elif success:
            # dont't need to upscale without debug
            # img_sz = (960,) * 2
            # frame = cv2.resize(frame, img_sz)

            out, frame = classifier.classify_swipe(
                frame, debug_img=True)
            if out:
                print('-----------')
                print(out.name)
                cv2.putText(frame, out.name, (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, RED, 12)

            # time.sleep(.15)
            cv2.imshow("DEBUG", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        print(traceback.format_exc())
