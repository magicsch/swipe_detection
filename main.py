import cv2
import time
from swipe_classifier import SwipeClassifier
from utils import *
import traceback


"""
    This script is meant to show how the Swipe Classifier is used
    And also for debugging
"""


def main():
    classifier = SwipeClassifier()
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("No video feed")
            break
        elif success:
            # size of debug image
            # dont't need to upscale without debug
            frame = cv2.resize(frame, (960, 960))

            out = classifier.classify_swipe(
                frame, debug_img=False)
            if out is not Swipe.none:
                print('-----------')
                print(out.name)
                cv2.putText(frame, out.name, (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 12)

            time.sleep(.15)
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
        pass
