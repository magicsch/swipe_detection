import cv2
import time
from movenet import Movenet
from swipe_classifier import SwipeClassifier
from collections import deque
from utils import *
import traceback
import numpy as np


"""
    This script is meant to show how the Swipe Classifier is used
    And also for debugging
"""

fps = 0


def main():
    classifier = SwipeClassifier()
    frame_count = 0
    start_time = time.time()
    model = Movenet()
    cap = cv2.VideoCapture(0)
    seq = deque(maxlen=10)
    dir_seq = deque(maxlen=10)

    while True:
        success, frame = cap.read()
        if not success:
            print("No video feed")
            break
        elif success:
            # size of debug image
            frame = cv2.resize(frame, (960, 960))

            # Here happens the processing of the frame
            # classifier.classify(frame, debug_frame=True)
            # if None no swipe detect else [4,1] array

            out = classifier.classify_swipe(frame)
            if out is not Swipe.none:
                print('-----------')
                print(out)

            cv2.imshow("DEBUG", frame)
            frame_count += 1
            global fps
            fps = frame_count//(time.time()-start_time)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print_fps(fps)
                break


def print_fps(fps):
    print('----------------------')
    print(f'Average fps {fps}')
    print('----------------------')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_fps(fps)
        pass
    except BaseException as e:
        print(traceback.format_exc())
        pass
