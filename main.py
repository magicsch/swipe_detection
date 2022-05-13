import cv2
import time
from movenet import Movenet
from swipe_classifier import SwipeClassifier
from utils import *
import traceback


"""
    This script is meant to show how the Swipe Classifier is used
    And also for debugging
"""

fps = 0


def print_fps(fps):
    print('----------------------')
    print(f'Average fps {fps}')
    print('----------------------')


def main():
    classifier = SwipeClassifier()
    frame_count = 0
    start_time = time.time()
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("No video feed")
            break
        elif success:
            global fps
            # size of debug image
            frame = cv2.resize(frame, (960, 960))

            out = classifier.classify_swipe(
                frame, fps=fps, debug_img=False)
            if out is not Swipe.none:
                print('-----------')
                print(out.name)
                cv2.putText(frame, out.name, (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 12)

            time.sleep(.15)
            cv2.imshow("DEBUG", frame)
            frame_count += 1
            fps = frame_count//(time.time()-start_time)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print_fps(fps)
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_fps(fps)
        pass
    except BaseException as e:
        print(traceback.format_exc())
        pass
