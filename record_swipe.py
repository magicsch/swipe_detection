import cv2
import time
from movenet import Movenet
from swipe_classifier import SwipeClassifier
from collections import deque
import matplotlib.pyplot as plt
import argparse
import numpy as np

"""
    Example usage:
    python record_gesture.py --handedness right --gesture right_swipe/short
    Record right hand movement to a file named short.npy
"""

parser = argparse.ArgumentParser()
parser.add_argument('--gesture', type=str,
                    default='right_swipe', help="The folder of the type of the swipe and the name of the .npy file")
# parser.add_argument('--handedness', type=str,
#                     default='right', help="'right' for right hand, 'left' for left")
args = parser.parse_args()

gesture_path = f'gesture_examples/{args.gesture}/example1.npy'


fps = 0
recording = False


def main():
    classifier = SwipeClassifier()
    frame_count = 0
    start_time = time.time()
    model = Movenet()
    cap = cv2.VideoCapture(0)

    seq = deque(maxlen=50)

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

            keypoints = model.infer(frame)
            frame = model.draw_keypoints(frame, keypoints, .3)
            shoulder_width, shoulder_nose_height = SwipeClassifier.get_normalization_factors(
                keypoints)
            norm_right_wrist, norm_left_wrist = SwipeClassifier.normalize_kps(
                keypoints, shoulder_width, shoulder_nose_height)

            # seq.append(norm_right_wrist)

            # direction = SwipeClassifier.displacement_direction(seq=seq)

            # if direction:
            #     print(direction.name)

            if recording:
                seq.append(norm_right_wrist)
                print('Recording!')

            cv2.namedWindow("DEBUG")
            cv2.setMouseCallback("DEBUG", count_clicks)
            cv2.imshow('DEBUG', frame)

            frame_count += 1
            global fps
            fps = frame_count//(time.time()-start_time)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print_fps(fps)
                break

    with open(gesture_path, 'wb') as f:
        np.save(f, seq)
    plot_seq(seq=seq)


def count_clicks(event, x, y, flags, param):
    """
        Left mouse button starts recording and right mouse button stops it,
        when you exit with 'q' the sequence gets saved.
    """
    global recording
    if event == cv2.EVENT_LBUTTONDOWN and not recording:
        recording = True
    if event == cv2.EVENT_RBUTTONDOWN and recording:
        recording = False


def plot_seq(seq):
    seq = np.array(seq)
    plt.subplot(2, 1, 1)
    plt.plot(seq[:, 1], '-b')
    plt.title('Horizontal displacement')
    plt.ylabel('Normalized value')
    plt.subplot(2, 1, 2)
    plt.plot(seq[:, 0], '-r')
    plt.title('Vertical displacement')
    plt.ylabel('Normalized value')
    plt.show()


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
        print(e)
        pass
