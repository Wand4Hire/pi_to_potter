#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cv2
import threading
from threading import Thread
import os
import sys
import traceback
import time
from pathlib import Path

# Try to import optional modules
try:
    import CameraLED
    CAMERA_LED_AVAILABLE = True
except ImportError:
    CAMERA_LED_AVAILABLE = False
    print("Warning: CameraLED module not available")

import music
import ble
from spells import Spells
import server
import ml

# Figure out where your code is...
home_address = str(Path.home())
parser = argparse.ArgumentParser(
    description='Cast some spells!  Recognize wand motions')
parser.add_argument(
    '--train',
    help='Causes wand movement images to be stored for training selection.',
    action="store_true")
parser.add_argument(
    '--setup',
    help='show camera view',
    action="store_true",
    default=True)
parser.add_argument(
    '--home',
    help='The path to your pi_to_potter download.',
    default=home_address)
parser.add_argument(
    '--background_subtract',
    help='User background subtraction',
    action="store_true")
parser.add_argument(
    '--use_ble',
    help='Use the BLE system for spells',
    action="store_true")

args = parser.parse_args()

spells = Spells(args)

print('\n\n\n\n-----------------')
print(f'Perform training? {args.train}')
print(f'Show the original camera view? {args.setup}')
print(f'Use background subtraction? {args.background_subtract}')

print(f'Make sure the files are all at: {home_address}/pi_to_potter/...')

# Initialize camera LED if available
if CAMERA_LED_AVAILABLE:
    try:
        camera = CameraLED.CameraLED()
        camera.off()
    except Exception as e:
        print(f"Warning: Could not initialize camera LED: {e}")

print("Initializing point tracking")

# get the size of the screen
width = 800
height = 480

# Parameters
lk_params = dict(winSize=(25, 25), maxLevel=10, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blur_params = (4, 4)
dilation_params = (5, 5)
movment_threshold = 80

active = False

# Start capturing
cap = cv2.VideoCapture(0)
p0 = None  # Points holder
# Current number of frames without points. (After finding a few.)
frameMissingPoints = 0

time.sleep(2.0)
run_request = True

# Use these to narrow the field of view.
yStart = 0
yEnd = 360
xStart = 0
xEnd = 480

ret, image_data = cap.read()
if not ret or image_data is None:
    print("Error: Could not read from camera")
    sys.exit(1)

frame_holder = image_data
frame_no_background = image_data
frame_holder = frame_holder[yStart:yEnd, xStart:xEnd]
cv2.flip(frame_holder, 1, frame_holder)
frame = None

print("About to start.")


def RemoveBackground():
    global frame_holder, frame_no_background

    fgbg = cv2.createBackgroundSubtractorMOG2()
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        try:
            frameCopy = frame_holder.copy()

            # Subtract Background
            fgmask = fgbg.apply(frameCopy, learningRate=0.001)
            frame_no_background = cv2.bitwise_and(
                frameCopy, frameCopy, mask=fgmask)
            time.sleep(0.03)
        except Exception as e:
            print(f"Error in RemoveBackground: {e}")
            time.sleep(0.1)


def FrameReader():
    global frame_holder
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        try:
            ret, image_data = cap.read()
            if ret and image_data is not None:
                frame = image_data[yStart:yEnd, xStart:xEnd]
                cv2.flip(frame, 1, frame)
                frame_holder = frame
            time.sleep(.03)
        except Exception as e:
            print(f"Error in FrameReader: {e}")
            time.sleep(0.1)


point_aging = []


def trim_points():
    global point_aging
    indexesToDelete = []
    index = 0
    for old_point in point_aging:
        if (time.time() - old_point["when"] > 15):
            old_point["times_seen"] = old_point["times_seen"] - 1
            if old_point["times_seen"] <= 0:
                indexesToDelete.append(index)
                break
        index += 1

    for i in reversed(indexesToDelete):
        del point_aging[i]


def nearPoints(point1, point2, threshold):
    """Check if two points are within threshold distance"""
    dx = point1[0] - point2["x"]
    dy = point1[1] - point2["y"]
    return (dx*dx + dy*dy) < (threshold*threshold)


def GetPoints(image):
    global point_aging
    start_points = cv2.goodFeaturesToTrack(
        image, maxCorners=5, qualityLevel=0.0001, minDistance=5)

    # Clean out aged points.
    trim_points()
    
    if start_points is not None:
        # The commented out code below was causing issues - simplified approach
        pass
    
    return start_points


kernel = np.ones((5, 5), np.uint8)


def ProcessImage():
    global frame_holder, frame_no_background
    try:
        if args.background_subtract:
            frame = frame_no_background.copy()
        else:
            frame = frame_holder.copy()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray,
                                (5 * (xEnd - xStart),
                                 5 * (yEnd - yStart)),
                                interpolation=cv2.INTER_CUBIC)
        th, frame_gray = cv2.threshold(frame_gray, 180, 255, cv2.THRESH_BINARY)
        frame_gray = cv2.dilate(frame_gray, kernel, iterations=3)

        return frame_gray, frame
    except Exception as e:
        print(f"Error in ProcessImage: {e}")
        return None, None


audioProcess = None


def FindWand():
    global old_frame, old_gray, p0, mask, line_mask, run_request, audioProcess
    try:
        last = time.time()
        t = threading.currentThread()
        print("Find wand...")
        while getattr(t, "do_run", True):
            now = time.time()
            if run_request:
                result = ProcessImage()
                if result[0] is not None:
                    old_gray, old_frame = result
                    p0 = GetPoints(old_gray)
                    if p0 is not None:
                        frameMissingPoints = 0
                        mask = np.zeros_like(old_frame)
                        line_mask = np.zeros_like(old_gray)
                        run_request = False
                        music.play_wav(
                            f'{home_address}/pi_to_potter/music/twinkle.wav')
                    else:
                        music.stop_wav()
                last = time.time()

            time.sleep(.3)
    except cv2.error as e:
        print(f"OpenCV Error in FindWand: {e}")
    except Exception as e:
        print(f"Error in FindWand: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)


def TrackWand():
    global old_frame, old_gray, p0, mask, frameMissingPoints, line_mask, color, active, run_request
    print("Starting wand tracking...")
    color = (0, 0, 255)
    frame_gray = None
    good_new = None

    # Create a mask image for drawing purposes
    noPt = 0
    while True:
        try:
            active = False
            if p0 is not None:
                active = True
                result = ProcessImage()
                if result[0] is not None:
                    frame_gray, frame = result
                    if frame is not None:
                        cv2.imshow("frame_gray", frame_gray)
                        small = cv2.resize(
                            frame, (120, 120), interpolation=cv2.INTER_CUBIC)
                        cv2.imshow("gray", small)
                        if (args.background_subtract):
                            cv2.imshow("frame_no_background", frame_no_background)
                    else:
                        print("No frame.")

                # calculate optical flow
                newPoints = False
                if p0 is not None and len(p0) > 0:
                    noPt = 0
                    try:
                        if old_gray is not None and frame_gray is not None:
                            p1, st, err = cv2.calcOpticalFlowPyrLK(
                                old_gray, frame_gray, p0, None, **lk_params)
                            newPoints = True
                    except cv2.error as e:
                        print(f"OpenCV error in optical flow: {e}")
                    except Exception as e:
                        print(f"Error in optical flow: {e}")
                        continue
                else:
                    noPt = noPt + 1
                    if noPt > 10:
                        try:
                            # Updated for OpenCV 4.x - cv2.findContours returns only 2 values
                            contours, hierarchy = cv2.findContours(
                                line_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if (contours is not None and len(contours) > 0):
                                cnt = contours[0]
                                x, y, w, h = cv2.boundingRect(cnt)
                                crop = line_mask[y - 10:y +
                                                 h + 10, x - 30:x + w + 30]
                                result = ml.CheckShape(crop, args)
                                cv2.putText(
                                    line_mask, result, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                                spells.cast(result)
                                if line_mask is not None:
                                    show_line_mask = cv2.resize(
                                        line_mask, (120, 120), interpolation=cv2.INTER_CUBIC)
                                    if args.setup is not True:
                                        show_line_mask = cv2.resize(
                                            line_mask, (width, height), interpolation=cv2.INTER_CUBIC)
                                    cv2.imshow(
                                        "Raspberry Potter", show_line_mask)
                                line_mask = np.zeros_like(line_mask)
                                print("")
                        except Exception as e:
                            print(f"Error in spell detection: {e}")
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            traceback.print_exception(
                                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                        finally:
                            noPt = 0
                            run_request = True

                if newPoints:
                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        cv2.line(
                            line_mask, (a, b), (c, d), (255, 255, 255), 10)

                    if line_mask is not None:
                        show_line_mask = cv2.resize(
                            line_mask, (120, 120), interpolation=cv2.INTER_CUBIC)
                        if args.setup is not True:
                            cv2.namedWindow(
                                "Raspberry Potter", cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty(
                                "Raspberry Potter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            show_line_mask = cv2.resize(
                                line_mask, (width, height), interpolation=cv2.INTER_CUBIC)
                        cv2.imshow("Raspberry Potter", show_line_mask)

                    # Now update the previous frame and previous points
                    if frame_gray is not None:
                        old_gray = frame_gray.copy()
                else:
                    # This frame didn't have any points... lets go a couple more
                    # keep the old image( don't update it )
                    frameMissingPoints += 1
                    if (frameMissingPoints >= 5 or p0 is None):
                        # Now update the previous frame and previous points
                        if frame_gray is not None:
                            old_gray = frame_gray.copy()
                        p0 = None
                    else:
                        print("Chance: " + str(frameMissingPoints))

            else:
                run_request = True
                time.sleep(.3)

            if good_new is not None:
                p0 = good_new.reshape(-1, 1, 2)

        except IndexError:
            run_request = True
        except cv2.error as e:
            print(f"OpenCV Error in TrackWand: {e}")
        except TypeError as e:
            print(f"Type error in TrackWand: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"Error at line {exc_tb.tb_lineno}")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"General error in TrackWand: {e}")
            
        key = cv2.waitKey(10)
        if key in [27, ord('Q'), ord('q')]:  # exit on ESC
            cv2.destroyAllWindows()
            break


try:
    ml.TrainShapes(f'{home_address}/pi_to_potter')
    t = Thread(target=FrameReader)
    t.do_run = True
    t.start()

    tr = Thread(target=RemoveBackground)
    if args.background_subtract:
        tr.do_run = True
        tr.start()

    find = Thread(target=FindWand)
    find.do_run = True
    find.start()

    server_thread = Thread(target=server.runServer)
    server_thread.do_run = True
    server_thread.start()

    print("\n\n\n----------------------------------------------------------------------------------\n")
    print("Windows will open when there are points to see!")
    print("There should only be white spots corresponding to the wand.  If there are MORE this wont work.")
    print("Use an IR light source and a reflector, and ensure the camera does not see halogen light,")
    print("nor sunlight - both are big IR sources.")
    print("----------------------------------------------------------------------------------\n\n\n\n")
    time.sleep(2)
    TrackWand()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    # Clean shutdown
    if 't' in locals():
        t.do_run = False
        t.join()
    if args.background_subtract and 'tr' in locals():
        tr.do_run = False
        tr.join()
    if 'find' in locals():
        find.do_run = False
        find.join()
    if 'server_thread' in locals():
        server_thread.do_run = False
        server_thread.join()
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
