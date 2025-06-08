#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Add before any cv2 imports
import numpy as np
import argparse
import cv2
import threading
import subprocess
import os
import sys
import traceback
import time
from pathlib import Path
from threading import Thread

# Import other modules
try:
    import music
    import ml
    import spells
    import server
except ImportError as e:
    print(f"Warning: Missing module - {str(e)}")

# Try to import optional modules
try:
    import CameraLED
    CAMERA_LED_AVAILABLE = True
except ImportError:
    CAMERA_LED_AVAILABLE = False
    print("Warning: CameraLED module not available")

# Setup camera using libcamera-vid
class LibcameraVid:
    def __init__(self, width=640, height=480, framerate=30):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.process = None
        self.pipe = None
        
    def start(self):
        cmd = [
            'libcamera-vid',
            '-t', '0',  # Run indefinitely
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.framerate),
            '--nopreview',
            '--codec', 'yuv420',
            '--flush',  # Reduce latency
            '--denoise', 'cdn_off',  # Disable denoise for faster processing
            '--brightness', '0.0',
            '--contrast', '1.0',
            '--sharpness', '1.0',
            '-o', '-'
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.pipe = self.process.stdout
        
    def read(self):
        if self.pipe is None:
            return False, None
            
        # Read YUV420 frame (1.5 bytes per pixel)
        frame_size = self.width * self.height * 3 // 2
        data = self.pipe.read(frame_size)
        if len(data) != frame_size:
            return False, None
            
        # Convert YUV420 to BGR
        yuv = np.frombuffer(data, dtype=np.uint8)
        yuv = yuv.reshape((self.height * 3 // 2, self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return True, bgr
        
    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.pipe = None

frame_lock = threading.Lock()
home_address = str(Path.home())

parser = argparse.ArgumentParser(
    description='Cast some spells! Recognize wand motions')
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

# Parameters
width = 800
height = 480
lk_params = dict(winSize=(25, 25), maxLevel=10, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
movement_threshold = 80
active = False

# Global variables
run_request = True
frame_holder = None
frame_no_background = None
p0 = None
frameMissingPoints = 0
point_aging = []
old_frame = None
old_gray = None
mask = None
line_mask = None
color = (0, 0, 255)
should_exit = False

# Camera view parameters
yStart, yEnd = 0, 360
xStart, xEnd = 0, 480
kernel = np.ones((5, 5), np.uint8)

# Initialize libcamera-vid
print("Initializing libcamera-vid")
camera = LibcameraVid(width=640, height=480, framerate=30)
camera.start()
time.sleep(2)  # Let camera warm up

# Initialize frames
ret, image_data = camera.read()
if ret and image_data is not None:
    frame_holder = image_data
    frame_no_background = image_data
    frame_holder = frame_holder[yStart:yEnd, xStart:xEnd]
    cv2.flip(frame_holder, 1, frame_holder)
    frame = None
    print(f"Initial frame captured - size: {frame_holder.shape}")
else:
    print("Error: Could not capture initial frame")
    camera.stop()
    sys.exit(1)

print("About to start.")

kernel = np.ones((5, 5), np.uint8)


def RemoveBackground():
    global frame_holder, frame_no_background, should_exit

    fgbg = cv2.createBackgroundSubtractorMOG2()
    print("Background subtraction thread started")
    
    while not should_exit:
        try:
            with frame_lock:
                if frame_holder is None:
                    time.sleep(0.03)
                    continue
                frameCopy = frame_holder.copy()

            # Subtract Background
            fgmask = fgbg.apply(frameCopy, learningRate=0.001)
            
            with frame_lock:
                frame_no_background = cv2.bitwise_and(frameCopy, frameCopy, mask=fgmask)
            
            time.sleep(0.03)
        except Exception as e:
            print(f"Error in RemoveBackground: {e}")
            time.sleep(0.1)
    
    print("Background subtraction thread exiting")


def FrameReader():
    global frame_holder, should_exit
    
    print("Frame reader thread started")
    
    while not should_exit:
        try:
            ret, image_data = camera.read()
            if not ret or image_data is None:
                time.sleep(0.03)
                continue

            frame = image_data[yStart:yEnd, xStart:xEnd]
            cv2.flip(frame, 1, frame)
            
            with frame_lock:
                frame_holder = frame
            
            time.sleep(.03)
        except Exception as e:
            print(f"Error in FrameReader: {e}")
            time.sleep(0.1)
    
    print("Frame reader thread exiting")

def trim_points():
    global point_aging
    indexesToDelete = []
    index = 0
    for old_point in point_aging:
        if (time.time() - old_point["when"] > 15):
            old_point["times_seen"] = old_point["times_seen"] - 1
            if old_point["times_seen"] <= 0:
                indexesToDelete.append(index)
                deleted = True
                break
        index += 1

    for i in reversed(indexesToDelete):
        del point_aging[i]


def GetPoints(image):
    global point_aging
    if image is None or image.size == 0:
        return None
        
    start_points = cv2.goodFeaturesToTrack(
        image, maxCorners=5, qualityLevel=0.0001, minDistance=5)

    # Clean out aged points.
    trim_points()
    return start_points


def ProcessImage():
    global frame_holder, frame_no_background
    try:
        with frame_lock:
            if args.background_subtract:
                if frame_no_background is None:
                    return None, None
                frame = frame_no_background.copy()
            else:
                if frame_holder is None:
                    return None, None
                frame = frame_holder.copy()

        if frame is None or frame.size == 0:
            return None, None

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


def FindWand():
    global old_frame, old_gray, p0, mask, line_mask, run_request, should_exit
    try:
        print("Find wand...")
        while not should_exit:
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
                        music.play_wav(f'{home_address}/pi_to_potter/music/twinkle.wav')
                    else:
                        music.stop_wav()

            time.sleep(.3)
    except cv2.error as e:
        print("OpenCV Error in FindWand:")
        print(e)
    except Exception as e:
        print(f"Error in FindWand: {e}")
        traceback.print_exc()


def TrackWand():
    global old_frame, old_gray, p0, mask, frameMissingPoints, line_mask, color, active, run_request, should_exit
    print("Starting wand tracking...")
    color = (0, 0, 255)
    frame_gray = None
    good_new = None

    # Create a mask image for drawing purposes
    noPt = 0
    
    while not should_exit:
        try:
            active = False
            if p0 is not None:
                active = True
                result = ProcessImage()
                if result[0] is not None:
                    frame_gray, frame = result
                    if frame is not None:
                        # Show the processed frame for debugging
                        cv2.imshow("frame_gray", frame_gray)
                        small = cv2.resize(frame, (120, 120), interpolation=cv2.INTER_CUBIC)
                        cv2.imshow("gray", small)
                        if (args.background_subtract):
                            with frame_lock:
                                if frame_no_background is not None:
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
                        print("cv err")
                        print(e)
                    except Exception as e:
                        print("Optical flow error:", e)
                        continue
                else:
                    noPt = noPt + 1
                    if noPt > 10:
                        try:
                            if line_mask is not None:
                                contours, hierarchy = cv2.findContours(
                                    line_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if (contours is not None and len(contours) > 0):
                                    cnt = contours[0]
                                    x, y, w, h = cv2.boundingRect(cnt)
                                    crop = line_mask[y - 10:y + h + 10, x - 30:x + w + 30]
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
                                        cv2.imshow("Raspberry Potter", show_line_mask)
                                    line_mask = np.zeros_like(line_mask)
                                    print("")
                        except Exception as e:
                            print(f"FindSpell error: {e}")
                            traceback.print_exc()
                        finally:
                            noPt = 0
                            run_request = True

                if newPoints:
                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        cv2.line(line_mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 10)

                    if line_mask is not None:
                        show_line_mask = cv2.resize(
                            line_mask, (120, 120), interpolation=cv2.INTER_CUBIC)
                        if args.setup is not True:
                            cv2.namedWindow("Raspberry Potter", cv2.WND_PROP_FULLSCREEN)
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
            print("OpenCV Error in TrackWand:")
            print(e)
        except TypeError as e:
            print("Type error.")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print((exc_type, exc_tb.tb_lineno))
        except KeyboardInterrupt as e:
            should_exit = True
            break
        except Exception as e:
            print(f"Tracking Error: {e}")
            traceback.print_exc()
            
        # Handle key presses
        key = cv2.waitKey(10)
        if key in [27, ord('Q'), ord('q')]:  # exit on ESC
            should_exit = True
            cv2.destroyAllWindows()
            break


# MAIN EXECUTION
try:
    ml.TrainShapes(f'{home_address}/pi_to_potter')
    
    t = Thread(target=FrameReader, daemon=True)
    t.start()

    if args.background_subtract:
        tr = Thread(target=RemoveBackground, daemon=True)
        tr.start()

    find = Thread(target=FindWand, daemon=True)
    find.start()

    # Start server if available
    try:
        server_thread = Thread(target=server.runServer, daemon=True)
        server_thread.start()
    except:
        pass

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
    should_exit = True
except Exception as e:
    print(f"Error in main: {e}")
    traceback.print_exc()
    should_exit = True
finally:
    should_exit = True
    time.sleep(0.5)
    
    # Clean up camera resources
    try:
        camera.stop()
        print("libcamera-vid stopped")
    except Exception as e:
        print(f"Error stopping camera: {e}")
    
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    try:
        music.stop_wav()
    except:
        pass
    
    print("Cleanup complete")
    sys.exit(1)
