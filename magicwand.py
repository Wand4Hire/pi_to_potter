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
    def __init__(self, width=320, height=240, framerate=30):  # REDUCED from 640x480
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

# Parameters - OPTIMIZED for performance
width = 800  # Keep original display size
height = 480  # Keep original display size
lk_params = dict(winSize=(15, 15), maxLevel=5,  # REDUCED from (25,25) and 10 levels
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Movement detection parameters
movement_threshold = 5.0  # Minimum distance to consider as movement
stationary_threshold = 2.0  # Maximum distance to consider as stationary
movement_frames_required = 3  # Frames of movement to start tracking
stationary_frames_required = 10  # Frames of no movement to stop tracking

# Tracking states
TRACKING_IDLE = 0
TRACKING_DETECTING_MOVEMENT = 1
TRACKING_ACTIVE = 2
TRACKING_DETECTING_STOP = 3

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

# Movement tracking state
tracking_state = TRACKING_IDLE
movement_counter = 0
stationary_counter = 0
last_point = None
spell_path = []  # Store the path for spell recognition

# Camera view parameters - SLIGHTLY REDUCED processing area
yStart, yEnd = 0, 240  # REDUCED from 0, 360
xStart, xEnd = 0, 320  # REDUCED from 0, 480
kernel = np.ones((3, 3), np.uint8)  # REDUCED from (5,5)

# Initialize libcamera-vid with REDUCED resolution
print("Initializing libcamera-vid")
camera = LibcameraVid(width=320, height=240, framerate=30)  # REDUCED from 640x480
camera.start()
time.sleep(1)  # REDUCED from 2 seconds

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

kernel = np.ones((3, 3), np.uint8)  # REDUCED kernel size

def filter_circular_objects(fgmask, min_circularity=0.7):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 2000:  # REDUCED for smaller resolution
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > min_circularity:
            valid_contours.append(cnt)
    
    filtered_mask = np.zeros_like(fgmask)
    cv2.drawContours(filtered_mask, valid_contours, -1, 255, -1)
    return filtered_mask

def prioritize_center_objects(fgmask, center_region_ratio=0.3):
    height, width = fgmask.shape
    center_x, center_y = width // 2, height // 2
    region_size = int(min(width, height) * center_region_ratio)
    
    # Create center region mask
    center_mask = np.zeros_like(fgmask)
    cv2.rectangle(center_mask, 
                 (center_x - region_size, center_y - region_size),
                 (center_x + region_size, center_y + region_size),
                 255, -1)
    
    # Prioritize objects in center
    prioritized = cv2.bitwise_and(fgmask, center_mask)
    return prioritized if np.count_nonzero(prioritized) > 0 else fgmask

def RemoveBackground():
    global frame_holder, frame_no_background, should_exit

    # OPTIMIZED: Better background subtraction parameters
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    print("Background subtraction thread started")
    
    while not should_exit:
        try:
            with frame_lock:
                if frame_holder is None:
                    time.sleep(0.01)  # REDUCED from 0.03
                    continue
                frameCopy = frame_holder.copy()

            # Subtract Background
            fgmask = fgbg.apply(frameCopy, learningRate=0.01)  # INCREASED learning rate
            
            # SIMPLIFIED: Morphological operations
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # ADDED: Filter for circular objects
            fgmask = filter_circular_objects(fgmask)
            
            # ADDED: Prioritize center objects
            fgmask = prioritize_center_objects(fgmask)
            
            with frame_lock:
                frame_no_background = cv2.bitwise_and(frameCopy, frameCopy, mask=fgmask)
            
            time.sleep(0.01)  # REDUCED from 0.03
        except Exception as e:
            print(f"Error in RemoveBackground: {e}")
            time.sleep(0.05)
    
    print("Background subtraction thread exiting")

def FrameReader():
    global frame_holder, should_exit
    
    print("Frame reader thread started")
    
    while not should_exit:
        try:
            ret, image_data = camera.read()
            if not ret or image_data is None:
                time.sleep(0.01)  # REDUCED from 0.03
                continue

            frame = image_data[yStart:yEnd, xStart:xEnd]
            cv2.flip(frame, 1, frame)
            
            with frame_lock:
                frame_holder = frame
            
            # REMOVED: time.sleep(.03) - let it run full speed
        except Exception as e:
            print(f"Error in FrameReader: {e}")
            time.sleep(0.05)
    
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
    # Find the brightest point in the image (wand tip)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    
    # REDUCED threshold for smaller resolution
    if max_val > 150:  # REDUCED from 200
        point = np.array([[max_loc]], dtype=np.float32)
        return point
    
    
    return None

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
        
        # CRITICAL FIX: Minimal upscaling instead of massive 5x upscale
        frame_gray = cv2.resize(frame_gray,
                                (2 * (xEnd - xStart),    # REDUCED from 5x to 2x
                                 2 * (yEnd - yStart)),   # REDUCED from 5x to 2x
                                interpolation=cv2.INTER_LINEAR)  # CHANGED from CUBIC to LINEAR
        
        th, frame_gray = cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY)  # REDUCED threshold
        frame_gray = cv2.dilate(frame_gray, kernel, iterations=2)  # REDUCED from 3

        return frame_gray, frame
    except Exception as e:
        print(f"Error in ProcessImage: {e}")
        return None, None

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    if p1 is None or p2 is None:
        return float('inf')
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def perform_spell_recognition():
    """Perform spell recognition on the collected path"""
    global spell_path
    
    if len(spell_path) < 10:  # Need minimum points for recognition
        print("Path too short for spell recognition")
        return False
    
    try:
        # Convert path to format expected by ML module
        path_array = np.array(spell_path)
        
        # Call spell recognition (assuming ml module has this function)
        if hasattr(ml, 'recognize_spell'):
            spell_result = ml.recognize_spell(path_array)
            if spell_result:
                print(f"Spell recognized: {spell_result}")
                # Execute the spell
                if hasattr(spells, 'cast_spell'):
                    spells.cast_spell(spell_result)
                return True
        else:
            print("Spell recognition not available")
            
    except Exception as e:
        print(f"Error in spell recognition: {e}")
    
    return False

def clear_spell_pattern():
    """Clear the current spell pattern and reset tracking state"""
    global spell_path, line_mask, tracking_state, movement_counter, stationary_counter
    
    print("Clearing spell pattern")
    spell_path = []
    if line_mask is not None:
        line_mask.fill(0)  # Clear the line mask
    tracking_state = TRACKING_IDLE
    movement_counter = 0
    stationary_counter = 0

def update_tracking_state(current_point):
    """Update the movement tracking state machine"""
    global tracking_state, movement_counter, stationary_counter, last_point, spell_path
    
    if last_point is not None:
        distance = calculate_distance(current_point, last_point)
        
        if tracking_state == TRACKING_IDLE:
            # Looking for initial movement
            if distance > movement_threshold:
                movement_counter += 1
                if movement_counter >= movement_frames_required:
                    print("Movement detected - starting spell tracking")
                    tracking_state = TRACKING_DETECTING_MOVEMENT
                    spell_path = [current_point]  # Start new path
                    movement_counter = 0
            else:
                movement_counter = 0
                
        elif tracking_state == TRACKING_DETECTING_MOVEMENT:
            # Confirming sustained movement
            if distance > movement_threshold:
                movement_counter += 1
                spell_path.append(current_point)
                if movement_counter >= movement_frames_required:
                    print("Sustained movement confirmed - active tracking")
                    tracking_state = TRACKING_ACTIVE
                    movement_counter = 0
            else:
                # Lost movement, back to idle
                movement_counter = 0
                tracking_state = TRACKING_IDLE
                clear_spell_pattern()
                
        elif tracking_state == TRACKING_ACTIVE:
            # Active tracking - add all points to path
            spell_path.append(current_point)
            
            if distance < stationary_threshold:
                stationary_counter += 1
                if stationary_counter >= stationary_frames_required:
                    print("Movement stopped - attempting spell recognition")
                    tracking_state = TRACKING_DETECTING_STOP
                    stationary_counter = 0
            else:
                stationary_counter = 0
                
        elif tracking_state == TRACKING_DETECTING_STOP:
            # Confirming the wand has stopped
            if distance < stationary_threshold:
                stationary_counter += 1
                if stationary_counter >= stationary_frames_required:
                    print("Wand stopped - performing spell recognition")
                    success = perform_spell_recognition()
                    if not success:
                        print("No spell match - clearing pattern")
                    clear_spell_pattern()
            else:
                # Movement resumed, back to active tracking
                print("Movement resumed")
                tracking_state = TRACKING_ACTIVE
                stationary_counter = 0
                spell_path.append(current_point)
    
    last_point = current_point
    return tracking_state in [TRACKING_ACTIVE, TRACKING_DETECTING_STOP]

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
            time.sleep(.1)  # REDUCED from .3
    except cv2.error as e:
        print("OpenCV Error in FindWand:")
        print(e)
    except Exception as e:
        print(f"Error in FindWand: {e}")
        traceback.print_exc()

def TrackWand():
    global old_frame, old_gray, p0, mask, frameMissingPoints, line_mask, color, active, run_request
    print("Starting wand tracking...")
    color = (0, 0, 255)
    frame_gray = None
    good_new = None

    # Create the Raspberry Potter window at the start - KEEP ORIGINAL LOGIC
    if args.setup is not True:
        cv2.namedWindow("Raspberry Potter", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Raspberry Potter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow("Raspberry Potter", cv2.WINDOW_NORMAL)

    # Create a mask image for drawing purposes
    noPt = 0
    
    while True:
        try:
            active = False
            if p0 is not None:
                active = True
                frame_gray, frame = ProcessImage()
                if frame is not None:
                    if args.setup:  # Only show debug windows in setup mode
                        cv2.imshow("frame_gray", frame_gray)
                        small = cv2.resize(
                            frame, (120, 120), interpolation=cv2.INTER_LINEAR)  # CHANGED to LINEAR
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
                        print("OpenCV error:", e)
                    except BaseException:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        print("Optical flow error:")
                        traceback.print_exception(
                            exc_type, exc_value, exc_traceback, 
                            limit=2, file=sys.stdout
                        )
                        continue
                else:
                    noPt = noPt + 1
                    if noPt > 5:  # REDUCED from 10
                        run_request = True

                if newPoints:
                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # Update tracking state based on movement
                    if len(good_new) > 0:
                        current_point = good_new[0].ravel()
                        should_draw = update_tracking_state(current_point)
                        
                        # Only draw lines when actively tracking
                        if should_draw:
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                cv2.line(
                                    line_mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 8)
                        
                        # Print tracking state for debugging
                        # if args.setup:
                        #    state_names = ["IDLE", "DETECTING_MOVEMENT", "ACTIVE", "DETECTING_STOP"]
                        #    print(f"State: {state_names[tracking_state]}, Point: {current_point}, Path length: {len(spell_path)}")

                # Always update the display
                if line_mask is not None:
                    # Create display image
                    display_img = np.zeros((height, width), dtype=np.uint8)
                    
                    # Scale and show the tracking visualization
                    if line_mask.shape[0] > 0 and line_mask.shape[1] > 0:
                        display_img = cv2.resize(line_mask, (width, height))
                    
                    # Add status text overlay
                    if tracking_state != TRACKING_IDLE:
                        state_names = ["IDLE", "DETECTING", "TRACKING", "STOPPING"]
                        status_text = f"Status: {state_names[tracking_state]}"
                        cv2.putText(display_img, status_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow("Raspberry Potter", display_img)

                # Now update the previous frame and previous points
                if frame_gray is not None:
                    old_gray = frame_gray.copy()
                    
                # Update points for next frame
                if good_new is not None:
                    p0 = good_new.reshape(-1, 1, 2)

            else:
                # Show black screen when not tracking
                blank = np.zeros((height, width), dtype=np.uint8)
                cv2.putText(blank, "Looking for wand...", (width//2-100, height//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Raspberry Potter", blank)
                
                run_request = True
                time.sleep(.05)  # REDUCED from .1

        except IndexError:
            run_request = True
        except cv2.error as e:
            print("OpenCV error:", e)
        except TypeError as e:
            print("Type error:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"Error at line {exc_tb.tb_lineno}")
        except KeyboardInterrupt as e:
            raise e
        except BaseException as e:
            print("Tracking Error:", e)
            traceback.print_exc()
        
        key = cv2.waitKey(5)  # REDUCED from 10
        if key in [27, ord('Q'), ord('q')]:  # exit on ESC
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
    
    time.sleep(1)  # REDUCED from 2
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
    time.sleep(0.2)
    
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
