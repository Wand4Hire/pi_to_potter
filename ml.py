from os import listdir
from os.path import isfile, join, isdir
import cv2
import numpy as np
import math
import time

knn = None
nameLookup = {}


def nearPoints(p1, p2, dist):
    """Check if two points are within a specified distance"""
    point2 = p2
    if isinstance(p2, dict) and "x" in p2:
        point2 = [p2["x"], p2["y"]]
    
    distance = math.sqrt(((p1[0] - point2[0])**2) + ((p1[1] - point2[1])**2))
    print("Comparing: " +
          str(p1[0]) +
          " " +
          str(p1[1]) +
          " " +
          str(point2[0]) +
          " " +
          str(point2[1]) +
          " distance: " +
          str(distance))
    return distance < dist


def TrainShapes(path_to_pictures):
    """Train the KNN classifier with shape images"""
    global knn, nameLookup
    labelNames = []
    labelIndexes = []
    trainingSet = []
    numPics = 0
    dirCount = 0
    mypath = path_to_pictures + "/Pictures/"
    
    # Check if Pictures directory exists
    if not isdir(mypath):
        print(f"Warning: Pictures directory not found at {mypath}")
        return
    
    for d in listdir(mypath):
        if isdir(join(mypath, d)):
            nameLookup[dirCount] = d
            dirCount = dirCount + 1
            for f in listdir(join(mypath, d)):
                if isfile(join(mypath, d, f)):
                    labelNames.append(d)
                    labelIndexes.append(dirCount - 1)
                    trainingSet.append(join(mypath, d, f))
                    numPics = numPics + 1

    print("Training set...")
    print(trainingSet)

    print("Labels...")
    print(labelNames)

    print("Indexes...")
    print(labelIndexes)

    print("Lookup...")
    print(nameLookup)

    if numPics == 0:
        print("Warning: No training images found")
        return

    samples = []
    for i in range(0, numPics):
        try:
            img = cv2.imread(trainingSet[i])
            if img is None:
                print(f"Warning: Could not load image {trainingSet[i]}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize to consistent size for training
            gray_resized = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_CUBIC)
            samples.append(gray_resized.flatten())
        except Exception as e:
            print(f"Error processing image {trainingSet[i]}: {e}")
            continue

    if len(samples) == 0:
        print("Error: No valid training samples found")
        return

    # Convert to numpy arrays
    samples_array = np.array(samples, dtype=np.float32)
    labels_array = np.array(labelIndexes[:len(samples)], dtype=np.int32)

    # Initialize and train kNN classifier
    knn = cv2.ml.KNearest_create()
    knn.train(samples_array, cv2.ml.ROW_SAMPLE, labels_array)
    print(f"Training completed with {len(samples)} samples")


lastTrainer = None


def CheckShape(img, args):
    """Check what shape the input image represents"""
    global knn, nameLookup, lastTrainer

    if knn is None:
        print("Error: KNN classifier not trained")
        return "error"

    size = (20, 20)
    try:
        # Ensure image is grayscale
        if len(img.shape) == 3:
            test_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = img.copy()
            
        test_gray = cv2.resize(test_gray, size, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return "error"

    # Save training images if requested
    if hasattr(args, 'train') and args.train and not np.array_equal(img, lastTrainer if lastTrainer is not None else np.array([])):
        try:
            cv2.imwrite("Pictures/char" + str(time.time()) + ".png", test_gray)
            lastTrainer = img.copy()
        except Exception as e:
            print(f"Error saving training image: {e}")

    try:
        # Prepare sample for prediction
        sample = test_gray.flatten().astype(np.float32).reshape(1, -1)
        
        # Find nearest neighbors
        ret, result, neighbours, dist = knn.findNearest(sample, k=5)
        
        print(f"KNN result: ret={ret}, result={result}, neighbours={neighbours}, dist={dist}")
        
        # Get the predicted class
        predicted_class = int(result[0][0])
        
        if predicted_class in nameLookup:
            match_name = nameLookup[predicted_class]
            print("Match: " + match_name)
            return match_name
        else:
            print(f"Error: Predicted class {predicted_class} not found in lookup")
            return "error"
            
    except Exception as e:
        print(f"Error in shape prediction: {e}")
        return "error"
