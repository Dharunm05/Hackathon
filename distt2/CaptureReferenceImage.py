import cv2 as cv
import numpy as np

# Distance constants (KNOWN_DISTANCE and CAR_WIDTH are set based on the car example)
KNOWN_DISTANCE = 10  # IN CENTIMETERS (Adjust this based on known distance for car detection)
CAR_WIDTH = 100  # IN CENTIMETERS (Average car width; adjust if needed)

# Object detector constants 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detection
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV DNN
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn.DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Object detector function (modified to exclude persons)
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]

        # Handle both scalar and array cases for classid
        class_name = class_names[int(classid)] if isinstance(classid, np.ndarray) else class_names[classid]
        label = f"{class_name} : {score:.2f}"

        # Draw rectangle and label only if the class is 'car'
        if class_name == 'car':
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)
            data_list.append([class_name, box[2], (box[0], box[1] - 2)])  # [class_name, width, (x, y)]

    return data_list

# Function to find focal length based on known distance and object width
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# Reading the reference images
ref_car = cv.imread(r'C:\Users\dharu\OneDrive\Desktop\distance\Yolov4-Detector-and-Distance-Estimator\image4.jpg')

if ref_car is None:
    print("Error: Could not read the reference image. Please check the file path.")
else:
    car_data = object_detector(ref_car)

    # Make sure that objects are detected in the reference images
    if len(car_data) > 0:
        car_width_in_rf = car_data[0][1]  # Width of the car in the reference image

        print(f"Car width in pixels: {car_width_in_rf}")

        # Finding focal length 
        focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)

        # Video capture for live detection
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            data = object_detector(frame)
            for d in data:
                if d[0] == 'car':  # Check for car detection only
                    distance = distance_finder(focal_car, CAR_WIDTH, d[1])
                    x, y = d[2]

                    # Draw black rectangle for text background
                    cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)

                    # Check if the user is fraud or not based on the detected distance
                    if distance < 150:
                        status = "FRAUD USER"
                        color = RED
                    elif distance > 150:
                        status = "NOT A FRAUD"
                        color = GREEN
                    else:
                        status = "INCONCLUSIVE"
                        color = (255, 255, 0)

                    # Display the status (fraud or not)
                    cv.putText(frame, f'Dis: {round(distance, 2)} cm', (x + 5, y + 13), FONTS, 0.48, color, 2)
                    cv.putText(frame, status, (x + 5, y + 30), FONTS, 0.6, color, 2)

            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Error: Could not detect the reference car in the image.")

