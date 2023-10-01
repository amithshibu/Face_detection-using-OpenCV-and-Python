#!/usr/bin/env python3

import cv2
import time
import sys

def benchmark(num_times):
    """
    Call face_cascade.detectMultiScale 'num_times' number of times 
    and return the execution time.
    """
    start_time = time.clock_gettime(time.CLOCK_REALTIME)
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the input image
    input_image = cv2.imread('test.jpg')

    # Convert the image into grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Calculate the overhead time for setup
    overhead_time = time.clock_gettime(time.CLOCK_REALTIME) - start_time

    start_time = time.clock_gettime(time.CLOCK_REALTIME)
    # Detect faces
    for _ in range(num_times):
        faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4)

    face_detection_time = time.clock_gettime(time.CLOCK_REALTIME) - start_time

    return (overhead_time, face_detection_time)

if __name__ == '__main__':
    num_times = int(sys.argv[1])
    overhead_time, face_detection_time = benchmark(num_times)
    print("Overhead time to load classifier and image: %.6f seconds" % overhead_time)
    print("Time to perform %d face detections: %.6f seconds" % (num_times, face_detection_time))
