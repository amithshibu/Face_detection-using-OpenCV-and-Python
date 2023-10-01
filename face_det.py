import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
input_image = cv2.imread('test.jpg')

# Convert the input image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around the detected faces
for (x, y, width, height) in detected_faces:
    cv2.rectangle(input_image, (x, y), (x + width, y + height), (255, 0, 0), 2)

# Display the output image with the detected faces
cv2.imshow('Detected Faces', input_image)
cv2.waitKey()
