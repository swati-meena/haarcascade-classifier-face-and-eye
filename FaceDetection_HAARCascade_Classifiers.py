import numpy as np
import cv2

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r"/home/hp/Desktop/deeplearning/haar_cascade_classifier_basic_project/Haarcascades/haarcascade_frontalface_default.xml")

# Load the image
image = cv2.imread(r"/home/hp/Desktop/deeplearning/Sentiment Analysis/Training/happy/12.jpeg")

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Check if faces are detected
if len(faces) == 0:
    print("No faces found!")
else:
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:  # (x, y) is the top-left corner, and (w, h) is the width and height of the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    # Display the output image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()
