
import cv2

# Load the Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(r"/home/hp/Desktop/deeplearning/haar_cascade_classifier_basic_project/Haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"/home/hp/Desktop/deeplearning/haar_cascade_classifier_basic_project/Haarcascades/haarcascade_eye.xml")

# Check if the cascade files were loaded properly
if face_cascade.empty():
    print("Error: Could not load face cascade classifier.")
    exit()

if eye_cascade.empty():
    print("Error: Could not load eye cascade classifier.")
    exit()

# Function to detect faces and eyes in the frame
def detect_faces_and_eyes(gray, frame):
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Region of interest for detecting eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

# Initialize webcam capture
video_capture = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam opened successfully. Starting face and eye detection...")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # If frame is not captured correctly, exit
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the captured frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes in the grayscale image
    result_frame = detect_faces_and_eyes(gray, frame)

    # Display the resulting frame with detected faces and eyes
    cv2.imshow('Face and Eye Detection', result_frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()










