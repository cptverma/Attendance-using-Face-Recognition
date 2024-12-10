import cv2
import os

# Configuration
OUTPUT_DIR = "known_faces"  # Directory to save photos
PHOTO_COUNT = 5  # Number of photos to capture
DELAY_BETWEEN_PHOTOS = 2  # Delay between photos in seconds

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

print("Starting the webcam. Press 'q' to quit manually.")
photo_number = 0

while photo_number < PHOTO_COUNT:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Display the video feed
    cv2.imshow("Webcam", frame)

    # Automatically take photos every few seconds
    if photo_number < PHOTO_COUNT:
        photo_path = os.path.join(OUTPUT_DIR, f"person_{photo_number + 1}.jpg")
        cv2.imwrite(photo_path, frame)
        print(f"Saved photo {photo_number + 1} at {photo_path}")
        photo_number += 1
        cv2.waitKey(DELAY_BETWEEN_PHOTOS * 1000)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()

print(f"Captured {photo_number} photos and saved them in '{OUTPUT_DIR}'.")
