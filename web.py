
import cv2
import streamlit as st
import face_recognition
from datetime import datetime
import os
import numpy as np


# Capture
OUTPUT_DIR = "known_faces" 
PHOTO_COUNT = 5  
DELAY_BETWEEN_PHOTOS = 2  

os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("Attendance using Face Recognition - Mini Project ")

person_name = st.text_input("Enter the name of the person:")

if st.button("Capture Photos"):
    if not person_name.strip():
        st.error("Please enter a valid name!")
    else:

        video_capture = cv2.VideoCapture(0)

        st.write(f"Starting photo capture for {person_name}. Please stay in front of the camera.")
        photo_number = 0

        frame_placeholder = st.empty()

        while photo_number < PHOTO_COUNT:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to access the webcam. Please ensure it's connected.")
                break

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            photo_path = os.path.join(OUTPUT_DIR, f"{person_name}_{photo_number + 1}.jpg")
            cv2.imwrite(photo_path, frame)
            st.write(f"Saved photo {photo_number + 1} at {photo_path}")
            photo_number += 1
            cv2.waitKey(DELAY_BETWEEN_PHOTOS * 1000)

        video_capture.release()
        st.success(f"Captured {photo_number} photos for {person_name} and saved them in '{OUTPUT_DIR}'.")


st.title("Live Face Detection App")

path = 'C://Users//hp//OneDrive//Desktop//LastMP//attendance//attendance//known_faces'

images = []
classNames = []
mylist = os.listdir(path)
print("I've reached here")
for cl in mylist:
    image_path = f'{path}/{cl}'
    curImg = cv2.imread(image_path)


    if curImg is None:
        print(f"Warning: Unable to load image {image_path}. Skipping.")
        continue

    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoded_face = face_recognition.face_encodings(img)[0]
            encodeList.append(encoded_face)
        except IndexError:
            print(f"No face found in image {idx + 1}. Skipping.")
        except Exception as e:
            print(f"Error processing image {idx + 1}: {e}")
    return encodeList

encoded_face_train = findEncodings(images)


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}\n')


start_webcam = st.checkbox("Start Webcam")

if start_webcam:

    frame_placeholder = st.empty()


    cap = cv2.VideoCapture(0) 


    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Please ensure it's connected.")
            break

        imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            print(matchIndex)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                print(name.split("_")[0])
                y1,x2,y2,x1 = faceloc
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(frame,name.split("_")[0], (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name.split("_")[0])

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)


    cap.release()
    st.info("Webcam stopped.")

else:
    st.write("Check 'Start Webcam' to begin live face detection.")