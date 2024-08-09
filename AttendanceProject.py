import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import Tk, Label, Button, Frame

path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


    
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)  # Move the file pointer to the beginning
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.write(f'\n{name},{tString},{dString}')
            print(f'Attendance marked for {name} at {tString} on {dString}')
        else:
            print(f'{name} already marked present.')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
        cv2.imshow('webcam', img)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def close_webcam():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

root = Tk()
root.title("Face Recognition Attendance System")

label = Label(root, text="Welcome to the Face Recognition Attendance System!")
label.pack()

button_frame = Frame(root)
button_frame.pack()

start_button =Button(button_frame, text="Start Webcam", command=start_webcam)
start_button.grid(row=0, column=0, padx=10, pady=10)

close_button = Button(button_frame, text="Close Webcam", command=close_webcam)
close_button.grid(row=0, column=1, padx=10, pady=10)

window = root

root.mainloop()
