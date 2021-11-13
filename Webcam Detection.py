import cv2

trained_face_data = cv2.CascadeClassifier('D:/Python Projects/Face Detection/haarcascade_frontalface_default.xml')

# Webcam opening.
webcam = cv2.VideoCapture(0)

while True:
    
    # read the current frame
    successful_frame_read, frame = webcam.read()
    
    #grayscaled_img
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 6)
    
    cv2.imshow('Webcam Face Detection', frame)
    key = cv2.waitKey(1)
    
    if key==81 or key==113 or key==27:
        break
    
    
