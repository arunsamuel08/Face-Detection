import cv2

#loading pre-trained data
trained_face_data = cv2.CascadeClassifier('D:/Python Projects/Face Detection/haarcascade_frontalface_default.xml')

# choosing image to detect from
img = cv2.imread('D:/Python Projects/Face Detection/rdj.png')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 10)

#
cv2.imshow('Face Detection App', img)
cv2.waitKey()

print("Code Completed")