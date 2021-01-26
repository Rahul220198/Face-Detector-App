import cv2
from random import randrange

# -------------- load some pre-trained data on face frontals from opencv(haar cascade algorithm) ------------- #
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# -------------- Choose an image to detect the faces in image ------------------ #
img = cv2.imread('SampleImage.jpg')

# -------------- convert to Grayscaled image -------------- #
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------- Detect Faces in image -------------- #
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)


# ------------- Draw rectangle around the faces ------------- #
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

# -------------- Display  the image with faces detected -------------- #
cv2.imshow("Image display", img)

# -------------- wait for the image to show -------------- #
cv2.waitKey()

print("Code Complete")
