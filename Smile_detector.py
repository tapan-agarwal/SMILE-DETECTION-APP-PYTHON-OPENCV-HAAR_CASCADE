import cv2

# import pre trained haar cascade algorithm to detect face
trained_face_algorithm = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# import pre trained algorith
trained_smile_algorithm = cv2.CascadeClassifier("haarcascade_smile.xml")

# to load an image on which smile will be detected
#img = cv2.imread("G2.jpg")

# to start webcame or camera for real time smile detection
video = cv2.VideoCapture(0)

# to convert image in black and white so that we can detect the co-ordinates as haar cascade algorithm works faster on black and white images
#grayScale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# to get the location of co-ordinates of rectangle enclosing a smile
#smile_coordinates = trained_smile_algorithm.detectMultiScale(grayScale_image)
# print(smile_coordinates)

# to draw box around all the smiles , we need to get co-ordinates of all the smiles in the image and then we will draw bow around it
# for r in smile_coordinates:
#(x, y, w, h) = r
#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

#successful_frame_read, frame = video.read()
# TO show the image
#cv2.imshow("Smile Detection APP", frame)
# cv2.waitKey(1)


while True:

    # to read current frame from the video
    successful_frame_read, frame = video.read()

    # if there is an error , then abort
    if not successful_frame_read:
        break

    # to change the the current frame in black and white
    grayScale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to detect faces or to get the location of co-ordinates of rectangle enclosing a face
    face_coordinates = trained_face_algorithm.detectMultiScale(grayScale_image)

    # to get the location of co-ordinates of rectangle enclosing a smile
    smile_coordinates = trained_smile_algorithm.detectMultiScale(
        grayScale_image, scaleFactor=1.7, minNeighbors=20)

    # to display box around all the faces in the live video
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # to display box around all the smiles in the live video
    for (x, y, w, h) in smile_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # TO show the image
    cv2.imshow("Smile Detection APP", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
cv2.destroyAllWindows()

print("Code Complete")
