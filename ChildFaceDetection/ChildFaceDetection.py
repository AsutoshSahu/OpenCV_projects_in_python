
import cv2

#opencv reads colour in (B, G, R)

#parameter specifying how much the image size is reduced at each image scale
scaleFactor = 1.4
#parameter speciying how many neighbours each candidate rect should have to retain it
minNeighbours = 1

##for 1 face scaleFcator=1.4 minNeighbours=1
##for 2 face scaleFcator=1.4 minNeighbours=1
##for 3 face scaleFcator=1.5 minNeighbours=2
##for 4 face scaleFcator=1.3 minNeighbours=2
##for 5 face scaleFcator=1.6 minNeighbours=2
##for 8 face scaleFcator=1.1 minNeighbours=5

#loads the required haarcascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#loads image
img = cv2.imread('ipot.jpg')
#converting the image to grayscale as grayscale is easy for analysis
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
f = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbours)

#as eyes lies within the face therefore the for loop for eye is within for loop for face
for (x, y, w, h) in f:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 1)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2 )
        
#shows the image in a window titled ASUTOSH    
cv2.imshow('ASUTOSH', img)
#waits for any key to be pressed by the user to proceed(i.e. to destroy all windows)
cv2.waitKey(0)
cv2.destroyAllWindows()


