
import cv2

#opencv reads colour in (B, G, R)

#scalefactor is the parameter specifying how much the image size is reduced at each image scale
#minNeighbours is the parameter speciying how many neighbours each candidate rect should have to retain it 

##for 1 face scaleFcator=1.4 minNeighbours=1
##for 2 face scaleFcator=1.4 minNeighbours=1
##for 3 face scaleFcator=1.5 minNeighbours=2
##for 4 face scaleFcator=1.3 minNeighbours=2
##for 5 face scaleFcator=1.6 minNeighbours=2
##for 8 face scaleFcator=1.1 minNeighbours=5

#loads the haarcascade(kind of dataset)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#loads image
img1 = cv2.imread('anw5.JPG')
#converting the image to grayscale as grayscale is easy for analysis
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#                               scalefactor, minNeighbours
f1 = faceCascade.detectMultiScale(gray1, 1.1, 5)

#loads image
img2 = cv2.imread('3.jpg')
#converting the image to grayscale as grayscale is easy for analysis
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
f2 = faceCascade.detectMultiScale(gray2, 1.1, 5)

#for detecting 5 faces
for (x1, y1, w1, h1) in f1:
    cv2.rectangle(img1, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
#for detecting 3 faces
for (x2, y2, w2, h2) in f2:
    cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

#shows the images in a windows titled ASUTOSH1 and ASUTOSH2    
cv2.imshow('ASUTOSH1', img1)
cv2.imshow('ASUTOSH2', img2)
#waits for any key to be pressed by the user to proceed(i.e. to destroy all windows)
cv2.waitKey(0)
cv2.destroyAllWindows()


