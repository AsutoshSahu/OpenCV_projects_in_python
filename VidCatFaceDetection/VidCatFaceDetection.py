
import cv2

#opencv reads colour in (B, G, R)
cap = cv2.VideoCapture('cats3.mp4')

#parameter specifying how much the image size is reduced at each image scale
scaleFactor = 1.1
#parameter speciying how many neighbours each candidate rect should have to retain it
minNeighbours = 2

##for 1 face scaleFcator=1.4 minNeighbours=1
##for 2 face scaleFcator=1.4 minNeighbours=1
##for 3 face scaleFcator=1.5 minNeighbours=2
##for 4 face scaleFcator=1.3 minNeighbours=2
##for 5 face scaleFcator=1.6 minNeighbours=2
##for 8 face scaleFcator=1.1 minNeighbours=5

#loads the haarcascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

while True:
    #analysing the video framewise and each frame is stored in img var
    ret, img = cap.read()
    #converting the img to grayscale as grayscale is easy for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cface = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbours)
    for (x, y, w, h) in cface:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #showing the video frame wise in a window titled ASUTOSH  
    cv2.imshow('ASUTOSH', img)
    #waits for any key to be pressed by the user to proceed
    if cv2.waitKey(0):
        break

#stops capturing
cap.release()    
cv2.destroyAllWindows()


