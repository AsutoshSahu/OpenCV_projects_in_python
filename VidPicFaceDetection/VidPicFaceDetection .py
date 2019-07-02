
import cv2

#opencv reads colour in (B, G, R)
#loads video
cap = cv2.VideoCapture('man.mp4')

'''func detects objects draws rects of different colours using corresponding haarcascades'''
def rect(img):
    #converting the img to grayscale as grayscale is easy for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbours)
    for (x, y, w, h) in m:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2 )
            
        smile = smileCascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2 )
        

#parameter specifying how much the image size is reduced at each image scale
scaleFactor = 1.1
#parameter speciying how many neighbours each candidate rect should have to retain it
minNeighbours = 1

##for 1 face scaleFcator=1.4 minNeighbours=1
##for 2 face scaleFcator=1.4 minNeighbours=1
##for 3 face scaleFcator=1.5 minNeighbours=2
##for 4 face scaleFcator=1.3 minNeighbours=2
##for 5 face scaleFcator=1.6 minNeighbours=2
##for 8 face scaleFcator=1.1 minNeighbours=5

#loading haarcascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#loading image
img2 = cv2.imread('man.png')

while True:
    #for framewise analysis of the video
    ret, img1 = cap.read()

    rect(img1)
    rect(img2)
    
    cv2.imshow('ASUTOSH1', img1)
    cv2.imshow('ASUTOSH2', img2)

    #waits for any key to be pressed by the user to proceed
    if cv2.waitKey(0):
        break

#stops capturing
cap.release()

cv2.destroyAllWindows()


