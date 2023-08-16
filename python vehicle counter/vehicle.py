import cv2
import numpy as np


capture = cv2.VideoCapture(r"vehicle.mp4")
min_width = 80
min_height = 80

count_line = 550

a = cv2.bgsegm.createBackgroundSubtractorMOG() #substract background and give output vehicle


def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy


detect = []
offset = 6
counter = 0


while True:
    ret,frame = capture.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    blur = cv2.GaussianBlur(grey, (3,5), 5)
    sub = a.apply(blur, kernel, -1)
    dilate = cv2.dilate(sub,np.ones((5,5)))
    
    dilateada = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilateada = cv2.morphologyEx(dilateada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilateada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame,(25,count_line),(1200,count_line),(0,0,0),3)
    

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width) and (h>=min_height)
        if not val_counter:
            continue

        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame, center, 4,(0,0,255), -1)


        for(x,y) in detect:
            if y<(count_line+offset) and y>(count_line-offset):
                counter+=1

            cv2.line(frame,(25,count_line),(1200,count_line),(142,255,0),3)
            detect.remove((x,y))
            print("Vehicle Counter :" + str(counter))


    cv2.putText(frame,"Vehicle Counter :" + str(counter), (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    

 

    cv2.imshow("Video Original",frame)

    if cv2.waitKey(8) == 27:
        break

cv2.destroyAllWindows()
capture.release()