import cv2

video = cv2.VideoCapture(0)
first_frame=None

while True:
    check,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray)


    thres_frame = cv2.threshold(delta_frame,35,255,cv2.THRESH_BINARY)[1]
    thres_frame = cv2.dilate(thres_frame,None,iterations=2)
    contours, hierachy = cv2.findContours(thres_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnts in contours:
        if cv2.contourArea(cnts) <1000:
            continue
        (x,y,w,h) = cv2.boundingRect(cnts)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    cv2.imshow("Color Image", frame)
    cv2.imshow("Gray Image", gray)
    cv2.imshow("Delta Image", delta_frame)
    cv2.imshow("Threshold Image", thres_frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()