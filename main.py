import cv2 as cv
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, vertices, 255)
    masked = cv.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
 

cap = cv.VideoCapture('moving car1.mp4')

if not cap.isOpened():
    print('error,video cant open')
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    height, width = frame.shape[:2]
    roi_vertices = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    
    
    masked_blurred = region_of_interest(blurred, roi_vertices)

    # Lane detection using Canny edge detection and Hough Line Transform
    edges = cv.Canny(masked_blurred, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)
    draw_lines(frame, lines)
 

 

    width=500
    height=500
    res = cv.resize(frame, (width, height))
    
    cv.imshow('Lane Detection', res)
    
    if cv.waitKey(1) == ord('s'):
        break

cap.release()
cv.destroyAllWindows()