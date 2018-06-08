import cv2
import numpy as np

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)

def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=35, counter = 0):
    matches = []


    im, contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)


    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue


        centroid = get_centroid(x, y, w, h)
        matches.append(((x, y, w, h), centroid))

        cv2.putText(frame, 'car', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 70, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)


    return matches


cap = cv2.VideoCapture('input.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

while True:

    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    frame = cv2.resize(frame, (600, 400))
    fgmask = cv2.resize(fgmask, (600, 400))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel, iterations=2)
    th = dilation[dilation < 240] = 0

    detect_vehicles(dilation)

    cv2.imshow('original',frame)
    cv2.imshow('fg',dilation)

    k = cv2.waitKey(30) & 0xFF
    if k ==  27:
        break

cap.release()
cv2.destroyAllWindows()

