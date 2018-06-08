import cv2

img = cv2.imread("/home/hasib/PycharmProjects/compVision/car_images/car_detect_fg3.png")
img = cv2.resize(img, (600, 400))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

dilation = cv2.dilate(opening, kernel, iterations=2)
th = dilation[dilation < 240] = 0 #threshold


cv2.imshow('initial image',img)
cv2.imshow('closing image',closing)
cv2.imshow('opening image',opening)
cv2.imshow('dilation image',dilation)
cv2.imshow('threshold output',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
