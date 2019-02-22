import cv2
im = cv2.imread('out31.tif', cv2.IMREAD_GRAYSCALE)
imC = cv2.applyColorMap(im, cv2.COLORMAP_JET)
cv2.imshow("result",imC)
cv2.waitKey()