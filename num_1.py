import cv2
import numpy as np

img = cv2.imread("variant-7.jpg")
processed = np.flip(img, axis=(0, 1))
    
cv2.imwrite("processed_marker.jpg", processed)
cv2.imshow("Result", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()