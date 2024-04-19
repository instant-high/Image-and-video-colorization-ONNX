import cv2
import numpy as np

def adjust_saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,1] *= factor
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result


