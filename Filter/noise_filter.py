from skimage.util import random_noise
import numpy as np
import cv2

img = cv2.imread('/home/huynth/ImageProcessing/data/VOC2020/JPEGImages/wildfire1.jpg')
noise_img = random_noise(img, mode='s&p', amount=0.2)
noise_img = np.array(255 * noise_img, dtype='uint8')

cv2.imshow("Salt and Pepper noise", noise_img)
cv2.imwrite('result/s_p_noise.jpg', noise_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()