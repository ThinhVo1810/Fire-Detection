import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('/home/huynth/ImageProcessing/data/VOC2020/JPEGImages/wildfire1.jpg')

# Mean filter
mean_filter = cv2.blur(img, (5,5))

# gaussian filter
gaussian_filter = cv2.GaussianBlur(img, (5,5), 0, 0)

# Median filter
median_filter = cv2.medianBlur(img, 3)

# Bilateral Filter
bilateral_filter = cv2.bilateralFilter(img, 9, 255, 255)

# 2D convolution
kernel = np.ones((9,9), np.float32) / 81
filter_2D = cv2.filter2D(img, -1, kernel)

titles = ['Original', 'Mean filter', 'Gaussian', 'Median', 'Bilateral', '2D convolution']

images = [img, mean_filter, gaussian_filter, median_filter, bilateral_filter, filter_2D]

for i in range(6):
    plt.subplot(3, 2, i+1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.imsave(f'result/{titles[i]}.jpg', cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()