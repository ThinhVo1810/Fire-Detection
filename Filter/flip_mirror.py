import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'data/VOC2020/JPEGImages/tree-fire.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_mirror = np.fliplr(img_rgb)
img_flip = np.flipud(img_rgb)

images = [img_rgb, img_mirror, img_flip]
titles = ['Original', 'Mirror', 'Flip']
for i in range(3):
    plt.subplot(1,3,i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    if i > 0:
        plt.imsave(f'result/{titles[i]}.jpg', images[i])
plt.show()