from PIL import Image
from PIL import ImageFilter
from PIL.ImageFilter import MinFilter, RankFilter, UnsharpMask
import matplotlib.pyplot as plt

# Unsharp Mask
img = Image.open('data/VOC2020/JPEGImages/wildfire1.jpg')
dimg = img.filter(UnsharpMask(radius=2, percent=150, threshold=3))

# Kernel
km = (
    -2, -1, 0,
    -1, 1, 1,
    0, 1, 2
)
k = ImageFilter.Kernel(size=(3,3), kernel=km, scale=sum(km), offset=0)
kernel_img = img.filter(k)


# Rank filter
rank_filter = img.filter(RankFilter(size=9, rank=2))

# Min filter
min_filter = img.filter(MinFilter(size=9))

images = [dimg, kernel_img, rank_filter, min_filter]
titles = ["UnsharpFilter", "Kernel", "Rankfilter", "MinFilter"]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
    plt.imsave(f'result/{titles[i]}.jpg', images[i])
    plt.xticks([]), plt.yticks([])
plt.show()