from PIL import Image
import matplotlib.pyplot as plt

def darkness(img):
    return img.point(lambda i: int(i ** 2 / 255))


if __name__ == '__main__':
    img_path = "/home/huynth/ImageProcessing/data/VOC2020/JPEGImages/wildfire1.jpg"
    img = Image.open(img_path)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    img = darkness(img)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imsave("result/result_darkness.jpg", img)
    plt.title('result')
    plt.axis('off')
    plt.show()