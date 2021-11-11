from PIL import Image
import matplotlib.pyplot as plt


def solarize(img):
    if img.mode != "RGB":
        img = img.convert('RGB')

    return img.point(lambda i: i ^ 0xFF if i < 128 else i)

if __name__ == '__main__':
    img = "data/VOC2020/JPEGImages/wildfire1.jpg"
    img = Image.open(img)
    img = solarize(img)
    plt.imshow(img)
    img.save('result/solarize.jpg', 'JPEG')
    plt.show()