from PIL import Image
import matplotlib.pyplot as plt

def aqua(img):
    if img.mode != "RGB":
        img.convert("RGB")

    width, height = img.size
    pix = img.load()

    for w in range(width):
        for h in range(height):
            r, g, b = pix[w, h]

            pix[w, h] = min(255, int((g - b) **2 / 128)), \
                        min(255, int((r - b) **2 / 128)), \
                        min(255, int((r - b) **2 / 128))

    return img


if __name__ == '__main__':
    img_path = "./data/VOC2020/JPEGImages/wildfire1.jpg"
    img = Image.open(img_path)
    img = aqua(img)
    plt.imshow(img)
    plt.title('result')
    plt.axis('off')
    plt.show()