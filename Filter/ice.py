from PIL import Image
import matplotlib.pyplot as plt

def ice(img):
    if img.mode != "RGB":
        img.convert("RGB")

    width, height = img.size
    pix = img.load()

    for w in range(width):
        for h in range(height):
            r, g, b = pix[w, h]

            pix[w, h] = min(255, int(abs(r - g - b) * 3 / 2)), \
                        min(255, int(abs(g - b - r) * 3 / 2)), \
                        min(255, int(abs(b - r - g) * 3 / 2))

    return img

if __name__ == '__main__':
    img_path = 'data/VOC2020/JPEGImages/1_5.jpg'
    img = Image.open(img_path)
    img = ice(img)
    plt.imshow(img)
    plt.title('Result')
#    plt.imsave('result/rs_ice_filter.png', img)
    plt.show()