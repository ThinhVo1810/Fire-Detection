import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import os

class warmth_filter:
    """warmth-filter--
        This filter will improve all tones and absorb
        the blues by adding a slight yellow tint.
        Ideally to be used on portraits.
    """

    def __init__(self):
        # create look-up tables for increasing and decreasing red and blue resp.
        self.increaseChannel = self.LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decreaseChannel = self.LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)
        #cv2.imshow("Original", img_rgb)
        r,g,b = cv2.split(img_rgb)
        b = cv2.LUT(b, self.increaseChannel).astype(np.uint8)
        r = cv2.LUT(r, self.decreaseChannel).astype(np.uint8)
        img_rgb = cv2.merge((r,g,b))

        # saturation increased
        h,s,v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        s = cv2.LUT(s, self.increaseChannel).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2RGB)

    def LUT_8UC1(self, x, y):
        #Create look-up table using scipy spline interpolation function
        spl = UnivariateSpline(x, y)
        return spl(range(256))


if __name__=='__main__':
    class_object = warmth_filter()
    result = '/Users/nguyenthaihuuhuy/Filter Image/result'
    file_name = "Lenna.png" #File_name will come here
    res = class_object.render(file_name)
    cv2.imwrite(os.path.join(result,"warm_image.jpg"), res)
    cv2.imshow("Warm-Filter version", res)
    cv2.waitKey(0)