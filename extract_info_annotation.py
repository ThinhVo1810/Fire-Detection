from numpy.core.fromnumeric import shape
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from make_datapath import make_datapath_list
import matplotlib.pyplot as plt

class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        # Include image annotation
        ret = []
        #read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            #information for bouding box
            bndbox = []
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1
                if pt == "xmin" or pt == "xmax":
                    pixel /= width 
                else:
                    pixel /= height
                bndbox.append(pixel)
            label_id = self.classes.index(name)
            bndbox.append(label_id)
            ret += [bndbox]
        
        return np.array(ret) # [[xmin,ymin,xmax, ymax, label_id]]


if __name__ == "__main__":
    classes = ["fire"]
    
    annotation_xml = Anno_xml(classes)

    root_path = "./data/VOC2020/"
    train_img_list, train_annotation_list, \
    val_img_list, val_annotation_list = make_datapath_list(root_path)
    idx = 1
    img_file_path = val_img_list[idx]

    img = cv2.imread(img_file_path)
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    plt.show()
    height, width, channels = img.shape
    print(f'Size image: {height} {width} {channels}')

    annotation_infor = annotation_xml(val_annotation_list[idx], width, height)
    print(annotation_infor)    
