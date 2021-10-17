import cv2
import matplotlib.pyplot as plt
import numpy as np
from extract_info_annotation import Anno_xml
from make_datapath import make_datapath_list
from utils.augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
                                PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
                                ToPercentCoords, Resize, SubtractMeans


class DataTransform():

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train" : Compose([ConvertFromInts(), # convert image from int to float32
            ToAbsoluteCoords(), #back annotation to normal type
            PhotometricDistort(),
            Expand(color_mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(input_size),
            SubtractMeans(color_mean)   # tru di subtract mean color BGR
            ]),   
            "val" : Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, bnd_box, label):
        return self.data_transform[phase](img, bnd_box, label)


if __name__ == '__main__':
    classes = ['fire']
    root_path = './data/VOC2020/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)

    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    # annotation information
    trans_anno = Anno_xml(classes)
    anno_infor_list = trans_anno(train_anno_list[0], width, height)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Find color mean image
#    avg_color_per_row = np.average(img, axis=0)
#    avg_color = np.average(avg_color_per_row, axis=0)
#    print(f'Color mean of image {avg_color}')

    color_mean = [ 8, 17, 32]
    input_size = 300

    #transform train image
    transform = DataTransform(input_size, color_mean)
    phase = 'train'
    img_transformed, boxes, label = transform(img, phase, anno_infor_list[:, :4], 
        anno_infor_list[:,4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

    # transform valid image
    phase = 'val'
    img_transformed, boxes, label = transform(img, phase, anno_infor_list[:, :4], 
        anno_infor_list[:,4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()