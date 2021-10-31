import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def make_datapath_list(root_path):
    image_path_template = os.path.join(root_path, "JPEGImages", "%s.jpg")
    annotation_path_template = os.path.join(root_path, "Annotations", "%s.xml")

    train_id_names = os.path.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = os.path.join(root_path, "ImageSets/Main/val.txt")

    train_img_list = []
    train_anno_list = []

    val_img_list = []
    val_anno_list = []

    for line in open(train_id_names):
        # Xoa ky tu xuong dong va space (neu co)
        file_id = line.strip() 
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


if __name__ == '__main__':
    root_path = './data/VOC2020/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)

    print(len(val_img_list))
    print(val_img_list[0])

    # Show test imgage
    img = plt.imread(val_img_list[5])
    plt.imshow(img)
    plt.show()