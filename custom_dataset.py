import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from extract_info_annotation import Anno_xml
from make_datapath import make_datapath_list
from transform import DataTransform


class MyDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt, height, width = self.pull_item(idx)
        return img, gt

    def pull_item(self, idx):
        img_file_path = self.img_list[idx]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        # Lay ra cac annotation information
        anno_file_path = self.anno_list[idx]
        anno_info = self.anno_xml(anno_file_path, width, height)

        # tien xu ly
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:,4])

        # BGR to RGB  (height, width, channels) -> (channels, height, width)
        torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0])) #sample[0]=img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]=annotation
    #[3, 300, 300]
    # (batch_size, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.permute(0,3,2,1)
    return imgs, targets


if __name__=='__main__':

    classes = ['fire']
    root_path = './data/VOC2020/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)

    color_mean = [ 8, 17, 32]
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_anno_list, phase='train', 
    transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    val_dataset = MyDataset(val_img_list, val_anno_list, phase="val",
    transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    print(train_dataset.__getitem__(2))
    print(len(train_dataset))

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size()) 
    print(len(targets))
    print(targets[0].size()) # xmin, ymin, xmax, ymax, label