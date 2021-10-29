import torch
import pandas as pd
import itertools
from math import sqrt


configs = {
    "num_classes" : 1, #we only have 1 class: fire
    "input_size" : 300, #SSD 300
    "bbox_aspect_num" : [4, 6, 6, 6, 4, 4], # ty le cho source 1 -> 6
    "feature_maps" : [38, 19, 10, 5, 3, 1],
    "steps" : [8, 16, 32, 64, 100, 300], # Size of default box 
    "min_size" : [30, 60, 111, 162, 213, 264],
    "max_size" : [60, 111, 162, 213, 264, 315],
    "aspect_ratios" : [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
}

class DefBox():
    def __init__(self, configs):
        self.img_size = configs["input_size"]
        self.feature_maps = configs["feature_maps"]
        self.min_size = configs["min_size"]
        self.max_size = configs["max_size"]
        self.aspect_ratios = configs["aspect_ratios"]
        self.steps = configs["steps"]

    def create_defbox(self):
        defbox_list = []

        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]

                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k

                # Small box
                s_k = self.min_size[k] / self.img_size 
                defbox_list += [cx, cy, s_k, s_k]

                # Big square box
                s_k = sqrt(s_k * self.max_size[k] / self.img_size)
                defbox_list += [cx, cy, s_k, s_k]

                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    defbox_list += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(defbox_list).view(-1, 4)
        output.clamp_(max=1, min=0)

        return output

if __name__ == "__main__":
    defbox = DefBox(configs)
    dbox_list =  defbox.create_defbox()
#    print(dbox_list)

    # To pandas format
    print(pd.DataFrame(dbox_list.numpy()))