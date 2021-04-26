import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
import data_load.util as util


#general testset (including LR and GT pair)
#bicubic downsampling
# class DataloadFromFolderTest(data.Dataset):  # load test dataset
#     def __init__(self, image_dir, scale, scene_name, transform):
#         super(DataloadFromFolderTest, self).__init__()
#
#         GT_dir = os.path.join(image_dir, 'raw', 'raw-189', scene_name)
#         LR_dir = os.path.join(image_dir, 'compressed', 'compressed-189', scene_name)
#
#         GT_alist = os.listdir(GT_dir)
#         LR_alist = os.listdir(LR_dir)
#
#         GT_alist.sort()
#         LR_alist.sort()
#
#         self.GT_image_filenames = [os.path.join(GT_dir, x) for x in GT_alist]
#         self.LR_image_filenames = [os.path.join(LR_dir, x) for x in LR_alist]
#
#         self.L = len(GT_alist)
#         self.scale = scale
#         self.transform = transform  # To_tensor
#
#     def __getitem__(self, index):
#         target = []
#         GT_temp = util.modcrop(Image.open(self.GT_image_filenames[2]).convert('RGB'), self.scale)
#         target.append(GT_temp)
#         target = [np.asarray(HR) for HR in target]
#         target = np.asarray(target)
#         t, h, w, c = target.shape
#         target = target.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT']
#         if self.transform:
#             target = self.transform(target)  # Tensor, [CT',H',W']
#         target = target.view(c, t, h, w)
#
#         LR = []
#         for i in range(self.L):
#             LR_temp = util.modcrop(Image.open(self.LR_image_filenames[i]).convert('RGB'), self.scale)
#             LR.append(LR_temp)
#         LR = [np.asarray(temp) for temp in LR]
#         LR = np.asarray(LR)
#         t, h, w, c = LR.shape
#         LR = LR.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT']
#         if self.transform:
#             LR = self.transform(LR)  # Tensor, [CT',H',W']
#         LR = LR.view(c, t, h, w)
#         return LR, target
#
#     def __len__(self):
#         return 1


# real data test (only including LR video)
# unknown downsampling type
class DataloadFromFolderTest(data.Dataset):  # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        ori_dir = os.path.join(image_dir, scene_name)
        com_alist = os.listdir(ori_dir)
        com_alist.sort()
        self.com_image_filenames = [os.path.join(ori_dir, x) for x in com_alist]
        self.L = len(com_alist)
        self.scale = scale
        self.transform = transform  # To_tensor

    def __getitem__(self, index):
        com = []
        img_list = [index - 2, index - 1, index, index + 1, index + 2]
        for i in range(5):
            temp_list = img_list[i]
            if temp_list < 0:
                temp_list = 0
            elif temp_list > self.L - 1:
                temp_list = self.L - 1
            com_temp = Image.open(self.com_image_filenames[temp_list]).convert('RGB')
            com.append(com_temp)
        com = [np.asarray(temp) for temp in com]
        com = np.asarray(com)
        t, h, w, c = com.shape
        com = com.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT']
        if self.transform:
            com = self.transform(com)  # Tensor, [CT',H',W']
        com = com.view(c, t, h, w)

        return com

    def __len__(self):
        return self.L
