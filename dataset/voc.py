import os
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset
import numpy as np

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}

coco_map = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

coco_voc = {
    0: 0,
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
    255: 255
}

web_voc_nums = {
    "airplane": 500,
    "bicycle": 500,
    "bird": 500,
    "boat": 500,
    "bottle": 500,
    "bus": 500,
    "car": 500,
    "cat": 500,
    "chair": 500,
    "cow": 500,
    "dining_table": 500,
    "dog": 500,
    "horse": 500,
    "motor_bike": 500,
    "person": 500,
    "potted_plant": 500,
    "sheep": 500,
    "sofa": 500,
    "train": 500,
    "tv_monitor": 500
}

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if as_coco:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug_ascoco.txt')
            else:
                split_f = os.path.join(splits_dir, 'val_ascoco.txt')
        else:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        if pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
            else:
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"PseudoLabels/{pseudo}/rw")) for x in
                               self.images]
        if as_coco:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"cocovoc_1h_labels_{self.image_set}.npy"))
        else:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))

        self.indices = indices if indices is not None else np.arange(len(self.images))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_lvl_lbls

    def __len__(self):
        return len(self.indices)


class VOCWebSegmentation(data.Dataset):

    def __init__(self,
                 root,
                 labels,
                 web_num = None,
                 web_path = None,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        base_dir = "voc"
        web_base_dir = web_path
        voc_root = os.path.join(self.root, base_dir)
        voc_web_root = os.path.join(voc_root, web_base_dir)
        print(voc_web_root)
        self.transform = transform
        if as_coco:
            self.labels = [90]
        else:
            self.labels = labels
        self.web_num = web_num
        self.web_voc_nums = web_voc_nums
        voc_web_len = len(labels)-1
        voc_web_start = labels[1]
        print(self.labels)
        if len( os.listdir(voc_web_root) ) == (len(self.labels)-1):
            # print(len(self.labels))
            voc_web_cls = sorted(os.listdir(voc_web_root))
        else:
            voc_web_cls = sorted(os.listdir(voc_web_root))[voc_web_start-1: voc_web_start+voc_web_len-1]
        total_file = []
        idx = 0
        for cls in voc_web_cls:
            if self.web_num is not None:
                file_names = sorted(os.listdir( os.path.join(voc_web_root, cls) ))[:web_num]
            else:
                file_names = sorted(os.listdir( os.path.join(voc_web_root, cls) ))[:self.web_voc_nums[cls]]
            if as_coco:  
                self.coco_voc_dict = coco_voc
                voc_cls_idx = voc_web_start+idx
                coco_cls_idx = self.coco_voc_dict[voc_cls_idx]
                total_file += [ [os.path.join(cls, file_name), coco_cls_idx] for file_name in file_names ]
            else:
                print(str(cls)+": "+str(len(file_names)))
                total_file += [ [os.path.join(cls, file_name), voc_web_start+idx] for file_name in file_names ]
            idx += 1


        self.images = [( os.path.join(voc_web_root, path[0]) , path[1]) for path in total_file]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.img_lvl_labels = 0

        # self.indices = indices if indices is not None else np.arange(len(self.images))

    def __getitem__(self, index):
        img = Image.open(self.images[index][0]).convert('RGB')
        target =self.images[index][1]
        img_lvl_lbls = np.zeros(self.labels[-1])
        img_lvl_lbls[target-1] = 1

        if self.transform is not None:
            img, _ = self.transform(img, img)

        return img, Image.fromarray(np.zeros((img.size[0], img.size[1]))), img_lvl_lbls
    
    def __len__(self) -> int:
        return len(self.images)

class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        return full_voc


class VOCasCOCOSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, as_coco=True,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc
    
class VOCWebSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        
        full_voc = VOCWebSegmentation(root, self.labels, self.web_num, self.web_path, transform=None, indices=indices,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc
    
class VOCWebasCOCOSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCWebSegmentation(root, self.labels, self.web_num, transform=None, indices=indices, as_coco=True,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])
