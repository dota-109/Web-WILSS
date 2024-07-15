import torch
import numpy as np
import bisect
import os
from PIL import Image
from torch.fft import fft2
from torch.fft import ifft2

def image_labels(dataset):
    images = []
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        images.append(cls)
    return images


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def group_images_bkg(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls, count_classes = np.unique(np.array(dataset[i][1]), return_counts=True)
        count = 0
        if all(x in labels_cum for x in cls):
            for j, cl in enumerate(cls):
                if cl == 0 or cl == 255:
                    count += count_classes[j]
            for x in cls:
                if x in labels:
                    idxs[x].append((i, count))
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.indices)


class ConcatDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample
    

class Replayset(torch.utils.data.Dataset):
    """
    A dataset that return the flickr downloaded dataset.
    Arguments:
        path (string): dir path of replay images
    """

    def rfft(self, x):
        t = fft2(x, dim = (-2,-1))
        return torch.stack((t.real, t.imag), -1)

    def irfft(self, x):
        return ifft2(torch.complex(x[...,0], x[...,1]), dim = (-2,-1))
    
    def extract_ampl_phase(self, fft_im):
    # fft_im: size should be bx3xhxwx2
        fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
        return fft_amp, fft_pha
    
    def low_freq_mutate(self,  amp_src, amp_trg, L=0.01 ):
        _, _, h, w = amp_src.size()
        b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
        amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
        amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
        amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
        amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
        return amp_src
    

    def __init__(self, path, labels_old, labels, num_per_class=2, transform=None):
        # import os
        print("Adding replay images")
        self.base_path = path
        self.labels_old = labels_old
        self.labels = labels
        self.num = num_per_class
        assert self.num>0, "number of per class should be larger than 0."
        self.transform = transform
        self.replay_lists = []
        self._get_datalist()
        # self.psc_real = 
        self.psc_real_t = torch.from_numpy(np.load("./psc_real_val.npy").astype(np.float32))
        
        print(f"Loading path: {self.base_path}.")
        print(f"Finish loading {len(self.replay_lists)} images, {len(self.replay_lists)//len(self.labels_old[1:])} images per class.")
        
    def _get_datalist(self):
        files = os.listdir(self.base_path)
        cls_idx = 0
        for i in range(len(files)):
            full_path = os.path.join(self.base_path, files[i] + "/train_fullPath.txt")
            # full_path = os.path.join(self.base_path, files[i])
            with open(full_path, 'r') as f:
                file_names = [x[:-1].split(' ') for x in f.readlines()]
            # print("Check the label path !!!")
            # print(file_names[0])
            tmp = [ (x[0][:], x[1][:], files[i]) for x in file_names]
            tmp_len = len(tmp)
            if not self.num>tmp_len:
                self.replay_lists+=(tmp[:self.num])
            else:
                # print(f"class {files[i-1]} not satisfaies the number {self.num}, actual number is {tmp_len}")    
                self.replay_lists+=(tmp)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.replay_lists[index][0]).convert('RGB')
        target = Image.open(self.replay_lists[index][1])
        # target = Image.open(self.replay_lists[index][0]).convert('L')
        assert self.transform is not None, "transformation of replayset is none"
        img, target = self.transform(img, target)
        l1h = np.zeros(len(self.labels)+len(self.labels_old)-2)
        target_cls = np.unique(target)
        target_cls = target_cls[target_cls!=0]
        for _ in target_cls:
            l1h[_-1] = 1
        return img, target, l1h

    def __len__(self):
        return len(self.replay_lists)