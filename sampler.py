import torch
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import sys
import argparser
import math
import os

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class InterleaveSampler(Sampler):
    def __init__(self, dataset, batch_size, max_iterations = None, shuffle=False):
        self.dataset = dataset
        self.datasets_num = len(self.dataset.datasets)
        self.batch_size = batch_size
        self.max_iter = max_iterations
        self.shuffle = shuffle
        self.largest_dataset_size = max([len(d) for d in self.dataset.datasets])
        self.minimum_dataset_size = min([len(d) for d in self.dataset.datasets])
        # print()

    def __iter__(self):
        # print("sampler")
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.datasets_num):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if self.shuffle:
                sampler = RandomSampler(cur_dataset)
            else:
                sampler = SequentialSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.datasets_num // 2
        samples_to_grab = self.batch_size//2
        if not self.max_iter is None:
            epoch_samples = self.max_iter*self.batch_size
        else:
            epoch_samples = self.largest_dataset_size * self.datasets_num
        
        final_sample_list = []
        for _ in range(0,epoch_samples, step):
            for i in range(self.datasets_num):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for __ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # print(i)
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_sample_list.extend(cur_samples)


        return iter(final_sample_list)

    def __len__(self):
        if not self.max_iter is None:
            return self.max_iter*self.batch_size
        else:
            return math.ceil(self.largest_dataset_size/self.batch_size) * self.datasets_num * self.batch_size



class DistributedInterleaveSampler(Sampler):
    def __init__(self, sampler, num_replicas, rank=None, shuffle=False):
        self.sampler = sampler
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(sampler.__len__() * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(self.sampler)
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    data1 = list([i for i in range(100)])
    data2 = list(i for i in range(1000, 1500))
    concate_dataset = ConcatDataset([data1, data2])
    sampler1 = InterleaveSampler(concate_dataset, world_size*2)
    distributed_sampler = DistributedInterleaveSampler(sampler1, world_size)
    print()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

    '''
    # parser = argparser.get_argparser()
    # opts = parser.parse_args()
    # opts = argparser.modify_command_options(opts)
    # opts.data_root = r"D:\\ADAXI\\Datasets\\VOC_SDR"
    # opts.replay = True
    # opts.mix = False
    # opts.task = '10-10s'
    # opts.dataset = 'voc'
    # opts.step = 1
    # train_dst, val_dst, test_dst, n_classes = get_dataset(opts, rank=0)
    # batch_size = 8
    # concate_dataset = ConcatDataset([train_dst.dataset, train_dst.replayset])
    # interleave_sampler = InterleaveSampler(concate_dataset, batch_size=8)
    # loader = DataLoader(concate_dataset, sampler=interleave_sampler, batch_size=8)
    # print()
    # interleave_sampler.__len__()
    # for _, (image, label) in enumerate(loader):
    #     # print(image)
    #     # print(label)
    #     print(_)
    # print("good")
    '''
    