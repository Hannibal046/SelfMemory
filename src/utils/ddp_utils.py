def set_available_port():
    import os
    import random
    
    def port_is_used(port,ip='127.0.0.1'):
        """
        test whether a port is used or not
        """
        import socket
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            s.connect((ip,port))
            s.shutdown(2)
            return True
        except:
            return False

    ports = list(range(12350,20000))
    random.shuffle(ports)

    os.environ['MASTER_ADDR'] = 'localhost'
    for port in ports:
        if not port_is_used(port):
            os.environ['MASTER_PORT'] = str(port)
            break
    else:
        raise RuntimeError("No Available Port for DDP")

def get_rank():
    import torch.distributed as dist
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

import torch
import math
class PadSequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class UnevenSequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    This is slightly different version of SequentialDistrbitedSample from 
    https://github.com/huggingface/transformers/blob/81ac45f85c35244831f11f73c09ea10eee4f953a/src/transformers/trainer_pt_utils.py
    In thie version, the datset is not evenly split. Since we don't need tensors of same shape to reduce or gather
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        import math
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas 
        indices = list(range(len(self.dataset)))
        self.indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples] ## a trick for python list ls[:infinity]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def wait_for_everyone():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        torch.distributed.barrier()
    else:
        return 

def mprint(*args,**kwargs):
    if is_main_process():
        print(*args,**kwargs)

def cleanup():
    import torch.distributed as dist
    dist.destroy_process_group()