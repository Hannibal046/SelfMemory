from dataclasses import dataclass
@dataclass
class LabelSmoother:
    """copied from huggingface/transformers"""
    
    ignore_index: int = -100

    def __call__(self, logits, labels, shift_labels=False,epsilon: float = 0.1):
        import torch
        import torch.nn as nn    
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]
def get_files(prefix):
    import os
    return [os.path.join(prefix,x) for x in os.listdir(prefix)]

def split_list(ls,n):
    # assert len(ls)%n == 0
    return [ls[idx:idx+n] for idx in range(0,len(ls),n)]



from dataclasses import dataclass
@dataclass
class bpe:
    
    bpe_code_path:str=None

    def __post_init__(self):
        import fastBPE
        self.bper = fastBPE.fastBPE(self.bpe_code_path)
    def __call__(self,x):
        return self.bper.apply([x])[0]

def get_model_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])

def debpe(bpe):
    import re
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def save_config(data_args,model_args,training_args,gen_args):
    from dataclasses import asdict
    import os
    import json
    ret = {
        "data_args":{},
        # "model_args":{},
        "training_args":{},
        "gen_args":gen_args,
    }
    for k,v in vars(data_args).items():ret["data_args"][k] = v
    # for k,v in vars(model_args).items():ret["model_args"][k] = v
    for k,v in vars(training_args).items():ret["training_args"][k] = v

    with open(os.path.join(training_args.output_dir,'other_config.json'),'w') as f:
        json.dump(ret,f,indent=4)

def move_to_device(maybe_tensor, device):
    
    import torch
    import numpy as np

    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor

def update_args(args,model_args):
    from dataclasses import asdict
    import argparse
    """
    args:argparse.Namespace/dict
    # data_args:dataclass
    model_args:class
    # training_args:dataclass
    """
    # data_keys = asdict(data_args).keys()
    model_keys = vars(model_args).keys()
    # training_keys = asdict(training_args).keys()
    
    if isinstance(args,dict):
        # for k,v in args['data_args'].items():
        #     if k in data_keys:
        #         setattr(data_args,k,v)
        for k,v in args.items():
            if k in model_keys:
                setattr(model_args,k,v)
        # for k,v in args['training_args'].items():
        #     if k in training_keys:
        #         setattr(training_args,k,v)

    elif isinstance(args,argparse.Namespace):
        for key in vars(args):
            value = getattr(args,key)
            if value:
                # if key in data_keys:
                #     setattr(data_args,key,value)
                if key in model_keys:
                    setattr(model_args,key,value)
                # if key in training_keys:
                #     setattr(training_args,key,value)
    return model_args
    # return synchronize_args(data_args,model_args,training_args)

def synchronize_args(data_args,model_args,training_args):
    # import os    
    data_args.train_batch_size = training_args.train_batch_size
    # data_args.eval_batch_size = training_args.eval_batch_size
    data_args.max_trg_len = model_args.max_trg_len
    data_args.max_src_len = model_args.max_src_len
    # if 'src.vocab' in os.listdir(data_args.data_path):
    #     ## separate vocab
    #     model_args.use_joint_bpe = False
    # else:
    #     model_args.use_joint_bpe = True
    return data_args,model_args,training_args

def get_remain_time(start,max_step,cur_step):
    import time
    end = time.time()
    past = end-start
    remain = (max_step/cur_step)*past - past
    return time2float(s2hm(remain))

def time2float(t):
    """
    t: 12:45,str
    """
    h,m = t.split(":")
    h,m = int(h),int(m)
    return round(h + m/60,2)

def s2hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return  "%02d:%02d:%02d" % (h, m, s)
    # return  "%02d:%02d" % (h, m)

def s2hm(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return  "%02d:%02d" % (h, m)

def s2ms(s):
    m,s = divmod(s,60)
    return "%02d:%02d" % (m, s)

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MetricsTracer:
    def __init__(self,valid_metrics):
        if valid_metrics == 'ppl':
            self.cur_best_metrics = 1000000
            self.better = "<"
        elif valid_metrics in  ['bleu','rouge1','acc','rouge',"mlm_acc"]:
            self.cur_best_metrics = -1
            self.better = ">"
    def is_better(self,new_metrics):
        if eval(str(new_metrics)+self.better+str(self.cur_best_metrics)):
            self.cur_best_metrics = new_metrics
            return True
        else:
            return False

def dump_vocab(p,toker,direction='joint'):
    import os
    vocab = toker.dump_vocab() # dict {token:id}
    with open(os.path.join(p,direction+".vocab"),'w') as f:
        for k,v in vocab.items():
            f.write(k+'\t'+str(v)+'\n')

def get_current_gpu_usage():
    import GPUtil
    gpu = GPUtil.getGPUs()[0]
    return f"{gpu.memoryUsed}/{gpu.memoryTotal}"

def get_jsonl(f):
    import json
    return [json.loads(x) for x in open(f,encoding='utf-8').readlines()]

def get_pickle(f):
    import pickle
    return pickle.load(open(f,'rb'))

def get_txt(f):
    return [x.rstrip('\n') for x in open(f,encoding='utf-8').readlines()]

def analysis_DBS(hyps,refs,num_return_sequences=None):
    import random
    from metrics_utils import get_bleu_score
    from tqdm import tqdm 
    num_samples = len(refs)
    num_return_sequences = len(hyps[0])
    if not len(hyps)==len(refs):
        if num_return_sequences is not None:
            assert int(len(hyps)/len(refs)) == num_return_sequences
        else:
            assert len(hyps)%len(refs)==0
            num_return_sequences = int(len(hyps)/len(refs))
        hyps = [hyps[idx:idx+num_return_sequences] for idx in range(0,len(hyps),num_return_sequences)]
        
        assert len(hyps) == num_samples
    
    # avg,worst,best,random
    
    ## avg
    repeated_refs = [[refs[idx]]*num_return_sequences for idx in range(len(refs))]
    repeated_refs = [x for y in repeated_refs for x in y]
    avg_bleu = get_bleu_score([x for y in hyps for x in y],repeated_refs)

    ## best,worst,random
    best = []
    worst = []
    _random = []
    for idx in range(len(hyps)):
        group = hyps[idx]
        ref = refs[idx]

        best_bleu = 0 
        best_hyp = ""
        worst_bleu = 100
        worst_hyp = ""

        for hyp in group:
            cur_bleu = get_bleu_score([hyp],[ref])
            if cur_bleu > best_bleu:
                best_hyp = hyp
                best_bleu = cur_bleu
            if cur_bleu < worst_bleu:
                worst_bleu = cur_bleu
                worst_hyp = hyp
        
        best.append(best_hyp)
        worst.append(worst_hyp)
        _random.append(random.choice(group))
    
    best_bleu = get_bleu_score(best,refs)
    worst_bleu = get_bleu_score(worst,refs)
    random_bleu = get_bleu_score(_random,refs)

    results = {
        "best":best_bleu,
        "worst":worst_bleu,
        "avg":avg_bleu,
        "random":random_bleu,
    }
    return results#,best
    # print(f"Best:{best_bleu:.2f} Worst:{worst_bleu:.2f} Avg:{avg_bleu:.2f} Random:{random_bleu:.2f}")

def run_pool(data,process_fn,num_works = 10,verbose=True):
    from multiprocessing import Pool
    from tqdm import tqdm
    with Pool(num_works) as p:
        if verbose:
            outputs = list(tqdm(p.imap(process_fn,data),total=len(data)))
        else:
            outputs = p.imap(process_fn,data)
    return outputs

def set_seed(seed: int = 19980406):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def write_txt(file=None,data=None):
    assert file
    assert data
    with open(file,'w') as f:
        for d in data:
            f.write(d+'\n')

def write_jonsl(file=None,data=None):
    assert file
    assert data
    import json
    with open(file,'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')

def get_json_dir(file_dir):
    import json,os
    files = [os.path.join(file_dir,x) for x in os.listdir(file_dir)]
    files.sort(key=lambda x:int(x.split("/")[-1].split(".")[0]))
    return [json.load(open(x)) for x in files]


def evaluate_candidates(candidates,refs):
    from .metrics_utils import get_rouge_score
    assert len(candidates) % len(refs) == 0
    num_candidates = int(len(candidates)/len(refs))
    candidates = split_list(candidates,num_candidates)
    ## random
    from random import choice
    random_results = []
    for _ in range(5):
        random_results.append(get_rouge_score([choice(x) for x in candidates],refs))
    r1 = [x[0] for x in random_results]
    r2 = [x[1] for x in random_results]
    rl = [x[2] for x in random_results]
    print(f"random=({sum(r1)/len(r1)},{sum(r2)/len(r2)},{sum(rl)/len(rl)})")
    ## best and worst
    best_hyps = []
    worst_hyps = []
    for ref,candidate_ls in zip(refs,candidates):
        candidate_ls = [[candidate,get_rouge_score([candidate],[ref])[0]] for candidate in candidate_ls]
        candidate_ls.sort(key=lambda x:float(x[1]))
        best_hyps.append(candidate_ls[-1][0])
        worst_hyps.append(candidate_ls[0][0])
    print(f"best={get_rouge_score(best_hyps,refs)}")
    print(f"worst={get_rouge_score(worst_hyps,refs)}")


def get_gpu_usage():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used / 1024**2
