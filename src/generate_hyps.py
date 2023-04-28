import json,os,time,argparse,warnings,time,yaml
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES']='1'
## torch
import torch
import torch.distributed as dist
## lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelSummary,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
## transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
## own
from utils.utils import (
    LabelSmoother,
    get_remain_time,
)
from utils.metrics_utils import (
    get_rouge_score,
    get_bleu_score,
)
from utils.optim_utils import (
    get_inverse_sqrt_schedule_with_warmup
)
from utils.ddp_utils import (
    UnevenSequentialDistributedSampler,
)
from model import (
    DualEncoderPegasusForConditionalGeneration,
    DualEncoderBartForConditionalGeneration,
    DualEncoderTransformerForConditionalGeneration,
)
from train_generator import (
    MemoryDataset,
    collate_fct,
    ConditionalGenerator,
)

class MemoryDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        memory=None,
        ):
        super().__init__()
        if memory is not None:
            if len(data) != len(memory):
                assert len(memory)%len(data)==0,(len(data),len(memory))
                multiple = int(len(memory)/len(data))
                data = [[x]*multiple for x in data]
                data = [x for y in data for x in y]
            assert len(data)==len(memory),(len(data),len(memory))
            for idx in range(len(data)):
                data[idx]['memory']=memory[idx]
        
        self.data = data
    
    def __getitem__(self,index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)

class Generator(ConditionalGenerator):
    @staticmethod
    def add_model_specific_args(parent_parser):
        
        parser = parent_parser.add_argument_group("model_args")
        ## data
        parser.add_argument('--data_path', )
        parser.add_argument('--memory_path')
        parser.add_argument('--output_path')
        parser.add_argument('--config_path')
        parser.add_argument('--memory_encoding')
        parser.add_argument('--src', )
        parser.add_argument('--trg', )
        parser.add_argument('--train_max_src_len',type=int)
        parser.add_argument('--train_max_trg_len',type=int)
        ## model
        parser.add_argument('--pretrained_model_path',)
        ## generation
        parser.add_argument('--num_return_sequences',type=int)
        parser.add_argument('--num_beam_groups',type=int)
        parser.add_argument('--num_beams',type=int)
        parser.add_argument('--length_penalty',type=float)
        parser.add_argument('--diversity_penalty',type=float)
        parser.add_argument('--gen_max_len',type=int)
        parser.add_argument('--gen_min_len',type=int)
        parser.add_argument('--no_repeat_ngram_size',type=int)
        parser.add_argument('--early_stopping',type=bool)
        parser.add_argument('--top_p',type=float)
        parser.add_argument('--temperature',type=float)
        parser.add_argument('--do_sample',type=bool)
        ## training_parameters
        parser.add_argument('--per_device_eval_batch_size',type=int)
        parser.add_argument('--eval_metrics',default='rouge1')
        parser.add_argument('--logging_steps',type=int)
        parser.add_argument('--seed',type=int)
        
        return parent_parser

    def configure_model(self):
        ## tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_path)
        self.vocab_size = len(self.tokenizer)
        ## model
        if self.hparams.memory_path is not None:
            ## retrieval-aug
            if self.hparams.memory_encoding == 'concate':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.pretrained_model_path)
                # if "<MEMORY_SPLITTER>" not in self.tokenizer.vocab:
                #     special_tokens_dict = {'additional_special_tokens': ["<MEMORY_SPLITTER>"]}
                #     self.tokenizer.add_special_tokens(special_tokens_dict)
                #     self.vocab_size = len(self.tokenizer)
                #     self.model.resize_token_embeddings(len(self.tokenizer))
            elif self.hparams.memory_encoding == 'separate':
                if 'pegasus' in self.hparams.pretrained_model_path:
                    self.model = DualEncoderPegasusForConditionalGeneration.from_pretrained(self.hparams.pretrained_model_path)
                elif 'bart' in self.hparams.pretrained_model_path:
                    # config = BartConfig.from_pretrained(self.hparams.pretrained_model_path)
                    self.model = DualEncoderBartForConditionalGeneration.from_pretrained(self.hparams.pretrained_model_path)
                elif 'transformer' in self.hparams.pretrained_model_path:
                    self.model = DualEncoderTransformerForConditionalGeneration.from_pretrained(self.hparams.pretrained_model_path)
        else:
            ## vanilla seq2seq
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.pretrained_model_path)
        
        self.model.resize_token_embeddings(len(self.tokenizer))

    def test_step(self, batch, batch_idx):
        hyps = self.generate(batch)
        return hyps,batch['refs']
    
    # def on_test_start(self) -> None:
    #     self.print(self.hparams)

    def test_epoch_end(self,outputs):
        hyps,refs = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        if len(hyps) == len(refs):
            self.eval_generation(hyps,refs,stage='test')
        if self.trainer.is_global_zero:
            if self.hparams.output_path is not None:
                os.makedirs(os.path.dirname(self.hparams.output_path),exist_ok=True)
                with open(self.hparams.output_path,'w',encoding='utf-8') as f:
                    for h in hyps:
                        f.write(h.replace("\n"," ")+"\n")
    
    def setup(self,stage):
        if stage == 'test':
            data = [json.loads(x) for x in open(self.hparams.data_path,encoding='utf-8').readlines()]
            self.test_data_cnt = len(data)
        memory = None
        if self.hparams.memory_path is not None:
            mem_path = os.path.join(self.hparams.memory_path)
            memory = [x.strip() for x in open(mem_path,encoding='utf-8').readlines()]
        
        if 'context' in data[0].keys():
            for idx in range(len(data)):
                data[idx]['context'] = " [EOU] ".join(data[idx]['context'])
            
            ## persona feature
            if 'persona' in data[0].keys():
                for idx in range(len(data)):
                    persona = " [EOU] ".join(data[idx]['persona'])
                    data[idx]['context'] = persona + " [EOU] " + data[idx]['context']

        self.test_dataset = MemoryDataset(
            data = data,
            memory=memory,
        )

    def test_dataloader(self):
        if self.trainer.num_devices > 1:
            sampler = UnevenSequentialDistributedSampler(self.test_dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.collate_fct,
                                           num_workers=8, pin_memory=True,sampler=sampler)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Generator.add_model_specific_args(parser)
    args = parser.parse_args()
    config = yaml.full_load(open(args.config_path,encoding='utf-8'))
    for k,v in config.items():
        if getattr(args,k) is None:
            setattr(args,k,v)
    pl.seed_everything(args.seed,workers=True)
    model = Generator(**vars(args))
    strategy = None
    if args.accelerator == 'gpu' and torch.cuda.device_count()>1:strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy = strategy,
        replace_sampler_ddp=False,
    )
    trainer.test(model)