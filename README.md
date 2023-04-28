# Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory

This repository contains the source code for this paper [Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory]().

With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation (we define this as __primal problem__), previous works mainly focus on how to retrieve better memory.

However, one fundamental limitation exists for current literature: the memory is retrieved from a fixed corpus and is bounded by the quality of the corpus. Due to the finite retrieval space, bounded memory would greatly limit the potential of the memory-augmented generation model.

In this paper, by exploring the __duality__ of the primal problem: better generation also prompts better memory, we propose a framework called **Selfmem**, which iteratively adopts a retrieval-augmented generator itself to generate an unbounded memory pool and uses a memory selector to pick one generated memory for the next generation round. 

By combining the primal and dual problem, a retrieval-augmented generation model could **lift itself up** with its own output in the infinite generation space.

<div align=center>
<img src=model.svg width=75% height=75% />
</div>

---

## Setup
Our code is mainly based on ⚡ [PyTorch Lightning]() and 🤗 [Transformers](https://github.com/huggingface/transformers). 

Specifically, the model definition and tokenizer is based on 🤗, and the Trainer is from ⚡.

```bash
## firstly install torch corresponding to the CUDA
pip install transformers==4.24.0 \
            pytorch-lightning==1.8.0.post1 \
            sacrebleu==2.2.0 \
            gputil==1.4.0

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
---
## Data
For PLM we use, `BART-base` and `Pegasus`, download from huggingface model hubs and put it in the [pretrained_model](pretrained_model) folder.


For dataset we use:
- **JRC-Acquis**. We use the data version from this [paper](https://aclanthology.org/2021.acl-long.567.pdf). For downloading, we refer to this [LINK](https://drive.google.com/file/d/1iuBH_YsnL28cTYjjpSq5BgukG7QhBLs_/view) to download the data and this [script](https://github.com/jcyk/copyisallyouneed/blob/master/scripts/prepare.sh) for data pre-processing.

- **XSum** is downloaded from this [LINK](https://github.com/yixinL7/BRIO).

- **DailyDialog** is downloaded from this [LINK](http://yanran.li/dailydialog.html).

- **BigPatent** is available [here](https://evasharma.github.io/bigpatent/).

After downloading the data, make it in the format of `Jsonline`  and put it in the `data` folder.

For initial memory retrieval, we use `ElasticSearch` to conduct first-stage memory retrieval based on BM25 score.

We also provide the final hypothesis and reference in the [output](output) directory for potential need. For evaluation scripts, please refer to [metrics_utils.py](src/utils/metrics_utils.py)

---
## Retrieval-Augmented Generator
Here we use JRC-Acqius EnDe dataset as example:
```bash
cd your/work/dir/src

## train a vanilla Transformer model
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16

## Transformer-Joint
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16 \
    --memory_encoding concate \
    --memory_dir ../data/jrc_ende/memory/bm25 

## Transformer-Dual
python train_generator.py \
    --config config/jrc_ende/train_generator.yaml \
    --precision 16 \
    --memory_encoding separate \
    --memory_dir ../data/jrc_ende/memory/bm25 
```


## Memory Selector
First we use the trained generator to generate candidates
```bash
cd your/work/dir/src

python generate_hyps.py \
	--config config/jrc_ende/generate_hyps.yaml \
    --num_beams 50 --num_return_sequences 50 \
	--data_path ../data/jrc_ende/test.jsonl \
	--pretrained_model_path your/trained/model/path
	--memory_encoding concate \
	--memory_path ../data/jrc_ende/memory/bm25/test.txt \
	--output_path output.txt
```
Then we using [this code](https://github.com/facebookresearch/fairseq/tree/main/examples/discriminative_reranking_nmt) to train a memory selector.

## Lift Yourself Up
Here is the pseudo code for the whole process:
```python
generator = Trainer(model_input,memory)
candidates = generator(model_input,memory)
selector = Trainer(model_input,candidates)

for _ in range(iteration_k):
    candidates = generator(model_input,memory)
    memory = selector(model_input,candidates)
    hypothesis = generator(model_input,memory)
    current_score = metrics(hypothesis,reference)
```
