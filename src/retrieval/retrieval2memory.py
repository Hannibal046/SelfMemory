from tqdm import tqdm
import json
import pickle,os

dataset = 'cnndm'
memory_key='summary'
memory_bank = []
with open(f"/data/{dataset}/train.jsonl") as f:
    with tqdm(total=1207222) as pbar:
        for line in f:
            line = json.loads(line)
            memory_bank.append(line[memory_key])
            pbar.update(1)

for _split in ['dev','test','train']:
    if _split == 'train':
        bm25 = []
        for i in range(15):
            p = f"/data/{dataset}/bm25/train_"+str(i)+".pkl"
            bm25.extend(pickle.load(open(p,'rb')))
        ## in case of empty list:
        for i in range(len(bm25)):
            if len(bm25[i])==0:
                bm25[i].append(0)
                bm25[i].append(1)
        print("sanity_check:",sum([1 for idx,i in enumerate(bm25) if idx==i[0]])/len(bm25))
        for idx,lst in enumerate(bm25):
            if idx in lst:
                lst.remove(idx)
        print("sanity_check:",sum([1 for idx,i in enumerate(bm25) if idx==i[0]])/len(bm25))
    else:
        with open(f"/data/{dataset}/bm25/"+_split+".pkl",'rb') as f:
            bm25 = pickle.load(f)    
    memory = [memory_bank[idx[0]] for idx in bm25]
    os.makedirs(f"/data/{dataset}/memory/bm25/",exist_ok=True)
    
    with open(f"/data/{dataset}/memory/bm25/"+_split+".txt",'w') as f:
        for m in memory:
            m = m.replace("\r\n"," ").replace('\n'," ")
            f.write(m+'\n')
    print(_split+" done")