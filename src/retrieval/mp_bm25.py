import time
import os
from os import system as shell

if __name__ == '__main__':

    dataset='cnndm'
    query_lang='document'
    ## build index
    cmd = ""
    cmd = f"python bm25.py "
    cmd += f"--build_index "
    cmd += f"--index_name {dataset} "
    cmd += f"--query_lang {query_lang} "
    cmd += f"--index_file /data/{dataset}/train.jsonl "
    print(cmd)
    shell(cmd)

    ## search with multi process
    
    ## train 
    os.makedirs(f"/data/{dataset}/bm25",exist_ok=True)
    total_cnt = 0
    with open(f"/data/{dataset}/train.jsonl") as f:
        for line in f:
            total_cnt += 1
    num_workers = 15
    num_samples_per_worker = (total_cnt // num_workers)+1
    for idx in range(num_workers):
        cmd = ""
        cmd += f"python bm25.py "
        cmd += f"--search_index "
        cmd += f"--index_name {dataset} "
        cmd += f"--query_lang {query_lang} "
        cmd += f"--output_file /data/{dataset}/bm25/train_{idx}.pkl "
        cmd += f"--search_file /data/{dataset}/train.jsonl "
        cmd += f"--start_index {idx*num_samples_per_worker} "
        cmd += f"--end_index {(idx+1)*num_samples_per_worker} "
        cmd += f"&"
        # print(cmd)
        shell(cmd)
    
    ## dev
    cmd = ""
    cmd += f"python bm25.py "
    cmd += f"--search_index "
    cmd += f"--index_name {dataset} "
    cmd += f"--query_lang {query_lang} "
    cmd += f"--output_file /data/{dataset}/bm25/dev.pkl "
    cmd += f"--search_file /data/{dataset}/dev.jsonl "
    shell(cmd)
## test
    cmd = ""
    cmd += f"python bm25.py "
    cmd += f"--search_index "
    cmd += f"--index_name {dataset} "
    cmd += f"--query_lang {query_lang} "
    cmd += f"--output_file /data/{dataset}/bm25/test.pkl "
    cmd += f"--search_file /data/{dataset}/test.jsonl "
    shell(cmd)
    while True:
        if len(os.listdir(f"/data/{dataset}/bm25")) == num_workers+2:
            break
        else:
            time.sleep(1000)

