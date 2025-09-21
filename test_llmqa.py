import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
import random
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import random
from typing import List, Tuple, Dict, Set
from find_path import find_paths_in_subgraph
from find_pruned_path import find_pruned_paths_in_subgraph
import time
import anthropic
import openai
import torch.optim as optim
import time
import signal
from sklearn.cluster import KMeans
import gc


openai_key='<key>'
llm_name='gpt3.5'

def chatgpt_response(api_key, input_text, model="gpt-3.5-turbo"):
    if llm_name=='gpt3.5':
        model="gpt-3.5-turbo"
    elif llm_name=='gpt4':
        model="gpt-4o-mini"
    openai.api_key = api_key
    f=0
    try_times=0
    while(f == 0 and try_times<=10):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text}
                ]
            )
            f = 1
        except:
            try_times+=1
            print("openai error, retry")
            time.sleep(2)
            if try_times>10:
                return 'openai error'
    return response['choices'][0]['message']['content']

def get_response(input_text, openai_api_key=openai_key, chatgpt_model="gpt-3.5-turbo"):
    if llm_name=='Claude':
        return Claude_response(input_text)
    elif llm_name =='gpt3.5' or llm_name =='gpt4':
        return chatgpt_response(openai_api_key, input_text, model=chatgpt_model)




def query_llm(question: str, context: str = ""):
    return context + " " + question  

def contains_answer(llm_response: str, answers: List[str]):
    return any(str(ans).lower() in llm_response.lower() for ans in answers)


if __name__ == "__main__":
    
    #dataset_name='wikimovie'
    #dataset_name='webqsp'
    #dataset_name='metaqa/1-hop'
    dataset_name='metaqa/2-hop'
    
    
    llmf='flan-'
    #llmf=''
    
    
    file_path=f""
    
    
    #sample_size=1000
    sample_size=None
    use_evidence=True
    #use_evidence=False
    
    total = 0
    correct = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if sample_size and sample_size < len(lines):
        lines = random.sample(lines, sample_size)
    
    for data in lines:
        
        
        question = data["question"]
        answers = data["answer"]
        evidence = data.get("evidence_text", "")

        prompt = question
        if use_evidence and evidence:
            prompt = f'''You are a knowledgeable assistant helping to answer questions based on evidence.
                    Given the following context:
                    {evidence}
                    Answer the question below as accurately as possible. If the answer is not in the context, make your best guess.  
                    Please return all the possible answers as a list. Given the reason of your thought.              
                    Question: {question}
                    Answer:'''
        else:
            prompt = f'''You are a knowledgeable assistant helping to answer questions.
                    Question: {question}
                    Answer:'''

        llm_response = get_response(prompt)
        
        
        if contains_answer(llm_response, answers):
            correct += 1
        total += 1
        print(correct,'/',total,'=',correct/total)
        
        train_note=f'{dataset_name}/{llmf}acc.txt'
        acc=correct/total
        with open(train_note, 'w') as f:
            f.write(f"{correct}/{total}={acc}")
        
    accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")