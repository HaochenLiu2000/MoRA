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


def print_memory(device='cuda:0'):
    
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    print(f"allocated: {allocated:.2f} MB")
    
def safe_cleanup(device):
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()



km=True
hops_communicate=True
moe=True

phase='test'
#phase='train'

openai_key=""

llm='llama2-7bchat'
llm='flant5-3b'

if llm=='llama2-7bchat':
    tokenizer = LlamaTokenizer.from_pretrained("./llama2-7bchat")
    model = LlamaForCausalLM.from_pretrained("./llama2-7bchat")
elif llm=='llama2-13bchat':
    tokenizer = LlamaTokenizer.from_pretrained("./llama2-13bchat")
    model = LlamaForCausalLM.from_pretrained("./llama2-13bchat")
elif llm=="flant5-3b":
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
elif llm=="flant5-11b":
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

tokenizer.pad_token_id = 0    

#dataset_name='wikimovie'
#dataset_name='webqsp'
#dataset_name='metaqa/1-hop'
dataset_name='metaqa/2-hop'
train_file=f"{dataset_name}/train.json"
dev_file=f"{dataset_name}/dev.json"
test_file=f"{dataset_name}/test.json"

if dataset_name=='webqsp':
    with open(f"webqsp/entity2name.json") as f:
        entities_names = json.load(f)


def map_entity(e_text, unknown_counter, unknown_entity_map, answer_dict=None):
    if answer_dict is not None:
        if e_text in answer_dict:
            return answer_dict[e_text]
    if e_text in entities_names:
        return entities_names[e_text]
    if e_text not in unknown_entity_map:
        unknown_entity_map[e_text] = f"<unknown{unknown_counter[0]}>"
        unknown_counter[0] += 1
    return unknown_entity_map[e_text]

def simplify_relation(rel_text):
    return rel_text.split('.')[-1].strip('<>')


def build_kg_dict(subgraph_triplets):
    kg_dict = {}
    for head, relation, tail in subgraph_triplets:
        if head not in kg_dict:
            kg_dict[head] = {}
        if relation not in kg_dict[head]:
            kg_dict[head][relation] = []
        if tail not in kg_dict[head][relation]:
            kg_dict[head][relation].append(tail)

        reverse_relation = f'~{relation}'
            
        if tail not in kg_dict:
            kg_dict[tail] = {}
        if reverse_relation not in kg_dict[tail]:
            kg_dict[tail][reverse_relation] = []
        if head not in kg_dict[tail][reverse_relation]:
            kg_dict[tail][reverse_relation].append(head)

    return kg_dict



def find_subgraph(question, query_entities, answer, metakg, k):
    
    if isinstance(answer, str):
        answer = {answer}
    else:
        answer = set(answer)

    visited_entities = set(query_entities)
    subgraph_triplets = []
    frontier = set(query_entities)

    for hop in range(k):
        next_frontier = set()

        for head in frontier:
            if head not in metakg:
                continue
            for relation, tails in metakg[head].items():
                for tail in tails:
                    subgraph_triplets.append([head, relation, tail])
                    if tail not in visited_entities:
                        next_frontier.add(tail)

        visited_entities.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    subgraph_entities = list(visited_entities)
    return subgraph_entities, subgraph_triplets



def normalize_triplets(triplets):
    normalized = []
    seen = set()

    for head, relation, tail in triplets:
        if relation.startswith('~'):
            relation = relation[1:]
            head, tail = tail, head

        triplet = (head, relation, tail)
        if triplet not in seen:
            seen.add(triplet)
            normalized.append([head, relation, tail])

    return normalized



    
# === MLP Modules ===
class ScoreNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class PromptNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)


# === Candidate Edge Construction ===
def get_candidate_triplets(
    current_entities: List[str],
    kg_dict: Dict[str, Dict[str, List[str]]],
    visited_triplets: Set[Tuple[str, str, str]],
    subgraph_triplets: List[List]
):
    candidates = []
    seen = set()
    for head in current_entities:
        if head not in kg_dict:
            continue
        
        for rel, tails in kg_dict[head].items():
            for tail in tails:
                if rel[0] != '~':
                    triplet = (head, rel, tail)
                    lookup = [head, rel, tail]
                else:
                    triplet = (tail, rel[1:], head)
                    lookup = [tail, rel[1:], head]

                if triplet in visited_triplets or triplet in seen:
                    continue
                idx = subgraph_triplets.index(lookup)
                try:
                    idx = subgraph_triplets.index(lookup)
                    candidates.append((triplet, idx))
                    seen.add(triplet)
                except ValueError:
                    continue

    return candidates




def split_to_agents_semantic_with_embedding_classifier(
    candidates: List[Tuple[Tuple[str, str, str], int]],
    num_agents: int,
    router: torch.Tensor,
    batch_size: int
):
    
    all_batch_outputs = []

    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i + batch_size]
        triplet_texts = [f"{h} {r} {t}" for (h, r, t), _ in batch_candidates]

        with torch.no_grad():
            embeddings = llm_get_embedding_batch(
                soft_prompts=[None] * num_agents,
                agent_id=None,
                inputs_text=triplet_texts,
                num_agents2=num_agents
            )

        if isinstance(embeddings, str) and embeddings == 'OOM error':
            print("too many candidates")
            return 'OOM error'

        probs = router(embeddings)
        
        all_batch_outputs.append((batch_candidates, probs))
    return all_batch_outputs 
    
    




def llm_get_embedding_batch(soft_prompts, agent_id, inputs_text, num_agents2, agent_tokens=None, hops_communicate_tokens=None):
    
    safe_cleanup(device)
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_embeds = model.get_input_embeddings()(input_ids)
    batch_size, seq_len, hidden_size = input_embeds.size()
    
    if moe:
        if agent_id==None:
            batch_input_embeds = input_embeds
            batch_attention_mask = attention_mask
        else:
            if hops_communicate and (hops_communicate_tokens is not None):
                batch_input_embeds = torch.cat([hops_communicate_tokens.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1),agent_tokens[agent_id].unsqueeze(0).expand(batch_size, -1, -1), input_embeds], dim=1) 
                prompt_mask = torch.ones((batch_size, 1 + agent_tokens[agent_id].size(0)), dtype=attention_mask.dtype, device=attention_mask.device)
                batch_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1) 
            else:
                batch_input_embeds = torch.cat([agent_tokens[agent_id].unsqueeze(0).expand(batch_size, -1, -1), input_embeds], dim=1)
                prompt_mask = torch.ones((batch_size, agent_tokens[agent_id].size(0)), dtype=attention_mask.dtype, device=attention_mask.device)
                batch_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
    
    
    decoder_input_ids = torch.tensor([[0]]*len(inputs_text)).to(device)
    
    flag=0
    for attempt in range(3):
        try:
            outputs = model(
                inputs_embeds=batch_input_embeds,
                attention_mask=batch_attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            )
            flag=1
            break
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            time.sleep(2)
    
    if flag==0:
        print(f"CUDA OOM...")
        return 'OOM error'
        
    
    if llm=="llama2-7bchat" or llm=="llama2-13bchat":
        last_hidden = outputs.hidden_states[-1]
        _ = outputs.hidden_states[-1].detach().cpu()
    elif llm=="flant5-3b" or llm=="flant5-11b":
        last_hidden = outputs.encoder_last_hidden_state
        _ = outputs.encoder_last_hidden_state.detach().cpu()
    mask = batch_attention_mask.unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1)
    embedding = summed / (count)
    safe_cleanup(device)
    return embedding


def run_reasoning(
    question: str,
    answer: List[str],
    query_entities: List[str],
    subgraph_triplets: List[List[str]],
    subgraph_entities: List[str],
    positive_entities: Set,
    positive_triplet_indices: Set,
    kg_dict: Dict[str, Dict[str, List[str]]],
    network: nn.Module,
    num_agents: int = 4,
    max_hop: int = 3,
    top_k: int = 10,
    batch_size: int = 30,
    phase: str = 'test'
):
    
    visited_triplets = set()
    current_entities = query_entities
    visited_entities = set(query_entities)
    soft_prompts = [None for _ in range(num_agents)]
    hops_sum = [None for _ in range(num_agents)]
    all_helpful_triplets = []
    all_losses = []
    last_hop_num_agents2=0
    for hop in range(max_hop):
        safe_cleanup(device)
        candidates = get_candidate_triplets(current_entities, kg_dict, visited_triplets, subgraph_triplets)
        
        if not candidates:
            break
        
        num_agents2=num_agents
        if km:
            if moe:
                all_batches = split_to_agents_semantic_with_embedding_classifier(candidates, num_agents, network.expert_classifier, batch_size)
                if all_batches=='OOM error':
                    return 0
                agent_tokens=network.agent_tokens

        
        safe_cleanup(device)
        
        if random_agent:
            trainable_agent_id = random.randint(0, num_agents2 - 1)
        
        if moe:
            triplets_flat = [triplet for triplet, _ in candidates]
            triplet_indices_flat = [index for _, index in candidates]
            scores_flat = []
            selected_indices=[]
            
            hop_embs = [[] for _ in range(num_agents2)]
            for ii in range(len(all_batches)):
                batch=all_batches[ii]
                batch_candidates=batch[0]
                probs=batch[1]
                triplets_batch = triplets_flat[ii*batch_size:ii*batch_size+batch_size]
                triplet_indices_batch = triplet_indices_flat[ii*batch_size:ii*batch_size+batch_size]
                
                text_batch=[]
                for idx in range(len(batch_candidates)):
                    triplet=batch_candidates[idx][0]
                    if len(triplet)==3:
                        h, r, t = triplet
                    else:
                        return 'error'
                    text = f"Please evaluate if the triplet is relevant to the given question.\nQuestion: {question}\nTriplet: ({h}, {r}, {t})"
                    text_batch.append(text)
                scores_list_batch = []
                for agent_id in range(num_agents2):
                    if hops_communicate and (hops_sum[0] is not None):
                        c_tokens=network.hops_communicate(hops_sum[agent_id])
                        emb1 = llm_get_embedding_batch(soft_prompts, agent_id, text_batch, last_hop_num_agents2, agent_tokens, c_tokens)
                    else:
                        emb1 = llm_get_embedding_batch(soft_prompts, agent_id, text_batch, last_hop_num_agents2, agent_tokens)
                    
                    if emb1=='OOM error':
                        if phase == 'train':
                            return 'OOM error','OOM error'
                        else:
                            return 'OOM error'
                    
                    hop_embs[agent_id].append(emb1.detach())
                    
                    scores = network.score_net_list[hop](emb1)
                    scores_list_batch.append(scores.unsqueeze(-1))
                    
                    
                
                if hops_communicate:
                    with torch.no_grad():
                        for agent_id in range(num_agents2):
                            all_embs = torch.cat(hop_embs[agent_id], dim=0)
                            mean_emb = all_embs.mean(dim=0)
                            hops_sum[agent_id]=mean_emb
                
                scores_list_batch = torch.cat(scores_list_batch,dim=-1)
                
                scores = (probs*scores_list_batch).sum(dim=1)
                if phase=='train':
                    labels = [1 if idx in positive_triplet_indices else 0 for idx in triplet_indices_batch]
                    positive_indices = [i for i, l in enumerate(labels) if l == 1]
                    negative_indices = [i for i, l in enumerate(labels) if l == 0]
                    pos_sample_size = min(len(positive_indices), top_k)
                    neg_sample_size = min(len(negative_indices), top_k)
                    pos_selected = random.sample(positive_indices, pos_sample_size) if pos_sample_size > 0 else []
                    neg_selected = random.sample(negative_indices, neg_sample_size) if neg_sample_size > 0 else []
                    
                    selected_indices_batch = pos_selected+neg_selected
                    selected_scores = scores[selected_indices_batch]
                    
                    selected_labels = torch.tensor([labels[i] for i in selected_indices_batch], device=scores.device, dtype=torch.float)  # [top_k]
                    selected_indices+=selected_indices_batch
                    optimizer.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(selected_scores, selected_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    del emb1, scores
                    safe_cleanup(device)
                else:
                    scores_flat.append(scores) 
            
            if phase=='test':
                scores = torch.cat(scores_flat, dim=0)
                selected_indices = torch.topk(scores, k=min(top_k, len(scores)))[1].tolist()
                
            selected_triplets = [candidates[i][0] for i in selected_indices]
            all_helpful_triplets.extend(selected_triplets)
            for triplet in selected_triplets:
                visited_triplets.add(triplet)
            next_entities = []
            for h, _, t in selected_triplets:
                if h not in visited_entities:
                    next_entities.append(h)
                    visited_entities.add(h)
                if t not in visited_entities:
                    next_entities.append(t)
                    visited_entities.add(t)
            current_entities = list(set(next_entities))
            
            
            continue        
                
                
    if phase == 'train':
        if len(all_losses)!=0:
            total_loss = sum(all_losses) / len(all_losses)
        else:
            total_loss=0
        return all_helpful_triplets, total_loss
    else:
        return all_helpful_triplets


class TripletRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_agents: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_agents)

    def forward(self, embeddings: torch.Tensor):
        logits = self.linear(embeddings)
        probs = F.softmax(logits, dim=-1)
        return probs

class Multi_Agent_Net(nn.Module):
    def __init__(self, max_hop):
        super().__init__()
        
        if llm=='llama2-7bchat':
            emb_dim=4096
        if llm=='flant5-3b':
            emb_dim=2048
        
        if km==False:
            self.score_net_list = torch.nn.ModuleList([ScoreNet(emb_dim) for i in range(max_hop)])
            self.prompt_net_list = torch.nn.ModuleList([PromptNet(emb_dim, emb_dim) for i in range(max_hop*2)])
        else:
            
            if moe:
                self.expert_classifier=TripletRouter(emb_dim, 3)
                self.agent_tokens=nn.Parameter(torch.randn(3, 5, emb_dim))
                self.score_net_list = torch.nn.ModuleList([ScoreNet(emb_dim) for i in range(max_hop)])
                if hops_communicate:
                    self.hops_communicate = nn.Linear(emb_dim, emb_dim)
            else:
                self.score_net_list = torch.nn.ModuleList([ScoreNet(emb_dim) for i in range(max_hop)])
                self.prompt_net_list = torch.nn.ModuleList([PromptNet(emb_dim, emb_dim) for i in range(max_hop)])
                self.topic_prompt=PromptNet(emb_dim, emb_dim)
    def forward(self, x):
        return 0
    

def handler(signum, frame):
    raise TimeoutError("Timeout")    

signal.signal(signal.SIGALRM, handler)


# === Usage ===
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    if llm=='flant5-3b':
        llmf='flan-'
    else:
        llmf=''
    if moe:
        num_agents=3
    else:
        num_agents=4
    
    
    #random_agent=True
    random_agent=False
    if phase=='train':
        
        if dataset_name=='wikimovie':
            save_size=5000
        elif dataset_name=='webqsp':
            save_size=500
        else:
            save_size=10000
        if dataset_name=='webqsp' or dataset_name=='wikimovie':
            max_hop=2
        else:
            max_hop=int(dataset_name[-5])
        network = Multi_Agent_Net(max_hop).to(device)
        
        
        
        optimizer = optim.Adam(network.parameters(), lr=1e-4)
        network.train()
        num_epochs = 50
         
        for epoch in range(num_epochs):
            top_k=5
            total_loss = 0
            p=0
            q=0

            if dataset_name!='webqsp' and dataset_name!='wikimovie':

                train_file=f"{dataset_name}/train.jsonl"
                dev_file=f"{dataset_name}/dev.jsonl"
                test_file=f"{dataset_name}/test.jsonl"
                with open('./metaqa/metaqa_kg.pickle', 'rb') as f:
                    metakg = pickle.load(f)
                train_data=[]
                dev_data=[]
                test_data=[]
                k=int(dataset_name[-5])
                with open(train_file, "r") as f:
                    total_lines = sum(1 for _ in f)
                gc.collect()
                with open(os.path.expanduser(train_file)) as f:
                    for line in tqdm(f, total=total_lines,desc=f"Epoch{epoch}"):
                        p+=1
                        q+=1
                        if not line:
                            continue
                        qus = json.loads(line)
                        question = qus["question"]
                        query_entities = qus["entity_set"]
                        answer = qus["Label"]
                        subgraph_entities,subgraph_triplets=find_subgraph(question,query_entities,answer,metakg,k)
                        subgraph_triplets=normalize_triplets(subgraph_triplets)
                        positive_entities, positive_triplet_indices=find_paths_in_subgraph(subgraph_triplets, query_entities, answer, k)        

                        kg_dict = build_kg_dict(subgraph_triplets)

                        helpful_triplets, loss = run_reasoning(
                            question, answer, query_entities, subgraph_triplets, subgraph_entities, positive_entities, positive_triplet_indices, kg_dict,
                            network, num_agents=num_agents, max_hop=max_hop, top_k=top_k, batch_size=20, phase='train'
                        )
                        if helpful_triplets=='OOM error':
                            print('OOM error')
                            continue
                        if p>=save_size:
                            if moe:
                                if hops_communicate:
                                    if dataset_name=='metaqa/1-hop':
                                        torch.save(network.state_dict(), f'metaqa/1-hop/{llmf}hc-moe-model{epoch}-{q}.pth')
                                    elif dataset_name=='metaqa/2-hop':
                                        torch.save(network.state_dict(), f'metaqa/2-hop/{llmf}hc-moe-model{epoch}-{q}.pth')
                                    else:
                                        torch.save(network.state_dict(), f'{dataset_name}/{llmf}hc-moe-model{epoch}-{q}.pth')
                                else:
                                    if dataset_name=='metaqa/1-hop':
                                        torch.save(network.state_dict(), f'metaqa/1-hop/moe-model{epoch}-{q}.pth')
                                    elif dataset_name=='metaqa/2-hop':
                                        torch.save(network.state_dict(), f'metaqa/2-hop/moe-model{epoch}-{q}.pth')
                                    else:
                                        torch.save(network.state_dict(), f'{dataset_name}/moe-model{epoch}-{q}.pth')
                            else:
                                if dataset_name=='metaqa/1-hop':
                                    torch.save(network.state_dict(), f'metaqa/1-hop/model{epoch}-{q}.pth')
                                elif dataset_name=='metaqa/2-hop':
                                    torch.save(network.state_dict(), f'metaqa/2-hop/model{epoch}-{q}.pth')
                                else:
                                    torch.save(network.state_dict(), f'{dataset_name}/model{epoch}-{q}.pth')
                            p-=save_size
                        total_loss += loss

            else:
                with open(train_file, "r") as f:
                    total_lines = sum(1 for _ in f)
                gc.collect()
                with open(train_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_lines,desc=f"Epoch{epoch}"):
                        p+=1
                        q+=1
                        item = json.loads(line)
                        if dataset_name=='webqsp':
                            question = item['question']
                            answer_dict={}
                            for answer in item['answers']:
                                answer_dict[answer['kb_id']]=answer['text']
                            
                            unknown_entity_map = {}
                            unknown_counter = [1]
                            answer=[map_entity(answer['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for answer in item['answers']]
                            query_entities = [map_entity(entity['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for entity in item['entities']]
                            subgraph_entities = [map_entity(entity['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for entity in item['subgraph']['entities']]

                            subgraph_triplets = []
                            for triplet in item['subgraph']['tuples']:
                                
                                h = map_entity(triplet[0]['kb_id'], unknown_counter, unknown_entity_map, answer_dict)
                                r = simplify_relation(triplet[1]['text'])
                                t = map_entity(triplet[2]['kb_id'], unknown_counter, unknown_entity_map, answer_dict)
                                subgraph_triplets.append([h, r, t])
                            
                            unknown_counter_merge = [1]
                            subgraph_triplets, positive_entities, positive_triplet_indices=find_pruned_paths_in_subgraph(subgraph_triplets, query_entities, answer, unknown_counter_merge, max_hops=2)
                            
                        else:
                            question = item['question']
                            answer=[answer['text'] for answer in item['answers']]
                            query_entities=[entity['text'] for entity in item['entities']]
                            subgraph_entities=[entity['text'] for entity in item['subgraph']['entities']]
                            subgraph_triplets=[[x['text'] for x in triplet] for triplet in item['subgraph']['tuples']]

                            positive_entities, positive_triplet_indices=find_paths_in_subgraph(subgraph_triplets, query_entities, answer, max_hops=2)
                        kg_dict = build_kg_dict(subgraph_triplets)

                        helpful_triplets, loss = run_reasoning(
                            question, answer, query_entities, subgraph_triplets, subgraph_entities, positive_entities, positive_triplet_indices, kg_dict,
                            network, num_agents=num_agents, max_hop=max_hop, top_k=top_k, batch_size=20, phase='train'
                        )

                        if helpful_triplets=='OOM error':
                            print('OOM error')
                            continue
                        if p>=save_size:
                            if moe:
                                if hops_communicate:
                                    torch.save(network.state_dict(), f'{dataset_name}/{llmf}hc-moe-model{epoch}-{q}.pth')
                                else:
                                    torch.save(network.state_dict(), f'{dataset_name}/moe-model{epoch}-{q}.pth')
                            else:
                                torch.save(network.state_dict(), f'{dataset_name}/model{epoch}-{q}.pth')
                            p-=save_size
                            total_loss += loss

        exit()



            
            
    elif phase=='test':
        
        top_k=20
        with torch.no_grad():
            test_epoch=1
            test_num=10000
            if moe:
                if hops_communicate:
                    save_file=f"{dataset_name}/{llmf}hc-moe-save_answer_km_{test_epoch}_{test_num}.json"
            llm_name = "chatgpt"
            if dataset_name=='webqsp' or dataset_name=='wikimovie':
                k=2
            else:
                k=int(dataset_name[-5])
            
            network = Multi_Agent_Net(k).to(device)
            network.eval()
            if moe:
                if hops_communicate:
                    state_dict = torch.load(f'{dataset_name}/{llmf}hc-moe-model{test_epoch}-{test_num}.pth')
            network.load_state_dict(state_dict)  
            
            
            correct=0
            p=0
            
            if dataset_name!='webqsp' and dataset_name!='wikimovie':
    
                train_file=f"{dataset_name}/train.jsonl"
                dev_file=f"{dataset_name}/dev.jsonl"
                test_file=f"{dataset_name}/test.jsonl"
                with open('./metaqa/metaqa_kg.pickle', 'rb') as f:
                    metakg = pickle.load(f)
                train_data=[]
                dev_data=[]
                test_data=[]
                k=int(dataset_name[-5])
                with open(test_file, "r") as f:
                    total_lines = sum(1 for _ in f)
                gc.collect()
                with open(os.path.expanduser(test_file)) as f:
                    for line in tqdm(f, total=total_lines):
                        p+=1
                        if not line:
                            continue
                        qus = json.loads(line)
                        question = qus["question"]
                        query_entities = qus["entity_set"]
                        answer = qus["Label"]
                        subgraph_entities,subgraph_triplets=find_subgraph(question,query_entities,answer,metakg,k)
                        subgraph_triplets=normalize_triplets(subgraph_triplets)
                        positive_entities, positive_triplet_indices=find_paths_in_subgraph(subgraph_triplets, query_entities, answer, k)    
                        kg_dict = build_kg_dict(subgraph_triplets)
                        helpful = run_reasoning(
                            question, answer, query_entities, subgraph_triplets, subgraph_entities, positive_entities, positive_triplet_indices, kg_dict,
                            network, num_agents=num_agents, max_hop=k, top_k=top_k, batch_size=20, phase='test'
                        )
                        if helpful=='OOM error':
                            updated_answer = {
                            "id": p,
                            "question": question,
                            "answer": answer,
                            "query_entities": query_entities,
                            "subgraph_triplets": subgraph_triplets,
                            "positive_entities": list(positive_entities),
                            "positive_triplet_indices": list(positive_triplet_indices),
                            "evidence_text": "",
                            "recall": None,
                            "recalled_ans": None,
                            "llm_answer": "OOM error"
                            }
                            with open(save_file, 'a') as f:
                                f.write(json.dumps(updated_answer) + '\n')
                                print('OOM error')
                                continue
                        
                        helpful_triplet_set = set(tuple(x) for x in helpful)
                        gt_triplet_set = set(tuple(subgraph_triplets[i]) for i in positive_triplet_indices)
                        num_correct = len(helpful_triplet_set & gt_triplet_set)
                        num_total_positive = len(gt_triplet_set)
                        if num_total_positive > 0:
                            recall = num_correct / num_total_positive
                        else:
                            recall=None
                            
                        evidence_text = "\n".join(
                                        [f"({triplet[0]}, {triplet[1]}, {triplet[2]});" for triplet in helpful]
                                    )
                        
                        answer_prompt = """Based on the knowledge triplets, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
                        llm_input_answers=f"{answer_prompt}\nQ: {question}\nKnowledge Triplets: {evidence_text}\nA: "
                        llm_output_answers=""
                        
                        updated_answer = {
                            "id": p,
                            "question": question,
                            "answer": answer,
                            "query_entities": query_entities,
                            "subgraph_triplets": subgraph_triplets,
                            "positive_entities": list(positive_entities),
                            "positive_triplet_indices": list(positive_triplet_indices),
                            "evidence_text": evidence_text,
                            "recall": recall,
                            "recalled_ans": recalled_ans,
                            "llm_answer": llm_output_answers
                            }
                        with open(save_file, 'a') as f:
                            f.write(json.dumps(updated_answer) + '\n')
                        continue
                    
            else:
                with open(test_file, "r") as f:
                    total_lines = sum(1 for _ in f)
                gc.collect()
                with open(test_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_lines):
                        p+=1
                        item = json.loads(line)
                        if dataset_name=='webqsp':
                            question = item['question']
                            answer_dict={}
                            for answer in item['answers']:
                                answer_dict[answer['kb_id']]=answer['text']
                            
                            unknown_entity_map = {}
                            unknown_counter = [1]
                            answer=[map_entity(answer['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for answer in item['answers']]
                            
                            query_entities = [map_entity(entity['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for entity in item['entities']]
                            subgraph_entities = [map_entity(entity['kb_id'], unknown_counter, unknown_entity_map, answer_dict) for entity in item['subgraph']['entities']]

                            subgraph_triplets = []
                            for triplet in item['subgraph']['tuples']:
                                
                                h = map_entity(triplet[0]['kb_id'], unknown_counter, unknown_entity_map, answer_dict)
                                r = simplify_relation(triplet[1]['text'])
                                t = map_entity(triplet[2]['kb_id'], unknown_counter, unknown_entity_map, answer_dict)
                                subgraph_triplets.append([h, r, t])
                                
                            unknown_counter_merge = [1]
                            subgraph_triplets, positive_entities, positive_triplet_indices=find_pruned_paths_in_subgraph(subgraph_triplets, query_entities, answer, unknown_counter_merge, max_hops=2)
                            
                        else:
                            question = item['question']
                            answer=[answer['text'] for answer in item['answers']]
                            query_entities=[entity['text'] for entity in item['entities']]
                            subgraph_entities=[entity['text'] for entity in item['subgraph']['entities']]
                            subgraph_triplets=[[x['text'] for x in triplet] for triplet in item['subgraph']['tuples']]

                            positive_entities, positive_triplet_indices=find_paths_in_subgraph(subgraph_triplets, query_entities, answer, max_hops=2)
                        
                        kg_dict = build_kg_dict(subgraph_triplets)

                        helpful = run_reasoning(
                            question, answer, query_entities, subgraph_triplets, subgraph_entities, positive_entities, positive_triplet_indices, kg_dict,
                            network, num_agents=num_agents, max_hop=2, top_k=top_k, batch_size=20, phase='test'
                        )
                        if helpful=='OOM error':
                            updated_answer = {
                            "id": p,
                            "question": question,
                            "answer": answer,
                            "query_entities": query_entities,
                            "subgraph_triplets": subgraph_triplets,
                            "positive_entities": list(positive_entities),
                            "positive_triplet_indices": list(positive_triplet_indices),
                            "evidence_text": "",
                            "recall": None,
                            "recalled_ans": None,
                            "llm_answer": "OOM error"
                            }
                            with open(save_file, 'a') as f:
                                f.write(json.dumps(updated_answer) + '\n')
                                print('OOM error')
                                continue
                        
                        helpful_triplet_set = set(tuple(x) for x in helpful)
                        gt_triplet_set = set(tuple(subgraph_triplets[i]) for i in positive_triplet_indices)
                        num_correct = len(helpful_triplet_set & gt_triplet_set)
                        num_total_positive = len(gt_triplet_set)
                        if num_total_positive > 0:
                            recall = num_correct / num_total_positive
                        else:
                            recall=None
                        evidence_text = "\n".join(
                                        [f"({triplet[0]}, {triplet[1]}, {triplet[2]});" for triplet in helpful]
                                    )
                        if evidence_text is None:
                            evidence_text=""
                        answer_prompt = """Based on the knowledge triplets, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
                        llm_input_answers=f"{answer_prompt}\nQ: {question}\nKnowledge Triplets: {evidence_text}\nA: "
                        llm_output_answers=""

                        updated_answer = {
                            "id": p,
                            "question": question,
                            "answer": answer,
                            "query_entities": query_entities,
                            "subgraph_triplets": subgraph_triplets,
                            "positive_entities": list(positive_entities),
                            "positive_triplet_indices": list(positive_triplet_indices),
                            "evidence_text": evidence_text,
                            "recall": recall,
                            "recalled_ans": recalled_ans,
                            "llm_answer": llm_output_answers
                            }
                        with open(save_file, 'a') as f:
                            f.write(json.dumps(updated_answer) + '\n')
                        
                        continue
            exit()
            
            
            
            