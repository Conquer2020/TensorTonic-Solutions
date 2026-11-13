import math
import torch

def continuous_batching_scheduler(requests:list,kv_capacity:int) -> list:
    waiting=sorted(enumerate(requests),key=lambda r:(r[1]['arrival_step'],r[0]))
    active=[];reserved=0;records=[];step=waiting[0][1]['arrival_step'] if waiting else 0
    while waiting or active:
        if not active and waiting and waiting[0][1]['arrival_step']>step:step=waiting[0][1]['arrival_step']
        admitted=[];finished=[]
        def admit():
            nonlocal waiting,reserved
            keep=[]
            for index,r in waiting:
                size=r['prompt_tokens']+r['max_new_tokens']
                if r['arrival_step']>step or reserved+size>kv_capacity:
                    keep.append((index,r));continue
                admitted.append(r['id'])
                if r['max_new_tokens']==0:finished.append(r['id'])
                else:
                    active.append(dict(r,generated=0,footprint=size));reserved+=size
            waiting=keep
        admit()
        alive=[]
        for r in active:
            r['generated']+=1
            if r['generated']==r['max_new_tokens']:
                finished.append(r['id']);reserved-=r['footprint']
            else:alive.append(r)
        active=alive
        admit()
        records.append({'step':step,'admitted':admitted,'active':[r['id'] for r in active],'finished':finished,'kv_tokens_in_use':sum(r['prompt_tokens']+r['generated'] for r in active)})
        step+=1
    return records
