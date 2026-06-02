#!/usr/bin/env python
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def dtype_from_name(name):
    name=str(name).lower()
    if name in {'bf16','bfloat16'}: return torch.bfloat16
    if name in {'fp16','float16'}: return torch.float16
    if name in {'fp32','float32'}: return torch.float32
    return torch.bfloat16

def read_jsonl(path):
    return [json.loads(line) for line in open(path)]

def choice_nll(model, tokenizer, prompt, choice, device):
    ans=' '+str(choice).strip(); full=str(prompt)+ans
    prompt_ids=tokenizer(str(prompt),add_special_tokens=False)['input_ids']
    enc=tokenizer(full,add_special_tokens=False,return_tensors='pt')
    input_ids=enc['input_ids'].to(device); attention_mask=enc.get('attention_mask')
    if attention_mask is not None: attention_mask=attention_mask.to(device)
    labels=input_ids.clone(); labels[:,:len(prompt_ids)]=-100
    if (labels!=-100).sum().item()==0:
        labels[:]=-100; labels[:,-1]=input_ids[:,-1]
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

@torch.no_grad()
def dev_accuracy(model, tokenizer, rows, device, limit=300):
    model.eval(); correct=0; n=0
    for ex in rows[:limit]:
        nlls=[float(choice_nll(model, tokenizer, ex['prompt'], c, device).detach().cpu()) for c in ex['choices']]
        pred=min(range(len(nlls)), key=lambda i:nlls[i]); correct += int(pred==int(ex['correct_choice'])); n+=1
    model.train(); return correct/max(n,1), n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_id',required=True); ap.add_argument('--train_jsonl',required=True); ap.add_argument('--dev_jsonl',required=True); ap.add_argument('--output_dir',required=True)
    ap.add_argument('--seed',type=int,default=1); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--weight_decay',type=float,default=0.0)
    ap.add_argument('--lora_r',type=int,default=2); ap.add_argument('--lora_alpha',type=int,default=4); ap.add_argument('--lora_dropout',type=float,default=0.0); ap.add_argument('--target_modules',default='q_proj,v_proj,o_proj')
    ap.add_argument('--dtype',default='bfloat16'); ap.add_argument('--attn_implementation',default='eager'); ap.add_argument('--max_grad_norm',type=float,default=1.0); ap.add_argument('--dev_limit',type=int,default=300)
    args=ap.parse_args(); torch.manual_seed(args.seed); out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train_rows=read_jsonl(args.train_jsonl); dev_rows=read_jsonl(args.dev_jsonl); dtype=dtype_from_name(args.dtype)
    tokenizer=AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model_id,torch_dtype=dtype,device_map='auto',trust_remote_code=False,attn_implementation=args.attn_implementation)
    lora_cfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,target_modules=[x.strip() for x in args.target_modules.split(',') if x.strip()],lora_dropout=args.lora_dropout,bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,lora_cfg); model.train(); device=next(model.parameters()).device
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
    print(json.dumps({'event':'lora_train_start','model_id':args.model_id,'train_examples':len(train_rows),'dev_examples':len(dev_rows),'trainable_parameters':trainable,'total_parameters':total,'target_modules':args.target_modules,'lora_r':args.lora_r,'lora_alpha':args.lora_alpha},indent=2),flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay); t0=time.time(); steps=0
    for epoch in range(args.epochs):
        pbar=tqdm(train_rows,desc=f'epoch {epoch+1}/{args.epochs}')
        for ex in pbar:
            opt.zero_grad(set_to_none=True); nlls=torch.stack([choice_nll(model,tokenizer,ex['prompt'],c,device) for c in ex['choices']]); logits=-nlls.unsqueeze(0); gold=torch.tensor([int(ex['correct_choice'])],device=logits.device)
            loss=F.cross_entropy(logits,gold)
            if torch.isnan(loss) or torch.isinf(loss): raise RuntimeError('Bad LoRA training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.max_grad_norm); opt.step(); steps+=1; pbar.set_postfix(loss=float(loss.detach().cpu()))
    dev_acc, dev_n=dev_accuracy(model,tokenizer,dev_rows,device,args.dev_limit)
    adapter_dir=out_dir/'lora_adapter'; model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    summary={'model_id':args.model_id,'seed':args.seed,'epochs':args.epochs,'steps':steps,'train_examples':len(train_rows),'dev_examples':len(dev_rows),'dev_eval_limit':args.dev_limit,'dev_choice_accuracy':dev_acc,'dev_n':dev_n,'trainable_parameters':trainable,'total_parameters':total,'trainable_fraction':trainable/total,'target_modules':args.target_modules,'lora_r':args.lora_r,'lora_alpha':args.lora_alpha,'lora_dropout':args.lora_dropout,'lr':args.lr,'weight_decay':args.weight_decay,'elapsed_s':time.time()-t0,'adapter_dir':str(adapter_dir),'cuda_available':torch.cuda.is_available()}
    (out_dir/'train_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
