#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, string, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
LETTERS=list(string.ascii_uppercase)
def get_dtype(name):
    name=str(name).lower()
    if name in {'bf16','bfloat16'}: return torch.bfloat16
    if name in {'fp16','float16'}: return torch.float16
    if name in {'fp32','float32'}: return torch.float32
    return 'auto'
def normalize(text):
    text=str(text).lower().strip(); text=re.sub(r'\s+',' ',text); return text.strip(' \n\t\r.,;:!?"\'`')
def strip_answer_cue(prompt): return re.sub(r'\n?\s*Answer:\s*$','',str(prompt).strip(),flags=re.I).strip()
def build_prompt(ex):
    choices=[str(c) for c in ex['choices']]; labels=LETTERS[:len(choices)]
    opts='\n'.join(f'{lab}. {choice}' for lab,choice in zip(labels,choices))
    return f"{strip_answer_cue(ex['prompt'])}\n\nChoose exactly one answer from the options below. Return only the option letter.\n{opts}\nAnswer:", labels
def parse_prediction(text, choices, labels):
    raw=str(text).strip(); first=raw.splitlines()[0].strip() if raw else ''; first_norm=normalize(first); full_norm=normalize(raw)
    if first.upper() in labels: return labels.index(first.upper()), True
    for pat in [r'^\s*\(?([A-Z])\)?[.)]?\s*$', r'\b(?:answer|option|choice)\s*[:\-]?\s*\(?([A-Z])\)?\b', r'\bthe\s+answer\s+is\s+\(?([A-Z])\)?\b']:
        m=re.search(pat,raw,flags=re.I)
        if m:
            lab=m.group(1).upper()
            if lab in labels: return labels.index(lab), True
    m=re.search(r'\b([0-9]{1,2})\b',raw)
    if m:
        k=int(m.group(1))
        if 0<=k<len(choices): return k, True
        if 1<=k<=len(choices): return k-1, True
    norms=[normalize(c) for c in choices]
    for i,c in enumerate(norms):
        if first_norm==c or full_norm.startswith(c): return i, True
    for i,c in enumerate(norms):
        if c and c in full_norm: return i, True
    return -1, False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model_id',required=True); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--method',choices=['vanilla_custom_gen','dola_custom_gen'],required=True); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--run_id',required=True); ap.add_argument('--dtype',default='bfloat16'); ap.add_argument('--max_new_tokens',type=int,default=16); ap.add_argument('--dola_layers',default='high'); ap.add_argument('--repetition_penalty',type=float,default=1.2); ap.add_argument('--attn_implementation',default='eager'); args=ap.parse_args()
    torch.manual_seed(args.seed); dtype=get_dtype(args.dtype)
    tokenizer=AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model_id,torch_dtype=dtype,device_map='auto',trust_remote_code=False,attn_implementation=args.attn_implementation); model.eval()
    if args.method=='dola_custom_gen':
        model.config.output_hidden_states=True; model.generation_config.output_hidden_states=True
    device=next(model.parameters()).device; out_path=Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True); n=0; parsed=0
    with open(args.data) as fin, out_path.open('w') as fout:
        for line in fin:
            ex=json.loads(line); prompt,labels=build_prompt(ex); choices=[str(c) for c in ex['choices']]; gold=int(ex['correct_choice'])
            if hasattr(tokenizer,'apply_chat_template') and tokenizer.chat_template:
                enc=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True)
            else: enc=tokenizer(prompt,return_tensors='pt')
            enc=enc.to(device) if hasattr(enc,'to') else {k:v.to(device) for k,v in enc.items()}
            gen_kwargs=dict(**enc,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            if args.method=='dola_custom_gen':
                gen_kwargs.update(dict(custom_generate='transformers-community/dola',trust_remote_code=True,dola_layers=args.dola_layers,repetition_penalty=args.repetition_penalty,output_hidden_states=True))
            t0=time.perf_counter();
            with torch.no_grad(): output_ids=model.generate(**gen_kwargs)
            latency=time.perf_counter()-t0; input_len=enc['input_ids'].shape[-1]; gen_text=tokenizer.decode(output_ids[0][input_len:],skip_special_tokens=True)
            pred,ok=parse_prediction(gen_text,choices,labels); parsed += int(ok)
            row={'id':ex.get('id'),'benchmark':ex.get('benchmark'),'model_id':args.model_id,'method':args.method,'evaluation_mode':'generation','run_id':args.run_id,'seed':args.seed,'prompt':prompt,'choices':choices,'target':ex.get('target'),'generated_text':gen_text,'correct_choice':gold,'pred_choice':int(pred),'prediction':choices[pred] if ok and pred>=0 else '', 'correct':bool(ok and pred==gold),'confidence':1.0 if ok else 0.0,'choice_nlls':[],'choice_probs':[],'latency_s':float(latency),'refusal':False,'parse_success':bool(ok)}
            fout.write(json.dumps(row)+'\n'); n+=1
            if n%25==0: print(f'[{args.method}] {n} examples parse={parsed}/{n}',flush=True)
    print(json.dumps({'method':args.method,'n':n,'parsed':parsed,'out':str(out_path)},indent=2))
if __name__=='__main__': main()
