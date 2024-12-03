import os
import sys
from pathlib import Path
import json
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
import argparse
from openai import OpenAI
tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)
from task_template import SYSTEM,META_INFO,CRITERIA,INSTRUCTION_RAT_GENERATE_PROMPT_WLABEL1,INSTRUCTION_RAT_GENERATE_PROMPT_WLABEL2,INPUT_PROMPT_RAT_GENERATE_WLABEL,task2task


client = OpenAI(
    api_key = "XXX",
)

def get_fewshot(fewshot_path,fewshot_number):
    t1,t2,t3,l,r = [],[],[],[],[]
    with open(fewshot_path,'r') as f:
        data = json.load(f)
    for d in data:      
        if len(t1) == fewshot_number:
            break 
        t1.append(d['text_a'])
        t2.append(d['text_b'])
        t3.append(d['text_c'])
        l.append(d['label'])
        r.append(d['reason'])
    return t1,t2,t3,l,r

def process_data(file_path,max_number):
    id_l,t1,t2,t3,label_l  = [],[],[],[],[]
    with open(file_path,'r') as f:
        res = json.load(f)
    for i,r in enumerate(res):
        if i == max_number:
            break
        id_l.append(r['id'])
        t1.append(r['text_a'])
        if "text_b" not in r:
            t2.append(None)
        else:
            t2.append(r['text_b'])
        if "text_c" not in r:
            t3.append(None)
        else:
            t3.append(r['text_c'])
        label_l.append(r["label"])

    return id_l,t1,t2,t3,label_l

def load_data(data_path,task,max_number,label_first,criteria,fewshot_number):
    dataset = []

    id_l,t1,t2,t3,label_l = process_data(data_path,max_number)

    task = task2task[task]
    system = SYSTEM[task] 
    meta = META_INFO[task]
    criteria = "判断标准\n"[:CRITERIA[task] ]+CRITERIA[task] if criteria else ""
    prefix = system+"\n"+meta+"\n\n"+criteria+"\n"[:len(criteria)]

    ins_p1 = INSTRUCTION_RAT_GENERATE_PROMPT_WLABEL1[task] 
    ins_p2 = INSTRUCTION_RAT_GENERATE_PROMPT_WLABEL2[task] 
    input_prompt = INPUT_PROMPT_RAT_GENERATE_WLABEL[task]
    
    if fewshot_number>0:
        fewshot_path = os.path.join("/".join(data_path.split("/")[:-1]),"few-shot.json")
        t1_fs,t2_fs,t3_fs,label_l_fs,rat_fs = get_fewshot(fewshot_path,fewshot_number)

        prompt_fs = "示例\n"
        for i in range(fewshot_number):
            prompt_fs += "[示例]"+str(i+1)+"\n"
            prompt_fs += input_prompt.format(t1_fs[i],t2_fs[i],t3_fs[i],label_l_fs[i])
            prompt_fs+= "\n"+rat_fs[i]+"\n\n"


        ins_p1 =  prompt_fs+ins_p1.replace("现在，请你","现在，请你参照上面的示例，")+"，给出得到该答案的推理过程"
        
        ins_p2 = ins_p2.replace("然后，","")



    for i in range(len(t1)):

        full_prompt = [{
            "role":"user",
            "content": prefix+ins_p1+"\n"+input_prompt.format(t1[i],t2[i],t3[i],label_l[i])+"\n\n"+ins_p2
            }]
        dataset.append({"case_id":id_l[i],"prompt":full_prompt,"label":label_l[i]})

    return dataset




def get_openai_response(prompt):
    completion = client.chat.completions.create(
        model = "XXX",
        messages = prompt,
        temperature = 0,
        top_p = 1,
    )
    return completion.choices[0].message.content


def fetch_response(entry):
    start_time = time.time()
    prompt = entry['prompt']
    try:
        response_content = get_openai_response(prompt)
    except Exception as e:
        print(e)
        response_content = "Wrong: "+str(e)
    spent_time = time.time() - start_time
    return entry["case_id"], entry["label"], response_content, spent_time



def main(data_path,task, save_path,max_number,label_first,criteria,fewshot_number):

    save_dir = "/".join(save_path.split("/")[:-1])
    Path(save_dir).mkdir(exist_ok=True, parents=True)
            

    data = load_data(data_path,task,max_number,label_first,criteria,fewshot_number)
    print(data[0]['prompt'])
    
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(fetch_response, entry) for entry in data[:]
        ]

        for idx, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=-1)
        ):
            case_id,label, response_content, spent_time = future.result()
            responses.append({'id':case_id,"pr":response_content,"gt":label})

        with open(os.path.join(save_path), "w",encoding='utf-8') as f:
            json.dump(responses, f, indent=1,ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",help="path for loading data")
    parser.add_argument("--task",help="task name of the dataset")
    parser.add_argument("--save_path",help="path for saving data")
    parser.add_argument("--max_number",type=int,default=-1,help="Max Processing sample number")



    parser.add_argument("--label_first",action="store_true",help="Give Label in the prompt")
    parser.add_argument("--criteria",action="store_true",help="Give Judge Criteria in the prompt")
    parser.add_argument("--fewshot_number",type=int,default=-1,help="Give few-shot Exemplars in the prompt")



    args = parser.parse_args()

    main(args.data_path, 
         args.task,
         args.save_path,
         args.max_number,
         args.label_first,
         args.criteria,
         args.fewshot_number,
        )


