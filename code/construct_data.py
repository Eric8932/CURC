import os
import sys
from pathlib import Path
import json
import numpy as np
import concurrent.futures
from retry import retry
from tqdm import tqdm
import argparse
tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)
from task_template import INPUT_PROMPT,PREDICT_PROMPT1,PREDICT_PROMPT2,REASON_PROMPT1,REASON_PROMPT2,EXPLAIN_PROMPT1,EXPLAIN_PROMPT2,task2task




def main(data_path,task, save_path,max_number,method):
    save_dir = "/".join(save_path.split("/")[:-1])
    Path(save_dir).mkdir(exist_ok=True, parents=True)
            

    
    with open(data_path,'r',encoding='utf-8') as f:
        train_l = json.load(f)[:max_number]
    with open(os.path.join("/".join(data_path.split("/")[:-1]),"rat.json"),'r',encoding='utf-8') as f:
        rat_l = json.load(f)[:max_number]
    rat_dic = {}
    for r in rat_l:
        rat_dic[r['id']] = r['rationale']


    task = task2task[task]
    if method == "PREDICT":
        prompt1,prompt2 = PREDICT_PROMPT1[task],PREDICT_PROMPT2[task]
        input_prompt = INPUT_PROMPT[task]
    elif method == "REASON":
        prompt1,prompt2 = REASON_PROMPT1[task],REASON_PROMPT2[task]
        input_prompt = INPUT_PROMPT[task].replace("答案：","").strip()+"\n\n让我们一步一步思考，\n"
    elif method == "EXPLAIN":
        prompt1,prompt2 = EXPLAIN_PROMPT1[task],EXPLAIN_PROMPT2[task]
        input_prompt = INPUT_PROMPT[task]

    dataset = []
    for i in range(len(train_l)):
        if i not in rat_dic:
            continue
        r = train_l[i]
        if method == 'PREDICT':
            dataset.append({"instruction":prompt1+prompt2+"\n\n"+input_prompt.format(r['text_a'],r['text_b'],r['text_c']),
                  'input':"","output":r['label']+"。"})
        elif method == "REASON":
            dataset.append({"instruction":prompt1+prompt2+"\n\n"+input_prompt.format(r['text_a'],r['text_b'],r['text_c']),
                  'input':"","output":rat_dic[int(r['id'])]})
        elif method == "EXPLAIN":
            dataset.append({"instruction":prompt1+prompt2+"\n\n"+input_prompt.format(r['text_a'],r['text_b'],r['text_c']),
                  'input':"","output":r['label']+"。\n推理过程："+rat_dic[int(r['id'])]})
    print(dataset[0])

    with open(os.path.join(save_path), "w",encoding='utf-8') as f:
        json.dump(dataset, f, indent=1,ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",help="path for loading data")
    parser.add_argument("--task",help="task name of the dataset")
    parser.add_argument("--save_path",help="path for saving data")
    parser.add_argument("--max_number",type=int,default=-1,help="Max Processing sample number")
    parser.add_argument("--method",choices=["PREDICT", "REASON","EXPLAIN"], default="PREDICT",help="Constructing Data for differnt methods",)

    parser.add_argument("--criteria",action="store_true",help="Give Judge Criteria in the prompt")
    parser.add_argument("--fewshot_number",type=int,default=-1,help="Give few-shot Exemplars in the prompt")



    args = parser.parse_args()

    main(args.data_path, 
         args.task,
         args.save_path,
         args.max_number,
         args.method
        )


