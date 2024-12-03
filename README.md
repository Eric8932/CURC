# NLU Rationale Collection: Exploring Rationales' Effects in Natural Language Understanding for Large Language Models

Code and Data for the paper *Exploring Rationales' Effects in Natural Language Understanding for Large Language Models*.

## Data

In the **data** folder, we provide the data for 34 datasets, including the original train, dev, and test sets, as well as the corresponding cleaned ID lists. Additionally, we also provide the cleaned rationales for each dataset, and for some datasets, we include few-shot exemplars.

In the **task_template** folder, we have information about each dataset, including the task to which it belongs, the dataset description, the instructions about the dataset , the Label Criteria, and so on


## Code

###LLMs-as-a-Judge
We have each of the three LLMs give the label for each sample in the dataset. In the prompt, we include the judge criteria for each label as well as 8 examples.

```
# Example for generating labels for TNEWS dataset.
python3 code/judge.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_train_judge1.json \
        --task tnews \
        --fewshot_number 8;
```

###Rationale Generation
We have an LLM generate the rationale for the label of each sample.
There are three prompt designs to choose from: 1. Providing only the label. 2.Adding Label Criteria. 3.Adding 8-shot exemplars.

```
# Example for generating rationales for TNEWS dataset.
python3 code/generate_rationale.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_rat_lab.json \
        --task tnews \
        --label_first ;

python3 code/generate_rationale.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_rat_lab_cri.json \
        --task tnews \
        --label_first  --criteria;

python3 code/generate_rationale.py ---data_path /data/TNEWS/train.json \
        --save_path /results/tnews_rat_lab_8shot.json \
        --task tnews \
        --label_first  --fewshot_number 8;

```

###Constructing Training Data
With the cleaned data and rationales for each sample, we construct the training data in the Alpaca format.
We have three training approaches, corresponding to three data construction methods:
PREDICT: Directly output the label.
REASON: Output the rationale first, then the label.
EXPLAIN: Output the label first, then the rationale.
The input instructions are adjusted accordingly based on the output type.

```
# Example for constructing training data for TNEWS dataset.
python3 code/construct_data.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_train_predict.json \
        --task tnews \
        --fewshot_number 8 --method PREDICT ;
		
python3 code/construct_data.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_train_reason.json \
        --task tnews \
        --fewshot_number 8 --method REASON ;
		
python3 code/construct_data.py --data_path /data/TNEWS/train.json \
        --save_path /results/tnews_train_explain.json \
        --task tnews \
        --fewshot_number 8 --method EXPLAIN ;
```


##TRAINING & INFERENCE
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training our models and implementing different training methods.
We use [vllm](https://github.com/vllm-project/vllm) for inferening the trained models.



