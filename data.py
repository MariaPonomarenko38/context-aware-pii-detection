from datasets import Dataset, load_dataset, concatenate_datasets
import datasets
import pandas as pd
import json
from transformers import AutoTokenizer

TRAINING_PROMPT = """USER: Estimate the importance of the PII '{pii}' for answering question, 
based on the question and the context where the PII appears.
Context: {context}
Question: {question} 

If it is highly important for the question output - high.
If it is not important for the question output - low.
Output just one word - either high or low.

ASSISTANT: {importance}"""

def format(exam_answer):
    data = eval(exam_answer)

    exam_answers = '\n'.join([f"{item['marker']} - {item['text']}" for item in data])

    return exam_answers

def prepare_instructions(context, question, importance):
    instructions = []

    prompt_sample = TRAINING_PROMPT

    for c, q, i in zip(context, question, importance):
        importance_iter = json.loads(i)
        for pii in importance_iter.keys(): 
            example = prompt_sample.format(
                pii=pii,
                context=c,
                question=q,
                importance=importance_iter[pii]
            )
            instructions.append(example)

    return instructions


def prepare_dataset(dataset_repo, context_field, question_field, importance_field):
    dataset = load_dataset(dataset_repo)
    train_dataset = dataset["train"]
    #val_dataset = dataset["test"]

    context = train_dataset[context_field]
    question = train_dataset[question_field]
    importance = train_dataset[importance_field]
    
    train_prompt_question = prepare_instructions(context, question, importance)

    train_prompt_question_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(data={"instructions": train_prompt_question})
    )

    #dataset_cc = concatenate_datasets([train_prompt_question_dataset, train_question_response_dataset])
    return train_prompt_question_dataset 

if __name__ == '__main__':
    train_dataset = prepare_dataset("ponoma16/context-aware-pii-detection", "context", "question", "importance")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(len(train_dataset))
    print("*** FIRST TRAINING EXAMPLE ***")
    print(train_dataset[0])
    print("\n*** TOKENIZED EXAMPLE ***")
    print(tokenizer(train_dataset[0]['instructions']).input_ids) # Check the token IDs
    print("\n*** DECODED TOKENS ***")
    print(tokenizer.decode(tokenizer(train_dataset[0]['instructions']).input_ids))