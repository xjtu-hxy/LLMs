from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from datasets import Dataset

if __name__ == '__main__':
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='Qwen3-4B-base',
        max_seq_length=512,
        fast_inference=True, #使用vllm
        max_lora_rank=32,
        gpu_memory_utilization=0.7
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r= 32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,
        use_gradient_checkpointing="unsloth",
        random_state=3407
    )

    reasoning_start = "<start_working_out>" # Acts as <think>
    reasoning_end   = "<end_working_out>"   # Acts as </think>
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""

    chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

    # Replace with out specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    tokenizer.apply_chat_template([
        {"role" : "user", "content" : "What is 1+1?"},
        {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "What is 2+2?"},
    ], tokenize = False, add_generation_prompt = True)

    # 加载数据集
    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    # Select only numbers
    dataset = dataset.iloc[np.where(is_number)[0]]

    def format_dataset(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # Remove generated <think> and </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting
        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + expected_answer + solution_end
        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
    dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

    dataset = dataset.loc[dataset["N"] <= 512/2].copy()

    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
    dataset = Dataset.from_pandas(dataset)

