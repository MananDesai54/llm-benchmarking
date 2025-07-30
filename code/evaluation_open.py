"""
 Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from hashlib import sha256
import datetime
from tqdm import tqdm
from func_timeout import func_set_timeout
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair


def test_open(messages, args, tools=None):
    try:
        texts = args.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            pad_token_id=args.tokenizer.pad_token_id,
        )
        model_inputs = args.tokenizer(texts, return_tensors="pt").to(args.device)
        generated_ids = args.model.generate(**model_inputs, **args.params)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = args.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = args.parse_output(response[0])

        return response
    except Exception as e:
        print(messages, response, e, flush=True)

    return None


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_feedback(tool_calls, functions):
    @func_set_timeout(10)
    def excute_func(functions, func_name, func_args):
        for func in functions:
            if func.split("def")[1].split("(")[0].strip() == func_name:
                exec(func)
        return eval(func_name)(**func_args)

    res = []
    for tool_call in tool_calls:
        try:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            feedback = excute_func(functions, tool_name, tool_args)

            res.append(
                {
                    "role": "tool",
                    "content": json.dumps(feedback, ensure_ascii=False),
                    "tool_call_id": tool_call["id"],
                }
            )
        except Exception as e:
            res.append(
                {
                    "role": "tool",
                    "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}",
                    "tool_call_id": tool_call["id"],
                }
            )

    return res


def sample_process_direct(sample, args):
    if sample.get("messages"):
        messages = sample["messages"]
    else:
        messages = [
            {
                "role": "user",
                "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}",
            }
        ]
    response = test_open(messages=messages, args=args)
    if response:
        messages.append(response.copy())
        sample["messages"] = messages.copy()
        return sample
    return None


def sample_process_mandatory(sample, args):
    turn = 0

    if sample.get("messages"):
        messages = sample["messages"]
    else:
        messages = [
            {
                "role": "user",
                "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}",
            }
        ]
    tools = [
        {"type": "function", "function": tool.copy()}
        for tool in sample["tools"].values()
    ]
    while turn < args.max_turns:
        response = test_open(messages=messages, tools=tools, args=args)
        if response:
            messages.append(response.copy())
            if response.get("tool_calls", None):
                feedback = get_feedback(response["tool_calls"], sample["functions"])
                messages += feedback.copy()
            if not response.get("tool_calls", None) or turn == args.max_turns - 1:
                sample["messages"] = messages.copy()
                sample["turn"] = turn
                return sample

        turn += 1
    return None


def sample_process_free(sample, args):
    turn = 0

    if sample.get("messages"):
        messages = sample["messages"]
    else:
        messages = [
            {
                "role": "user",
                "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}",
            }
        ]
    tools = [
        {"type": "function", "function": tool.copy()}
        for tool in sample["tools"].values()
    ]
    while turn < args.max_turns:
        response = test_open(messages=messages, tools=tools, args=args)
        if response:
            messages.append(response.copy())
            if response.get("tool_calls", None):
                feedback = get_feedback(response["tool_calls"], sample["functions"])
                messages += feedback.copy()
            if not response.get("tool_calls", None) or turn == args.max_turns - 1:
                sample["messages"] = messages.copy()
                sample["turn"] = turn
                return sample

        turn += 1
    return None


def sample_process(sample, args):
    if args.scenario == "Direct":
        sample = sample_process_direct(sample, args)
    elif args.scenario == "Mandatory":
        sample = sample_process_mandatory(sample, args)
    elif args.scenario == "Free":
        sample = sample_process_free(sample, args)
    else:
        raise NotImplementedError

    if sample:
        if sample["messages"][-1]["role"] == "assistant":
            if sample["messages"][-1]["content"]:
                if (
                    sample["answer"].rstrip(".0").lower()
                    in sample["messages"][-1]["content"]
                    .rstrip(".0")
                    .replace(",", "")
                    .lower()
                ):
                    sample["answer_correct"] = 1
                elif (
                    sample["messages"][-2]["role"] == "tool"
                    and sample["answer"].rstrip(".0").lower()
                    in sample["messages"][-2]["content"]
                    .rstrip(".0")
                    .replace(",", "")
                    .lower()
                ):
                    sample["answer_correct"] = 1
                else:
                    sample["answer_correct"] = 0

    return sample


def parse_qwen25(inputs: str):
    output = {"role": "assistant", "content": None, "tool_calls": None}
    start_tool = "<tool_call>"
    end_tool = "</tool_call>"

    start = inputs.find(start_tool)

    if start == -1:
        output["content"] = inputs.strip()
    else:
        output["content"] = inputs[:start].strip() if start != 0 else None
        tool_calls = inputs[start:].lstrip(start_tool)
        tool_calls = tool_calls.split(end_tool)
        for tool_call in tool_calls:
            if tool_call.strip():
                try:
                    if tool_call.strip().startswith(start_tool):
                        tool_call = tool_call.strip().lstrip(start_tool).strip()
                    tool_call = json_repair.loads(tool_call)
                    if type(tool_call["arguments"]) != dict:
                        tool_call["arguments"] = json_repair.loads(
                            tool_call["arguments"]
                        )
                    if not output["tool_calls"]:
                        output["tool_calls"] = []
                    output["tool_calls"].append(
                        {
                            "id": "call_"
                            + sha256(str(datetime.datetime.now()).encode()).hexdigest(),
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(
                                    tool_call["arguments"], ensure_ascii=False
                                ),
                                "name": tool_call["name"],
                            },
                        }
                    )
                except Exception as e:
                    print(e)
                    pass

    return output


def parse_llama31(inputs: str):
    output = {"role": "assistant", "content": None, "tool_calls": None}
    start_tool = "<|python_tag|>"

    start = inputs.find(start_tool)

    if start == -1:
        try:
            tool_calls = json_repair.loads(inputs.strip())
            tool_calls = inputs.strip()
        except:
            output["content"] = inputs.strip()
    else:
        tool_calls = inputs[start:].lstrip(start_tool).strip()
        output["content"] = inputs[:start].strip()
    tool_calls = json_repair.loads(tool_calls)
    if type(tool_calls) != list:
        tool_calls = [tool_calls]
    for tool_call in tool_calls:
        try:
            if type(tool_call["parameters"]) != dict:
                tool_call["arguments"] = json_repair.loads(tool_call["parameters"])
            else:
                tool_call["arguments"] = tool_call["parameters"].copy()

            if not output["tool_calls"]:
                output["tool_calls"] = []
            output["tool_calls"].append(
                {
                    "id": "call_"
                    + sha256(str(datetime.datetime.now()).encode()).hexdigest(),
                    "type": "function",
                    "function": {
                        "arguments": json.dumps(
                            tool_call["arguments"], ensure_ascii=False
                        ),
                        "name": tool_call["name"],
                    },
                }
            )
            break
        except Exception as e:
            print(e)
            pass
    if not output["tool_calls"]:
        output["content"] = inputs.strip()
    return output


def get_params(model):
    if model == "qwen25":
        return dict(do_sample=False, max_new_tokens=1024, eos_token_id=[151645, 151643])
    elif model == "llama31":
        return dict(
            do_sample=False, max_new_tokens=1024, eos_token_id=[128001, 128008, 128009]
        )
    else:
        raise NotImplementedError


def get_parse_output(model):
    if model == "qwen25":
        return parse_qwen25
    elif model == "llama31":
        return parse_llama31
    else:
        raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="Direct",
        choices=["Direct", "Mandatory", "Free"],
    )
    parser.add_argument(
        "--series", type=str, default="qwen25", choices=["llama31", "qwen25"]
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_file", type=str, default="../data/ToolHop.json")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_turns", type=int, default=9)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    with open(args.input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf8") as f:
            ids = [json.loads(line)["id"] for line in f.readlines()]

    args.params = get_params(args.series)
    args.parse_output = get_parse_output(args.series)
    args.model, args.tokenizer = load_model(args)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(
        args.output_file, "a"
    ) as f:
        futures = [
            executor.submit(sample_process, sample, args)
            for sample in data
            if sample["id"] not in ids
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            response = future.result()
            if response:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")
                f.flush()

    with open(args.output_file, "r", encoding="utf8") as f:
        data = [json.loads(line) for line in f.readlines()]
        result = round(
            100 * sum([sample["answer_correct"] for sample in data]) / 995, 2
        )
        print(
            f"Scenario: {args.scenario}\nSeries: {args.series}\nModel: {args.model_path}\nValid Items: {len(data)}\nResult: {result}\n"
        )


if __name__ == "__main__":
    main()
