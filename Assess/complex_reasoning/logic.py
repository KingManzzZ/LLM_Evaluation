import json
import time
import os
import re
import sys

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api

def deal(model, item):
    """
    针对常识逻辑推理，强化选项提取
    """
    print(f"常识逻辑--处理第{item['rowIdx']}条数据")

    # 强化 Prompt 指导，强制要求输出格式
    add_prompt = (
        f"请基于常识和基本逻辑解答以下问题。\n"
        f"问题：{item['question']}\n"
        f"选项：{item['options']}\n"
        f"请在推理后直接给出选项字母，并务必在最后一行以'【答案】选项字母'的形式结束（例如：【答案】B）。"
    )

    try:
        answer = call_api(model, add_prompt)
        print(f"原始响应摘要: {answer[:50]}...")

        # 1. 正则尝试提取【答案】标识
        match = re.search(r"【答案】\s*([A-D])", answer)
        if match:
            pred = match.group(1)
        else:
            # 2. 备选提取：取响应中出现频率最高的选项字母或最后一个字母
            clean_ans = re.sub(r"[^A-D]", "", answer.upper())
            pred = clean_ans[-1] if clean_ans else ""

        is_correct = 1 if pred == item['answer'] else 0
        print(f"回答是{pred}, 答案是{item['answer']}, 判定: {'通过' if is_correct else '失败'}")
        return answer, is_correct
    except Exception as e:
        print(f"逻辑推理 API 调用异常: {e}")
        return "ERROR", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        response, score = deal(model, item)
        result.append(score)

        response_dic = {
            "dataId": item['rowIdx'],
            "response": re.sub(r"\n\s*\n", "\n", response),
            "is_correct": bool(score)
        }
        response_ls.append(response_dic)

    final_score = (sum(result) / len(result)) * 100 if result else 0.0
    print(f"逻辑评估完成: 已处理 {len(result)} 条数据，平均得分: {final_score}")
    return response_ls, round(final_score, 2)

if __name__ == "__main__":
    ls=[]
    for model in ['DeepSeek-V3',"qwen-max"]:
        print("正在执行{}模型".format(model))
        result=  evaluate(model)
        print(result)
        ls.append(result)
    print(ls)
