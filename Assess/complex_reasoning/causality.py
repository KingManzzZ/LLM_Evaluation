import json
import re
import sys
import os

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api


def deal(model, item):
    """
    加强因果推理的 Prompt 指导和结果提取逻辑
    """
    print(f"因果关系--处理第{item['rowIdx']}条数据")

    # 构建更严谨的 Prompt，引导模型输出固定格式
    options_str = "，".join(item['options'])
    add_prompt = (
        f"请分析以下因果关系问题，并从选项中选择最合理的答案。\n"
        f"问题：{item['question']}\n"
        f"选项：{options_str}\n"
        f"要求：请先简要说明推理过程，最后务必以'【答案】选项字母'的形式结束（例如：【答案】A）。"
    )

    try:
        answer = call_api(model, add_prompt)

        # 1. 优先使用正则提取【答案】后的字母
        match = re.search(r"【答案】\s*([A-D])", answer)
        if match:
            pred = match.group(1)
        else:
            # 2. 备选提取：查找最后出现的大写字母（防止模型话多但没写格式）
            clean_answer = re.sub(r'[^A-D]', '', answer.upper())
            pred = clean_answer[-1] if clean_answer else ""

        score = 1 if pred == item['answer'] else 0
        print(f"模型回答提取点: {pred}, 正确答案: {item['answer']}, 得分: {score}")
        return answer, score
    except Exception as e:
        print(f"因果推理 API 调用异常: {e}")
        return "ERROR", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        response, score = deal(model, item)
        result.append(score)

        # 清理响应中的多余空行，保持结果整洁
        clean_response = re.sub(r"\n\s*\n", "\n", response)

        response_ls.append({
            "dataId": item['rowIdx'],
            "response": clean_response,
            "is_correct": bool(score)
        })

    # 计算百分制得分
    final_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(final_score, 2)



if __name__ == "__main__":
    # for model in ['yi-lightning',"ernie-4.0-8k"]:
    #     print("正在执行{}模型".format(model))
    #     try:
    #         result=  evaluate(model)
    #         print(result)
    #     except:
    #         print("模型{}执行错误".format(model))
    #         continue
    result=  evaluate("gpt-4o-mini")
    print(result)