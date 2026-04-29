import time
import json
import os
import sys
import re

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api
from largemodel_create_and_evaluate.promot_hub import prompt_hub


def deal(model, item):
    """
    上下文理解模块的处理逻辑。
    要求模型在长文本中理解隐含的因果、情感或主题。
    """
    print(f"----- [上下文理解] 处理第{item['rowIdx'] + 1}条数据 -----")

    # 强化 Prompt，要求模型进行上下文关联思考
    prompt = (
        f"请基于以下提供的长文本内容进行深度理解，并回答问题。\n\n"
        f"由于文本较长，请特别注意上下文之间的逻辑关联、隐含情感或主题。\n"
        f"问题：{item['question']}\n\n"
        f"要求：说明你的理由，最后以'【答案】选项/内容'的形式结束。"
    )

    try:
        answer = call_api(model, prompt)
        print(f"模型回答: {answer[:100]}...")

        # 结果判定：支持选择题和简答题
        standard_answer = str(item['answer'])
        # 提取【答案】后的内容
        match = re.search(r"【答案】\s*(.+)", answer)
        pred = match.group(1).strip() if match else answer

        if standard_answer in pred or pred in standard_answer:
            return answer, 1.0
        else:
            # 柔性判定：如果关键词匹配成功
            keywords = standard_answer.split(',') # 假设简答题答案以逗号分隔关键词
            match_count = sum(1 for kw in keywords if kw.strip() in answer)
            score = match_count / len(keywords) if keywords else 0
            return answer, score
    except Exception as e:
        print(f"上下文理解请求失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None):
    """
    上下文理解子指标评测方法。
    """
    if qs_list is None:
        # 保持对原有文件读取方式的兼容
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "上下文关联", "contextual_relevance.json")
        with open(direction, 'r', encoding='utf-8') as f:
            qs_list = json.load(f)

    result = []
    response_ls = []

    try:
        for item in qs_list:
            response, score = deal(model, item)
            result.append(score)

            response_ls.append({
                "dataId": item.get('rowIdx', 0),
                "response": re.sub(r"\n\s*\n", "\n", response),
                "score": round(score, 2)
            })
    except Exception as e:
        print(f"评测中断: {e}")

    avg_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(avg_score, 2)


if __name__ == "__main__":
    ls = []
    for model in ["DeepSeek-V3"]:
        print("正在执行{}模型".format(model))
        result = evaluate(model)
        print(result)
        ls.append(result)
    print(ls)

#0.4834375