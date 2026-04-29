import json
import time
import os
import sys
import re

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api
def deal(model,item):
    """
    信息提取模块的处理逻辑。
    要求模型从海量文本中定位并提取特定事实。
    """
    print(f"----- [信息提取] 处理第{item['rowIdx'] + 1}条数据 -----")

    # 强化 Prompt，要求模型进行精确提取
    prompt = (
        f"请在以下提供的长文本中定位并提取问题的答案。\n\n"
        f"注意：仅回答提取到的具体事实，不要包含无用的解释。\n"
        f"问题：{item['question']}\n\n"
        f"要求：首先输出你的提取逻辑或原文出处，最后以'【答案】提取内容'的形式结束。"
    )

    try:
        answer = call_api(model, prompt)
        print(f"提取结果: {answer[:100]}...")

        # 结果判定：基于 F1 或者 Rouge-L 设计更复杂的算法（此处使用关键词+柔性匹配）
        standard_answer = str(item.get('answer', ''))
        # 匹配【答案】后的标记
        match = re.search(r"【答案】\s*(.+)", answer)
        pred = match.group(1).strip() if match else answer

        if standard_answer in pred or pred in standard_answer:
            return answer, 1.0
        else:
            # 柔性判定：如果关键词匹配成功
            keywords = standard_answer.split(',')
            match_count = sum(1 for kw in keywords if kw.strip() in pred)
            score = match_count / len(keywords) if keywords else 0
            return answer, score
    except Exception as e:
        print(f"信息提取请求失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None):
    """
    信息提取指标评测方法。
    """
    if qs_list is None:
        # 保持对原有文件读取方式的兼容
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "信息提取", "information_extraction.json")
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
    ls=[]
    for model in ['DeepSeek-V3',"qwen-max","yi-lightning"]:
        print("正在执行{}模型".format(model))
        result=  evaluate(model)
        print(result)
        ls.append(result)
    print(ls)


# def locate_answer_paragraphs(item, labels):
#     """根据labels中的字符位置定位答案所在段落"""
#     paragraphs = item["question"]["content"]
#     answer_pars = set()
#
#     # 计算每个段落的字符偏移量
#     char_offset = 0
#     for idx, par in enumerate(paragraphs):
#         par_length = len(par) + 4  # 包括分隔符占位长度
#         for label in labels:
#             start, end = label["start"][0], label["end"][0]  # 取第一个位置范围
#             if char_offset <= start < char_offset + par_length:
#                 answer_pars.add(idx + 1)  # 段落编号从1开始
#         char_offset += par_length
#     return sorted(list(answer_pars))  # 返回有序段落编号列表