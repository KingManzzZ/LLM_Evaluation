import os
import re
import sys

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api
# 改为使用专用的数学推理评测模型
try:
    from model.math_judge_model import get_math_similarity
except ImportError:
    # 尝试不同路径层级的引用
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model"))
        from math_judge_model import get_math_similarity
    except:
        from content_model import compare_texts as get_math_similarity

def deal(model, item):
    """
    数学推理评测：整合推理过程相似度与最终结果匹配度
    使用 MathBERT 实现高精度的过程逻辑判定
    """
    try:
        r_parts = item['answer'].split("#### ")
        r_process = r_parts[0]
        r_answer = r_parts[1].strip()
    except:
        # 兼容简易格式的参考答案
        r_process = ""
        r_answer = str(item['answer']).strip()

    # 此处优化了 prompt，加入更强的风格约束
    add_prompt = (
        f"请逐步解答以下数学题。确保最后一行以'#### 数字'的格式给出最终数值答案。\n"
        f"题目：{item['question']}\n"
        f"输出要求：\n1. 请先提供清晰的解题步骤；\n2. 最后以格式'#### 最终答案数字'结尾。"
    )

    try:
        answer = call_api(model, add_prompt)

        #--- 提取预测结果与过程 ---
        process = answer
        num = ""
        if "####" in answer:
            parts = answer.split("####")
            process = parts[0].strip()
            # 强化正则匹配提取最后一个数值
            num_match = re.search(r"[-+]?\d*\.?\d+", parts[-1])
            num = num_match.group() if num_match else ""
        else:
            # 备用：从全文最后一个数值尝试匹配
            all_nums = re.findall(r"[-+]?\d*\.?\d+", answer)
            num = all_nums[-1] if all_nums else ""

        #--- 结果与过程判定 ---
        # 1. 语义相似度 (Similarity)：50分权重，针对过程逻辑
        similar = get_math_similarity(process, r_process) if r_process else 1.0

        # 2. 数值匹配
        try:
            # 尝试数值化对比，忽略小数点位数差异
            is_match = abs(float(num) - float(r_answer)) < 1e-6
        except:
            # 兜底字符串对比
            is_match = (str(num).strip() == str(r_answer).strip())

        #--- 计分模型 ---
        if is_match:
            # 结果正确：50基础分 + 过程相关性奖励分
            score = 50 + (50 * similar)
        else:
            # 结果错误：仅根据推理逻辑给予少量分（最高不超过30分）
            score = 30 * similar

        print(f"[数学评测] 模型预测: {num}, 参考: {r_answer} | 相似度: {similar:.2f} | 得分: {score:.2f}")
        return answer, score

    except Exception as e:
        print(f"数学评测运行时异常: {e}")
        return f"RUNTIME_ERROR: {str(e)}", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        print(f"数学--处理第{item['rowIdx']}条数据")
        response, score = deal(model, item)

        result.append(score)
        response_ls.append({
            "dataId": item['rowIdx'],
            "response": re.sub(r"\n\s*\n", "\n", response),
            "score": round(score, 2)
        })

    final_score = (sum(result) / len(result)) if result else 0.0
    return response_ls, round(final_score, 2)

if __name__ == "__main__":
    ls=[]
    for model in ["qwen-max"]:
            result=  evaluate(model)
            print(result)
            ls.append(result)
    print(ls)