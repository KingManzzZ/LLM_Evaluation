import time
import json
import os
import sys
import re

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api


def split_text_into_chunks(text, max_chars=3000):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # 按段落分割（以换行符为分隔符）
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= max_chars:
            if current_chunk:
                current_chunk += '\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            if len(paragraph) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                sentences = paragraph.split('。')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        if current_chunk:
                            current_chunk += '。' + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def send_long_text_as_conversation(model, long_text, question, max_chars=3000):
    """将长文本分块发送给模型，使用系统级消息堆叠优化逻辑"""
    text_chunks = split_text_into_chunks(long_text, max_chars)
    print(f"长文本被切割为 {len(text_chunks)} 个块")

    messages = [
        {"role": "system", "content": "你是一个拥有超强记忆力的助手。接下来的消息将分段发送一段长文本，请你保持极简回复（仅回复OK），直到最后接收到具体的提问后，再根据全文内容进行详细回答。"}
    ]

    for idx, chunk in enumerate(text_chunks):
        content = f"[长文本数据 {idx + 1}/{len(text_chunks)}]:\n\n{chunk}"
        messages.append({"role": "user", "content": content})

        response = call_api(model, None, messages=messages)
        print(f"第 {idx + 1} 块发送完成，模型状态: {response.strip()[:10]}")

        messages.append({"role": "assistant", "content": response})
        time.sleep(0.5)

    final_query = f"以上是全部参考文本。现在请回答问题：{question}"
    messages.append({"role": "user", "content": final_query})

    final_answer = call_api(model, None, messages=messages)
    return final_answer


def llm_judge(pred, ref, question, judge_model="Qwen-max"):
    """
    引入第三方模型作为裁判，判定回答准确性
    """
    judge_prompt = f"""你是一个专业的测评裁判。请根据以下提供的[参考答案]和[模型回答]，判定[模型回答]是否准确解答了[问题]。

[问题]: {question}
[参考答案]: {ref}
[模型回答]: {pred}

要求：
1. 忽略语气、措辞的差异，只关注事实一致性。
2. 给出 0 到 1 之间的分数（0为完全错误，1为完全正确）。
3. 请严格按照以下格式返回结果：【分数】你的评分数字
"""
    try:
        response = call_api(judge_model, judge_prompt)
        match = re.search(r'【分数】\s*([0-1](?:\.\d+)?)', response)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"裁判模型调用失败: {e}")

    # 兜底逻辑：如果裁判失败，使用备用的简单子串匹配
    if ref in pred or pred in ref:
        return 1.0
    return 0.0


def deal_with_judge(model, item, judge_model="Qwen-max"):
    """使用裁判模型的处理逻辑"""
    print("----- [记忆能力-含裁判] 处理第{}条数据 -----".format(item['rowIdx'] + 1))

    long_text = item.get('context', '') or item.get('text', '') or item.get('question', '')
    if isinstance(long_text, dict):
        long_text = long_text.get('content', '') or long_text.get('text', '')

    question = item.get('query', '') or item.get('question', '')
    if isinstance(question, dict):
        question = question.get('query', '') or question.get('content', '')

    if not long_text or not question:
        return "ERROR", 0.0

    try:
        answer = send_long_text_as_conversation(model, long_text, question)
        standard_answer = str(item.get('answer', ''))

        # 调用裁判模型进行评分
        score = llm_judge(answer, standard_answer, question, judge_model)
        print(f"模型回答: {answer[:100]}...")
        print(f"裁判得分: {score}")

        return answer, score
    except Exception as e:
        print(f"处理失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None, use_judge=True):
    """评估模型的记忆能力"""
    if qs_list is None:
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "记忆能力", "memory_retention.json")
        with open(direction, 'r', encoding='utf-8') as f:
            qs_list = json.load(f)

    result = []
    response_ls = []

    for item in qs_list:
        if use_judge:
            response, score = deal_with_judge(model, item)
        else:
            # 兼容旧逻辑
            from memory_retention import deal
            response, score = deal(model, item)

        result.append(score)
        response_ls.append({
            "dataId": item.get('rowIdx', 0),
            "response": re.sub(r"\n\s*\n", "\n", str(response)),
            "score": round(score, 2)
        })

    avg_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(avg_score, 2)


if __name__ == "__main__":
    for model in ["DeepSeek-V3"]:
        print(f"正在执行{model}模型 (含第三方裁判)")
        responses, score = evaluate(model, use_judge=True)
        print(f"平均得分: {score}")

