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
        # 如果当前段落加入后不超过限制，则加入
        if len(current_chunk) + len(paragraph) + 1 <= max_chars:
            if current_chunk:
                current_chunk += '\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            # 如果当前段落本身超过限制，需要进一步分割
            if len(paragraph) > max_chars:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 按句子分割长段落
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
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def send_long_text_as_conversation(model, long_text, question, max_chars=3000):
    """将长文本分块发送给模型，使用系统级消息堆叠优化逻辑
    """
    # 切割文本
    text_chunks = split_text_into_chunks(long_text, max_chars)
    print(f"长文本被切割为 {len(text_chunks)} 个块")
    
    # 构造消息序列
    # 初始系统消息，明确告知后续是长文本输入
    messages = [
        {"role": "system", "content": "你是一个拥有超强记忆力的助手。接下来的消息将分段发送一段长文本，请你保持极简回复（仅回复OK），直到最后接收到具体的提问后，再根据全文内容进行详细回答。"}
    ]

    # 逐块添加文本并发送，要求模型保持极简回复以减少上下文污染
    for idx, chunk in enumerate(text_chunks):
        content = f"[长文本数据 {idx + 1}/{len(text_chunks)}]:\n\n{chunk}"
        messages.append({"role": "user", "content": content})

        # 实时发送并获取简单的确认，确保模型“看到”了这部分数据
        response = call_api(model, None, messages=messages)
        print(f"第 {idx + 1} 块发送完成，模型状态: {response.strip()[:10]}")

        # 将模型简单的确认存入历史
        messages.append({"role": "assistant", "content": response})

        # 避免请求频率过快
        time.sleep(0.5)

    # 最终提问
    final_query = f"以上是全部参考文本。现在请回答问题：{question}"
    messages.append({"role": "user", "content": final_query})

    # 调用 API 获取最终详细答案
    final_answer = call_api(model, None, messages=messages)
    return final_answer


def deal(model, item):
    """处理单个测试项，针对长文本记忆稳定性"""
    print("----- [记忆能力] 处理第{}条数据 -----".format(item['rowIdx'] + 1))

    # 获取长文本和问题
    long_text = item.get('context', '') or item.get('text', '') or item.get('question', '')
    if isinstance(long_text, dict):
        long_text = long_text.get('content', '') or long_text.get('text', '')

    question = item.get('query', '') or item.get('question', '')
    if isinstance(question, dict):
        question = question.get('query', '') or question.get('content', '')

    if not long_text or not question:
        print("警告：内容缺失")
        return "ERROR", 0.0

    try:
        # 使用分块发送的方式处理长文本（保持原有的大海捞针/分块逻辑）
        answer = send_long_text_as_conversation(model, long_text, question)
        print(f"模型回答摘要: {answer[:100]}...")

        # 结果判定：基于子串匹配
        standard_answer = str(item.get('answer', ''))

        # 精确或子串包含
        if standard_answer in answer or answer in standard_answer:
            return answer, 1.0
        else:
            # 关键词柔性分
            keywords = standard_answer.split(',')
            match_count = sum(1 for kw in keywords if kw.strip() in answer)
            score = match_count / len(keywords) if keywords else 0
            return answer, score
    except Exception as e:
        print(f"处理失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None):
    """评估模型的记忆能力"""
    if qs_list is None:
        # 从文件读取数据
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "记忆能力", "memory_retention.json")
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
                "response": re.sub(r"\n\s*\n", "\n", str(response)),
                "score": round(score, 2)
            })
    except Exception as e:
        print(f"记忆评测中断: {e}")

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
