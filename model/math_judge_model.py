import os
import torch
from transformers import AutoTokenizer, AutoModel
import re


# 获取当前项目的根目录 (对应 QA 根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将模型缓存路径设置到项目下的 ModelCache 文件夹
CACHE_DIR = os.path.join(BASE_DIR, "ModelCache")

# 使用指定的 MathBERT 模型: https://huggingface.co/tbs17/MathBERT
# 该模型专为数学公式和推理文本的语义对齐设计
MODEL_NAME = "tbs17/MathBERT"

class MathBERTJudge:
    def __init__(self):
        print(f"正在加载数学推理专用模型: {MODEL_NAME}...")
        try:
            # 加载专为数学任务预训练的 MathBERT
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            self.model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            print("MathBERT 模型加载成功。")
        except Exception as e:
            print(f"加载 MathBERT 失败: {e}，尝试使用备用模型 bert-base-uncased")
            fallback_model = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, cache_dir=CACHE_DIR)
            self.model = AutoModel.from_pretrained(fallback_model, cache_dir=CACHE_DIR)

    def clean_math_text(self, text):
        """
        简单的数学文本清洗，标准化 LaTeX 符号以提高匹配精度
        """
        if not isinstance(text, str):
            text = str(text)
        # 统一常见的 LaTeX 括号表示
        text = text.replace('\\(', '$').replace('\\)', '$')
        text = text.replace('\\[', '$$').replace('\\]', '$$')
        return text.strip()

    def get_embedding(self, text):
        """获取文本向量（[CLS] token）"""
        if not text or not text.strip():
            # 返回全零向量，维度对应 BERT 的 768
            return torch.zeros((1, 768))

        cleaned_text = self.clean_math_text(text)

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=512,  # 增加长度以适应复杂的推理过程
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取 [CLS] 向量作为句向量，捕获整体推理逻辑
        return outputs.last_hidden_state[:, 0, :]

    def compare_steps(self, step1, step2):
        """
        计算两个数学推理步骤/过程之间的余弦相似度
        """
        if not step1.strip() or not step2.strip():
            return 0.0

        try:
            emb1 = self.get_embedding(step1)
            emb2 = self.get_embedding(step2)

            # 计算余弦相似度
            sim = torch.cosine_similarity(emb1, emb2).item()
            # 限制在 0-1 之间并保留 4 位精度
            return max(0.0, min(1.0, round(sim, 4)))
        except Exception as e:
            print(f"相似度计算异常: {e}")
            return 0.0

# 单例模式，避免在批处理评测中重复加载大型模型
math_judge = None

def get_math_similarity(text1, text2):
    """
    外部调用接口：获取两段数学解题过程的语义相似度得分
    """
    global math_judge
    if math_judge is None:
        math_judge = MathBERTJudge()
    return math_judge.compare_steps(str(text1), str(text2))

if __name__ == "__main__":
    # 测试代码
    t1 = "Let x be the number of apples. x + 5 = 10, so x = 5."
    t2 = "Suppose there are x apples. The equation is x + 5 = 10, which means x equals 5."
    print(f"[测试] 推理过程1: {t1}")
    print(f"[测试] 推理过程2: {t2}")
    print(f"[测试] MathBERT 语义相似度得分: {get_math_similarity(t1, t2)}")

