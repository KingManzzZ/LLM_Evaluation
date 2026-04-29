import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 设置缓存路径为项目根目录下的 ModelCache 文件夹
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ModelCache")

class ModelCache:
    """单例类，用于在评估套件中缓存大型 NLP 模型，减少重复加载。"""
    _pipelines = {}
    _models = {}

    @classmethod
    def get_pipeline(cls, task, model_name, **kwargs):
        """获取共享 pipeline 的统一方法（例如文本分类）。"""
        key = (task, model_name)
        if key not in cls._pipelines:
            print(f"信息: 正在加载任务 '{task}' 的模型 '{model_name}'...")
            # 如果有 GPU 则使用
            device = 0 if torch.cuda.is_available() else -1
            # 设置 model_kwargs 包含 cache_dir
            cls._pipelines[key] = pipeline(task, model=model_name, device=device, model_kwargs={"cache_dir": CACHE_DIR}, **kwargs)
        return cls._pipelines[key]

    @classmethod
    def get_model_and_tokenizer(cls, model_class, tokenizer_class, model_name):
        """标准化的模型加载方法，用于更精细的控制。"""
        if model_name not in cls._models:
            print(f"信息: 正在加载模型和分词器 '{model_name}'...")
            tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=CACHE_DIR)
            model = model_class.from_pretrained(model_name, cache_dir=CACHE_DIR)
            if torch.cuda.is_available():
                model = model.to('cuda')
            cls._models[model_name] = (model, tokenizer)
        return cls._models[model_name]

# 辅助别名
def get_classifier(model_name, **kwargs):
    return ModelCache.get_pipeline("text-classification", model_name, **kwargs)

def get_feature_extractor(model_name, **kwargs):
    return ModelCache.get_pipeline("feature-extraction", model_name, **kwargs)


