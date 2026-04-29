import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

api_setting = {
        "DeepSeek-V3": {
            "model_name": "deepseek-ai/DeepSeek-V3",
            "api_key": os.getenv("DeepSeek-V3_API_KEY"),
            "base_url": os.getenv("DeepSeek-V3_BASE_URL")},
        "Qwen-max": {
            "model_name": "qwen-max-2025-01-25",
            "api_key": os.getenv("Qwen-max_API_KEY"),
            "base_url": os.getenv("Qwen-max_BASE_URL")},
        "yi-lightning": {
            "model_name": "yi-lightning",
            "api_key": os.getenv("yi-lightning_API_KEY"),
            "base_url": os.getenv("yi-lightning_BASE_URL")},
        "ERNIE-4.0-8k": {
            "model_name": "ernie-4.0-8k",
            "api_key": os.getenv("ERNIE-4.0-8k_API_KEY"),
            "base_url": os.getenv("ERNIE-4.0-8k_BASE_URL")},
        "gpt-4o-mini": {
            "model_name": "gpt-4o-mini",
            "api_key": os.getenv("gpt-4o-mini_API_KEY")
        }
    }

def call_api(model, prompt=None, retry_limit=3, messages=None, temperature=0.9) -> str:
    """统一的 API 调用函数，具有改进的错误处理和重试逻辑。

    参数:
        model: api_setting 中的模型标识符。
        prompt: 单个字符串提示词（如果 messages 为 None 则使用）。
        retry_limit: 尝试次数。
        messages: 消息字典列表 [{"role": "user/assistant", "content": "..."}]。
        temperature: 采样温度。
    """
    from openai import OpenAI
    import time

    if model not in api_setting:
        print(f"错误: 配置文件中未找到模型 {model}。")
        return "ERROR_CONFIG"

    if messages is None:
        if prompt is None:
            return "ERROR_INPUT"
        messages = [{"role": "user", "content": str(prompt)}]

    for attempt in range(retry_limit):
        try:
            config = api_setting[model]
            client_args = {"api_key": config["api_key"]}
            if "base_url" in config:
                client_args["base_url"] = config["base_url"]

            client = OpenAI(**client_args)
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                presence_penalty=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"模型 {model} API 调用失败 (尝试 {attempt + 1}/{retry_limit}): {e}")
            if attempt < retry_limit - 1:
                time.sleep(2)

    print(f"严重错误: 模型 {model} 在 {retry_limit} 次尝试后 API 调用依然失败。")
    return "ERROR"

