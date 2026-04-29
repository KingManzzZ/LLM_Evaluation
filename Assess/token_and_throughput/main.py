import sys
import os
import statistics
import time
import json
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from time import perf_counter_ns

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import api_setting


@dataclass
class PerformanceMetrics:
    first_token_latency_ms: float  # 首token延迟(毫秒)
    tokens_per_second: float  # token生成速度(tokens/秒)
    total_time_sec: float  # 总耗时(秒)
    total_tokens: int  # 总token数
    inter_token_latencies_ms: List[float]  # token间延迟列表(毫秒)
    time_to_last_token_ms: float  # 到最后token的时延(毫秒)
    p95_latency_ms: float  # 95分位延迟(毫秒)
    p99_latency_ms: float  # 99分位延迟(毫秒)
    min_latency_ms: float  # 最小延迟(毫秒)
    max_latency_ms: float  # 最大延迟(毫秒)


class StreamAnalyzer:
    def __init__(self):
        self.start_time_ns: Optional[int] = None
        self.first_token_time_ns: Optional[int] = None
        self.last_token_time_ns: Optional[int] = None
        self.token_count: int = 0
        self.prev_token_time_ns: Optional[int] = None
        self.inter_token_latencies_ns: List[int] = []
        self.token_lengths: List[int] = []

    def reset(self):
        """在每次请求前重置所有时间相关变量"""
        self.start_time_ns = None
        self.first_token_time_ns = None
        self.last_token_time_ns = None
        self.prev_token_time_ns = None
        self.inter_token_latencies_ns = []
        self.token_lengths = []
    def record_start(self):
        """记录请求开始时间(纳秒精度)"""
        self.start_time_ns = perf_counter_ns()

    def record_token(self, token: str):
        """记录token到达"""
        current_time_ns = perf_counter_ns()
        token_length = len(token)

        # 如果是第一个token
        if self.first_token_time_ns is None:
            self.first_token_time_ns = current_time_ns
            self.prev_token_time_ns = current_time_ns
        else:
            # 计算token间延迟
            delta_ns = current_time_ns - self.prev_token_time_ns
            self.inter_token_latencies_ns.append(delta_ns)
            self.prev_token_time_ns = current_time_ns

        self.token_count += 1
        self.token_lengths.append(token_length)
        self.last_token_time_ns = current_time_ns

    def calculate_metrics(self) -> PerformanceMetrics:
        """计算所有性能指标"""

        if None in [self.start_time_ns, self.first_token_time_ns, self.last_token_time_ns]:
            raise ValueError("Insufficient data to calculate metrics")

        # 转换纳秒为秒
        ns_to_ms = 1_000_000
        ns_to_sec = 1_000_000_000

        # 计算基本指标
        first_token_latency_ms = (self.first_token_time_ns - self.start_time_ns) / ns_to_ms
        total_time_sec = (self.last_token_time_ns - self.start_time_ns) / ns_to_sec#开始时间到最后一个token的时长
        time_to_last_token_ms = (self.last_token_time_ns - self.first_token_time_ns) / ns_to_ms#首token和最后一个token间的时长

        # 计算吞吐量和生成速度
        tokens_per_second = self.token_count / total_time_sec if total_time_sec > 0 else 0

        # 计算延迟统计信息(转换为毫秒)
        latencies_ms = [lat_ns / ns_to_ms for lat_ns in self.inter_token_latencies_ns]
        if latencies_ms:
            quantiles = statistics.quantiles(latencies_ms, n=100)
            p95_latency_ms = quantiles[94]
            p99_latency_ms = quantiles[98]
            min_latency_ms = min(latencies_ms)
            max_latency_ms = max(latencies_ms)


        else:
            p95_latency_ms = 0
            p99_latency_ms = 0
            min_latency_ms = 0
            max_latency_ms = 0


        return PerformanceMetrics(
            first_token_latency_ms=first_token_latency_ms,
            tokens_per_second=tokens_per_second,
            total_time_sec=total_time_sec,
            total_tokens=self.token_count,
            inter_token_latencies_ms=latencies_ms,
            time_to_last_token_ms=time_to_last_token_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
        )


def analyze_openai_stream(analyzer: StreamAnalyzer, prompt: str, model):
    """使用OpenAI流式API并分析性能，返回性能指标和响应内容"""
    response_content = ""
    
    for attempt in range(3):
        try:
            client_args = {"api_key": api_setting[model]["api_key"]}
            if "base_url" in api_setting[model]:
                client_args["base_url"] = api_setting[model]["base_url"]
            client = OpenAI(**client_args)

            analyzer.record_start()
            stream = client.chat.completions.create(
                model=api_setting[model]["model_name"],  # 可根据需要切换模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # 控制生成随机性
                top_p=0.95,
                presence_penalty=0.5,  # 降低重复短语概率
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    analyzer.record_token(token)
                    response_content += token
            return analyzer.calculate_metrics(), response_content
        except Exception as e:
            print(f"API调用失败（第{attempt + 1}次重试）: {e}")
            time.sleep(4)  # 错误等待时间
    raise ConnectionError("API调用失败")


def print_metrics(metrics: PerformanceMetrics,n):
    """打印性能指标"""
    print(("\n=== 性能指标报告{}===").format(n))
    print(f"首token延迟: {metrics.first_token_latency_ms:.2f} 毫秒")
    print(f"Token生成速度: {metrics.tokens_per_second:.2f} tokens/秒")
    print(f"总token数: {metrics.total_tokens}")
    print(f"总耗时: {metrics.total_time_sec:.2f} 秒")
    print(f"到最后token的时延: {metrics.time_to_last_token_ms:.2f} 毫秒")
    print(f"P95延迟: {metrics.p95_latency_ms:.2f} 毫秒")
    print(f"P99延迟: {metrics.p99_latency_ms:.2f} 毫秒")
    print(f"最小延迟: {metrics.min_latency_ms:.2f} 毫秒")
    print(f"最大延迟: {metrics.max_latency_ms:.2f} 毫秒")
    print("==================\n")


def calculate_scores(all_metrics: List[PerformanceMetrics]) -> List[float]:
    METRIC_WEIGHTS = {
        'first_token_latency_ms': 0.20,
        'tokens_per_second': 0.20,
        'total_time_sec': 0.10,
        'total_tokens': 0.10,
        'time_to_last_token_ms': 0.10,
        'p95_latency_ms': 0.10,
        'p99_latency_ms': 0.10,
        'min_latency_ms': 0.05,
        'max_latency_ms': 0.05
    }
    """根据权重和稳定性计算综合得分（0-100）"""
    if not all_metrics:
        return []

    # 收集指标极值和波动范围
    field_data = {field: {'values': [], 'volatility': 0} for field in METRIC_WEIGHTS}
    for metric in all_metrics:
        for field in METRIC_WEIGHTS:
            field_data[field]['values'].append(getattr(metric, field))

    # 计算全局波动性基准
    max_volatility = 0
    for field in METRIC_WEIGHTS:
        values = field_data[field]['values']
        if not values:
            continue
        field_volatility = max(values) - min(values)
        field_data[field]['volatility'] = field_volatility
        max_volatility = max(max_volatility, field_volatility)

    # 计算每个测试项的得分
    scores = []
    for metric in all_metrics:
        total = 0.0
        for field, weight in METRIC_WEIGHTS.items():
            values = field_data[field]['values']
            if not values:
                continue
            # 计算位置得分
            val = getattr(metric, field)
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                pos_score = 100
            else:
                if field == 'tokens_per_second':
                    pos_score = ((val - min_val) / (max_val - min_val)) * 100#正向指标，越大越好
                else:
                    pos_score = ((max_val - val) / (max_val - min_val)) * 100#反向指标，越小越好
                pos_score = max(0, min(100, pos_score))

            # 计算稳定性系数（波动越小系数越高）
            stability_coeff = 1 - (field_data[field]['volatility'] / max_volatility) if max_volatility > 0 else 1
            adjusted_score = pos_score * stability_coeff

            total += adjusted_score * weight
        scores.append(round(total, 2))
    return scores


def main(model, qs_list):
    """
    Standardized entry point for performance assessment.
    Returns: (all_response, final_score_dict)
    """
    return run_benchmark(model, qs_list)

def run_benchmark(model, qs_list):
    all_metrics = []
    all_response = []
    analyzer = StreamAnalyzer()

    print(f"INFO: Starting Performance Benchmark for {model}")

    for item in qs_list:
        idx = item.get("rowIdx", 0)
        analyzer.reset()

        # Consistent prompt preparation
        if item.get("type_index") == "choice":
            prompt = str(item.get("question", "")) + "\nOptions: " + str(item.get("options", ""))
        elif item.get("type_index") == "compare_question":
            prompt = str(item.get("question", {}).get("A", "")) + "\n" + str(item.get("options", {}).get("B", ""))
        else:
            prompt = str(item.get("question", ""))

        try:
            metrics, response_content = analyze_openai_stream(analyzer, prompt, model)
            all_metrics.append(metrics)

            # Record result
            all_response.append({
                "dataId": idx,
                "response": response_content
            })
        except Exception as e:
            print(f"ERROR: Item {idx} performance test failed: {e}")

    if all_metrics:
        scores = calculate_scores(all_metrics)
        final_score = sum(scores) / len(scores)

        final_result = {
            "single_scores": scores,
            "final_score": round(final_score, 2)
        }
        return all_response, final_result

    return [], 0.0





# 示例用法
if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])  # 从命令行参数获取
    else:
        ls=[{
        "rowIdx": 0,
        "question": "《红楼梦》中'林黛玉焚稿断痴情'的章节发生在哪个传统节日期间？",
        "options": [
            "A: 端午节",
            "B: 重阳节",
            "C: 元宵节",
            "D: 清明节"
        ],
        "answer": "B",
        "type_index": "choice"
    },
    {
        "rowIdx": 1,
        "question": "太阳系中木星的卫星数量超过80颗",
        "answer": "正确",
        "type_index": "judgment"
    },
    {
        "rowIdx": 2,
        "question": "中国航天史上首位女航天员刘洋执行的是哪次载人飞行任务？",
        "options": [
            "A: 神舟五号",
            "B: 神舟七号",
            "C: 神舟九号",
            "D: 天宫二号"
        ],
        "answer": "C",
        "type_index": "choice"
    }] # 从标准输入获取
        run_benchmark("yi-lightning",ls)

