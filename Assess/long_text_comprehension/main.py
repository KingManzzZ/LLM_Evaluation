from Assess.long_text_comprehension import context_understanding
from Assess.long_text_comprehension import extract
from Assess.long_text_comprehension import memory_retention

def main(model, qs_list):
    """
    大模型长文本理解能力评测统一入口。
    子指标：上下文理解、信息提取、记忆能力。
    """
    return evaluate(model, qs_list)

def evaluate(model, qs_list):
    # 分组题目
    groups = {
        'context_understanding': [],
        'information_extraction': [],
        'memory_retention': []
    }

    # 根据 min_metric 或其他标识进行分组
    # 假设数据集中的 min_metric 对应如下：
    # context_understanding: 上下文理解
    # information_extraction: 信息提取
    # memory_retention: 记忆能力
    for item in qs_list:
        metric = item.get("min_metric", "")
        if "context" in metric.lower():
            groups['context_understanding'].append(item)
        elif "extract" in metric.lower():
            groups['information_extraction'].append(item)
        elif "memory" in metric.lower() or "retention" in metric.lower():
            groups['memory_retention'].append(item)
        else:
            # 默认分组逻辑，可以根据实际情况调整
            groups['context_understanding'].append(item)

    all_response = []
    scores = {
        'context_understanding': 0.0,
        'information_extraction': 0.0,
        'memory_retention': 0.0
    }

    # 各子指标权重 (可根据需求调整，当前设为均权或 4:3:3)
    weights = {
        'context_understanding': 0.4,
        'information_extraction': 0.3,
        'memory_retention': 0.3
    }

    # 1. 上下文理解评测
    if groups['context_understanding']:
        try:
            print(f"开始评测 [上下文理解] (数量: {len(groups['context_understanding'])})")
            resp, score = context_understanding.evaluate(model, groups['context_understanding'])
            scores['context_understanding'] = score
            all_response.extend(resp)
        except Exception as e:
            print(f"上下文理解评测出错: {e}")

    # 2. 信息提取评测
    if groups['information_extraction']:
        try:
            print(f"开始评测 [信息提取] (数量: {len(groups['information_extraction'])})")
            resp, score = extract.evaluate(model, groups['information_extraction'])
            scores['information_extraction'] = score
            all_response.extend(resp)
        except Exception as e:
            print(f"信息提取评测出错: {e}")

    # 3. 记忆能力评测
    if groups['memory_retention']:
        try:
            print(f"开始评测 [记忆能力] (数量: {len(groups['memory_retention'])})")
            resp, score = memory_retention.evaluate(model, groups['memory_retention'])
            scores['memory_retention'] = score
            all_response.extend(resp)
        except Exception as e:
            print(f"记忆能力评测出错: {e}")

    # 按照 dataId 排序
    all_response.sort(key=lambda x: x.get("dataId", 0))

    # 计算总分
    final_score = sum(scores[k] * weights[k] for k in weights)

    final_report = {
        'context_understanding': round(scores['context_understanding'], 2),
        'information_extraction': round(scores['information_extraction'], 2),
        'memory_retention': round(scores['memory_retention'], 2),
        'final_score': round(final_score, 2)
    }

    print(f"\n长文本理解能力评测完成！最终得分: {final_report['final_score']}")
    return all_response, final_report

if __name__ == "__main__":
    # 模拟测试
    test_qs = [
        {'rowIdx': 0, 'question': '这是一段长文本...', 'answer': '答案', 'min_metric': 'context_understanding'}
    ]
    responses, report = evaluate("DeepSeek-V3", test_qs)
    print(report)

