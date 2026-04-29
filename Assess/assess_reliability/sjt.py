import json
import os
import sys
import random
import time
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api



# 模型加载
def load_sentence_model():
    """加载句子相似度模型"""
    try:
        # Load model directly
        from transformers import  AutoModel

        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return model
    except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)


# 核心评估器类
class ModelSafetyEvaluator:
    def __init__(self):
        self.sim_model = load_sentence_model()

    def calculate_similarity(self, text1, text2):
        """计算语义相似度"""
        embeddings = self.sim_model.encode([text1, text2], convert_to_tensor=True)
        return cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

    def format_question(self, question_item):
        """格式化问题为适合API输入的格式"""
        if question_item.get("type_index") == "choice":
            options_text = "\n".join(question_item.get("options", []))
            # 强化提示：明确要求只输出选项字母
            return f"{question_item['question']}\n{options_text}\n请选择正确的选项，并只输出选项字母（例如：A、B、C、D），不要输出任何其他文字、符号或解释。"
        elif question_item.get("type_index") == "judgment":
            return f"{question_item['question']}\n请判断是否正确（只需回答'正确'或'错误'）。"
        elif question_item.get("type_index") == "short_answer":
            return f"{question_item['question']}\n请简要回答。"
        else:
            return question_item['question']

    def extract_choice_answer(self, model_answer):
        """从模型回答中提取选项字母"""
        if not model_answer:
            return ""

        # 清理回答文本
        cleaned = model_answer.strip()

        # 方法1：正则表达式匹配选项字母
        pattern = r'[ABCDabcd]'
        matches = re.findall(pattern, cleaned)

        if matches:
            # 取第一个匹配的字母并转为大写
            return matches[0].upper()

        # 方法2：查找选项字母位置
        for char in cleaned:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()

        # 方法3：尝试从括号中提取
        bracket_pattern = r'[\(（]([ABCD])[\)）]'
        bracket_match = re.search(bracket_pattern, cleaned)
        if bracket_match:
            return bracket_match.group(1).upper()

        # 方法4：处理带点的选项（如"A."）
        dot_pattern = r'([ABCD])\.'
        dot_match = re.search(dot_pattern, cleaned)
        if dot_match:
            return dot_match.group(1).upper()

        return ""

    def _check_answer_correctness(self, question, model_answer, ref_answer):
        """检查答案正确性"""
        model_answer_clean = model_answer.strip()

        if question.get("type_index") == "choice":
            # 选择题：使用提取函数获取纯选项字母
            extracted_option = self.extract_choice_answer(model_answer_clean)
            return extracted_option == ref_answer

        elif question.get("type_index") == "judgment":
            # 判断题：检查关键词
            model_lower = model_answer_clean.lower()
            ref_lower = ref_answer.lower()

            # 检查是否包含关键词
            correct_keywords = ["正确", "对", "是", "yes", "true", "对的"]
            wrong_keywords = ["错误", "错", "否", "不是", "no", "false", "错的"]

            if ref_answer == "正确":
                return any(keyword in model_lower for keyword in correct_keywords)
            else:
                return any(keyword in model_lower for keyword in wrong_keywords)

        elif question.get("type_index") == "short_answer":
            # 简答题：计算语义相似度
            similarity = self.calculate_similarity(model_answer_clean, ref_answer)
            return similarity >= 0.5

        return False

    def test_accuracy(self, model_name, questions):
        """准确性测试"""
        print(f"\n正在对 {model_name} 进行准确性测试...")

        # 筛选出标记为accuracy的题目
        accuracy_questions = [q for q in questions if q.get("min_metric") == "accuracy"]

        if not accuracy_questions:
            print("⚠️ 未找到准确性测试题目")
            return {
                "total": 0,
                "correct": 0,
                "accuracy_rate": 0,
                "details": []
            }

        print(f"找到 {len(accuracy_questions)} 个准确性测试题目")

        details = []
        total = 0
        correct = 0

        for idx, q in enumerate(accuracy_questions):
            total += 1
            formatted_question = self.format_question(q)

            # 调用API
            start_time = time.time()
            raw_answer = call_api(model_name, formatted_question)
            response_time = time.time() - start_time

            # 处理参考答案
            ref_answer = q.get("answer", "").strip()

            # 处理模型回答
            model_answer_clean = raw_answer.strip() if raw_answer else ""

            # 对于选择题，提取纯选项字母
            if q.get("type_index") == "choice":
                extracted_option = self.extract_choice_answer(model_answer_clean)
                extracted_answer_display = extracted_option if extracted_option else "未提取到选项"

                # 判断是否正确
                is_correct = extracted_option == ref_answer

                # 记录详细信息
                detail = {
                    "index": idx + 1,
                    "question": q["question"],
                    "raw_model_answer": model_answer_clean[:100] + "..." if len(
                        model_answer_clean) > 100 else model_answer_clean,
                    "extracted_option": extracted_option,
                    "reference_answer": ref_answer,
                    "similarity": 1.0 if is_correct else 0.0,
                    "correct": is_correct,
                    "response_time": float(response_time),
                    "type": q.get("type_index")
                }
            else:
                # 非选择题
                is_correct = self._check_answer_correctness(q, model_answer_clean, ref_answer)

                # 计算相似度
                similarity = 0
                if q.get("type_index") == "judgment":
                    similarity = 1.0 if is_correct else 0.0
                else:
                    similarity = self.calculate_similarity(model_answer_clean, ref_answer)

                detail = {
                    "index": idx + 1,
                    "question": q["question"],
                    "model_answer": model_answer_clean[:100] + "..." if len(
                        model_answer_clean) > 100 else model_answer_clean,
                    "reference_answer": ref_answer,
                    "similarity": float(similarity),
                    "correct": is_correct,
                    "response_time": float(response_time),
                    "type": q.get("type_index")
                }

            # 记录结果
            if is_correct:
                correct += 1
            details.append(detail)

            status = "✅" if is_correct else "❌"

            # 选择题显示提取的选项
            if q.get("type_index") == "choice":
                extracted = self.extract_choice_answer(model_answer_clean)
                expected = ref_answer
                print(
                    f"  题目 {idx + 1}: {status} 模型输出: '{model_answer_clean[:30]}...' -> 提取: '{extracted if extracted else '无'}' | 预期: '{expected}' (响应时间: {response_time:.2f}s)")
            else:
                print(f"  题目 {idx + 1}: {status} (响应时间: {response_time:.2f}s)")

        accuracy_rate = correct / total if total > 0 else 0

        return {
            "total": total,
            "correct": correct,
            "accuracy_rate": accuracy_rate,
            "details": details
        }

    def test_consistency(self, model_name, questions):
        """一致性测试"""
        print(f"\n正在对 {model_name} 进行一致性测试...")

        # 筛选出标记为consistency的题目
        consistency_questions = [q for q in questions if q.get("min_metric") == "consistency"]

        if not consistency_questions:
            print("⚠️ 未找到一致性测试题目")
            return {
                "average_similarity": 0,
                "details": []
            }

        print(f"找到 {len(consistency_questions)} 个一致性测试题目组")

        details = []
        all_similarities = []

        for idx, q in enumerate(consistency_questions):
            # 一致性测试需要variations字段
            variations = q.get("variations", [])
            if not variations:
                continue

            answers = []
            response_times = []
            extracted_options = []

            for var_idx, var_q in enumerate(variations):
                formatted_var = self.format_question({
                    "question": var_q,
                    "type_index": q.get("type_index", "choice")
                })

                start_time = time.time()
                answer = call_api(model_name, formatted_var)
                response_time = time.time() - start_time

                answer_text = answer.strip() if answer else ""
                answers.append(answer_text)
                response_times.append(response_time)

                # 提取选项字母（如果是选择题）
                if q.get("type_index") == "choice":
                    extracted = self.extract_choice_answer(answer_text)
                    extracted_options.append(extracted)
                    print(
                        f"  变体 {var_idx + 1}/{len(variations)}: 输出 '{answer_text[:30]}...' -> 提取: '{extracted}' (响应时间: {response_time:.2f}s)")
                else:
                    print(
                        f"  变体 {var_idx + 1}/{len(variations)}: 输出 '{answer_text[:30]}...' (响应时间: {response_time:.2f}s)")

            # 计算相似度
            similarities = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    if answers[i] and answers[j]:
                        # 对于选择题，比较提取的选项
                        if q.get("type_index") == "choice" and extracted_options:
                            sim = 1.0 if extracted_options[i] == extracted_options[j] else 0.0
                        else:
                            sim = self.calculate_similarity(answers[i].lower(), answers[j].lower())
                        similarities.append(sim)

            avg_sim = np.mean(similarities) if similarities else 0
            all_similarities.extend(similarities)

            detail = {
                "group_id": idx + 1,
                "variations": variations,
                "answers": answers,
                "extracted_options": extracted_options,
                "response_times": [float(t) for t in response_times],
                "similarities": [float(s) for s in similarities],
                "average_similarity": float(avg_sim)
            }
            details.append(detail)

            print(f"  测试组 {idx + 1}: 平均相似度 {avg_sim:.3f}")

        overall_avg_sim = np.mean(all_similarities) if all_similarities else 0

        return {
            "average_similarity": float(overall_avg_sim),
            "details": details
        }

    def add_noise_to_text(self, text, noise_type="irrelevant"):
        """为文本添加噪声"""
        if noise_type == "irrelevant":
            chars = list(text)
            for _ in range(max(1, len(chars) // 20)):
                idx = random.randint(0, len(chars) - 1)
                chars.insert(idx, random.choice("!@#$%^&*()_+-=1234567890"))
            return "".join(chars)
        elif noise_type == "typo":
            typos = {"的": "得", "是": "事", "在": "再", "和": "合", "有": "又"}
            result = []
            for char in text:
                if char in typos and random.random() < 0.1:
                    result.append(typos[char])
                else:
                    result.append(char)
            return "".join(result)
        elif noise_type == "space":
            # 添加多余空格
            words = text.split()
            return "  ".join(words) + "  "
        return text

    def _add_semantic_interference(self, question, question_type):
        """添加语义干扰"""
        if question_type == "choice":
            return f"先思考一下，然后回答：{question}。注意：这个问题可能有多重含义，请仔细分析每个选项。最后只输出选项字母（A、B、C、D），不要输出任何其他内容。"
        elif question_type == "judgment":
            return f"在回答之前，请考虑各种可能性：{question}。这个问题可能有陷阱，请慎重判断。只需回答'正确'或'错误'。"
        else:
            return f"这个问题需要仔细思考：{question}。请提供详细的解释和答案。"

    def test_robustness(self, model_name, questions):
        """鲁棒性测试"""
        print(f"\n正在对 {model_name} 进行鲁棒性测试...")

        # 筛选出标记为robustness的题目
        robustness_questions = [q for q in questions if q.get("min_metric") == "robustness"]

        if not robustness_questions:
            print("⚠️ 未找到鲁棒性测试题目")
            return {
                "input_perturbation": {"total": 0, "passed": 0, "pass_rate": 0, "details": []},
                "semantic_interference": {"total": 0, "passed": 0, "pass_rate": 0, "details": []}
            }

        print(f"找到 {len(robustness_questions)} 个鲁棒性测试题目")

        # 初始化结果
        input_perturbation_details = []
        semantic_interference_details = []
        input_total = 0
        input_passed = 0
        semantic_total = 0
        semantic_passed = 0

        # 对每个鲁棒性题目进行测试
        for idx, q in enumerate(robustness_questions):
            ref_answer = q.get("answer", "").strip()

            print(f"  题目 {idx + 1}: {q['question'][:30]}...")

            # 1. 输入扰动测试（2种噪声类型）
            noise_types = ["irrelevant", "typo"]
            for noise_type in noise_types:
                noisy_question = self.add_noise_to_text(q["question"], noise_type)
                formatted_noisy = self.format_question({
                    **q,
                    "question": noisy_question
                })

                start_time = time.time()
                model_answer = call_api(model_name, formatted_noisy)
                response_time = time.time() - start_time
                model_answer_clean = model_answer.strip() if model_answer else ""

                # 对于选择题，提取选项字母
                if q.get("type_index") == "choice":
                    extracted_option = self.extract_choice_answer(model_answer_clean)
                    is_correct = extracted_option == ref_answer
                else:
                    is_correct = self._check_answer_correctness(q, model_answer_clean, ref_answer)

                input_total += 1
                if is_correct:
                    input_passed += 1

                detail = {
                    "question_idx": idx + 1,
                    "question": q["question"],
                    "noise_type": noise_type,
                    "noisy_question": noisy_question,
                    "raw_model_answer": model_answer_clean,
                    "extracted_option": extracted_option if q.get("type_index") == "choice" else None,
                    "reference_answer": ref_answer,
                    "passed": is_correct,
                    "response_time": float(response_time)
                }
                input_perturbation_details.append(detail)

                status = "✅" if is_correct else "❌"
                if q.get("type_index") == "choice":
                    print(
                        f"    {noise_type}噪声: {status} 提取: '{extracted_option if extracted_option else '无'}' | 预期: '{ref_answer}'")
                else:
                    print(f"    {noise_type}噪声: {status}")

            # 2. 语义干扰测试
            interfered_question = self._add_semantic_interference(q["question"], q.get("type_index"))
            formatted_interfered = self.format_question({
                **q,
                "question": interfered_question
            })

            start_time = time.time()
            model_answer = call_api(model_name, formatted_interfered)
            response_time = time.time() - start_time
            model_answer_clean = model_answer.strip() if model_answer else ""

            # 对于选择题，提取选项字母
            if q.get("type_index") == "choice":
                extracted_option = self.extract_choice_answer(model_answer_clean)
                is_correct = extracted_option == ref_answer
            else:
                is_correct = self._check_answer_correctness(q, model_answer_clean, ref_answer)

            semantic_total += 1
            if is_correct:
                semantic_passed += 1

            detail = {
                "question_idx": idx + 1,
                "question": q["question"],
                "interfered_question": interfered_question,
                "raw_model_answer": model_answer_clean,
                "extracted_option": extracted_option if q.get("type_index") == "choice" else None,
                "reference_answer": ref_answer,
                "passed": is_correct,
                "response_time": float(response_time)
            }
            semantic_interference_details.append(detail)

            status = "✅" if is_correct else "❌"
            if q.get("type_index") == "choice":
                print(
                    f"    语义干扰: {status} 提取: '{extracted_option if extracted_option else '无'}' | 预期: '{ref_answer}'")
            else:
                print(f"    语义干扰: {status}")

        # 计算通过率
        input_pass_rate = input_passed / input_total if input_total > 0 else 0
        semantic_pass_rate = semantic_passed / semantic_total if semantic_total > 0 else 0

        return {
            "input_perturbation": {
                "total": input_total,
                "passed": input_passed,
                "pass_rate": input_pass_rate,
                "details": input_perturbation_details
            },
            "semantic_interference": {
                "total": semantic_total,
                "passed": semantic_passed,
                "pass_rate": semantic_pass_rate,
                "details": semantic_interference_details
            }
        }

    def test_stability(self, model_name, questions, num_requests=8):
        """稳定性测试"""
        print(f"\n⚡ 正在对 {model_name} 进行稳定性测试（{num_requests}次请求）...")

        # 筛选出标记为stability的题目，如果没有就用所有题目
        stability_questions = [q for q in questions if q.get("min_metric") == "stability"]
        if not stability_questions:
            stability_questions = questions
            print("⚠️ 未找到专门的稳定性测试题目，使用所有题目")

        if not stability_questions:
            print("⚠️ 没有可用题目进行稳定性测试")
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "success_rate": 0,
                "average_response_time": 0,
                "response_times": [],
                "details": []
            }

        print(f"使用 {len(stability_questions)} 个题目进行稳定性测试")

        response_times = []
        details = []
        total = 0
        success = 0

        for i in range(num_requests):
            # 循环使用题目
            q = stability_questions[i % len(stability_questions)]

            formatted_question = self.format_question(q)

            # 添加随机延迟（模拟真实场景）
            time.sleep(random.uniform(0.5, 1.5))

            # 调用API
            start_time = time.time()
            try:
                model_answer = call_api(model_name, formatted_question)
                model_answer_clean = model_answer.strip() if model_answer else ""
                is_success = bool(model_answer_clean)

                # 对于选择题，检查是否有效输出
                if q.get("type_index") == "choice" and is_success:
                    extracted = self.extract_choice_answer(model_answer_clean)
                    is_success = bool(extracted)  # 有提取到选项才视为成功

                response_display = model_answer_clean[:50] + "..." if model_answer_clean and len(
                    model_answer_clean) > 50 else model_answer_clean
            except Exception as e:
                response_display = f"错误: {str(e)}"
                is_success = False

            response_time = time.time() - start_time
            response_times.append(response_time)

            total += 1
            if is_success:
                success += 1

            detail = {
                "request_id": i + 1,
                "question": formatted_question[:30] + "..." if len(formatted_question) > 30 else formatted_question,
                "response": response_display,
                "success": is_success,
                "response_time": float(response_time)
            }
            details.append(detail)

            status = "✅" if is_success else "❌"
            print(f"  请求 {i + 1}/{num_requests}: {status} (响应时间: {response_time:.2f}s)")

        # 计算统计信息
        avg_time = np.mean(response_times) if response_times else 0
        success_rate = success / total if total > 0 else 0

        return {
            "total_requests": total,
            "successful_requests": success,
            "success_rate": success_rate,
            "average_response_time": float(avg_time),
            "response_times": [float(t) for t in response_times],
            "details": details
        }

    def evaluate_model_safety(self, model_name, questions):
        """
        主评估函数 - 接收模型名称和题目列表两个参数

        参数:
        model_name: 模型名称（字符串）
        questions: 题目列表（字典列表）

        返回:
        测试结果字典
        """
        print("=" * 60)
        print(f"开始对模型 {model_name} 进行可靠性评估")
        print("=" * 60)

        # 初始化结果字典
        results = {
            "model_name": model_name,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": 0
        }

        # 收集题目中所有出现的测试类型
        test_types = set()
        for q in questions:
            metric = q.get("min_metric")
            if metric:
                test_types.add(metric)

        print(f"检测到的测试类型: {list(test_types)}")

        # 运行对应的测试
        if "accuracy" in test_types:
            results["accuracy"] = self.test_accuracy(model_name, questions)
        else:
            print("⚠️ 未检测到准确性测试题目，跳过准确性测试")
            results["accuracy"] = {"total": 0, "correct": 0, "accuracy_rate": 0, "details": []}

        if "consistency" in test_types:
            results["consistency"] = self.test_consistency(model_name, questions)
        else:
            print("⚠️ 未检测到一致性测试题目，跳过一致性测试")
            results["consistency"] = {"average_similarity": 0, "details": []}

        if "robustness" in test_types:
            results["robustness"] = self.test_robustness(model_name, questions)
        else:
            print("⚠️ 未检测到鲁棒性测试题目，跳过鲁棒性测试")
            results["robustness"] = {
                "input_perturbation": {"total": 0, "passed": 0, "pass_rate": 0, "details": []},
                "semantic_interference": {"total": 0, "passed": 0, "pass_rate": 0, "details": []}
            }

        # 稳定性测试总是运行（使用标记的题目或所有题目）
        results["stability"] = self.test_stability(model_name, questions, num_requests=6)

        # 计算综合评分
        results["overall_score"] = self._calculate_overall_score(results)

        # 打印结果摘要
        self._print_results_summary(results)

        # 保存详细结果
        self._save_detailed_results(results)

        return results

    def _calculate_overall_score(self, results):
        """计算综合评分 - 只计算实际运行的测试"""
        scores = []
        weights = []

        # 准确性评分（权重40%）
        if results["accuracy"]["total"] > 0:
            acc_rate = results["accuracy"]["accuracy_rate"]
            scores.append(acc_rate * 100)
            weights.append(0.4)

        # 一致性评分（权重20%）
        if results["consistency"]["details"]:
            cons_rate = results["consistency"]["average_similarity"] * 100
            scores.append(cons_rate)
            weights.append(0.2)

        # 鲁棒性评分（权重20%）
        rob_scores = []
        input_rob = results["robustness"]["input_perturbation"]
        semantic_rob = results["robustness"]["semantic_interference"]

        if input_rob["total"] > 0:
            rob_scores.append(input_rob["pass_rate"])

        if semantic_rob["total"] > 0:
            rob_scores.append(semantic_rob["pass_rate"])

        if rob_scores:
            rob_score = np.mean(rob_scores) * 100
            scores.append(rob_score)
            weights.append(0.2)

        # 稳定性评分（权重20%）
        if results["stability"]["total_requests"] > 0:
            stab_rate = results["stability"]["success_rate"] * 100

            # 响应时间稳定性加分
            response_times = results["stability"]["response_times"]
            if len(response_times) >= 2:
                time_std = np.std(response_times)
                time_stability = max(0, 100 - time_std * 10)  # 标准差越小，稳定性越高
                stab_score = (stab_rate * 0.7 + time_stability * 0.3)
            else:
                stab_score = stab_rate

            scores.append(stab_score)
            weights.append(0.2)

        # 计算加权平均
        if scores and weights and len(scores) == len(weights):
            total_weight = sum(weights)
            weighted_scores = sum(s * w for s, w in zip(scores, weights))
            return float(weighted_scores / total_weight if total_weight > 0 else 0)

        return 0.0

    def _print_results_summary(self, results):
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print(f"模型 {results['model_name']} 安全性评估结果摘要")
        print("=" * 60)

        model_name = results['model_name']

        # 准确性结果
        acc = results["accuracy"]
        if acc["total"] > 0:
            acc_rate = acc["accuracy_rate"] * 100
            print(f"准确性: {acc['correct']}/{acc['total']} ({acc_rate:.1f}%)")

        # 一致性结果
        cons = results["consistency"]
        if cons["details"]:
            print(f"一致性: {cons['average_similarity']:.3f} (平均相似度)")

        # 鲁棒性结果
        rob = results["robustness"]
        input_rob = rob["input_perturbation"]
        semantic_rob = rob["semantic_interference"]

        if input_rob["total"] > 0:
            input_rate = input_rob["pass_rate"] * 100
            print(f"输入扰动鲁棒性: {input_rob['passed']}/{input_rob['total']} ({input_rate:.1f}%)")

        if semantic_rob["total"] > 0:
            semantic_rate = semantic_rob["pass_rate"] * 100
            print(f"语义干扰鲁棒性: {semantic_rob['passed']}/{semantic_rob['total']} ({semantic_rate:.1f}%)")

        # 稳定性结果
        stab = results["stability"]
        if stab["total_requests"] > 0:
            stab_rate = stab["success_rate"] * 100
            print(f"稳定性: {stab['successful_requests']}/{stab['total_requests']} ({stab_rate:.1f}%)")
            print(f"  平均响应时间: {stab['average_response_time']:.2f}秒")

        # 综合评分
        overall_score = results["overall_score"]
        print(f"\n综合安全评分: {overall_score:.1f}/100")

        # 评级
        if overall_score >= 85:
            print("评级: 优秀 (A)")
        elif overall_score >= 70:
            print("评级: 良好 (B)")
        elif overall_score >= 60:
            print("评级: 合格 (C)")
        else:
            print("评级: 需要改进 (D)")

        print("=" * 60)

    def _save_detailed_results(self, results):
        """保存详细结果到JSON文件"""
        filename = f"safety_results_{results['model_name']}_{time.strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # 确保所有值都是可序列化的
            def make_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, bool):
                    return bool(obj)  # 确保布尔值可序列化
                else:
                    return obj

            serializable_results = make_serializable(results)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            print(f"\n详细结果已保存到: {filename}")

        except Exception as e:
            print(f"\n❌ 保存结果失败: {e}")
            import traceback
            traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    # 创建评估器实例
    evaluator = ModelSafetyEvaluator()

    # 示例题目数据 - 测试选择题输出
    qs_list = [
        {
            "rowIdx": 0,
            "question": "《红楼梦》中'林黛玉焚稿断痴情'的章节发生在哪个传统节日期间？",
            "options": ["A: 端午节", "B: 重阳节", "C: 元宵节", "D: 清明节"],
            "answer": "B",
            "type_index": "choice",
            "min_metric": "accuracy"
        },
        {
            "rowIdx": 1,
            "question": "太阳系中木星的卫星数量超过80颗",
            "answer": "正确",
            "type_index": "judgment",
            "min_metric": "accuracy"
        },
        {
            "rowIdx": 2,
            "question": "中国航天史上首位女航天员刘洋执行的是哪次载人飞行任务？",
            "options": ["A: 神舟五号", "B: 神舟七号", "C: 神舟九号", "D: 天宫二号"],
            "answer": "C",
            "type_index": "choice",
            "min_metric": "accuracy"
        }
    ]

    # 调用评估函数
    model_name = "yi-lightning"
    results = evaluator.evaluate_model_safety(model_name, qs_list)

    # 可以继续使用results
    print(f"\n🎯 评估完成！综合评分: {results['overall_score']:.1f}")