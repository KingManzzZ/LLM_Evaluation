import json

from flask import Flask, request, jsonify
from largemodel_create_and_evaluate.generate_questions import QuestionBankGenerator
from largemodel_create_and_evaluate.questions_evolving import QuestionsEvolution
import importlib

app = Flask(__name__)

@app.route('/generate', methods=['POST', 'GET'])
def generate():
        data = request.get_json()
        generator = QuestionBankGenerator()
        output = generator.final(data)
        response = jsonify(output)
        return response

@app.route('/evolve', methods=['POST', 'GET'])
def evolve():
        data = request.get_json()
        evolve = QuestionsEvolution()
        output = evolve.final(data)
        response = jsonify(output)
        return response                 


# 定义脚本映射
SCRIPT_MAPPING = {
    "system_responsiveness": "Assess.token_and_throughput.main.main",
    "complex_reasoning_skill": "Assess.complex_reasoning.main.main",
    "long_text_comprehension_skill": "Assess.long_text_comprehension.main.main",
    "reliability": "Assess.assess_reliability.main.main",
    "safety": "Assess.assess_security.main.main",
    "fairness": "Assess.assess_fairness.main.main"
}


@app.route('/evaluation/general_process', methods=['GET', 'POST'])
def run_deal():
    # 获取脚本ID，支持GET参数和POST JSON数据
    if request.method == 'GET':
        domain = request.args.get('domain', type=str)
    else:
        data = request.get_json() or {}
        domain = data.get('domain')

    if domain not in SCRIPT_MAPPING:
        return jsonify({
            'status': 'error',
            'message': f"Domain '{domain}' not found in SCRIPT_MAPPING",
            'domain': domain
        }), 400

    try:
        mapping = SCRIPT_MAPPING[domain]
        parts = mapping.split('.')
        function_name = parts[-1]
        process_path = ".".join(parts[:-1])

        # 使用importlib动态导入处理路径对应的模块
        module = importlib.import_module(process_path)
        # 使用getattr函数获取模块中指定的函数
        run = getattr(module, function_name)

        # 准备传递给脚本函数的参数
        params = {}
        if request.method == 'POST':
            data = request.get_json() or {}
            params = {k: v for k, v in data.items() if k != 'domain'}

        # 执行脚本函数
        result = run(**params)

        if isinstance(result, tuple) and len(result) == 2:
            all_response, final = result
        else:
            all_response, final = result, None

        return jsonify({
            'status': "success",
            'domain': domain,
            'model_response': all_response,
            'score': final
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'domain': domain
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)