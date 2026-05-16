// 测试脚本：验证TestResultView组件对test/test2接口数据的处理

// 模拟test/test2接口返回的数据
const mockTest2Data = {
  code: 200,
  msg: 'ok',
  data: {
    testId: 6,
    finalScore: 98.4,
    singleScore: null,
    metricScores: {
      common_sense_logical_reasoning: 100,
      mathematical_reasoning: 96,
      casual_reasoning: 100,
    },
  },
  timestamp: 1767627409201,
}

// 模拟TestResultView组件的核心函数
class MockTestResultView {
  constructor(apiResponseData) {
    this.apiResponseData = apiResponseData
  }

  // 从apiResponseData中提取metricScores
  getMetricScores() {
    return this.apiResponseData?.metricScores || this.apiResponseData?.data?.metricScores
  }

  // 获取总得分
  getFinalScore() {
    const finalScore = this.apiResponseData?.finalScore || this.apiResponseData?.data?.finalScore
    return finalScore !== null ? finalScore.toFixed(2) : '0.00'
  }

  // 格式化指标名称
  formatMetricName(metric) {
    // 指标名称映射表
    const metricMap = {
      common_sense_logical_reasoning: '常识逻辑推理',
      mathematical_reasoning: '数学推理',
      casual_reasoning: '因果推理',
      system_responsiveness: '系统响应效率',
      complex_reasoning_skill: '复杂推理能力',
      long_text_comprehension_skill: '长文本理解能力',
      accuracy: '准确性',
      robustness: '鲁棒性',
      consistency: '一致性',
      stability: '稳定性',
      randomly_generated_samples: '随机生成样本',
      command_hijacking: '指令挟持',
      jailbreak_attacks: '越狱攻击',
      content_distortions: '内容扭曲',
      prompt_blocking: '提示屏蔽',
      disrupt_conversations: '干扰对话',
      black_box: '黑盒',
      white_box: '白盒',
      gender: '性别',
      race: '种族',
      age: '年龄',
      religion: '宗教',
      politics: '政治',
    }

    // 如果在映射表中找到，返回对应的中文名称
    if (metricMap[metric]) {
      return metricMap[metric]
    }

    // 否则进行通用格式化：下划线转空格，首字母大写
    return metric.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())
  }

  // 测试组件行为
  testComponentBehavior() {
    console.log('=== TestResultView组件测试开始 ===')

    // 测试1：验证metricScores提取
    console.log('1. 验证metricScores提取:')
    const metricScores = this.getMetricScores()
    console.log('   提取结果:', metricScores)
    console.log('   测试通过:', metricScores !== null && typeof metricScores === 'object')

    // 测试2：验证总得分计算
    console.log('2. 验证总得分计算:')
    const finalScore = this.getFinalScore()
    console.log('   总得分:', finalScore)
    console.log('   测试通过:', finalScore === '98.40')

    // 测试3：验证指标名称格式化
    console.log('3. 验证指标名称格式化:')
    for (const metric in metricScores) {
      const formattedName = this.formatMetricName(metric)
      console.log(`   ${metric} -> ${formattedName}`)
    }

    // 测试4：验证singleScore为null时的行为
    console.log('4. 验证singleScore为null时的行为:')
    const singleScore = this.apiResponseData?.singleScore || this.apiResponseData?.data?.singleScore
    console.log('   singleScore:', singleScore)
    console.log('   应该不显示平均分和正确率:', singleScore === null)

    console.log('=== TestResultView组件测试结束 ===')
  }
}

// 创建测试实例
const testInstance = new MockTestResultView(mockTest2Data.data)

// 运行测试
testInstance.testComponentBehavior()
