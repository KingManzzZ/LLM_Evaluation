<template>
  <div class="test-result-view">
    <div class="module-header">
      <h2 class="module-title"><i class="fas fa-chart-line"></i> 测试结果</h2>
      <div class="module-actions">
        <button class="export-button" @click="handleExport" title="生成测试报告">
          <i class="fas fa-file-export"></i>
          生成报告
        </button>
      </div>
    </div>

    <!-- 测试信息概览 -->
    <div class="result-section">
      <h3 class="section-title">测试概览</h3>
      <div class="overview-grid">
        <div class="overview-item">
          <label>测试名称：</label>
          <span>{{ testData.name }}</span>
        </div>
        <div class="overview-item">
          <label>测试时间：</label>
          <span>{{ testData.time }}</span>
        </div>
        <div class="overview-item">
          <label>被测模型：</label>
          <span>{{ testData.models.join('、') }}</span>
        </div>
        <div class="overview-item">
          <label>主维度：</label>
          <span>{{ getDimensionLabel(testData.dimension) }}</span>
        </div>
        <div class="overview-item">
          <label>分维度：</label>
          <span>{{ getMetricLabel(testData.metric) }}</span>
        </div>
        <div class="overview-item">
          <label>子指标：</label>
          <span>{{ getSubMetricLabel(testData.subMetric) }}</span>
        </div>
        <div class="overview-item">
          <label>测试ID：</label>
          <span>{{ getTestId() }}</span>
        </div>
        <div class="overview-item">
          <label>总得分：</label>
          <span class="final-score">{{ getFinalScore() }}</span>
        </div>
      </div>
    </div>

    <!-- 性能指标 -->
    <div v-if="getMetricScores()" class="result-section">
      <h3 class="section-title">性能指标</h3>
      <div class="metrics-grid">
        <div v-for="(score, metric) in getMetricScores()" :key="metric" class="metric-card">
          <div class="metric-header">
            <span class="metric-title">{{ formatMetricName(metric) }}</span>
            <span class="metric-badge">满分100</span>
          </div>
          <div class="metric-value">{{ score.toFixed(2) }}</div>
          <div class="metric-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: score + '%' }"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 得分统计 -->
    <div class="result-section">
      <h3 class="section-title">得分统计</h3>
      <div class="score-grid">
        <!-- 总得分卡片 -->
        <div class="score-card">
          <div class="score-header">
            <span class="score-title">总得分</span>
            <span class="score-badge">满分100</span>
          </div>
          <div class="score-value">{{ getFinalScore() }}</div>
          <div class="score-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: getFinalScoreWidth() }"></div>
            </div>
          </div>
        </div>

        <!-- 动态显示指标得分 -->
        <template v-if="getMetricScores()">
          <!-- 显示所有指标得分，确保只显示有有效数值的指标 -->
          <div
            v-for="(score, metric) in getMetricScores()"
            :key="metric"
            class="score-card"
            v-if="score !== null && score !== undefined"
          >
            <div class="score-header">
              <span class="score-title">{{ formatMetricName(metric) }}</span>
              <span class="score-badge">指标得分</span>
            </div>
            <div class="score-value">{{ score.toFixed(2) }}</div>
            <div class="score-progress">
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: score + '%' }"></div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- 题目详情 -->
    <div class="result-section">
      <h3 class="section-title">题目详情</h3>
      <div class="question-table">
        <!-- 修改表格列结构 -->
        <div class="table-header">
          <div>序号</div>
          <div>题干</div>
          <div>回答模型</div>
          <div>模型回答</div>
          <div>得分</div>
          <div>是否通过</div>
          <div>操作</div>
        </div>
        <!-- 修改数据行 -->
        <div v-for="(item, index) in processedQuestions" :key="index" class="table-row">
          <div>{{ index + 1 }}</div>
          <div class="question-content">{{ item.content }}</div>
          <div>{{ testData.models.join('、') }}</div>
          <div class="answer-content">{{ truncateAnswer(item.modelAnswer) }}</div>
          <div class="score-cell">{{ item.score === null ? 'N/A' : item.score.toFixed(2) }}</div>
          <div>
            <span :class="getCorrectnessClass(item.isCorrect)">
              {{ item.isCorrect === null ? 'N/A' : item.isCorrect ? '通过' : '不通过' }}
            </span>
          </div>
          <div>
            <button class="view-button" @click="showQuestionDetail(item)">
              <i class="fas fa-eye"></i> 详情
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 添加模态框 -->
    <div v-if="selectedQuestion" class="modal-mask">
      <div class="modal-container">
        <div class="modal-header">
          <h3>题目详情</h3>
          <button @click="selectedQuestion = null" class="modal-close">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="modal-content">
          <div class="detail-item">
            <label> 题干：</label>
            <p class="question-text">{{ selectedQuestion.content }}</p>
          </div>
          <!-- 新增选项显示 -->
          <div class="detail-item" v-if="selectedQuestion.options">
            <label> 选项：</label>
            <div class="options-container">
              <div
                v-for="option in parseOptions(selectedQuestion.options)"
                :key="option.key"
                class="option-item"
              >
                <span class="option-key">{{ option.key }}:</span>
                <span class="option-value">{{ option.value }}</span>
              </div>
            </div>
          </div>
          <div class="detail-item">
            <label> 正确答案：</label>
            <p class="answer-text">{{ selectedQuestion.correctAnswer }}</p>
          </div>
          <div class="detail-item">
            <label> 模型回答：</label>
            <p class="answer-text">{{ selectedQuestion.modelAnswer }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps, ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { DataSetService, type TestDetailResponseData } from '../api/dataSet'

const router = useRouter()

const props = defineProps({
  testData: {
    type: Object,
    required: true,
  },
  apiResponseData: {
    type: Object,
    default: () => ({
      finalScore: 0,
      singleScore: [],
    }),
  },
})

// 新增变量和方法
const showDetailModal = ref(false)
const selectedQuestion = ref<any>(null)
const realQuestions = ref<any[]>([]) // 存储真实题目数据
const testDetailData = ref<any[]>([]) // 存储测试详情数据

// 获取测试ID
const getTestId = () => {
  // 详细调试信息
  console.log('=== getTestId() 开始执行 ===')
  console.log('apiResponseData:', JSON.stringify(props.apiResponseData))
  console.log('testData:', JSON.stringify(props.testData))

  // 支持多种API响应格式和testData对象
  const testId =
    props.apiResponseData?.testId ||
    props.apiResponseData?.data?.testId ||
    (props.apiResponseData?.data && props.apiResponseData.data.testId) ||
    props.testData?.testId ||
    (props.testData && props.testData.data && props.testData.data.testId) ||
    null

  console.log('=== getTestId() 执行结果 ===')
  console.log('testId:', testId)
  console.log('apiResponseData.testId:', props.apiResponseData?.testId)
  console.log('apiResponseData.data?.testId:', props.apiResponseData?.data?.testId)
  console.log('testData.testId:', props.testData?.testId)

  // 如果testId为null，尝试从testData的所有属性中查找
  if (testId === null) {
    console.warn('testId为null，尝试从testData中查找...')
    // 遍历testData的所有属性
    for (const key in props.testData) {
      if (props.testData[key] && typeof props.testData[key] === 'object') {
        console.log(`testData.${key}:`, props.testData[key])
        if (props.testData[key].testId !== undefined) {
          console.log(`找到testId: ${props.testData[key].testId}`)
          return props.testData[key].testId
        }
      } else if (key === 'testId') {
        console.log(`找到直接testId: ${props.testData[key]}`)
        return props.testData[key]
      }
    }
  }

  return testId !== undefined && testId !== null && testId !== '' ? testId : 'N/A'
}

// 获取总得分
const getFinalScore = () => {
  // 支持多种API响应格式
  const finalScore = props.apiResponseData?.finalScore || props.apiResponseData?.data?.finalScore
  return finalScore !== undefined && finalScore !== null ? finalScore.toFixed(2) : 'N/A'
}

// 获取总得分宽度（用于进度条）
const getFinalScoreWidth = () => {
  const finalScore = props.apiResponseData?.finalScore || props.apiResponseData?.data?.finalScore
  return finalScore !== undefined && finalScore !== null ? finalScore + '%' : '0%'
}

// 获取指标得分
const getMetricScores = () => {
  // 支持多种API响应格式，同时考虑props.testData（从ReviewPage传递的数据）
  const metricScores =
    props.apiResponseData?.metricScores ||
    props.apiResponseData?.data?.metricScores ||
    props.testData?.metricScores || // 从testData中获取（从ReviewPage传递的数据）
    props.testData?.data?.metricScores || // 从testData.data中获取
    null
  return metricScores || null
}

// 获取平均分宽度（用于进度条）
const getAverageScoreWidth = () => {
  const avgScore = getAverageScore()
  return avgScore !== null ? avgScore + '%' : '0%'
}

// 获取正确率宽度（用于进度条）
const getCorrectRateWidth = () => {
  const correctRate = getCorrectRate()
  return correctRate !== null ? correctRate + '%' : '0%'
}

// 获取测试详情
const fetchTestDetail = async (testId: number) => {
  try {
    console.log(`开始获取测试ID ${testId} 的详情数据...`)
    const response = await DataSetService.getTestDetail(testId)

    if (response.code === 200 && response.data) {
      console.log(`成功获取测试ID ${testId} 的详情数据:`, response.data)
      testDetailData.value = response.data
      return response.data
    } else {
      throw new Error(response.msg || '获取测试详情失败')
    }
  } catch (error) {
    console.warn(`获取测试ID ${testId} 的详情数据失败:`, error)
    ElMessage.warning('获取测试详情失败，将使用模拟数据')
    return null
  }
}

// 通过题目ID获取题目信息的API调用
const fetchQuestionData = async (questionId: number) => {
  try {
    console.log(`开始获取题目ID ${questionId} 的数据...`)

    // 添加超时控制
    const controller = new AbortController()
    const timeoutId = setTimeout(() => {
      console.warn(`获取题目ID ${questionId} 的数据超时`)
      controller.abort()
    }, 5000) // 5秒超时

    // 尝试使用不同的请求模式
    // 方案1: 直接请求（可能遇到CORS问题）
    const directUrl = `http://101.33.219.200:12345/dataInfo/select/${questionId}`

    // 方案2: 使用CORS代理（通过本地服务器转发）
    const proxyUrl = `/api/dataInfo/select/${questionId}`

    // 优先尝试代理模式，如果失败则尝试直接请求
    let url = proxyUrl
    let useProxy = true

    console.log(`请求URL: ${url} (使用代理模式)`)

    let response
    try {
      response = await fetch(url, {
        signal: controller.signal,
        mode: 'cors',
        credentials: 'omit', // 不发送凭据
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          'Cache-Control': 'no-cache',
          Pragma: 'no-cache',
        },
      })
    } catch (proxyError) {
      // 如果代理请求失败，尝试直接请求
      if (useProxy) {
        console.warn(`代理请求失败，尝试直接请求: ${proxyError.message}`)
        url = directUrl
        useProxy = false
        console.log(`切换到直接请求URL: ${url}`)

        response = await fetch(url, {
          signal: controller.signal,
          mode: 'no-cors', // 直接请求使用no-cors模式
          credentials: 'omit',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        })
      } else {
        throw proxyError
      }
    }

    clearTimeout(timeoutId)
    console.log(`题目ID ${questionId} 请求成功，状态码: ${response.status}`)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    console.log(`题目ID ${questionId} 返回数据:`, data)

    if (data.code === 200 && data.data) {
      console.log(`成功获取题目ID ${questionId} 的数据`)
      return data.data
    } else {
      throw new Error(data.msg || '获取题目数据失败')
    }
  } catch (error) {
    console.warn(`获取题目ID ${questionId} 的数据失败:`, error)

    // 根据错误类型提供更详细的错误信息
    if (error.name === 'AbortError') {
      console.warn(`题目ID ${questionId} 请求超时`)
    } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      console.warn(`题目ID ${questionId} 网络连接失败，可能是CORS策略或网络问题`)
    }

    // 返回更友好的模拟数据作为fallback
    return {
      dataId: questionId,
      question: `题目 ${questionId} - 这是题目的题干内容，用于测试模型的回答质量`,
      answer: `这是题目 ${questionId} 的正确答案`,
    }
  }
}

// 批量获取题目数据
const fetchRealQuestions = async () => {
  if (!props.testData.questionIds || props.testData.questionIds.length === 0) {
    console.warn('没有可用的题目ID列表')
    return
  }

  try {
    // 如果有testId，先获取测试详情数据
    const testId = getTestId()
    // 确保testId不是'N/A'且是有效的数字
    if (testId !== 'N/A' && !isNaN(Number(testId))) {
      await fetchTestDetail(parseInt(testId))
    }

    // 使用Promise.allSettled来处理部分成功的情况
    const promises = props.testData.questionIds.map((id: number) => fetchQuestionData(id))
    const results = await Promise.allSettled(promises)

    // 提取所有成功的结果
    realQuestions.value = results
      .filter((result): result is PromiseFulfilledResult<any> => result.status === 'fulfilled')
      .map((result) => result.value)

    // 统计失败的数量
    const failedCount = results.filter((result) => result.status === 'rejected').length

    if (realQuestions.value.length === 0) {
      console.warn('未能获取到任何题目数据')
      ElMessage.warning('无法获取题目数据，将显示模拟数据')
    } else {
      console.log(
        `成功获取到 ${realQuestions.value.length} 道题目的真实数据，${failedCount} 道题目获取失败`,
      )
      if (failedCount > 0) {
        ElMessage.warning(
          `成功获取 ${realQuestions.value.length} 道题目数据，${failedCount} 道题目获取失败`,
        )
      }
    }
  } catch (error) {
    console.error('批量获取题目数据失败:', error)
    ElMessage.error('获取题目数据失败，将显示模拟数据')
  }
}

// 处理题目数据，合并后端得分
const processedQuestions = computed(() => {
  if (!props.testData.questions) {
    return []
  }

  // 处理singleScore为null的情况
  const singleScores = Array.isArray(props.apiResponseData.singleScore)
    ? props.apiResponseData.singleScore
    : props.apiResponseData.singleScore === null
      ? null
      : []

  return props.testData.questions.map((question, index) => {
    // 当singleScore为null时，score也设置为null
    const score = singleScores === null ? null : singleScores[index] || 0

    // 尝试使用真实题目数据，如果获取失败则使用模拟数据
    const realQuestion = realQuestions.value.find((q: any) => q.dataId === question.id)

    // 从测试详情数据中获取对应的模型回答，通过dataId匹配而不是索引
    const matchingTestDetail = testDetailData.value.find((td: any) => td.dataId === question.id)
    let modelAnswer = matchingTestDetail?.modelOutput || question.modelAnswer || '暂无模型回答'

    // 如果测试详情数据中没有模型回答，尝试从真实题目数据中获取
    if (modelAnswer === '暂无模型回答' && realQuestion && realQuestion.modelAnswer) {
      modelAnswer = realQuestion.modelAnswer
    }

    // 获取options字段，优先从测试详情数据中获取，然后是真实题目数据，最后是原始数据
    const options = matchingTestDetail?.options || realQuestion?.options || question.options

    return {
      ...question,
      content: realQuestion ? realQuestion.question : question.content, // 使用真实题干
      correctAnswer: realQuestion ? realQuestion.answer : question.correctAnswer, // 使用真实答案
      modelAnswer: modelAnswer, // 使用从测试详情获取的模型回答
      score: score,
      options: options, // 添加选项字段
      // 当score为null时，isCorrect也设置为null
      isCorrect: score === null ? null : score >= 60, // 假设60分以上为正确
      realData: realQuestion, // 保存真实题目数据
    }
  })
})

// 计算平均分
const getAverageScore = () => {
  const singleScores = Array.isArray(props.apiResponseData.singleScore)
    ? props.apiResponseData.singleScore
    : props.apiResponseData.singleScore === null
      ? null
      : []

  // 当singleScore为null时，返回null
  if (singleScores === null || singleScores.length === 0) {
    return null
  }
  const sum = singleScores.reduce((acc, score) => acc + score, 0)
  return sum / singleScores.length
}

// 获取主维度标签
const getDimensionLabel = (dimension: string) => {
  const dimensionMap = {
    performance: '性能',
    reliability: '可靠性',
    security: '安全性',
    fairness: '公平性',
  }
  return dimensionMap[dimension] || dimension || '未设置'
}

// 获取分维度标签
const getMetricLabel = (metric: string) => {
  const metricMap = {
    // 性能分维度
    system_responsiveness: '系统响应效率',
    complex_reasoning_skill: '复杂推理能力',
    long_text_understanding: '长文本理解能力',
    // 可靠性分维度
    accuracy: '准确性',
    robustness: '鲁棒性',
    consistency: '一致性',
    stability: '稳定性',
    // 安全性分维度
    random: '随机生成样本',
    evaluation: '测评维度',
    hijacking: '指令挟持',
    jailbreak: '越狱攻击',
    distortion: '内容扭曲',
    blocking: '提示屏蔽',
    interference: '干扰对话',
    blackbox: '黑盒测试',
    whitebox: '白盒测试',
    // 公平性分维度
    gender: '性别',
    race: '种族',
    age: '年龄',
    religion: '宗教',
    politics: '政治',
    // 指标得分标签
    common_sense_logical_reasoning: '常识逻辑推理',
    mathematical_reasoning: '数学推理',
    casual_reasoning: '因果推理',
  }
  return metricMap[metric] || metric || '未设置'
}

// 获取子指标标签
const getSubMetricLabel = (subMetric: string) => {
  const subMetricMap = {
    // 复杂推理能力子指标
    mathematical_reasoning: '数学推理',
    common_sense_logical_reasoning: '常识逻辑推理',
    casual_reasoning: '因果推理',
    // 长文本理解能力子指标
    information_extraction: '信息提取',
    contextual_relevance: '上下文关联',
    memory_ability: '记忆能力',
  }
  return subMetricMap[subMetric] || subMetric || '未设置'
}

// 计算正确率
const getCorrectRate = () => {
  const singleScores = Array.isArray(props.apiResponseData.singleScore)
    ? props.apiResponseData.singleScore
    : props.apiResponseData.singleScore === null
      ? null
      : []

  // 当singleScore为null时，返回null
  if (singleScores === null || singleScores.length === 0) {
    return null
  }
  const correctCount = singleScores.filter((score) => score >= 60).length
  return (correctCount / singleScores.length) * 100
}

// 获取正确性样式类
const getCorrectnessClass = (isCorrect: boolean) => {
  return isCorrect ? 'correct-badge' : 'incorrect-badge'
}

// 截断答案文本
const truncateAnswer = (text: string) => {
  if (!text) return ''
  return text.length > 50 ? text.substring(0, 50) + '...' : text
}

// 格式化指标名称
const formatMetricName = (metricName: string) => {
  // 先尝试从metricMap中查找对应的中文名称
  const metricMap = {
    common_sense_logical_reasoning: '常识逻辑推理',
    mathematical_reasoning: '数学推理',
    casual_reasoning: '因果推理',
    system_responsiveness: '系统响应效率',
    complex_reasoning_skill: '复杂推理能力',
    long_text_understanding: '长文本理解能力',
    accuracy: '准确性',
    robustness: '鲁棒性',
    consistency: '一致性',
    stability: '稳定性',
    random: '随机生成样本',
    evaluation: '测评维度',
    hijacking: '指令挟持',
    jailbreak: '越狱攻击',
    distortion: '内容扭曲',
    blocking: '提示屏蔽',
    interference: '干扰对话',
    blackbox: '黑盒测试',
    whitebox: '白盒测试',
    gender: '性别',
    race: '种族',
    age: '年龄',
    religion: '宗教',
    politics: '政治',
  }

  // 如果在映射表中找到，直接返回
  if (metricMap[metricName]) {
    return metricMap[metricName]
  }

  // 否则进行通用格式化：下划线转空格，首字母大写
  return metricName.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

// 解析选项字符串
const parseOptions = (optionsStr: string) => {
  if (!optionsStr) return []

  // 支持两种格式：A:选项内容|B:选项内容 或 A选项内容|B选项内容
  const optionRegex = /([A-Z])(?::\s*)?(.*?)(?:\||$)/g
  const options = []
  let match

  while ((match = optionRegex.exec(optionsStr)) !== null) {
    options.push({
      key: match[1],
      value: match[2].trim(),
    })
  }

  return options
}

// 显示题目详情
const showQuestionDetail = (question: any) => {
  selectedQuestion.value = question
  showDetailModal.value = true
}

// 关闭详情模态框
const closeDetailModal = () => {
  showDetailModal.value = false
  selectedQuestion.value = null
}

// 新增导出方法
const handleExport = () => {
  if (!props.testData) {
    ElMessage.error('没有可用的测试数据')
    return
  }

  try {
    // 准备报告数据
    const reportData = {
      testData: props.testData,
      apiResponseData: props.apiResponseData || {},
      generateTime: new Date().toLocaleString(),
    }

    // 生成唯一报告ID
    const reportId = `test_report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

    // 保存到localStorage
    localStorage.setItem(reportId, JSON.stringify(reportData))

    // 在当前应用中跳转到报告页面
    router.push(`/test-report?reportId=${encodeURIComponent(reportId)}`)

    ElMessage.success('测试报告生成成功！')
  } catch (error) {
    console.error('生成报告失败:', error)
    ElMessage.error(`生成报告失败: ${(error as Error).message}`)
  }
}

// 组件挂载时获取真实题目数据和测试详情数据
onMounted(async () => {
  if (props.testData && props.testData.questionIds && props.testData.questionIds.length > 0) {
    fetchRealQuestions()
  }

  // 获取测试详情数据
  const testId = getTestId()
  // 确保testId不是'N/A'且是有效的数字
  if (testId && testId !== 'N/A' && !isNaN(Number(testId))) {
    await fetchTestDetail(Number(testId))
  }
})
</script>

<style scoped>
.test-result-view {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  margin-top: 20px; /* 与导航栏保持间距 */
}

.section-title {
  font-size: 1.25rem;
  color: #1e293b;
  margin-bottom: 1.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #6366f1;
}

.overview-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  margin-bottom: 2rem;
}

.overview-item {
  background: #f8fafc;
  padding: 1.2rem;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.overview-item label {
  color: #64748b;
  margin-right: 8px;
}

.metrics-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.metric-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  transition: box-shadow 0.2s;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.metric-card:hover {
  box-shadow: 0 4px 6px rgba(99, 102, 241, 0.1);
}

.metric-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}

.metric-title {
  font-weight: 600;
  color: #1e293b;
  font-size: 1rem;
}

.metric-badge {
  background: #f1f5f9;
  color: #64748b;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
}

.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: #6366f1;
  text-align: center;
  margin: 1rem 0;
}

.metric-progress {
  margin-top: 1rem;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #f1f5f9;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  transition: width 0.3s ease;
  border-radius: 4px;
}

.score-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.score-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  transition: box-shadow 0.2s;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.score-card:hover {
  box-shadow: 0 4px 6px rgba(99, 102, 241, 0.1);
}

.score-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}

.score-title {
  font-weight: 600;
  color: #1e293b;
  font-size: 1rem;
}

.score-badge {
  background: #f1f5f9;
  color: #64748b;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
}

.score-value {
  font-size: 2rem;
  font-weight: 700;
  color: #6366f1;
  text-align: center;
  margin: 1rem 0;
}

.score-progress {
  margin-top: 1rem;
}

.question-table {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.table-header {
  display: grid;
  grid-template-columns: 80px 2fr 1fr 1fr 1fr 1fr 1fr;
  background: #f8fafc;
  color: #64748b;
  font-weight: 500;
}

.table-row {
  display: grid;
  grid-template-columns: 80px 2fr 1fr 1fr 1fr 1fr 1fr;
  border-bottom: 1px solid #e2e8f0;
}

.table-header > div,
.table-row > div {
  padding: 12px 16px;
  min-height: 50px;
  display: flex;
  align-items: center;
}

.question-content {
  line-height: 1.6;
  color: #475569;
}

.answer-content {
  color: #64748b;
  white-space: pre-wrap;
}

@media (max-width: 768px) {
  .table-header,
  .table-row {
    grid-template-columns: 1fr;
  }

  .table-header > div,
  .table-row > div {
    padding: 8px 12px;
  }
}

/* 在原有样式后新增 */
.module-actions {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.export-button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.95rem;
}

.export-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
}

.export-button i {
  font-size: 1.1rem;
}

.view-button {
  background: #f1f5f9;
  color: #64748b;
  padding: 8px 16px;
  border-radius: 6px;
  transition: background 0.2s;
}

.view-button:hover {
  background: #e2e8f0;
}

/* 正确性徽章样式 */
.correct-badge {
  padding: 2px 8px;
  background-color: #f6ffed;
  border: 1px solid #b7eb8f;
  color: #52c41a;
  border-radius: 4px;
  font-size: 12px;
}

.incorrect-badge {
  padding: 2px 8px;
  background-color: #fff2f0;
  border: 1px solid #ffccc7;
  color: #ff4d4f;
  border-radius: 4px;
  font-size: 12px;
}

/* 选项容器样式 */
.options-container {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
}

/* 单个选项样式 */
.option-item {
  display: flex;
  align-items: flex-start;
  padding: 8px 12px;
  background: #f8fafc;
  border-radius: 6px;
  transition: background-color 0.2s ease;
}

.option-item:hover {
  background: #f1f5f9;
}

/* 选项字母样式 */
.option-key {
  font-weight: 600;
  color: #3498db;
  margin-right: 12px;
  min-width: 20px;
  text-align: center;
}

/* 选项内容样式 */
.option-value {
  color: #334155;
  line-height: 1.5;
  flex: 1;
}

/* 模态框样式 - 优化以符合整体风格 */
.modal-mask {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-container {
  background: white;
  border-radius: 12px;
  width: 600px;
  max-width: 90%;
  max-height: 80vh;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  transform: translateY(-20px);
  opacity: 0;
  animation: modalSlide 0.3s ease-out forwards;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

@keyframes modalSlide {
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  background: #8b5cf6;
  border-radius: 12px 12px 0 0;
  padding: 18px 24px;
  position: relative;
  padding-right: 50px;
}

.modal-header h3 {
  color: white;
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

.modal-close {
  color: rgba(255, 255, 255, 0.8);
  transition: all 0.2s ease;
  position: absolute;
  right: 24px;
  top: 50%;
  transform: translateY(-50%);
  padding: 8px;
  width: 32px;
  height: 32px;
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-close:hover {
  color: white;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.modal-content {
  padding: 24px;
  overflow-y: auto;
  flex: 1;
}

.detail-item {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid #f1f5f9;
  align-items: start;
  text-align: left;
}

.detail-item:last-child {
  border-bottom: none;
}

.detail-item label {
  color: #64748b;
  font-weight: 500;
  text-align: left !important;
  margin: 0;
}

.question-text {
  background: #f8fafc;
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  line-height: 1.6;
  color: #4b5563;
  margin: 0;
}

.answer-text {
  background: #f8fafc;
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  line-height: 1.6;
  color: #4b5563;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}

/* 模态框动画 */
.modal-fade-enter-active {
  transition: opacity 0.3s;
}

.modal-fade-enter-from {
  opacity: 0;
}

.modal-fade-enter-to {
  opacity: 1;
}

.modal-fade-leave-active {
  transition: opacity 0.3s;
}

.modal-fade-leave-from {
  opacity: 1;
}

.modal-fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .table-header,
  .table-row {
    grid-template-columns: 1fr;
  }

  .table-header > div,
  .table-row > div {
    padding: 8px 12px;
  }
}
</style>
