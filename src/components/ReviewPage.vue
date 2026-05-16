<template>
  <div class="review-page">
    <div class="module-header">
      <h2 class="module-title">测试审核</h2>
      <span class="beta-badge">审核中</span>
    </div>

    <div class="review-container">
      <!-- 测试信息概览 -->
      <div class="test-overview">
        <h3 class="section-title">测试信息</h3>
        <div class="overview-grid">
          <div class="overview-item">
            <label>测试名称：</label>
            <span>{{ testInfo.name }}</span>
          </div>
          <div class="overview-item">
            <label>被测模型：</label>
            <span>{{ testInfo.model }}</span>
          </div>
          <div class="overview-item">
            <label>题目数量：</label>
            <span>{{ reviewQuestions.length }}题</span>
          </div>
          <div class="overview-item">
            <label>审核状态：</label>
            <span class="status-badge">
              <i class="fas fa-clock"></i>
              {{ getReviewStatus() }}
            </span>
          </div>
        </div>
      </div>

      <!-- 题目审核表格 -->
      <div class="review-table-section">
        <h3 class="section-title">题目审核</h3>
        <div class="table-container">
          <table class="review-table">
            <thead>
              <tr>
                <th width="60">序号</th>
                <th width="300">题干</th>
                <th width="200">模型回答</th>
                <th width="100">回答模型</th>
                <th width="100">是否通过</th>
                <th width="150">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(question, index) in reviewQuestions" :key="question.id">
                <td class="text-center">{{ index + 1 }}</td>
                <td class="question-content">
                  <div class="content-preview">
                    {{ truncateText(question.content, 50) }}
                  </div>
                </td>
                <td class="model-answer">
                  <div class="answer-preview">
                    {{ truncateText(question.modelAnswer, 40) }}
                  </div>
                </td>
                <td class="text-center">{{ question.modelName }}</td>
                <td class="text-center">
                  <span :class="getCorrectnessClass(question.isCorrect)">
                    {{ question.isCorrect ? '通过' : '不通过' }}
                  </span>
                </td>
                <td class="text-center">
                  <div class="action-buttons">
                    <button class="detail-button" @click="showQuestionDetail(question)">
                      <i class="fas fa-eye"></i> 查看详情
                    </button>
                    <button
                      class="toggle-button"
                      @click="toggleCorrectStatus(question)"
                      :class="{ correct: question.isCorrect, incorrect: !question.isCorrect }"
                    >
                      <i :class="question.isCorrect ? 'fas fa-times' : 'fas fa-check'"></i>
                      {{ question.isCorrect ? '设为不通过' : '设为通过' }}
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 审核操作按钮 -->
      <div class="review-actions">
        <button class="back-button" @click="goBack">
          <i class="fas fa-arrow-left"></i> 返回修改
        </button>
        <button class="submit-button" @click="submitReview" :disabled="!isReviewComplete">
          <i class="fas fa-check-circle"></i> 提交审核
        </button>
      </div>
    </div>

    <!-- 题目详情模态框 -->
    <div v-if="showDetailModal" class="modal-mask">
      <div class="modal-container">
        <div class="modal-header">
          <h3>题目详情</h3>
          <button @click="closeDetailModal" class="modal-close">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="modal-content">
          <div class="detail-section">
            <label>题干：</label>
            <p class="question-text">{{ currentDetailQuestion?.content }}</p>
          </div>
          <!-- 添加选项显示 -->
          <div v-if="currentDetailQuestion?.options" class="detail-section">
            <label>选项：</label>
            <div class="options-container">
              <div
                v-for="option in parseOptions(currentDetailQuestion.options)"
                :key="option.key"
                class="option-item"
              >
                <span class="option-key">{{ option.key }}</span>
                <span class="option-value">{{ option.value }}</span>
              </div>
            </div>
          </div>
          <div class="detail-section">
            <label>模型回答：</label>
            <p class="answer-text">{{ currentDetailQuestion?.modelAnswer }}</p>
          </div>
          <div class="detail-section">
            <label>回答模型：</label>
            <span>{{ currentDetailQuestion?.modelName }}</span>
          </div>
          <div class="detail-section">
            <label>当前状态：</label>
            <span :class="getCorrectnessClass(currentDetailQuestion?.isCorrect)">
              {{ currentDetailQuestion?.isCorrect ? '通过' : '不通过' }}
            </span>
          </div>
        </div>
        <div class="modal-actions">
          <button
            class="toggle-button"
            @click="toggleCurrentQuestionCorrectness"
            :class="{
              correct: currentDetailQuestion?.isCorrect,
              incorrect: !currentDetailQuestion?.isCorrect,
            }"
          >
            <i :class="currentDetailQuestion?.isCorrect ? 'fas fa-times' : 'fas fa-check'"></i>
            {{ currentDetailQuestion?.isCorrect ? '设为不通过' : '设为通过' }}
          </button>
          <button class="close-button" @click="closeDetailModal">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { DataSetService, type TestDetailResponseData } from '../api/dataSet'

const router = useRouter()
const route = useRoute()

// 测试信息
const testInfo = ref({
  name: '',
  model: '',
  dimension: '',
  metric: '',
  os: '',
  cpu: '',
  gpu: '',
  questionIds: [] as number[],
})

// 审核题目数据
const reviewQuestions = ref<any[]>([])

// 模态框相关
const showDetailModal = ref(false)
const currentDetailQuestion = ref<any>(null)

// 生成随机正确/错误状态
const generateRandomCorrectness = () => Math.random() > 0.5

// 后端返回的测试结果数据
const apiResponseData = ref<any>(null)

// 初始化审核数据
const initReviewData = async () => {
  // 从路由参数获取测试信息
  if (route.query.testData) {
    try {
      const testData = JSON.parse(decodeURIComponent(route.query.testData as string))
      testInfo.value = {
        name: testData.testName || '',
        model: testData.modelName || '',
        dimension: testData.dimension || '',
        metric: testData.metric || '',
        os: testData.os || '',
        cpu: testData.cpu || '',
        gpu: testData.gpu || '',
        questionIds: testData.questionList || [],
      }

      // 获取后端返回的数据
      if (route.query.apiResponse) {
        try {
          apiResponseData.value = JSON.parse(decodeURIComponent(route.query.apiResponse as string))
          console.log('后端返回的测试结果数据:', apiResponseData.value)

          // 如果有testId，则调用测试详情接口获取模型回答，支持多种数据结构
          const testId =
            apiResponseData.value?.testId || apiResponseData.value?.data?.testId || null
          if (testId) {
            await fetchTestDetail(testId)
          }
        } catch (error) {
          console.warn('解析后端返回数据失败:', error)
        }
      }

      // 生成审核题目数据（异步）
      await generateReviewQuestions(testData.questionList || [])
    } catch (error) {
      console.error('解析测试数据失败:', error)
      ElMessage.error('加载测试数据失败')
    }
  }
}

// 获取测试详情
const fetchTestDetail = async (testId: number) => {
  try {
    console.log(`开始获取测试ID ${testId} 的详情数据...`)
    const response = await DataSetService.getTestDetail(testId)

    if (response.code === 200 && response.data) {
      console.log(`成功获取测试ID ${testId} 的详情数据:`, response.data)
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
    const url = proxyUrl
    const useProxy = true

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
const fetchRealQuestions = async (questionIds: number[]) => {
  try {
    // 使用Promise.allSettled来处理部分成功的情况
    const promises = questionIds.map((id: number) => fetchQuestionData(id))
    const results = await Promise.allSettled(promises)

    // 提取所有成功的结果
    const realQuestions = results
      .filter((result): result is PromiseFulfilledResult<any> => result.status === 'fulfilled')
      .map((result) => result.value)

    // 统计失败的数量
    const failedCount = results.filter((result) => result.status === 'rejected').length

    if (realQuestions.length === 0) {
      console.warn('未能获取到任何题目数据')
      ElMessage.warning('无法获取题目数据，将显示模拟数据')
      return null
    } else {
      console.log(
        `成功获取到 ${realQuestions.length} 道题目的真实数据，${failedCount} 道题目获取失败`,
      )
      if (failedCount > 0) {
        ElMessage.warning(
          `成功获取 ${realQuestions.length} 道题目数据，${failedCount} 道题目获取失败`,
        )
      }
      return realQuestions
    }
  } catch (error) {
    console.error('批量获取题目数据失败:', error)
    ElMessage.error('获取题目数据失败，将显示模拟数据')
    return null
  }
}

// 生成审核题目数据
const generateReviewQuestions = async (questionIds: number[]) => {
  // 先尝试获取真实题目数据
  const realQuestions = await fetchRealQuestions(questionIds)

  // 尝试获取测试详情数据，支持多种数据结构
  let testDetailData = null
  const testId = apiResponseData.value?.testId || apiResponseData.value?.data?.testId || null
  if (testId) {
    testDetailData = await fetchTestDetail(testId)
  }

  if (realQuestions) {
    // 使用真实题目数据
    reviewQuestions.value = questionIds.map((id, index) => {
      const realQuestion = realQuestions.find((q: any) => q.dataId === id)

      // 从测试详情数据中获取对应的模型回答，通过dataId匹配而不是索引
      let modelAnswer = `这是第${index + 1}道题的模型回答，当前为模拟数据。`
      if (testDetailData) {
        const matchingTestDetail = testDetailData.find((td: any) => td.dataId === id)
        if (matchingTestDetail) {
          modelAnswer = matchingTestDetail.modelOutput || modelAnswer
        }
      }

      // 获取options字段，优先从测试详情数据中获取，然后是真实题目数据
      let options = ''
      if (testDetailData) {
        const matchingTestDetail = testDetailData.find((td: any) => td.dataId === id)
        if (matchingTestDetail) {
          options = matchingTestDetail.options || ''
        }
      }
      if (!options && realQuestion) {
        options = realQuestion.options || ''
      }

      return {
        id: id,
        content: realQuestion
          ? realQuestion.question
          : `这是第${index + 1}道题的题干内容，用于测试模型的回答质量。`,
        modelAnswer: modelAnswer,
        modelName: testInfo.value.model,
        isCorrect: generateRandomCorrectness(),
        originalCorrectness: generateRandomCorrectness(), // 保存原始状态
        options: options, // 添加选项字段
      }
    })
  } else {
    // 使用模拟数据
    reviewQuestions.value = questionIds.map((id, index) => {
      // 从测试详情数据中获取对应的模型回答，通过dataId匹配而不是索引
      let modelAnswer = `这是第${index + 1}道题的模型回答，当前为模拟数据。`
      let options = ''
      if (testDetailData) {
        const matchingTestDetail = testDetailData.find((td: any) => td.dataId === id)
        if (matchingTestDetail) {
          modelAnswer = matchingTestDetail.modelOutput || modelAnswer
          options = matchingTestDetail.options || ''
        }
      }

      return {
        id: id,
        content: `这是第${index + 1}道题的题干内容，用于测试模型的回答质量。`,
        modelAnswer: modelAnswer,
        modelName: testInfo.value.model,
        isCorrect: generateRandomCorrectness(),
        originalCorrectness: generateRandomCorrectness(), // 保存原始状态
        options: options, // 添加选项字段
      }
    })
  }
}

// 文本截断
const truncateText = (text: string, maxLength: number) => {
  if (!text) return ''
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text
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

// 获取审核状态
const getReviewStatus = () => {
  const reviewedCount = reviewQuestions.value.filter(
    (q) => q.isCorrect !== q.originalCorrectness,
  ).length
  if (reviewedCount === 0) return '未开始审核'
  if (reviewedCount === reviewQuestions.value.length) return '审核完成'
  return `审核中 (${reviewedCount}/${reviewQuestions.value.length})`
}

// 是否正确样式类
const getCorrectnessClass = (isCorrect: boolean) => {
  return {
    'correct-status': isCorrect,
    'incorrect-status': !isCorrect,
  }
}

// 显示题目详情
const showQuestionDetail = (question: any) => {
  currentDetailQuestion.value = { ...question }
  showDetailModal.value = true
}

// 关闭详情模态框
const closeDetailModal = () => {
  showDetailModal.value = false
  currentDetailQuestion.value = null
}

// 切换当前题目正确状态
const toggleCurrentQuestionCorrectness = () => {
  if (currentDetailQuestion.value) {
    currentDetailQuestion.value.isCorrect = !currentDetailQuestion.value.isCorrect
    // 同步更新表格中的数据
    const index = reviewQuestions.value.findIndex((q) => q.id === currentDetailQuestion.value.id)
    if (index !== -1) {
      reviewQuestions.value[index].isCorrect = currentDetailQuestion.value.isCorrect
    }
  }
}

// 切换题目正确状态
const toggleCorrectStatus = (question: any) => {
  question.isCorrect = !question.isCorrect
}

// 返回上一页
const goBack = () => {
  router.back()
}

// 提交审核
const submitReview = () => {
  ElMessage.success('审核提交成功！正在跳转到测试结果页面...')

  // 详细记录apiResponseData的完整结构以便调试testId问题
  console.log('=== ReviewPage submitReview - API响应数据完整分析 ===')
  console.log('apiResponseData.value:', JSON.stringify(apiResponseData.value))
  console.log('apiResponseData.value type:', typeof apiResponseData.value)
  if (apiResponseData.value) {
    console.log('apiResponseData.value keys:', Object.keys(apiResponseData.value))
    console.log('apiResponseData.value.testId:', apiResponseData.value.testId)
    console.log(
      'apiResponseData.value.data:',
      apiResponseData.value.data ? JSON.stringify(apiResponseData.value.data) : 'undefined',
    )
    if (apiResponseData.value.data) {
      console.log('apiResponseData.value.data.testId:', apiResponseData.value.data.testId)
      console.log('apiResponseData.value.data.keys:', Object.keys(apiResponseData.value.data))
      if (apiResponseData.value.data.data) {
        console.log(
          'apiResponseData.value.data.data.testId:',
          apiResponseData.value.data.data.testId,
        )
      }
    }
  }

  // 构造测试结果数据，使用后端返回的得分数据
  // 从apiResponseData中获取testId，支持多种数据结构
  const testId =
    apiResponseData.value?.testId ||
    apiResponseData.value?.data?.testId ||
    apiResponseData.value?.data?.data?.testId ||
    null

  console.log('最终获取的testId:', testId)

  const testResult = {
    testId: testId, // 添加测试ID
    name: testInfo.value.name,
    time: new Date().toLocaleString(),
    models: [testInfo.value.model],
    dimension: testInfo.value.dimension || '', // 添加主维度
    metric: testInfo.value.metric || '', // 添加分维度
    subMetric: testInfo.value.subMetric || '', // 添加子指标
    finalScore: apiResponseData.value?.finalScore || apiResponseData.value?.data?.finalScore || 0,
    singleScores:
      apiResponseData.value?.singleScore || apiResponseData.value?.data?.singleScore || null, // 处理singleScore为null的情况
    // 添加metricScores字段
    metricScores:
      apiResponseData.value?.metricScores || apiResponseData.value?.data?.metricScores || null,
    questionIds: reviewQuestions.value.map((q) => q.id), // 添加题目ID列表
    questions: reviewQuestions.value.map((q, index) => ({
      id: q.id, // 添加题目ID
      content: q.content,
      correctAnswer: '正确答案',
      modelAnswer: q.modelAnswer,
      // 当singleScore为null时，score也设置为null
      score:
        apiResponseData.value?.singleScore || apiResponseData.value?.data?.singleScore
          ? (apiResponseData.value?.singleScore || apiResponseData.value?.data?.singleScore)[
              index
            ] || 0
          : null,
      isCorrect: q.isCorrect,
    })),
  }

  console.log('=== 提交审核 - 测试结果数据 ===')
  console.log('testResult.testId:', testResult.testId)
  console.log('完整testResult:', JSON.stringify(testResult, null, 2))

  // 跳转到测试结果页面，传递测试结果数据
  router.push({
    path: '/test-result',
    query: {
      data: encodeURIComponent(JSON.stringify(testResult)),
    },
  })
}

// 计算审核是否完成
const isReviewComplete = computed(() => {
  return reviewQuestions.value.length > 0
})

// 组件挂载时初始化数据
onMounted(() => {
  initReviewData()
})
</script>

<style scoped>
.review-page {
  min-height: 100vh;
  background: #f8fafc;
  padding: 2rem;
}

.module-header {
  margin-bottom: 2rem;
  padding: 1.5rem;
  border-radius: 12px;
  background: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.module-title {
  font-size: 1.75rem;
  color: #1f2937;
  margin: 0;
}

.beta-badge {
  font-size: 0.75rem;
  background: #6366f1;
  color: white;
  padding: 4px 12px;
  border-radius: 20px;
  font-weight: 500;
}

.review-container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.test-overview {
  padding: 2rem;
  border-bottom: 1px solid #e5e7eb;
}

.section-title {
  font-size: 1.25rem;
  color: #374151;
  margin-bottom: 1.5rem;
  font-weight: 600;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.overview-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.overview-item label {
  font-weight: 500;
  color: #6b7280;
  min-width: 80px;
}

.overview-item span {
  color: #1f2937;
}

.status-badge {
  background: #fef3c7;
  color: #d97706;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;
}

.review-table-section {
  padding: 2rem;
}

.table-container {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
}

.review-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.review-table th,
.review-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

.review-table th {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.review-table tr:hover {
  background: #f8fafc;
}

.text-center {
  text-align: center;
}

.question-content,
.model-answer {
  max-width: 300px;
}

.content-preview,
.answer-preview {
  line-height: 1.4;
  color: #4b5563;
}

.correct-status {
  color: #10b981;
  font-weight: 500;
}

.incorrect-status {
  color: #ef4444;
  font-weight: 500;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.detail-button {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: background 0.2s;
}

.detail-button:hover {
  background: #2563eb;
}

.toggle-button {
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
}

.toggle-button.correct {
  background: #10b981;
  color: white;
}

.toggle-button.correct:hover {
  background: #059669;
}

.toggle-button.incorrect {
  background: #ef4444;
  color: white;
}

.toggle-button.incorrect:hover {
  background: #dc2626;
}

.review-actions {
  padding: 2rem;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.back-button {
  background: #6b7280;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.back-button:hover {
  background: #4b5563;
}

.submit-button {
  background: #10b981;
  color: white;
  border: none;
  padding: 12px 32px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.submit-button:hover:not(:disabled) {
  background: #059669;
}

.submit-button:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

/* 模态框样式 - 复用题库模块样式 */
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

/* 更新后的模态框容器样式 */
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

/* 更新后的模态框头部样式 */
.modal-header {
  background: #3498db;
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

/* 更新后的模态框关闭按钮样式 */
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

/* 更新后的详情项样式 - 文字顶格展示 */
.detail-section {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid #f1f5f9;
  align-items: start;
  text-align: left;
}

.detail-section:last-child {
  border-bottom: none;
}

.detail-section label {
  color: #64748b;
  font-weight: 500;
  text-align: left !important;
  margin: 0;
}

.detail-section span,
.detail-section p {
  color: #334155;
  line-height: 1.6;
}

.question-text,
.answer-text {
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
  line-height: 1.6;
  color: #4b5563;
}

.modal-actions {
  padding: 16px 24px;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: flex-end;
}

.close-button {
  padding: 10px 20px;
  border: none;
  background: #3498db;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.close-button:hover {
  background: #2980b9;
}

@media (max-width: 768px) {
  .review-page {
    padding: 1rem;
  }

  .overview-grid {
    grid-template-columns: 1fr;
  }

  .review-table {
    font-size: 0.75rem;
  }

  .action-buttons {
    flex-direction: column;
  }
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
</style>
