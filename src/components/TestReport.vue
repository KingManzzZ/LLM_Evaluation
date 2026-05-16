<template>
  <div class="test-report" ref="reportContent">
    <div v-if="loading" class="loading-overlay">
      <i class="fas fa-spinner fa-spin"></i>
      正在生成报告...
    </div>
    <div v-else>
      <div class="header">
        <h1>{{ testData.name }} 测试报告</h1>
        <p>生成时间：{{ generateTime }}</p>
        <div class="header-actions">
          <button class="download-button" @click="exportToPDF" ref="downloadButton">
            <i class="fas fa-download"></i> 下载报告
          </button>
          <button class="print-button" @click="printReport">
            <i class="fas fa-print"></i> 打印报告
          </button>
        </div>
      </div>

      <!-- 测试概览 -->
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
            <span>{{ apiResponseData.testId || 'N/A' }}</span>
          </div>
          <div class="overview-item">
            <label>总得分：</label>
            <span class="final-score">{{
              apiResponseData.finalScore === null ? 'N/A' : apiResponseData.finalScore.toFixed(2)
            }}</span>
          </div>
          <div class="overview-item">
            <label>题目数量：</label>
            <span>{{ testData.questions ? testData.questions.length : 0 }}</span>
          </div>
        </div>
      </div>

      <!-- 性能指标 -->
      <div v-if="apiResponseData.metricScores" class="result-section">
        <h3 class="section-title">性能指标</h3>
        <div class="metrics-grid">
          <div
            v-for="(score, metric) in apiResponseData.metricScores"
            :key="metric"
            class="metric-card"
          >
            <div class="metric-header">
              <span class="metric-title">{{ getMetricLabel(metric) }}</span>
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
          <!-- 总得分卡片 - 始终显示 -->
          <div class="score-card">
            <div class="score-header">
              <span class="score-title">总得分</span>
              <span class="score-badge">满分100</span>
            </div>
            <div class="score-value">
              {{
                apiResponseData.finalScore === null ? 'N/A' : apiResponseData.finalScore.toFixed(2)
              }}
            </div>
            <div class="score-progress">
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{
                    width:
                      (apiResponseData.finalScore === null ? 0 : apiResponseData.finalScore) + '%',
                  }"
                ></div>
              </div>
            </div>
          </div>

          <!-- 动态显示指标得分 (test2接口) -->
          <template v-if="getMetricScores()">
            <div
              v-for="(score, metric) in getMetricScores()"
              :key="metric"
              class="score-card"
              v-if="score !== null && score !== undefined"
            >
              <div class="score-header">
                <span class="score-title">{{ getMetricLabel(metric) }}</span>
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

          <!-- 默认得分统计 (test1接口) -->
          <template v-else>
            <div class="score-card">
              <div class="score-header">
                <span class="score-title">平均分</span>
                <span class="score-badge">每题得分</span>
              </div>
              <div class="score-value">
                {{ getAverageScore() === null ? 'N/A' : getAverageScore().toFixed(2) }}
              </div>
              <div class="score-progress">
                <div class="progress-bar">
                  <div
                    class="progress-fill"
                    :style="{ width: (getAverageScore() === null ? 0 : getAverageScore()) + '%' }"
                  ></div>
                </div>
              </div>
            </div>

            <div class="score-card">
              <div class="score-header">
                <span class="score-title">正确率</span>
                <span class="score-badge">正确题目</span>
              </div>
              <div class="score-value">
                {{ getCorrectRate() === null ? 'N/A' : getCorrectRate().toFixed(1) }}%
              </div>
              <div class="score-progress">
                <div class="progress-bar">
                  <div
                    class="progress-fill"
                    :style="{ width: (getCorrectRate() === null ? 0 : getCorrectRate()) + '%' }"
                  ></div>
                </div>
              </div>
            </div>

            <div class="score-card">
              <div class="score-header">
                <span class="score-title">通过题目</span>
                <span class="score-badge">数量统计</span>
              </div>
              <div class="score-value">{{ getPassedCount() }}</div>
              <div class="score-progress">
                <div class="progress-bar">
                  <div
                    class="progress-fill"
                    :style="{ width: (getPassedRate() === null ? 0 : getPassedRate()) + '%' }"
                  ></div>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>

      <!-- 题目详情部分已经移除，根据用户要求 -->
      <!-- 详细题目详情也已经移除 -->
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import html2pdf from 'html2pdf.js'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()
const reportContent = ref<HTMLElement>()
const testData = ref<any>(null)
const apiResponseData = ref<any>(null)
const generateTime = ref(new Date().toLocaleString())
const loading = ref(true)
const downloadButton = ref<HTMLButtonElement>()

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

// 获取指标得分
const getMetricScores = () => {
  // 检查apiResponseData及其data属性中是否存在metricScores
  if (apiResponseData.value && apiResponseData.value.metricScores) {
    return apiResponseData.value.metricScores
  }
  if (
    apiResponseData.value &&
    apiResponseData.value.data &&
    apiResponseData.value.data.metricScores
  ) {
    return apiResponseData.value.data.metricScores
  }
  return null
}

// 计算平均分
const getAverageScore = () => {
  if (!apiResponseData.value || !apiResponseData.value.singleScore) return null
  const singleScores = Array.isArray(apiResponseData.value.singleScore)
    ? apiResponseData.value.singleScore
    : []
  if (singleScores.length === 0) return null
  const sum = singleScores.reduce((acc, score) => acc + score, 0)
  return sum / singleScores.length
}

// 计算正确率
const getCorrectRate = () => {
  if (!apiResponseData.value || !apiResponseData.value.singleScore) return null
  const singleScores = Array.isArray(apiResponseData.value.singleScore)
    ? apiResponseData.value.singleScore
    : []
  if (singleScores.length === 0) return null
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

// 处理题目数据
const processedQuestions = computed(() => {
  if (!testData.value || !testData.value.questions) return []

  const singleScores =
    apiResponseData.value && apiResponseData.value.singleScore
      ? apiResponseData.value.singleScore
      : []

  return testData.value.questions.map((question, index) => {
    const score = singleScores[index] || 0
    return {
      ...question,
      score: score,
      isCorrect: score >= 60, // 假设60分以上为正确
    }
  })
})

onMounted(() => {
  try {
    const reportId = route.query.reportId
    if (reportId && typeof reportId === 'string') {
      const decodedId = decodeURIComponent(reportId)
      const data = localStorage.getItem(decodedId)
      if (data) {
        const parsedData = JSON.parse(data)
        testData.value = parsedData.testData || parsedData
        apiResponseData.value = parsedData.apiResponseData || {}
      } else {
        ElMessage.error('找不到测试报告数据')
        router.push({ name: 'TestResult' })
      }
    } else {
      router.push({ name: 'TestResult' })
    }
  } catch (e) {
    console.error('数据解析失败:', e)
    router.push({ name: 'TestResult' })
  } finally {
    loading.value = false
  }
})

const exportToPDF = async () => {
  if (!reportContent.value) return

  // 临时隐藏下载按钮
  if (downloadButton.value) {
    downloadButton.value.style.display = 'none'
  }

  const opt = {
    margin: 10,
    filename: `测试报告_${testData.value.name}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2 },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
  }

  try {
    await html2pdf().from(reportContent.value).set(opt).save()
  } finally {
    // 导出完成后显示下载按钮
    if (downloadButton.value) {
      downloadButton.value.style.display = 'inline-block'
    }
  }
}

// 获取通过题目数量
const getPassedCount = () => {
  if (!apiResponseData.value || !apiResponseData.value.singleScore) return 0
  const singleScores = Array.isArray(apiResponseData.value.singleScore)
    ? apiResponseData.value.singleScore
    : []
  return singleScores.filter((score) => score >= 60).length
}

// 获取通过率
const getPassedRate = () => {
  if (!apiResponseData.value || !apiResponseData.value.singleScore) return null
  const singleScores = Array.isArray(apiResponseData.value.singleScore)
    ? apiResponseData.value.singleScore
    : []
  if (singleScores.length === 0) return null
  const passedCount = singleScores.filter((score) => score >= 60).length
  return (passedCount / singleScores.length) * 100
}

// 打印报告
const printReport = () => {
  window.print()
}
</script>

<style scoped>
.test-report {
  padding: 2rem;
  background: white;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
  border-bottom: 2px solid #6366f1;
  padding-bottom: 1rem;
}

.header h1 {
  color: #1e293b;
  margin-bottom: 0.5rem;
}

.header p {
  color: #64748b;
  margin-bottom: 1rem;
}

.download-button {
  background: #6366f1;
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background 0.3s;
  font-size: 1rem;
}

.download-button:hover {
  background: #4f46e5;
}

/* 测试概览样式 */
.result-section {
  margin-bottom: 2rem;
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
}

.overview-item {
  background: #f8fafc;
  padding: 1.2rem;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.overview-item label {
  font-weight: 600;
  color: #374151;
}

.overview-item span {
  color: #6366f1;
  font-weight: 500;
}

.final-score {
  font-size: 1.2rem;
  font-weight: 700;
  color: #10b981 !important;
}

/* 性能指标样式 */
.metrics-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

.metric-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  transition:
    transform 0.2s,
    box-shadow 0.2s;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.metric-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.metric-title {
  font-weight: 600;
  color: #1e293b;
  font-size: 1rem;
}

.metric-badge {
  background: #e0e7ff;
  color: #6366f1;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
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
  background: #f1f5f9;
  border-radius: 10px;
  height: 8px;
  overflow: hidden;
}

.progress-fill {
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  height: 100%;
  border-radius: 10px;
  transition: width 0.3s ease;
}

/* 得分统计样式 */
.score-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

.score-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  transition:
    transform 0.2s,
    box-shadow 0.2s;
}

.score-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.score-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.score-title {
  font-weight: 600;
  color: #1e293b;
  font-size: 1rem;
}

.score-badge {
  background: #e0e7ff;
  color: #6366f1;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
}

.score-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #6366f1;
  margin: 1rem 0;
}

.score-progress {
  margin-top: 1rem;
}

/* 题目详情样式 */
.question-table {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.table-header {
  display: grid;
  grid-template-columns: 60px 1fr 120px 1fr 80px 100px;
  background: #f8fafc;
  padding: 1rem;
  font-weight: 600;
  color: #374151;
  border-bottom: 1px solid #e2e8f0;
}

.table-row {
  display: grid;
  grid-template-columns: 60px 1fr 120px 1fr 80px 100px;
  padding: 1rem;
  border-bottom: 1px solid #f1f5f9;
  align-items: start;
}

.table-row:last-child {
  border-bottom: none;
}

.table-row:hover {
  background: #f8fafc;
}

.question-content,
.answer-content {
  word-break: break-word;
  line-height: 1.4;
}

.answer-content {
  color: #64748b;
  font-size: 0.9rem;
}

.score-cell {
  font-weight: 600;
  color: #6366f1;
  text-align: center;
}

.correct-badge {
  background: #d1fae5;
  color: #065f46;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
}

.incorrect-badge {
  background: #fee2e2;
  color: #991b1b;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
}

.loading-overlay {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  font-size: 1.2rem;
  color: #6366f1;
  z-index: 1000;
}

.loading-overlay i {
  margin-right: 0.5rem;
}

@media print {
  .download-button {
    display: none !important;
  }

  .test-report {
    padding: 0 !important;
    font-size: 12pt;
    max-width: none;
  }

  .header {
    padding-bottom: 0;
    border-bottom: none;
  }

  .section-title {
    page-break-before: avoid;
    margin-top: 1rem;
  }

  .metric-card,
  .score-card {
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .question-table {
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .table-row {
    break-inside: avoid;
    page-break-inside: avoid;
  }

  * {
    color: #000 !important;
  }
}
@media (max-width: 768px) {
  .test-report {
    padding: 1rem;
  }

  .overview-grid {
    grid-template-columns: 1fr;
  }

  .metrics-grid,
  .score-grid {
    grid-template-columns: 1fr;
  }

  .table-header,
  .table-row {
    grid-template-columns: 40px 1fr 80px 1fr 60px 80px;
    font-size: 0.8rem;
    padding: 0.5rem;
  }

  .metric-value,
  .score-value {
    font-size: 1.5rem;
  }
}

/* 详细题目分析样式 */
.detailed-questions {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.detailed-question {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  transition: box-shadow 0.2s;
}

.detailed-question:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.question-header {
  background: #f8fafc;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.question-number {
  font-weight: 600;
  color: #1e293b;
}

.question-score {
  color: #6366f1;
  font-weight: 500;
}

.question-detail {
  padding: 1.5rem;
}

.detail-item {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 1rem;
  align-items: start;
  margin-bottom: 1.5rem;
}

.detail-item:last-child {
  margin-bottom: 0;
}

.detail-item label {
  font-weight: 600;
  color: #374151;
  text-align: left;
}

.question-text,
.answer-text {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  line-height: 1.6;
  word-break: break-word;
  margin: 0;
}

.answer-text {
  background: #f0f9ff;
  border-color: #bae6fd;
}

/* 打印按钮样式 */
.print-button {
  background: #10b981;
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background 0.3s;
  font-size: 1rem;
  margin-left: 1rem;
}

.print-button:hover {
  background: #059669;
}

.header-actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
}
</style>
