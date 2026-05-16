<template>
  <div class="test-list px-4 py-6">
    <div class="max-w-7xl mx-auto">
      <div class="module-header mb-6">
        <div class="flex justify-between items-center">
          <h2 class="text-2xl font-semibold text-gray-800">
            <i class="fas fa-clipboard-list mr-2"></i>测试记录
          </h2>
          <div class="flex gap-4">
            <div class="stat-card">
              <i class="fas fa-vial"></i>
              <div>
                <div class="stat-title">总测试数</div>
                <div class="stat-value">{{ totalTests }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="control-bar mb-4">
        <div class="flex justify-between items-center">
          <div class="test-id-jump flex items-center gap-2">
            <label for="test-id-input" class="jump-label">测试ID跳转：</label>
            <input
              id="test-id-input"
              v-model="jumpTestId"
              type="number"
              placeholder="输入测试ID"
              class="jump-input"
              min="1"
              @keyup.enter="handleTestIdJump"
            />
            <button
              @click="handleTestIdJump"
              class="jump-button"
              :disabled="loading || !jumpTestId"
            >
              <i class="fas fa-arrow-right"></i> 跳转
            </button>
          </div>

          <div class="pagination">
            <button
              @click="handlePageChange(currentPage - 1)"
              :disabled="currentPage === 1"
              class="page-button"
            >
              ◀
            </button>
            <span>第 {{ currentPage }} / {{ totalPages }} 页</span>
            <button
              @click="handlePageChange(currentPage + 1)"
              :disabled="currentPage >= totalPages"
              class="page-button"
            >
              ▶
            </button>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="table-container">
          <table class="modern-table">
            <thead>
              <tr>
                <th style="width: 80px; text-align: center">序号</th>
                <th style="width: 20%; text-align: left">测试名称</th>
                <th style="width: 15%; text-align: left">被测模型</th>
                <th style="width: 15%; text-align: left">维度</th>
                <th style="width: 12%; text-align: center">题目数量</th>
                <th style="width: 12%; text-align: center">总得分</th>
                <th style="width: 18%; text-align: center">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(test, index) in tests" :key="test.testId">
                <td style="text-align: center">{{ (currentPage - 1) * pageSize + index + 1 }}</td>
                <td style="text-align: left">{{ test.name }}</td>
                <td style="text-align: left">{{ test.modelName }}</td>
                <td style="text-align: left">{{ getDimensionLabel(test.dimension) }}</td>
                <td style="text-align: center; color: #3b82f6; font-weight: 500">
                  {{ test.count }}题
                </td>
                <td style="text-align: center; font-weight: 600; color: #6366f1">
                  {{ test.finalScore }}
                </td>
                <td style="text-align: center">
                  <button class="detail-button" @click="viewTestDetail(test)">
                    <i class="fas fa-chart-bar"></i> 详情
                  </button>
                </td>
              </tr>
              <tr v-if="loading">
                <td colspan="7" style="text-align: center; padding: 2rem">
                  <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #6366f1"></i>
                  <p style="margin-top: 1rem">加载中...</p>
                </td>
              </tr>
              <tr v-if="!loading && tests.length === 0">
                <td colspan="7" style="text-align: center; padding: 2rem">
                  <p>暂无测试记录</p>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div v-if="selectedTest" class="modal-mask">
    <div class="modal-container">
      <div class="modal-header">
        <h3>{{ selectedTest.name }} 测试详情</h3>
        <button @click="selectedTest = null" class="modal-close">
          <i class="fas fa-times"></i>
        </button>
      </div>

      <div class="modal-content">
        <div class="info-grid">
          <div class="info-item">
            <label>测试名称：</label>
            <span class="info-value">{{ selectedTest.name }}</span>
          </div>
          <div class="info-item">
            <label>测试ID：</label>
            <span class="info-value">{{ selectedTest.testId }}</span>
          </div>
          <div class="info-item">
            <label>被测模型：</label>
            <span class="info-value">{{ selectedTest.modelName }}</span>
          </div>
          <div class="info-item">
            <label>维度：</label>
            <span class="info-value">{{ getDimensionLabel(selectedTest.dimension) }}</span>
          </div>
          <div class="info-item">
            <label>分维度：</label>
            <span class="info-value">{{ getMetricLabel(selectedTest.metricName) }}</span>
          </div>
          <div class="info-item">
            <label>题目数量：</label>
            <span class="info-value">{{ selectedTest.count }}题</span>
          </div>
          <div class="info-item">
            <label>总得分：</label>
            <span class="info-value">{{ selectedTest.finalScore }}</span>
          </div>
          <div class="info-item">
            <label>操作系统：</label>
            <span class="info-value">{{ selectedTest.os }}</span>
          </div>
          <div class="info-item">
            <label>CPU：</label>
            <span class="info-value">{{ selectedTest.cpu }}</span>
          </div>
          <div class="info-item">
            <label>GPU：</label>
            <span class="info-value">{{ selectedTest.gpu }}</span>
          </div>
          <div class="info-item">
            <label>测试描述：</label>
            <span class="info-value">{{ selectedTest.testDescription || '无' }}</span>
          </div>
          <div class="info-item">
            <label>测试时间：</label>
            <span class="info-value">{{ selectedTest.updateTime }}</span>
          </div>
        </div>

        <div
          v-if="selectedTest.metricScores && Object.keys(selectedTest.metricScores).length > 0"
          class="result-section"
        >
          <h4><i class="fas fa-chart-line"></i> 指标得分</h4>
          <div class="metrics-grid">
            <div
              v-for="(score, metric) in selectedTest.metricScores"
              :key="metric"
              class="metric-item"
            >
              <span class="metric-label">{{ getSubMetricLabel(metric) }}</span>
              <span class="metric-value">{{ score }}</span>
            </div>
          </div>
        </div>

        <div
          v-if="selectedTest.singleScores && selectedTest.singleScores.length > 0"
          class="result-section"
        >
          <h4><i class="fas fa-list"></i> 题目得分</h4>
          <div class="single-scores">
            <span
              v-for="(score, index) in selectedTest.singleScores"
              :key="index"
              class="score-badge"
            >
              题{{ index + 1 }}: {{ score }}
            </span>
          </div>
        </div>

        <div v-if="selectedTest.resultDescription" class="result-section">
          <h4><i class="fas fa-file-alt"></i> 结果描述</h4>
          <div class="result-description">
            {{ selectedTest.resultDescription }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { DataSetService, TestListResponseData } from '../api/dataSet' // 使用项目已有的API服务

// 定义测试记录类型
interface TestRecord {
  testId: number
  name: string
  modelName: string
  dimension: string
  metricName: string | null
  os: string
  cpu: string
  gpu: string
  count: number
  testDescription: string | null
  finalScore: number
  resultDescription: string
  updateTime: string
  metricScores: Record<string, number>
  singleScores: number[]
}

// 定义API响应类型
interface TestResponse {
  code: number
  msg: string
  data: {
    records: TestRecord[]
    total: number
    size: number
    current: number
    pages: number
  }
  timestamp: number
}

// 响应式数据
const tests = ref<TestRecord[]>([])
const selectedTest = ref<TestRecord | null>(null)
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(15)
const totalPages = ref(0)
const totalTests = ref(0)
const loading = ref(false)
const jumpTestId = ref<number | null>(null)

// 获取主维度标签
const getDimensionLabel = (dimension: string) => {
  const dimensionMap = {
    performance: '性能',
    reliability: '可靠性',
    security: '安全性',
    fairness: '公平性',
  }
  return dimensionMap[dimension as keyof typeof dimensionMap] || dimension || '未设置'
}

// 获取分维度标签
const getMetricLabel = (metric: string | null) => {
  if (!metric) return '未设置'
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
  return metricMap[metric as keyof typeof metricMap] || metric
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
  return subMetricMap[subMetric as keyof typeof subMetricMap] || subMetric
}

// 从接口获取测试记录
const fetchTests = async () => {
  loading.value = true
  try {
    // 使用项目已有的API服务方法
    const response = await DataSetService.getTestList({
      pageNum: currentPage.value,
    })

    console.log('Response data:', response)

    if (response.code === 200) {
      tests.value = response.data.records
      totalTests.value = response.data.total
      totalPages.value = response.data.pages
      currentPage.value = response.data.current
      pageSize.value = response.data.size
    } else {
      ElMessage.error('获取测试记录失败: ' + response.msg)
    }
  } catch (error) {
    console.error('获取测试记录失败:', error)
    if (error instanceof Error) {
      ElMessage.error(`网络错误: ${error.message}`)
    } else {
      ElMessage.error('网络错误，获取测试记录失败')
    }
  } finally {
    loading.value = false
  }
}

// 处理页面变化
const handlePageChange = (page: number) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
    fetchTests()
  }
}

// 处理搜索
const handleSearch = () => {
  // 重置到第一页
  currentPage.value = 1
  // 搜索功能需要后端支持，这里先只刷新数据
  fetchTests()
}

// 查看测试详情
const viewTestDetail = (test: TestRecord) => {
  selectedTest.value = test
}

// 组件挂载时获取数据
onMounted(() => {
  fetchTests()
})

// 处理根据testId跳转
const handleTestIdJump = async () => {
  if (!jumpTestId.value) {
    ElMessage.warning('请输入测试ID')
    return
  }

  // 先检查当前页是否有该testId
  const currentPageTest = tests.value.find((test) => test.testId === jumpTestId.value)
  if (currentPageTest) {
    ElMessage.success('已找到测试记录')
    viewTestDetail(currentPageTest)
    return
  }

  // 从第一页开始查找
  let found = false
  let searchPage = 1
  loading.value = true

  try {
    while (searchPage <= totalPages.value && !found) {
      const response = await DataSetService.getTestList({ pageNum: searchPage })

      if (response.code === 200) {
        const targetTest = response.data.records.find((test) => test.testId === jumpTestId.value)
        if (targetTest) {
          // 跳转到对应页面
          currentPage.value = searchPage
          tests.value = response.data.records
          found = true
          ElMessage.success('已跳转到测试所在页面')
          // 自动打开详情
          viewTestDetail(targetTest)
        } else {
          // 继续查找下一页
          searchPage++
        }
      } else {
        ElMessage.error('查询失败: ' + response.msg)
        break
      }
    }

    if (!found) {
      ElMessage.warning(`未找到testId为${jumpTestId.value}的测试记录`)
    }
  } catch (error) {
    console.error('跳转失败:', error)
    if (error instanceof Error) {
      ElMessage.error(`跳转失败: ${error.message}`)
    } else {
      ElMessage.error('跳转失败，网络错误')
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.control-bar {
  background-color: white;
  padding: 16px;
  border-radius: 8px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
}

.modern-input {
  width: 256px;
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  outline: none;
}

.modern-input:focus {
  box-shadow: 0 0 0 4px rgba(147, 197, 253, 0.5);
}

.page-button {
  padding: 6px 12px;
  border-radius: 4px;
  background-color: #f3f4f6;
  cursor: pointer;
}

.page-button:hover {
  background-color: #e5e7eb;
}

.page-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 以下样式保持不变 */
.stat-card {
  background-color: white;
  padding: 16px 24px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  gap: 1rem;
  min-width: 200px;
}

.stat-card i {
  font-size: 2rem;
  color: #6366f1;
  background-color: #ede9fe;
  padding: 0.75rem;
  border-radius: 50%;
}

.stat-card .stat-title {
  font-size: 0.875rem;
  color: #6b7280;
}

.stat-card .stat-value {
  font-size: 2rem;
  font-weight: 600;
  color: #374151;
}

.model-tags {
  display: inline-block;
  max-width: 240px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}

/* 详情按钮样式 */
.detail-button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.detail-button:hover {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.detail-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 5px rgba(99, 102, 241, 0.2);
}

.detail-button i {
  font-size: 13px;
}

/* 保持与DataSet.vue一致的单元格内边距 */
.modern-table td,
.modern-table th {
  padding: 12px 20px;
}

/* 优化表格间距 */
.modern-table th,
.modern-table td {
  font-size: 15px;
}

.modern-table td:nth-child(3) {
  font-weight: 500;
  color: #3b82f6;
}

.modern-table {
  min-width: 100%;
  border-collapse: collapse;
  border-spacing: 0;
  font-size: 0.875rem;
}

.modern-table th,
.modern-table td {
  vertical-align: middle;
}

.modern-table th {
  padding: 0.75rem 1rem;
  background-color: #f9fafb;
  color: #374151;
  font-weight: 500;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

.modern-table td {
  padding: 0.75rem 1rem;
  color: #4b5563;
  white-space: nowrap;
  border-bottom: 1px solid #f3f4f6;
}

.modern-table tr:hover td {
  background-color: #eff6ff;
}

.table-container {
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.modal-mask {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
}

.modal-container {
  background: #ffffff;
  border-radius: 12px;
  width: 80%;
  max-width: 800px;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  border: 1px solid #e5e7eb;
}

.modal-header {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  border-radius: 12px 12px 0 0;
  padding: 18px 24px;
  position: relative;
}

.modal-header h3 {
  color: white;
  margin: 0;
}

.modal-close {
  position: absolute;
  right: 24px;
  top: 50%;
  transform: translateY(-50%);
  color: white;
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-content {
  padding: 24px;
}

/* 信息网格 */
.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.info-item label {
  font-weight: 600;
  color: #4b5563;
  font-size: 0.875rem;
}

.info-value {
  color: #1f2937;
  font-size: 1rem;
  padding: 0.5rem 1rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

/* 结果部分 */
.result-section {
  margin-bottom: 2rem;
}

.result-section h4 {
  color: #1e293b;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.metric-label {
  font-weight: 500;
  color: #4b5563;
}

.metric-value {
  font-weight: 600;
  color: #6366f1;
  font-size: 1.25rem;
}

/* 题目得分 */
.single-scores {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.score-badge {
  background: #dbeafe;
  color: #1e40af;
  padding: 0.5rem 0.875rem;
  border-radius: 9999px;
  font-size: 0.875rem;
}

/* 结果描述 */
.result-description {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 8px;
  padding: 1rem;
  color: #0c4a6e;
  line-height: 1.6;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .modern-table th,
  .modern-table td {
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
  }

  /* 移动端测试ID跳转布局优化 */
  .test-id-jump {
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
  }
}

/* 测试ID跳转样式 */
.jump-label {
  font-weight: 600;
  color: #4b5563;
  font-size: 14px;
  white-space: nowrap;
}

.jump-input {
  width: 160px;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: all 0.3s ease;
}

.jump-input:focus {
  box-shadow: 0 0 0 4px rgba(147, 197, 253, 0.5);
  border-color: #6366f1;
}

.jump-button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.jump-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.jump-button:active:not(:disabled) {
  transform: translateY(0);
  box-shadow: 0 2px 5px rgba(99, 102, 241, 0.2);
}

.jump-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
</style>
