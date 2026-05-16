<template>
  <div class="data-set">
    <el-skeleton :loading="loading" animated>
      <div class="module-header">
        <h2 class="module-title"><i class="fas fa-database"></i> 题库</h2>
        <div class="header-actions">
          <!-- 修改筛选组件结构 -->
          <div class="filter-group">
            <!-- 原有题型筛选（移除@change事件） -->
            <el-select v-model="selectedType" placeholder="请选择题型" class="filter-select">
              <el-option value="全部" label="全部题型" />
              <el-option
                v-for="(value, key) in questionTypeMap"
                :key="key"
                :value="value"
                :label="value"
              />
            </el-select>

            <!-- 维度筛选组件（保持不变） -->
            <el-select
              v-model="selectedDimension"
              placeholder="选择主维度"
              class="filter-select"
              @change="handleDimensionChange"
            >
              <el-option
                v-for="dim in dimensionOptions"
                :key="dim.value"
                :label="dim.label"
                :value="dim.value"
              />
            </el-select>

            <el-select
              v-model="selectedMetric"
              placeholder="选择分维度"
              class="filter-select"
              :disabled="!selectedDimension"
              @change="handleMetricChange"
            >
              <el-option
                v-for="metric in metricOptions"
                :key="metric.value"
                :label="metric.label"
                :value="metric.value"
              />
            </el-select>

            <el-select
              v-model="selectedSubMetric"
              placeholder="选择小指标"
              class="filter-select"
              :disabled="!showSubMetrics"
            >
              <el-option
                v-for="sub in subMetricOptions"
                :key="sub.value"
                :label="sub.label"
                :value="sub.value"
              />
            </el-select>

            <!-- 修改后的查询按钮 -->
            <button class="query-button" @click="filterQuestions">
              <i class="fas fa-search"></i> 查询
            </button>
          </div>
        </div>
      </div>
      <div class="table-container">
        <table class="modern-table">
          <thead>
            <tr>
              <th class="index">序号</th>
              <th class="question">题干</th>
              <th class="type">类型</th>
              <th class="source">来源</th>
              <th class="date">日期</th>
              <th class="actions">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(question, index) in paginatedQuestions" :key="index">
              <td>{{ question.rowIdx }}</td>
              <td class="question-content">{{ question.question }}</td>
              <td class="type">
                <span class="type-badge" :class="getTypeClass(question.type)">
                  {{ question.type }}
                </span>
              </td>
              <td class="source">
                <span class="source-badge" :class="getSourceClass(question.source)">
                  {{ question.source }}
                </span>
              </td>
              <td class="date">{{ question.updateTime }}</td>
              <td class="actions">
                <div class="action-container">
                  <button class="icon-button" @click.stop="toggleActions(index)">
                    <i class="fas fa-ellipsis-v"></i>
                  </button>
                  <button
                    class="icon-button view-button"
                    @click="selectedQuestion = { index, ...question }"
                  >
                    <i class="fas fa-eye"></i>
                  </button>
                  <div v-if="activeIndex === index" class="action-menu">
                    <button class="menu-item edit" @click="modifyQuestion(index)">
                      <i class="fas fa-edit"></i> 编辑
                    </button>
                    <button
                      class="menu-item delete"
                      @click="deleteQuestion(question.rawData.dataId)"
                    >
                      <i class="fas fa-trash-alt"></i> 删除
                    </button>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="pagination-controls">
        <button @click="currentPage--" :disabled="currentPage === 1">◀</button>
        <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页（共 {{ total }} 条）</span>
        <div class="page-input-container">
          <input
            v-model="pageInput"
            type="number"
            min="1"
            :max="totalPages"
            placeholder="页码"
            class="page-input"
            @keyup.enter="jumpToPage"
          />
          <button @click="jumpToPage" class="jump-button">跳转</button>
        </div>
        <button
          @click="currentPage++"
          :disabled="currentPage >= totalPages || questions.length < 10"
        >
          ▶
        </button>
      </div>

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
              <label>类型：</label>
              <span class="type-badge" :class="getTypeClass(selectedQuestion.type)">
                {{ selectedQuestion.type }}
              </span>
            </div>
            <div class="detail-item">
              <label>来源：</label>
              <span class="source-badge" :class="getSourceClass(selectedQuestion.source)">
                {{ selectedQuestion.source }}
              </span>
            </div>
            <div class="detail-item">
              <label>日期：</label>
              <span>{{ selectedQuestion.updateTime }}</span>
            </div>
            <div class="detail-item">
              <label>题干：</label>
              <p class="question-text">{{ selectedQuestion.question.split('\n选项：')[0] }}</p>
            </div>
            <!-- 新增选项展示 -->
            <div v-if="selectedQuestion.type === '选择题'" class="detail-item">
              <label>选项：</label>
              <div class="options-container">
                <div
                  v-for="(option, optIndex) in selectedQuestion.question
                    .split('\n选项：')[1]
                    ?.split('\n')
                    .filter(Boolean)"
                  :key="optIndex"
                  class="option-item"
                >
                  {{ option }}
                </div>
              </div>
            </div>
            <div class="detail-item">
              <label>答案：</label>
              <p class="answer-text">{{ selectedQuestion.answer }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 编辑题目模态框 -->
      <div v-if="editingQuestion" class="modal-mask">
        <div class="modal-container edit-modal">
          <div class="modal-header">
            <h3>编辑题目</h3>
            <button @click="editingQuestion = null" class="modal-close">
              <i class="fas fa-times"></i>
            </button>
          </div>
          <div class="modal-content">
            <div class="edit-form">
              <div class="form-item">
                <label>题干：</label>
                <el-input
                  v-model="editingQuestion.question"
                  type="textarea"
                  :rows="3"
                  placeholder="请输入题干"
                />
              </div>
              <div class="form-item">
                <label>选项：</label>
                <el-input
                  v-model="editingQuestion.options"
                  type="textarea"
                  :rows="3"
                  placeholder="请输入选项，用 | 分隔"
                />
              </div>
              <div class="form-item">
                <label>答案：</label>
                <el-input v-model="editingQuestion.answer" placeholder="请输入答案" />
              </div>
              <div class="form-item">
                <label>维度：</label>
                <el-select v-model="editingQuestion.dimension" placeholder="请选择维度">
                  <el-option
                    v-for="dim in dimensionOptions.filter((d) => d.value)"
                    :key="dim.value"
                    :label="dim.label"
                    :value="dim.value"
                  />
                </el-select>
              </div>
              <div class="form-item">
                <label>题型：</label>
                <el-select v-model="editingQuestion.questionType" placeholder="请选择题型">
                  <el-option
                    v-for="(value, key) in questionTypeMap"
                    :key="key"
                    :label="value"
                    :value="key"
                  />
                </el-select>
              </div>
              <div class="form-item">
                <label>来源：</label>
                <el-select v-model="editingQuestion.dataSource" placeholder="请选择来源">
                  <el-option
                    v-for="(value, key) in dataSourceMap"
                    :key="key"
                    :label="value"
                    :value="key"
                  />
                </el-select>
              </div>
            </div>
            <div class="modal-actions">
              <button @click="cancelEdit" class="cancel-button">取消</button>
              <button @click="confirmEdit" class="confirm-button">确定编辑</button>
            </div>
          </div>
        </div>
      </div>
    </el-skeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { DataSetService } from '@/api/dataSet'
import { ElMessage, ElMessageBox } from 'element-plus'

// 初始化加载数据
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  only_question: '仅问题',
  compare_question: '问题对比组',
}

const dataSourceMap = {
  crawler: '爬虫获取',
  model_generation: '大模型生成',
  input: '手动录入',
}

const questions = ref<any[]>([])
const selectedType = ref('全部')
const currentPage = ref(1)
// 假设后端固定每页15条
const total = ref(0)
const totalPages = computed(() => Math.ceil(total.value / 15))
const loading = ref(false)
const selectedQuestion = ref<{ index: number; [key: string]: any } | null>(null)
const activeIndex = ref<number | null>(null)
const editingQuestion = ref<{
  dataId: number
  question: string
  options: string
  answer: string
  dimension: string
  questionType: string
  dataSource: string
} | null>(null)

// 维度映射配置
const dimensionMap = {
  performance: '性能',
  reliability: '可靠性',
  safety: '安全性',
  fairness: '公平性',
}

const dimensionOptions = [
  { value: '', label: '全部' }, // 新增全部选项
  ...Object.entries(dimensionMap).map(([value, label]) => ({ value, label })),
]

const metricOptionsMap = {
  // 性能分维度
  performance: [
    { value: 'system_responsiveness', label: '系统响应效率' },
    { value: 'complex_reasoning_skill', label: '复杂推理能力' },
    { value: 'long_text_comprehension_skill', label: '长文本理解能力' },
  ],
  // 可靠性分维度
  reliability: [
    { value: 'accuracy', label: '准确性' },
    { value: 'robustness', label: '鲁棒性' },
    { value: 'consistency', label: '一致性' },
    { value: 'stability', label: '稳定性' },
  ],
  // 安全性分维度
  safety: [
    { value: 'randomly_generated_samples', label: '随机生成样本' },
    { value: 'command_hijacking', label: '指令挟持' },
    { value: 'jailbreak_attacks', label: '越狱攻击' },
    { value: 'content_distortions', label: '内容扭曲' },
    { value: 'prompt_blocking', label: '提示屏蔽' },
    { value: 'disrupt_conversations', label: '干扰对话' },
    { value: 'black_box', label: '黑盒' },
    { value: 'white_box', label: '白盒' },
  ],
  // 公平性分维度
  fairness: [
    { value: 'gender', label: '性别' },
    { value: 'race', label: '种族' },
    { value: 'age', label: '年龄' },
    { value: 'religion', label: '宗教' },
    { value: 'politics', label: '政治' },
  ],
}

const subMetricMap = {
  // 复杂推理能力子指标
  complex_reasoning_skill: [
    { value: 'mathematical_reasoning', label: '数学推理' },
    { value: 'common_sense_logical_reasoning', label: '常识逻辑推理' },
    { value: 'casual_reasoning', label: '因果推理' },
  ],
  // 长文本理解能力子指标
  long_text_comprehension_skill: [
    { value: 'information_extraction', label: '信息提取' },
    { value: 'contextual_relevance', label: '上下文关联' },
    { value: 'memory_ability', label: '记忆能力' },
  ],
}

// 响应式数据
const selectedDimension = ref('')
const selectedMetric = ref('')
const selectedSubMetric = ref('')
const metricOptions = ref<Array<{ value: string; label: string }>>([])
const subMetricOptions = ref<Array<{ value: string; label: string }>>([])
const pageInput = ref('')
const showSubMetrics = computed(() =>
  ['complex_reasoning_skill', 'long_text_comprehension_skill'].includes(selectedMetric.value),
)

// 处理维度变化
const handleDimensionChange = () => {
  selectedMetric.value = ''
  selectedSubMetric.value = ''
  // 当选择全部时清空分维度选项
  metricOptions.value = selectedDimension.value
    ? metricOptionsMap[selectedDimension.value] || []
    : []
}

// 处理分维度变化
const handleMetricChange = () => {
  selectedSubMetric.value = ''
  subMetricOptions.value = subMetricMap[selectedMetric.value] || []
}

// 优化数据加载方法
const loadQuestions = async () => {
  loading.value = true
  try {
    const response = await DataSetService.getQuestionsPaginated({
      pageNum: currentPage.value,
      questionType:
        selectedType.value === '全部'
          ? ''
          : Object.keys(questionTypeMap).find((k) => questionTypeMap[k] === selectedType.value) ||
            '',
      dimension: selectedDimension.value,
      metric: selectedMetric.value,
      subMetric: selectedSubMetric.value,
    })
    processResponseData(response)
  } catch (error) {
    ElMessage.error('题库加载失败')
  } finally {
    loading.value = false
  }
}
const processResponseData = (response: any) => {
  // 计算连续序号：当前页码 * 每页数量 + 当前索引 + 1
  const pageSize = 15 // 假设每页固定15条数据
  const startIndex = (currentPage.value - 1) * pageSize

  questions.value = response.data.records.map((item, index) => ({
    rowIdx: startIndex + index + 1, // 全局连续序号
    question: formatQuestion(item),
    answer: item.answer,
    type: questionTypeMap[item.questionType],
    source: dataSourceMap[item.dataSource],
    dimension: dimensionMap[item.dimension] || '未分类',
    metric: metricOptionsMap[item.dimension]?.find((m) => m.value === item.metric)?.label || '',
    subMetric: subMetricMap[item.metric]?.find((s) => s.value === item.subMetric)?.label || '',
    updateTime: formatMysqlTimestamp(item.updateTime), // 修改日期处理方式
    rawData: item,
  }))
  total.value = response.data.total
}

// 新增 MySQL timestamp 格式化方法
const formatMysqlTimestamp = (timestamp: string) => {
  if (!timestamp) return '无日期'

  // 处理 MySQL timestamp 格式：YYYY-MM-DD HH:mm:ss
  const [datePart, timePart] = timestamp.split(' ')
  const [year, month, day] = datePart.split('-')
  const [hours, minutes] = timePart?.split(':') || ['00', '00']

  return `${year}年${month}月${day}日 ${hours}:${minutes}`
}

// 题目格式化方法
const formatQuestion = (item: any) => {
  if (item.questionType === 'choice') {
    return `${item.question}\n选项：${item.options
      .split('|')
      .map((o: string) => o.trim().replace(/([A-Z])\./, '$1. '))
      .join('\n')}`
  }
  return item.question
}
// 修改筛选方法调用新接口
const filterQuestions = () => {
  currentPage.value = 1
  loadQuestions()
}

// 修改删除方法对接接口
const deleteQuestion = async (dataId: number) => {
  try {
    // 添加确认对话框
    await ElMessageBox.confirm('确定要删除这道题目吗？此操作不可恢复。', '确认删除', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })

    // 调用删除接口
    await DataSetService.deleteQuestion(dataId)
    ElMessage.success('删除成功')
    loadQuestions() // 重新加载数据
  } catch (error) {
    // 如果是用户取消操作，不显示错误信息
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 编辑题目方法
const modifyQuestion = (index: number) => {
  const question = questions.value[index]
  const rawData = question.rawData

  // 初始化编辑数据
  editingQuestion.value = {
    dataId: rawData.dataId,
    question: rawData.question,
    options: rawData.options || '',
    answer: rawData.answer,
    dimension: rawData.dimension,
    questionType: rawData.questionType,
    dataSource: rawData.dataSource,
  }

  // 关闭操作菜单
  activeIndex.value = null
}

// 取消编辑
const cancelEdit = () => {
  editingQuestion.value = null
}

// 确认编辑
const confirmEdit = async () => {
  if (!editingQuestion.value) return

  try {
    // 验证必填字段
    if (!editingQuestion.value.question.trim()) {
      ElMessage.warning('请输入题干')
      return
    }

    if (!editingQuestion.value.answer.trim()) {
      ElMessage.warning('请输入答案')
      return
    }

    // 准备提交数据
    const submitData = {
      ...editingQuestion.value,
      // 当题型不为选择题时，options字段可为空
      options: editingQuestion.value.questionType === 'choice' ? editingQuestion.value.options : '',
    }

    // 调用编辑接口
    await DataSetService.updateQuestion(submitData)
    ElMessage.success('编辑成功')
    editingQuestion.value = null
    loadQuestions() // 重新加载数据
  } catch (error) {
    console.error('编辑失败:', error)
    ElMessage.error('编辑失败')
  }
}

// 页码跳转功能
const jumpToPage = () => {
  if (!pageInput.value) {
    ElMessage.warning('请输入页码')
    return
  }

  const targetPage = parseInt(pageInput.value)

  if (isNaN(targetPage) || targetPage < 1 || targetPage > totalPages.value) {
    ElMessage.warning(`请输入有效的页码（1-${totalPages.value}）`)
    pageInput.value = ''
    return
  }

  currentPage.value = targetPage
  pageInput.value = ''
}

// 分页计算方法调整
const paginatedQuestions = computed(() => {
  // 直接使用后端返回的完整数据
  return questions.value
})

const toggleActions = (index: number) => {
  // 如果点击的是当前已激活的菜单，则关闭
  if (activeIndex.value === index) {
    activeIndex.value = null
    return
  }

  // 设置新的激活索引
  activeIndex.value = index

  // 延迟执行，确保DOM已更新
  nextTick(() => {
    const actionMenu = document.querySelector('.action-menu') as HTMLElement
    const actionContainer = document.querySelector('.action-container') as HTMLElement

    if (actionMenu && actionContainer) {
      // 获取菜单和容器的位置信息
      const containerRect = actionContainer.getBoundingClientRect()
      const menuRect = actionMenu.getBoundingClientRect()
      const viewportHeight = window.innerHeight

      // 检查菜单是否会超出视口底部
      const menuBottom = containerRect.bottom + menuRect.height
      const willOverflowBottom = menuBottom > viewportHeight

      // 检查菜单是否会超出视口顶部
      const menuTop = containerRect.top - menuRect.height
      const willOverflowTop = menuTop < 0

      // 智能调整菜单位置
      if (willOverflowBottom && !willOverflowTop) {
        // 如果会超出底部但不会超出顶部，则向上展开
        actionMenu.style.bottom = '100%'
        actionMenu.style.top = 'auto'
        actionMenu.style.marginBottom = '5px'
        actionMenu.style.marginTop = '0'
      } else if (willOverflowTop && !willOverflowBottom) {
        // 如果会超出顶部但不会超出底部，则向下展开
        actionMenu.style.top = '100%'
        actionMenu.style.bottom = 'auto'
        actionMenu.style.marginTop = '5px'
        actionMenu.style.marginBottom = '0'
      } else {
        // 默认向上展开
        actionMenu.style.bottom = '100%'
        actionMenu.style.top = 'auto'
        actionMenu.style.marginBottom = '5px'
        actionMenu.style.marginTop = '0'
      }
    }
  })
}

const getTypeClass = (type: string) => {
  return {
    'bg-blue-100 text-blue-800': type.includes('选择'),
    'bg-green-100 text-green-800': type.includes('判断'),
    'bg-purple-100 text-purple-800': type.includes('简答'),
    'bg-yellow-100 text-yellow-800': type.includes('论述'),
    'bg-orange-100 text-orange-800': type.includes('分析'),
    'bg-pink-100 text-pink-800': type.includes('讨论'),
  }
}

const getSourceClass = (source: string) => {
  return {
    'bg-blue-100 text-blue-800': source === '大模型生成',
    'bg-green-100 text-green-800': source === '爬虫获取',
    'bg-purple-100 text-purple-800': source === '手动录入',
  }
}

watch(currentPage, () => {
  loadQuestions()
})

onMounted(loadQuestions)
</script>

<style scoped>
.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-radius: 12px;
}

.module-title {
  font-size: 1.75rem;
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 12px;
}

/* 表格样式 */
.table-container {
  border-radius: 12px;
  overflow-x: auto;
}

.modern-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  background: white;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.modern-table th {
  background: #3498db;
  color: white;
  padding: 16px 20px;
  border-bottom: 2px solid #2c3e50;
}

.modern-table td {
  padding: 14px 20px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: middle; /* 确保单元格内容垂直居中 */
}
/* 调整列宽设置（约170-185行） */
.modern-table th.index,
.modern-table td.index {
  width: 60px;
  padding: 0 8px;
}

.modern-table th.question,
.modern-table td.question-content {
  width: 55%; /* 从60%缩减 */
  max-width: 400px;
  white-space: normal; /* 允许题干换行 */
  text-align: left;
}

.modern-table th.type,
.modern-table td.type {
  width: 120px;
}

.modern-table th.source,
.modern-table td.source {
  width: 120px;
  text-align: center;
  vertical-align: middle;
}

.modern-table th.date,
.modern-table td.date {
  width: 120px;
}

.modern-table th.actions,
.modern-table td.actions {
  width: 100px;
}
.type-badge,
.source-badge {
  display: inline-flex;
  align-items: center; /* 新增垂直居中 */
  justify-content: center; /* 新增水平居中 */
  min-width: 60px;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  height: 32px; /* 固定高度保证对齐一致性 */
}

/* 更新样式部分 */
.action-container {
  position: relative;
  display: inline-block;
}

.action-menu {
  position: absolute;
  right: 0;
  bottom: 100%;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  z-index: 100;
  min-width: 120px;
  margin-bottom: 5px;
}

.menu-item {
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  background: none;
  border: none;
  cursor: pointer;
  transition: background 0.2s;
}

.menu-item:hover {
  background: #f8fafc;
}

.menu-item.edit {
  color: #3b82f6;
}

.menu-item.delete {
  color: #ef4444;
}
/*

/* 响应式设计 */
@media (max-width: 768px) {
  .module-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }

  .modern-table th {
    padding: 12px 15px;
  }

  .question-content {
    max-width: 300px;
    white-space: normal;
  }
}
.icon-button {
  width: 32px;
  height: 32px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
}

.view-button {
  width: 32px;
  height: 32px;
  margin-left: 8px;
}

.action-container {
  display: flex;
  gap: 8px;
  align-items: center;
}
/* 优化筛选按钮样式 */
.action-button.filter {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-button.filter select {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 8px; /* 增加圆角 */
  background-color: white;
  transition: all 0.2s ease;
  appearance: none; /* 移除默认样式 */
  cursor: pointer;
}

.action-button.filter select:hover {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.action-button.filter select:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* 添加下拉箭头图标 */
.action-button.filter {
  position: relative;
}

.action-button.filter::after {
  content: '\f078';
  font-family: 'Font Awesome 5 Free';
  font-weight: 900;
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: #64748b;
}

/* 新增样式 */
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
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  transform: translateY(-20px);
  opacity: 0;
  animation: modalSlide 0.3s ease-out forwards;
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
  /* 添加样式调整 */
  position: relative; /* 新增定位基准 */
  padding-right: 50px; /* 为关闭按钮留出空间 */
}

.modal-header h3 {
  color: white;
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

/* 更新后的模态框关闭按钮样式 */
.modal-close {
  color: rgba(248, 7, 7, 0.8);
  transition: all 0.2s ease;
  /* 添加样式调整 */
  position: absolute;
  right: 24px;
  top: 50%;
  transform: translateY(-50%);
  padding: 8px;
  width: 32px;
  height: 32px;
}

.modal-content {
  padding: 24px;
  max-height: 70vh;
  overflow-y: auto;
  /* 删除可能的居中属性 */
}

/* 更新后的详情项样式 */
.detail-item {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px solid #f1f5f9;
  /* 新增对齐方式 */
  align-items: start;
  text-align: left;
}

.detail-item:last-child {
  border-bottom: none;
}

.detail-item label {
  color: #64748b;
  font-weight: 500;
  /* 强制左对齐 */
  text-align: left !important;
}

.detail-item span,
.detail-item p {
  color: #334155;
  line-height: 1.6;
}

/* 更新后的题干文本样式 */
.question-text {
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
}

/* 编辑模态框样式 */
.edit-modal {
  width: 700px;
  max-width: 95%;
}

.edit-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-item label {
  font-weight: 500;
  color: #374151;
  font-size: 14px;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #e5e7eb;
}

.cancel-button {
  padding: 10px 20px;
  border: 1px solid #d1d5db;
  background: white;
  color: #374151;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-button:hover {
  background: #f9fafb;
  border-color: #9ca3af;
}

.confirm-button {
  padding: 10px 20px;
  border: none;
  background: #3498db;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.confirm-button:hover {
  background: #2980b9;
}

.confirm-button:disabled {
  background: #9ca3af;
  cursor: not-allowed;
  border: 1px solid #e2e8f0;
}

/* 更新后的答案文本样式 */
.answer-text {
  color: #10b981;
  font-weight: 500;
}

/* 更新后的查看按钮样式 */
.view-button {
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-radius: 8px;
  padding: 8px;
  margin-left: 6px;
}

.view-button:hover {
  background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
}

/* 添加分页样式 */
.pagination-controls {
  margin-top: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
}

.pagination-controls button {
  padding: 8px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: white;
  transition: all 0.2s;
}

.pagination-controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 页码输入框样式 */
.page-input-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-input {
  width: 80px;
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  text-align: center;
  font-size: 14px;
  transition: all 0.2s;
}

.page-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.page-input::placeholder {
  color: #9ca3af;
}

.jump-button {
  padding: 8px 16px;
  border: 1px solid #3b82f6;
  border-radius: 6px;
  background: #3b82f6;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.jump-button:hover {
  background: #2563eb;
  border-color: #2563eb;
}

/* 选项容器样式 */
.options-container {
  background: #f8fafc;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid #e2e8f0;
}

.option-item {
  padding: 6px 12px;
  margin: 4px 0;
  background: white;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* 调整筛选组样式 */
.filter-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  width: 100%;
}

.filter-select,
.query-button {
  flex: 1;
  min-width: 200px;
  height: 40px;
}

/* 调整查询按钮样式 */
.query-button {
  background: #3b82f6;
  color: white;
  padding: 0 20px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 40px;
  flex-shrink: 0;
  width: auto;
  height: 32px;
  min-width: 120px;
}
</style>
