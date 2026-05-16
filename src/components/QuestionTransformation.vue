<template>
  <div class="question-transformation">
    <div class="module-header">
      <h2 class="module-title"><i class="fas fa-shapes"></i> 题库变形中心</h2>
    </div>

    <div class="control-panel">
      <!-- 筛选控件组 -->
      <div class="filter-container">
        <el-select
          v-model="selectedType"
          placeholder="请选择题型"
          size="small"
          class="filter-select"
        >
          <el-option value="全部" label="全部题型" />
          <el-option
            v-for="(value, key) in questionTypeMap"
            :key="key"
            :value="value"
            :label="value"
          />
        </el-select>

        <el-select
          v-model="selectedDimension"
          placeholder="请选择主维度"
          size="small"
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
          :placeholder="selectedDimension ? '请选择分维度' : '请先选择主维度'"
          size="small"
          class="filter-select"
          @change="handleMetricChange"
          :disabled="!selectedDimension"
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
          :placeholder="
            !selectedMetric
              ? '请先选择分维度'
              : !showSubMetrics
                ? '当前分维度无小指标'
                : '请选择小指标'
          "
          size="small"
          class="filter-select"
          :disabled="!selectedMetric || !showSubMetrics"
        >
          <el-option
            v-for="sub in subMetricOptions"
            :key="sub.value"
            :label="sub.label"
            :value="sub.value"
          />
        </el-select>

        <button class="query-button" @click="handleSearch">查询</button>
      </div>
    </div>

    <!-- 更新为类题库的表格展示 -->
    <div class="table-container">
      <el-table
        ref="tableRef"
        :data="paginatedQuestions"
        style="width: 100%"
        @selection-change="handleSelectionChange"
        row-key="id"
        :reserve-selection="true"
        @select="handleSelect"
        @select-all="handleSelectAll"
      >
        <!-- 添加勾选框列 -->
        <el-table-column type="selection" width="55" />

        <!-- 序号列 -->
        <el-table-column label="序号" width="80">
          <template #default="{ $index }">
            {{ getQuestionNumber($index) }}
          </template>
        </el-table-column>

        <!-- 题干列 -->
        <el-table-column prop="question" label="题干" />

        <!-- 类型列 -->
        <el-table-column label="类型" width="120">
          <template #default="{ row }">
            <span :class="getTypeClass(row.type)">
              {{ row.type }}
            </span>
          </template>
        </el-table-column>

        <!-- 来源列 -->
        <el-table-column label="来源" width="120">
          <template #default="{ row }">
            <span :class="getSourceClass(row.source)">
              {{ row.source }}
            </span>
          </template>
        </el-table-column>

        <!-- 创建日期列 -->
        <el-table-column label="创建日期" width="120">
          <template #default="{ row }">
            {{ formatDate(row.createTime) }}
          </template>
        </el-table-column>

        <!-- 变形操作列 -->
        <el-table-column label="变形操作" width="200">
          <template #default="{ row, $index }">
            <div class="transform-control">
              <select
                v-model="row.transformationOption"
                class="operation-select"
                :disabled="!row.transformationOptions?.length"
              >
                <option value="">请选择操作</option>
                <option
                  v-for="option in transformationOptions"
                  :key="option.value"
                  :value="option.value"
                >
                  {{ option.label }}
                </option>
              </select>
              <button
                class="transform-button"
                :disabled="!row.transformationOption"
                @click="handleSingleTransformation(row)"
              >
                <i class="fas fa-magic"></i>
                变形
              </button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 统一变形操作区域 -->
      <div class="batch-transformation-container">
        <div class="batch-controls">
          <div class="batch-select">
            <label>选择变形操作：</label>
            <select v-model="batchTransformationOption" class="batch-operation-select">
              <option value="">请选择操作</option>
              <option
                v-for="option in transformationOptions"
                :key="option.value"
                :value="option.value"
              >
                {{ option.label }}
              </option>
            </select>
          </div>
          <button
            class="batch-transform-button"
            :disabled="!batchTransformationOption || selectedQuestions.length === 0"
            @click="handleBatchTransformation"
          >
            <i class="fas fa-magic"></i>
            统一变形
          </button>
        </div>
        <div class="batch-info">
          <span>已选择 {{ selectedQuestions.length }} 个题目</span>
        </div>
      </div>

      <!-- 分页控件 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="pageSize"
          layout="prev, pager, next"
          :total="total"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/api/index'
import { DataSetService } from '@/api/dataSet'
import { transformQuestions } from '@/api/transformationApi'

// 路由实例
const router = useRouter()

// 在script setup中添加
const transformationOptions = ref([
  { value: 'rewrite', label: '问法改写' },
  { value: 'add_noise', label: '添加噪音' },
  { value: 'reverse_polarity', label: '反向极化' },
  { value: 'complicate', label: '表达复杂化' },
  { value: 'substitute', label: '同义替代' },
])

// 新增响应式数据
const questions = ref<any[]>([])
const selectedType = ref('全部')
const selectedDimension = ref('')
const selectedMetric = ref('')
const selectedSubMetric = ref('')
const tableRef = ref()
const currentPage = ref(1)
const pageSize = ref(15) // 后端接口固定每页15条数据
const selectedQuestions = ref<number[]>([])
const total = ref(0) // 新增总条数
const batchTransformationOption = ref('') // 统一变形操作选项
const loading = ref(false) // 加载状态

// 筛选相关响应式数据
const metricOptions = ref<Array<{ value: string; label: string }>>([])
const subMetricOptions = ref<Array<{ value: string; label: string }>>([])

// 计算属性：是否显示小指标
const showSubMetrics = computed(() =>
  ['complex_reasoning_skill', 'long_text_comprehension_skill'].includes(selectedMetric.value),
)

// 数据源映射
const dataSourceMap = {
  input: '输入',
  generated: '智能生成',
  crawler: '爬虫获取',
  manual: '人工生成',
}

// 题型映射
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  question_only: '仅问题',
  question_group: '问题对比组',
}

// 维度映射
const dimensionMap = {
  performance: '性能',
  reliability: '可靠性',
  safety: '安全性',
  fairness: '公平性',
}

// 筛选选项映射（与题库筛选一致）
const dimensionOptions = [
  { value: '', label: '全部' },
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

// 加载题库数据
const loadQuestionBank = async () => {
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

    if (response.code === 200 || response.code === 0) {
      processResponseData(response)
    }
  } catch (error) {
    console.error('加载题库数据失败:', error)
  }
}

// 处理响应数据
const processResponseData = (response: any) => {
  // 确保数据存在
  if (!response.data || !response.data.records) {
    console.warn('API响应数据格式异常:', response)
    questions.value = []
    total.value = 0
    return
  }

  // 计算连续序号：当前页码 * 每页数量 + 当前索引 + 1
  const startIndex = (currentPage.value - 1) * pageSize.value

  questions.value = response.data.records.map((item, index) => ({
    id: item.dataId || index,
    question: item.question || '无题干',
    type: questionTypeMap[item.questionType] || item.questionType || '未知类型',
    source: dataSourceMap[item.dataSource] || item.dataSource || '未知来源',
    createTime: item.createTime || new Date().toISOString(),
    answer: item.answer || '',
    transformationOptions: ['rewrite', 'add_noise', 'substitute'],
    transformationOption: '',
    transformedQuestions: {},
    rawData: item,
  }))

  // 更新总条数
  total.value = response.data.total || 0

  // 数据加载完成后，恢复当前页的选择状态
  nextTick(() => {
    restoreSelection()
  })
}

// 筛选后的题目
const filteredQuestions = computed(() => {
  if (selectedType.value === '全部') {
    return questions.value
  }
  return questions.value.filter((question) => question.type === selectedType.value)
})

// 分页相关计算属性
const paginatedQuestions = computed(() => {
  // 直接使用后端返回的当前页数据，不再进行前端分页
  return filteredQuestions.value
})

// 序号计算
const getQuestionNumber = (index: number) => {
  return (currentPage.value - 1) * pageSize.value + index + 1
}

// 全选当前页
const selectCurrentPage = () => {
  const currentPageIds = paginatedQuestions.value.map((q) => q.id)
  selectedQuestions.value = [...new Set([...selectedQuestions.value, ...currentPageIds])]

  nextTick(() => {
    const table = tableRef.value
    if (table) {
      paginatedQuestions.value.forEach((row) => {
        table.toggleRowSelection(row, true)
      })
    }
  })
}

// 处理选择变化（优化逻辑，只处理当前页的选择状态）
const handleSelectionChange = (selection: any[]) => {
  console.log(
    '选择变化:',
    selection.map((q) => q.id),
  )

  // 获取当前页的所有题目ID
  const currentPageIds = paginatedQuestions.value.map((q) => q.id)

  // 保留之前页面的选择状态，只更新当前页的选择状态
  const previousSelections = selectedQuestions.value.filter((id) => !currentPageIds.includes(id))
  const currentPageSelections = selection.map((q) => q.id)

  selectedQuestions.value = [...previousSelections, ...currentPageSelections]
  console.log('当前已选择题目:', selectedQuestions.value)
}

// 处理单个选择
const handleSelect = (selection: any[], row: any) => {
  const isSelected = selection.includes(row)

  if (isSelected && !selectedQuestions.value.includes(row.id)) {
    // 添加选择
    selectedQuestions.value.push(row.id)
  } else if (!isSelected && selectedQuestions.value.includes(row.id)) {
    // 移除选择
    const index = selectedQuestions.value.indexOf(row.id)
    if (index > -1) {
      selectedQuestions.value.splice(index, 1)
    }
  }

  console.log('单个选择后已选择题目:', selectedQuestions.value)
}

// 处理全选/取消全选
const handleSelectAll = (selection: any[]) => {
  if (selection.length === 0) {
    // 取消全选当前页
    paginatedQuestions.value.forEach((row) => {
      const index = selectedQuestions.value.indexOf(row.id)
      if (index > -1) {
        selectedQuestions.value.splice(index, 1)
      }
    })
  } else {
    // 全选当前页
    paginatedQuestions.value.forEach((row) => {
      if (!selectedQuestions.value.includes(row.id)) {
        selectedQuestions.value.push(row.id)
      }
    })
  }

  console.log('全选/取消全选后已选择题目:', selectedQuestions.value)
}

// 恢复选择状态
const restoreSelection = () => {
  const table = tableRef.value
  if (table && selectedQuestions.value.length > 0) {
    // 清空当前选择
    table.clearSelection()

    // 恢复当前页的选择状态
    paginatedQuestions.value.forEach((row) => {
      if (selectedQuestions.value.includes(row.id)) {
        nextTick(() => {
          table.toggleRowSelection(row, true)
        })
      }
    })
  }
}

const filterQuestions = () => {
  currentPage.value = 1 // 重置到第一页
  loadQuestionBank() // 重新加载数据
}

// 查询按钮处理函数
const handleSearch = () => {
  currentPage.value = 1
  loadQuestionBank()
}

// 单个题目变形处理函数
const handleSingleTransformation = async (question: any) => {
  if (!question.transformationOption) {
    alert('请先选择变形操作')
    return
  }

  try {
    loading.value = true

    // 调用题目变形API
    const transformationResults = await transformQuestions({
      dataIds: [question.id],
      transformationType: question.transformationOption,
    })

    console.log('单个题目变形结果:', transformationResults)

    // 确保变形结果包含原题内容
    const resultsWithOriginal = transformationResults.map((result, index) => {
      // 如果后端返回的数据中已经包含原题内容，则使用后端数据
      // 否则使用前端保存的原题内容
      const originalQuestion =
        result.originalQuestion || question.question || question.originalQuestion || '原题内容缺失'

      return {
        ...result,
        originalQuestion: originalQuestion,
        originalDataId: question.id, // 保存原始dataId
        originalQuestionFromFrontend: question.question, // 保存前端获取的原题内容
      }
    })

    console.log('添加原题内容后的变形结果:', resultsWithOriginal)

    // 跳转到变形结果页面
    router.push({
      name: 'TransformationResults',
      query: {
        results: JSON.stringify(resultsWithOriginal),
      },
    })
  } catch (error) {
    console.error('单个题目变形失败:', error)
    alert('题目变形失败，请重试')
  } finally {
    loading.value = false
  }
}

// 统一变形处理函数
const handleBatchTransformation = async () => {
  if (!batchTransformationOption.value || selectedQuestions.value.length === 0) {
    alert('请先选择变形操作并选中题目')
    return
  }

  // 获取选中的题目ID
  const selectedQuestionIds = selectedQuestions.value

  // 由于questions只包含当前页数据，我们需要从所有页面加载选中题目
  // 这里直接使用选中题目的ID数组，不需要从questions中过滤
  const selectedCount = selectedQuestionIds.length

  // 显示确认对话框
  const operationLabel =
    transformationOptions.value.find((o) => o.value === batchTransformationOption.value)?.label ||
    batchTransformationOption.value

  const confirmMessage = `确定要对选中的 ${selectedCount} 个题目执行"${operationLabel}"操作吗？`

  if (confirm(confirmMessage)) {
    // 执行统一变形操作，直接传递选中题目ID
    await performBatchTransformation(selectedQuestionIds, batchTransformationOption.value)
  }
}

// 执行批量变形操作
const performBatchTransformation = async (selectedQuestionIds: number[], operation: string) => {
  try {
    loading.value = true

    console.log(
      `开始对 ${selectedQuestionIds.length} 个题目执行 ${operation} 操作，题目ID:`,
      selectedQuestionIds,
    )

    // 调用批量变形API
    const transformationResults = await transformQuestions({
      dataIds: selectedQuestionIds,
      transformationType: operation,
    })

    console.log('批量变形结果:', transformationResults)

    // 确保批量变形结果包含原题内容
    const resultsWithOriginal = transformationResults.map((result, index) => {
      // 由于我们无法从questions中获取原题内容（questions只包含当前页数据）
      // 我们依赖后端返回的原题内容
      const finalOriginalQuestion = result.originalQuestion || '原题内容缺失'

      return {
        ...result,
        originalQuestion: finalOriginalQuestion,
        originalDataId: result.id || result.originalDataId, // 保存原始dataId
        originalQuestionFromFrontend: undefined, // 无法从前端获取原题内容
      }
    })

    console.log('添加原题内容后的批量变形结果:', resultsWithOriginal)

    // 跳转到变形结果页面
    router.push({
      name: 'TransformationResults',
      query: {
        results: JSON.stringify(resultsWithOriginal),
      },
    })

    // 清空选择
    selectedQuestions.value = []
    batchTransformationOption.value = ''

    // 刷新表格选择状态
    nextTick(() => {
      const table = tableRef.value
      if (table) {
        table.clearSelection()
      }
      console.log('批量变形后已清空选择状态')
    })
  } catch (error) {
    console.error('批量变形操作失败:', error)
    alert('批量变形操作失败，请重试')
  } finally {
    loading.value = false
  }
}

const getTypeClass = (type: string) => {
  return {
    'type-badge-选择题': type.includes('选择'),
    'type-badge-判断题': type.includes('判断'),
    'type-badge-简答题': type.includes('简答'),
    'type-badge-仅问题': type.includes('仅问题'),
    'type-badge-问题对比组': type.includes('问题对比组'),
  }
}

const getSourceClass = (source: string) => {
  return {
    'source-badge-输入': source.includes('输入'),
    'source-badge-智能生成': source.includes('智能生成'),
    'source-badge-爬虫获取': source.includes('爬虫获取'),
    'source-badge-人工生成': source.includes('人工生成'),
  }
}

const formatDate = (dateString: string) => {
  if (!dateString) return '未知'
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN')
}

// 监听分页变化
watch(currentPage, () => {
  loadQuestionBank()
})

// 组件挂载时加载数据
onMounted(() => {
  loadQuestionBank()
})

const getTransformedQuestion = (question: any) => {
  const operation = question.transformationOption
  const transformedList = question.transformedQuestions[operation]

  if (!transformedList) return '暂无变形数据'

  const randomIndex = Math.floor(Math.random() * transformedList.length)
  const transformedItem = transformedList[randomIndex]

  // 选择题处理逻辑
  if (question.type === '选择题' && transformedItem.options) {
    return `${transformedItem.question}\n选项：${transformedItem.options.join('、')}`
  }

  return transformedItem.question
}
</script>

<style scoped>
/* 模块头部 */
.module-header {
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
}

/* 筛选控件组样式 */
.filter-container {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}

.filter-select {
  width: 180px;
  margin-right: 10px;
}

.count-input {
  width: 150px;
  margin-right: 10px;
}

.query-button {
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
}

.query-button:hover {
  background: linear-gradient(135deg, #2980b9, #2471a3);
  transform: translateY(-1px);
  box-shadow: 0 4px 6px rgba(41, 128, 185, 0.3);
}

.query-button:active {
  transform: translateY(0);
}

.type-selector {
  padding: 8px 32px 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  appearance: none;
  background: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364758b'%3e%3cpath d='M7 10l5 5 5-5z'/%3e%3c/svg%3e")
    no-repeat right 0.75rem center/12px;
}

.modern-table th {
  background: #3498db;
}
/* 调整列宽设置 */
.modern-table th.index,
.modern-table td.index-cell {
  width: 60px;
  min-width: 60px;
}

.modern-table th.type,
.modern-table td.type-badge {
  width: 120px;
  min-width: 120px;
}

.modern-table th.question,
.modern-table td.question-content {
  width: calc(100% - 300px); /* 动态宽度 */
  max-width: 600px;
  white-space: normal; /* 允许换行 */
  text-align: left;
}
.modern-table td {
  padding: 12px 16px;
  vertical-align: middle;
  border-bottom: 1px solid #e2e8f0;
}
.transform-button {
  background: #3498db; /* 统一主色调 */
  color: white;
  border-radius: 8px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
}

.transform-button:not(:disabled):hover {
  background: #2980b9; /* 加深色调 */
  transform: translateY(-1px);
  box-shadow: 0 4px 6px rgba(41, 128, 185, 0.3);
}

.transform-button:disabled {
  background: #bdc3c7; /* 禁用状态灰色 */
  cursor: not-allowed;
  opacity: 0.7;
}

/* 沿用DataSet的模态框动画 */
.modal-slide-enter-active {
  animation: modalSlide 0.3s;
}

/* 沿用DataSet的模态框动画关键帧 */
@keyframes modalSlide {
  from {
    transform: translateY(-20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

/* 其他样式保持不变 */
.question-transformation {
  margin: 20px;
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

th,
td {
  padding: 12px 15px;
  text-align: center;
  border-bottom: 1px solid #e0e0e0;
  border-right: 1px solid #e0e0e0;
}

th:last-child,
td:last-child {
  border-right: none;
}

th {
  background-color: #f5f5f5;
  font-weight: 600;
}

tr:hover {
  background-color: #f9f9f9;
}

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
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  animation: modalSlide 0.3s ease-out;
}

.modal-header {
  background: #3498db;
  padding: 1rem 1.5rem;
  border-radius: 12px 12px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  color: white;
  margin: 0;
  font-size: 1.25rem;
}

.modal-close {
  background: none;
  border: none;
  color: white;
  font-size: 1.25rem;
  cursor: pointer;
}

.modal-content {
  background-color: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  width: 700px;
  max-width: 90%;
}

.modal-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #333;
}

.modal-info {
  display: flex;
  flex-direction: column;
  margin-bottom: 30px;
}

.modal-info div {
  padding: 15px 0;
  font-size: 16px;
  color: #555;
}

.modal-buttons {
  display: flex;
  justify-content: flex-end;
}

.modal-button {
  padding: 12px 25px;
  border: none;
  cursor: pointer;
  border-radius: 6px;
  font-size: 16px;
  margin-left: 15px;
  transition: background-color 0.3s;
}

.modal-button.confirm {
  background-color: #28a745;
  color: white;
}

.modal-button.confirm:hover {
  background-color: #218838;
}

.modal-button.cancel {
  background-color: #dc3545;
  color: white;
}

.modal-button.cancel:hover {
  background-color: #c82333;
}

/* 在样式中新增 */
.operation-select {
  padding: 8px 28px 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  appearance: none;
  background: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364758b'%3e%3cpath d='M7 10l5 5 5-5z'/%3e%3c/svg%3e")
    no-repeat right 12px center/12px;
  transition: all 0.2s ease;
}

.operation-select:hover {
  border-color: #cbd5e1;
  box-shadow: 0 0 0 3px rgba(203, 213, 225, 0.1);
}

.operation-select:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* 新增样式规则 */
.transform-control {
  display: flex;
  gap: 8px;

  .transform-button {
    padding: 8px 16px;
    border-radius: 8px;
    background: #3498db;
    color: white;
    transition: all 0.2s ease;

    &:disabled {
      opacity: 0.6;
      background: #e2e8f0;
      cursor: not-allowed;
    }

    &:not(:disabled):hover {
      box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
      transform: translateY(-1px);
    }
  }
}

.modal-section {
  margin-bottom: 1.5rem;
  text-align: left;

  label {
    display: block;
    color: #64748b;
    margin-bottom: 0.5rem;
    text-align: left;
  }

  .transformed-preview {
    background: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    white-space: pre-wrap;
    text-align: left;
  }
}

.original-question {
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
}

.modal-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  margin-top: 2rem;

  .modal-action {
    padding: 10px 24px;
    border-radius: 8px;

    &.confirm {
      background: #3b82f6;
      color: white;
    }

    &.cancel {
      background: #f1f5f9;
      color: #64748b;
    }
  }
}

/* 统一变形操作区域样式 */
.batch-transformation-container {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.batch-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.batch-select {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.batch-select label {
  font-weight: 600;
  color: #374151;
  white-space: nowrap;
}

.batch-operation-select {
  padding: 8px 28px 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  appearance: none;
  background: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364758b'%3e%3cpath d='M7 10l5 5 5-5z'/%3e%3c/svg%3e")
    no-repeat right 12px center/12px;
  transition: all 0.2s ease;
  min-width: 150px;
}

.batch-operation-select:hover {
  border-color: #cbd5e1;
  box-shadow: 0 0 0 3px rgba(203, 213, 225, 0.1);
}

.batch-operation-select:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.batch-transform-button {
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
}

.batch-transform-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #2980b9, #2471a3);
  transform: translateY(-1px);
  box-shadow: 0 4px 6px rgba(41, 128, 185, 0.3);
}

.batch-transform-button:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  opacity: 0.7;
  transform: none;
  box-shadow: none;
}

.batch-info {
  color: #64748b;
  font-size: 0.875rem;
}
</style>
