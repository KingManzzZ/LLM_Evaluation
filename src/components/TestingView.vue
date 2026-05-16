<template>
  <div class="testing-view">
    <div class="module-header">
      <h2 class="module-title">新建测试</h2>
    </div>

    <div class="test-form">
      <!-- 修改表单容器结构 -->
      <div class="form-row">
        <div class="form-group horizontal">
          <label class="form-label">测试名称：</label>
          <input v-model="testName" type="text" placeholder="请输入测试名称" class="modern-input" />
        </div>

        <div class="form-group horizontal">
          <label class="form-label">被测模型：</label>
          <el-select v-model="selectedModel" placeholder="请选择模型" class="modern-select">
            <el-option
              v-for="model in availableModels"
              :key="model.id"
              :label="model.name"
              :value="model.id"
            />
          </el-select>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group horizontal">
          <label class="form-label">主维度：</label>
          <el-select
            v-model="selectedMainDimension"
            placeholder="请选择主维度"
            class="modern-select"
          >
            <el-option
              v-for="dim in mainDimensions"
              :key="dim.value"
              :label="dim.label"
              :value="dim.value"
            />
          </el-select>
        </div>

        <div class="form-group horizontal" v-if="showSubDimension">
          <label class="form-label">分维度：</label>
          <el-select
            v-model="selectedSubDimension"
            :placeholder="subDimensionPlaceholder"
            class="modern-select"
          >
            <el-option
              v-for="sub in subDimensions[selectedMainDimension]"
              :key="sub.value"
              :label="sub.label"
              :value="sub.value"
            />
          </el-select>
        </div>
      </div>

      <!-- 修改测试环境部分 -->
      <div class="form-row">
        <div class="form-group horizontal">
          <label class="form-label">操作系统：</label>
          <el-select v-model="selectedOS" placeholder="请选择" class="modern-select">
            <el-option label="Windows 11 22H2" value="Windows 11 22H2" />
          </el-select>
        </div>

        <div class="form-group horizontal">
          <label class="form-label">CPU：</label>
          <el-select v-model="selectedCPU" placeholder="请选择" class="modern-select">
            <el-option label="Intel i9-13900K" value="Intel i9-13900K" />
          </el-select>
        </div>

        <div class="form-group horizontal">
          <label class="form-label">GPU：</label>
          <el-select v-model="selectedGPU" placeholder="请选择" class="modern-select">
            <el-option label="NVIDIA RTX 4090" value="NVIDIA RTX 4090" />
          </el-select>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label"> <i class="fas fa-search"></i> 题目筛选 </label>

        <!-- 添加维度筛选组件 -->
        <div class="filter-group">
          <el-select
            v-model="selectedDimension"
            placeholder="选择主维度"
            class="filter-select"
            @change="handleDimensionChange"
          >
            <el-option value="" label="全部维度" />
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
            <el-option value="" label="全部分维度" />
            <el-option
              v-for="metric in metricOptions"
              :key="metric.value"
              :label="metric.label"
              :value="metric.value"
            />
          </el-select>

          <el-select
            v-model="selectedSubMetric"
            placeholder="选择子指标"
            class="filter-select"
            :disabled="!showSubMetrics"
            @change="handleSubMetricChange"
          >
            <el-option value="" label="全部子指标" />
            <el-option
              v-for="sub in subMetricOptions"
              :key="sub.value"
              :label="sub.label"
              :value="sub.value"
            />
          </el-select>

          <button class="query-button" @click="filterQuestions">
            <i class="fas fa-search"></i> 查询
          </button>
        </div>

        <!-- 添加提示信息 -->
        <div class="filter-hint"><i class="fas fa-info-circle"></i> 可以先后选择多个子指标</div>

        <!-- 显示已选择的子指标及对应题目数量 -->
        <div v-if="selectedSubMetrics.length > 0" class="selected-submetrics">
          <h4 class="selected-submetrics-title">已选择的子指标</h4>
          <div class="selected-submetrics-list">
            <div
              v-for="item in selectedSubMetrics"
              :key="item.subMetric"
              class="selected-submetric-item"
            >
              <span class="submetric-name">{{ item.subMetricLabel }}</span>
              <span class="submetric-count">{{ item.questionCount }}题</span>
              <button class="remove-submetric" @click="removeSelectedSubMetric(item.subMetric)">
                <i class="fas fa-times"></i>
              </button>
            </div>
          </div>
        </div>

        <div class="question-table">
          <!-- 修改表格组件 -->
          <el-table
            ref="tableRef"
            :data="paginatedQuestions"
            style="width: 100%"
            @selection-change="handleSelectionChange"
            row-key="id"
            :reserve-selection="true"
            v-loading="loading"
          >
            <el-table-column type="selection" width="55" />
            <!-- 修改表格列定义 -->
            <el-table-column label="序号" width="80">
              <template #default="{ $index }">
                {{ getQuestionNumber($index) }}
              </template>
            </el-table-column>
            <el-table-column prop="content" label="题干" />
            <el-table-column label="类型" width="120">
              <template #default="{ row }">
                <span :class="getTypeClass(row.type)">
                  {{ row.type }}
                </span>
              </template>
            </el-table-column>
            <el-table-column label="来源" width="120">
              <template #default="{ row }">
                <span :class="getSourceClass(row.source)">
                  {{ row.source }}
                </span>
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination-container">
            <el-pagination
              v-model:current-page="currentPage"
              :page-size="pageSize"
              layout="prev, pager, next"
              :total="total"
              @current-change="handlePageChange"
            />
            <span class="page-info">共 {{ total }} 条</span>
            <button class="select-all-button" @click="selectCurrentPage">全选本页</button>
          </div>
        </div>
      </div>

      <div class="form-actions">
        <button class="reset-button" @click="resetForm"><i class="fas fa-undo"></i> 重置</button>
        <button @click="submitTest" class="submit-button">
          <i class="fas fa-plus-circle"></i> 创建测试
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
const router = useRouter()
import TestResultView from './TestResultView.vue'
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { DataSetService } from '@/api/dataSet'
// 确保表格引用正确 (在setup顶部添加)
const tableRef = ref(null)

// 模拟数据
const testName = ref('')
const selectedMethod = ref('performance')
const selectedModel = ref('')
const availableModels = ref([
  { id: 'Qwen-max', name: 'Qwen-max' },
  { id: 'DeepSeek-V3', name: 'DeepSeek-V3' },
  { id: 'GPT-4o-mini', name: 'GPT-4o-mini' },
  { id: 'ERNIE-4.0-8K', name: 'ERNIE-4.0-8K' },
  { id: 'Yi-Lightning', name: 'Yi-lightning' },
])
const selectedQuestions = ref<string[]>([])

// 题库数据相关变量
const questions = ref<any[]>([])
const loading = ref(false)
const currentPage = ref(1)
const total = ref(0)
const totalPages = computed(() => Math.ceil(total.value / pageSize))
const pageSize = 15

// 题目类型映射
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  only_question: '仅问题',
  compare_question: '问题对比组',
}

// 数据来源映射
const dataSourceMap = {
  crawler: '爬虫获取',
  model_generation: '大模型生成',
  input: '手动录入',
}

// 维度映射配置
const dimensionMap = {
  performance: '性能',
  reliability: '可靠性',
  safety: '安全性',
  fairness: '公平性',
}

// 题目类型样式映射
const typeClassMap = {
  选择题: 'bg-blue-100 text-blue-800',
  判断题: 'bg-green-100 text-green-800',
  简答题: 'bg-purple-100 text-purple-800',
  仅问题: 'bg-orange-100 text-orange-800',
  问题对比组: 'bg-indigo-100 text-indigo-800',
}

// 加载题库数据
const loadQuestions = async () => {
  loading.value = true
  try {
    // 使用后端分页，每次只获取当前页的数据，支持筛选参数
    const response = await DataSetService.getQuestionsPaginated({
      pageNum: currentPage.value,
      pageSize: pageSize,
      questionType: '',
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

// 处理响应数据
const processResponseData = (response: any) => {
  questions.value = response.data.records.map((item, index) => ({
    id: item.dataId.toString(), // 使用dataId作为唯一标识
    content: formatQuestion(item),
    type: questionTypeMap[item.questionType],
    source: dataSourceMap[item.dataSource],
    dimension: dimensionMap[item.dimension] || '未分类',
    updateTime: formatMysqlTimestamp(item.updateTime),
    rawData: item,
  }))
  total.value = response.data.total || 0 // 使用后端返回的总数

  // 数据加载完成后，恢复选中状态
  nextTick(() => {
    restoreSelection()
  })
}

// 恢复选中状态
const restoreSelection = () => {
  const table = tableRef.value
  if (table && selectedQuestions.value.length > 0) {
    // 使用双重nextTick确保表格完全渲染完成
    nextTick(() => {
      nextTick(() => {
        // 清除当前页的所有选中状态
        table.clearSelection()

        // 遍历当前页数据，选中已选择的题目
        paginatedQuestions.value.forEach((row) => {
          if (selectedQuestions.value.includes(row.id)) {
            table.toggleRowSelection(row, true)
          }
        })
      })
    })
  }
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

// MySQL timestamp 格式化方法
const formatMysqlTimestamp = (timestamp: string) => {
  if (!timestamp) return '无日期'
  const [datePart, timePart] = timestamp.split(' ')
  const [year, month, day] = datePart.split('-')
  const [hours, minutes] = timePart?.split(':') || ['00', '00']
  return `${year}年${month}月${day}日 ${hours}:${minutes}`
}

// 筛选后的题目列表
const filteredQuestions = computed(() => {
  return questions.value
})

// 分页后的题目数据
const paginatedQuestions = computed(() => {
  // 直接使用后端返回的当前页数据，不再进行前端分页
  return filteredQuestions.value
})

// 添加序号计算逻辑
const getQuestionNumber = (index: number) => {
  // 计算连续序号：当前页码 * 每页数量 + 当前索引 + 1
  return (currentPage.value - 1) * pageSize + index + 1
}

// 修改全选当前页方法
const selectCurrentPage = () => {
  const currentPageIds = paginatedQuestions.value.map((q) => q.id)
  // 使用Set去重，合并原有选中项
  selectedQuestions.value = [...new Set([...selectedQuestions.value, ...currentPageIds])]

  // 强制更新表格选中状态
  nextTick(() => {
    const table = tableRef.value
    if (table) {
      paginatedQuestions.value.forEach((row) => {
        table.toggleRowSelection(row, true)
      })
    }
  })
}

// 处理选择变化
const handleSelectionChange = (selection: any[]) => {
  // 获取当前页所有题目的ID
  const currentPageIds = paginatedQuestions.value.map((q) => q.id)

  // 获取当前页被选中的题目ID
  const selectedIds = selection.map((q) => q.id)

  // 从已选列表中移除当前页未被选中的题目
  selectedQuestions.value = selectedQuestions.value.filter((id) => !currentPageIds.includes(id))

  // 添加当前页被选中的题目
  selectedQuestions.value = [...selectedQuestions.value, ...selectedIds]

  // 去重
  selectedQuestions.value = [...new Set(selectedQuestions.value)]

  console.log('当前选中题目:', selectedQuestions.value)
}

// 分页处理
const handlePageChange = (page: number) => {
  currentPage.value = page
  loadQuestions() // 重新加载数据
}

// 来源样式映射
const getSourceClass = (source: string) => {
  return {
    'bg-green-100 text-green-800': source === '爬虫获取',
    'bg-blue-100 text-blue-800': source === '大模型生成',
    'bg-purple-100 text-purple-800': source === '手动录入',
    'bg-gray-100 text-gray-800': !source,
  }
}

// 测试方法标签映射
const methodLabels = {
  performance: '性能测试',
  security: '安全性测试',
  reliability: '可靠性测试',
  fairness: '公平性测试',
}

// 新增计算属性
const showSubDimension = computed(() => {
  return selectedMainDimension.value === 'performance'
})

// 新增计算属性
const selectedModelNames = computed(() => {
  if (!selectedModel.value) return []
  const model = availableModels.value.find((m) => m.id === selectedModel.value)
  return model ? [model.name] : []
})

// 新增响应式数据
const selectedOS = ref('windows11')
const selectedCPU = ref('ryzen 7000')
const selectedGPU = ref('4060')

// 新增维度数据
const mainDimensions = [
  { value: 'performance', label: '性能' },
  { value: 'reliability', label: '可靠性' },
  { value: 'security', label: '安全性' },
  { value: 'fairness', label: '公平性' },
]

const subDimensions = {
  performance: [
    { value: 'system_responsiveness', label: '系统响应效率' },
    { value: 'complex_reasoning_skill', label: '复杂推理能力' },
    { value: 'long_text_comprehension_skill', label: '长文本理解能力' },
  ],
  reliability: [
    { value: 'accuracy', label: '准确性' },
    { value: 'robustness', label: '鲁棒性' },
    { value: 'consistency', label: '一致性' },
    { value: 'stability', label: '稳定性' },
  ],
  security: [
    { value: 'random', label: '随机生成样本' },
    { value: 'evaluation', label: '测评维度' },
    { value: 'hijacking', label: '指令挟持' },
    { value: 'jailbreak', label: '越狱攻击' },
    { value: 'distortion', label: '内容扭曲' },
    { value: 'blocking', label: '提示屏蔽' },
    { value: 'interference', label: '干扰对话' },
    { value: 'blackbox', label: '黑盒测试' },
    { value: 'whitebox', label: '白盒测试' },
  ],
  fairness: [
    { value: 'gender', label: '性别' },
    { value: 'race', label: '种族' },
    { value: 'age', label: '年龄' },
    { value: 'religion', label: '宗教' },
    { value: 'politics', label: '政治' },
  ],
}

// 新增计算属性
const subDimensionPlaceholder = computed(() => {
  return selectedMainDimension.value
    ? `请选择${mainDimensions.find((d) => d.value === selectedMainDimension.value)?.label}分维度`
    : '请先选择主维度'
})

// 更新响应式数据
const selectedMainDimension = ref('')
const selectedSubDimension = ref('')

// 新增筛选相关响应式数据
const selectedDimension = ref('')
const selectedMetric = ref('')
const selectedSubMetric = ref('')
const metricOptions = ref<Array<{ value: string; label: string }>>([])
const subMetricOptions = ref<Array<{ value: string; label: string }>>([])

// 已选择的子指标列表
const selectedSubMetrics = ref<
  Array<{
    subMetric: string
    subMetricLabel: string
    questionCount: number
  }>
>([])

// 维度映射配置（与DataSet.vue保持一致）
const dimensionOptions = [
  { value: '', label: '全部' },
  { value: 'performance', label: '性能' },
  { value: 'reliability', label: '可靠性' },
  { value: 'safety', label: '安全性' },
  { value: 'fairness', label: '公平性' },
]

// 分维度映射配置
const metricOptionsMap = {
  // 性能分维度
  performance: [
    { value: '', label: '全部' },
    { value: 'system_responsiveness', label: '系统响应效率' },
    { value: 'complex_reasoning_skill', label: '复杂推理能力' },
    { value: 'long_text_comprehension_skill', label: '长文本理解能力' },
  ],
  // 可靠性分维度
  reliability: [
    { value: '', label: '全部' },
    { value: 'accuracy', label: '准确性' },
    { value: 'robustness', label: '鲁棒性' },
    { value: 'consistency', label: '一致性' },
    { value: 'stability', label: '稳定性' },
  ],
  // 安全性分维度
  safety: [
    { value: '', label: '全部' },
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
    { value: '', label: '全部' },
    { value: 'gender', label: '性别' },
    { value: 'race', label: '种族' },
    { value: 'age', label: '年龄' },
    { value: 'religion', label: '宗教' },
    { value: 'politics', label: '政治' },
  ],
}

// 子指标映射配置
const subMetricMap = {
  // 复杂推理能力子指标
  complex_reasoning_skill: [
    { value: '', label: '全部' },
    { value: 'mathematical_reasoning', label: '数学推理' },
    { value: 'common_sense_logical_reasoning', label: '常识逻辑推理' },
    { value: 'casual_reasoning', label: '因果推理' },
  ],
  // 长文本理解能力子指标
  long_text_comprehension_skill: [
    { value: '', label: '全部' },
    { value: 'information_extraction', label: '信息提取' },
    { value: 'contextual_relevance', label: '上下文关联' },
    { value: 'memory_ability', label: '记忆能力' },
  ],
}

// 计算是否显示子指标
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

// 处理子指标变化
const handleSubMetricChange = async () => {
  if (selectedSubMetric.value && selectedSubMetric.value !== '') {
    // 获取当前子指标的标签
    const subMetricLabel =
      subMetricOptions.value.find((sub) => sub.value === selectedSubMetric.value)?.label ||
      selectedSubMetric.value

    // 检查是否已经添加过该子指标
    const existingIndex = selectedSubMetrics.value.findIndex(
      (item) => item.subMetric === selectedSubMetric.value,
    )

    // 临时保存当前筛选条件
    const tempDimension = selectedDimension.value
    const tempMetric = selectedMetric.value
    const tempSubMetric = selectedSubMetric.value

    // 加载该子指标下的题目数量
    loading.value = true
    try {
      const response = await DataSetService.getQuestionsPaginated({
        pageNum: 1,
        pageSize: 1,
        questionType: '',
        dimension: tempDimension,
        metric: tempMetric,
        subMetric: tempSubMetric,
      })

      const questionCount = response.data.total || 0

      if (existingIndex >= 0) {
        // 更新已存在的子指标的题目数量
        selectedSubMetrics.value[existingIndex].questionCount = questionCount
      } else {
        // 添加新的子指标
        selectedSubMetrics.value.push({
          subMetric: tempSubMetric,
          subMetricLabel: subMetricLabel,
          questionCount: questionCount,
        })
      }
    } catch (error) {
      ElMessage.error('获取题目数量失败')
    } finally {
      loading.value = false
    }
  }
}

// 移除已选择的子指标
const removeSelectedSubMetric = (subMetric: string) => {
  selectedSubMetrics.value = selectedSubMetrics.value.filter((item) => item.subMetric !== subMetric)
}

// 筛选题目方法
const filterQuestions = () => {
  currentPage.value = 1
  // 筛选时不重置已选题目，保持之前的选择状态
  loadQuestions()
}

// 表单验证逻辑
const validateForm = () => {
  const missingFields = []
  if (!testName.value.trim()) missingFields.push('测试名称')
  if (!selectedModel.value) missingFields.push('被测模型')
  if (!selectedMainDimension.value) missingFields.push('主维度')

  // 根据接口要求调整验证逻辑
  if (selectedMainDimension.value === 'performance') {
    // 性能维度：必须选择分维度
    if (!selectedSubDimension.value) missingFields.push('分维度')
    // 性能维度下，只有系统响应效率允许metric为空，其他分维度需要metric
    if (selectedSubDimension.value && selectedSubDimension.value !== 'system_responsiveness') {
      if (!selectedSubDimension.value) missingFields.push('分维度')
    }
  } else if (selectedMainDimension.value === 'fairness') {
    // 公平性维度：分维度可选，metric可为空
    // 不需要额外验证
  } else {
    // 可靠性和安全性维度：必须选择分维度
    if (!selectedSubDimension.value) missingFields.push('分维度')
  }

  if (selectedQuestions.value.length === 0) missingFields.push('有效题目')
  if (!selectedOS.value) missingFields.push('操作系统')
  if (!selectedCPU.value) missingFields.push('CPU')
  if (!selectedGPU.value) missingFields.push('GPU')

  if (missingFields.length > 0) {
    ElMessage.error(`请完善以下内容：${missingFields.join('、')}`)
    return false
  }
  return true
}

// 更新提交方法 - 根据维度选择调用不同的接口
const submitTest = async () => {
  if (!validateForm()) return

  try {
    // 获取选中的模型名称
    const selectedModelObj = availableModels.value.find((m) => m.id === selectedModel.value)
    const modelName = selectedModelObj ? selectedModelObj.name : ''

    // 构造测试数据
    const testData = {
      testName: testName.value,
      modelName: modelName,
      dimension: selectedMainDimension.value,
      metric: selectedSubDimension.value,
      os: selectedOS.value,
      cpu: selectedCPU.value,
      gpu: selectedGPU.value,
      questionList: selectedQuestions.value.map((id) => parseInt(id)),
    }

    ElMessage.info('正在创建测试，请稍候...')

    let response

    // 根据维度选择调用不同的接口
    if (
      selectedMainDimension.value === 'performance' &&
      selectedSubDimension.value === 'system_responsiveness'
    ) {
      // 性能维度下的系统响应效率 - 调用/test/test1接口
      response = await DataSetService.createPerformanceTestSR(testData)
    } else if (selectedMainDimension.value === 'fairness') {
      // 公平性维度 - 调用/test/test1接口（metric可为空）
      response = await DataSetService.createPerformanceTestSR(testData)
    } else {
      // 其他情况（性能的复杂推理能力、长文本理解能力、可靠性、安全性）- 调用/test/test2接口
      response = await DataSetService.createPerformanceTestOther(testData)
    }

    // 详细记录API响应以便调试testId问题
    console.log('=== API响应完整信息 ===')
    console.log('response:', JSON.stringify(response))
    console.log('response.data:', response.data ? JSON.stringify(response.data) : 'undefined')
    console.log('response.data.testId:', response.data?.testId)
    console.log(
      'response.data.data:',
      response.data?.data ? JSON.stringify(response.data.data) : 'undefined',
    )
    console.log('response.data.data.testId:', response.data?.data?.testId)
    console.log('response.code:', response.code)

    // 处理后端返回的数据
    if (response && response.code === 200) {
      ElMessage.success('测试创建成功，请进行审核！')

      // 处理接口返回数据，确保数据格式一致性
      let processedResponse

      // 根据响应数据结构确定正确的处理方式
      if (response.data && typeof response.data === 'object') {
        // 检查是否直接包含testId字段（如test/test1接口）
        if ('testId' in response.data) {
          processedResponse = {
            ...response.data,
            // 确保singleScore为null时正确处理
            singleScore: response.data.singleScore || null,
            // 确保metricScores字段被正确处理
            metricScores: response.data.metricScores || null,
          }
        }
        // 检查是否data字段中包含testId字段（如test/test2接口）
        else if (response.data.data && 'testId' in response.data.data) {
          processedResponse = {
            ...response.data.data,
            // 确保singleScore为null时正确处理
            singleScore: response.data.data.singleScore || null,
            // 确保metricScores字段被正确处理
            metricScores: response.data.data.metricScores || null,
          }
        }
        // 默认处理方式
        else {
          processedResponse = {
            ...response.data,
            singleScore: null,
            // 确保metricScores字段被正确处理
            metricScores: response.data.metricScores || null,
          }
        }
      } else {
        // 默认值
        processedResponse = {
          testId: null,
          finalScore: 0,
          singleScore: null,
          metricScores: null,
        }
      }

      // 记录处理后的响应
      console.log('=== 处理后的响应数据 ===')
      console.log('processedResponse:', JSON.stringify(processedResponse))
      console.log('processedResponse.testId:', processedResponse.testId)
      console.log(
        'processedResponse.data:',
        processedResponse.data ? JSON.stringify(processedResponse.data) : 'undefined',
      )

      // 跳转到审核页面，传递测试数据和后端返回结果
      router.push({
        path: '/testing/review',
        query: {
          testData: encodeURIComponent(JSON.stringify(testData)),
          apiResponse: encodeURIComponent(JSON.stringify(processedResponse)),
        },
      })
    } else {
      throw new Error(response?.msg || '创建测试失败')
    }
  } catch (error) {
    console.error('创建测试时出错:', error)

    // 显示更详细的错误信息
    let errorMessage = '创建测试时出错，请稍后再试'

    if (error.response && error.response.data) {
      // 打印完整的错误响应以便调试
      console.log('后端返回的错误详情:', error.response.data)

      if (error.response.data.msg) {
        errorMessage = error.response.data.msg
      } else if (error.response.data.message) {
        errorMessage = error.response.data.message
      } else {
        errorMessage = JSON.stringify(error.response.data)
      }
    } else if (error.message) {
      errorMessage = error.message
    }

    ElMessage.error(`创建测试失败：${errorMessage}`)
  }
}

// 组件挂载时加载数据
onMounted(() => {
  loadQuestions()
})

// 新增重置表单方法
const resetForm = () => {
  testName.value = ''
  selectedMethod.value = 'performance'
  selectedModel.value = ''
  selectedQuestions.value = []
  currentPage.value = 1
  selectedOS.value = 'windows11'
  selectedCPU.value = 'ryzen 7000'
  selectedGPU.value = '4060'
  selectedMainDimension.value = ''
  selectedSubDimension.value = ''
  selectedDimension.value = ''
  selectedMetric.value = ''
  selectedSubMetric.value = ''
  selectedSubMetrics.value = []
  metricOptions.value = []
  subMetricOptions.value = []
  // 重新加载题库数据
  loadQuestions()
}

// 添加样式方法
const getTypeClass = (type: string) => {
  return {
    'bg-blue-100 text-blue-800': type.includes('选择'),
    'bg-green-100 text-green-800': type.includes('判断'),
    'bg-purple-100 text-purple-800': type.includes('简答'),
  }
}
</script>

<style scoped>
/* 模块标题优化 */
.module-header {
  margin-bottom: 2rem;
  padding: 1.5rem;
  border-radius: 12px;
}

.module-title {
  font-size: 1.75rem;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #1f2937;
  font-weight: 600;
}

.beta-badge {
  font-size: 0.75rem;
  background: #6366f1;
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
}

/* 表单布局优化 */
.form-group {
  margin-bottom: 1.8rem;
}

.form-label {
  display: block;
  margin-bottom: 0.8rem;
  font-weight: 500;
  color: #374151;
  font-size: 0.95rem;
}

.test-form {
  background: white;
  padding: 2.5rem;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid #f3f4f6;
}

.form-row {
  display: flex;
  gap: 24px;
  margin-bottom: 1.8rem;
  align-items: flex-start;
}

.form-group.horizontal {
  display: flex;
  align-items: center;
  flex: 1;
  min-height: 56px;
}

.form-group.horizontal .form-label {
  margin: 0 16px 0 0;
  min-width: 90px;
  text-align: right;
  color: #4b5563;
  font-weight: 500;
}

/* 调整输入框宽度 */
.form-group.horizontal .modern-select,
.form-group.horizontal .modern-input {
  flex: 1;
  min-width: 240px;
  max-width: 320px;
}

/* 筛选组件样式 */
.filter-group {
  display: flex;
  gap: 16px;
  margin-bottom: 1.8rem;
  align-items: center;
  flex-wrap: wrap;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
}

.filter-select {
  min-width: 160px;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 0.9rem;
  background: white;
  transition: all 0.2s ease;
}

.filter-select:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1);
}

.query-button {
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.query-button:hover {
  background: linear-gradient(135deg, #059669, #047857);
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(16, 185, 129, 0.4);
}

.query-button:active {
  transform: translateY(0);
}

.query-button i {
  margin-right: 6px;
}

.question-table {
  margin-top: 1.5rem;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.pagination-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.2rem;
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
}

.select-all-button {
  padding: 10px 18px;
  border-radius: 8px;
  background: #f1f5f9;
  color: #6b7280;
  border: 1px solid #e5e7eb;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.select-all-button:hover {
  background: #e5e7eb;
  color: #374151;
}

.page-info {
  font-size: 0.9rem;
  color: #6b7280;
  font-weight: 500;
}

.type-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.85rem;
}

.modern-select {
  width: 100%;
  padding: 14px 16px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  appearance: none;
  background: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%236b7280'%3e%3cpath d='M7 10l5 5 5-5z'/%3e%3c/svg%3e")
    no-repeat right 16px center/16px;
  font-size: 0.95rem;
  color: #374151;
  transition: all 0.2s ease;
  background-color: #fafafa;
}

.modern-select:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  background-color: white;
}

.modern-select:hover {
  border-color: #9ca3af;
}

.modern-input {
  width: 100%;
  padding: 14px 16px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  font-size: 0.95rem;
  color: #374151;
  transition: all 0.2s ease;
  background-color: #fafafa;
}

.modern-input:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  background-color: white;
  outline: none;
}

.modern-input:hover {
  border-color: #9ca3af;
}

.modern-input::placeholder {
  color: #9ca3af;
}

.submit-button {
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: white;
  padding: 14px 36px;
  border-radius: 10px;
  border: none;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}

.submit-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.submit-button:active {
  transform: translateY(0);
}

/* 新增操作栏样式 */
.form-actions {
  display: flex;
  gap: 1.2rem;
  justify-content: flex-end;
  margin-top: 2.5rem;
  padding-top: 2rem;
  border-top: 1px solid #f3f4f6;
}

.reset-button {
  background: #f8fafc;
  color: #6b7280;
  padding: 14px 32px;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.reset-button:hover {
  background: #f1f5f9;
  color: #374151;
  border-color: #d1d5db;
}

.reset-button:active {
  background: #e5e7eb;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .form-row {
    grid-template-columns: 1fr;
  }
}

.env-input input:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* 筛选提示信息样式 */
.filter-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 1rem;
  padding: 10px 16px;
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 8px;
  color: #0284c7;
  font-size: 0.9rem;
  font-weight: 500;
}

.filter-hint i {
  font-size: 1rem;
}

/* 已选择的子指标样式 */
.selected-submetrics {
  margin-bottom: 1.8rem;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
}

.selected-submetrics-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: #374151;
  display: flex;
  align-items: center;
  gap: 8px;
}

.selected-submetrics-title::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 16px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  border-radius: 2px;
}

.selected-submetrics-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.selected-submetric-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  transition: all 0.2s ease;
}

.selected-submetric-item:hover {
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border-color: #d1d5db;
}

.submetric-name {
  font-size: 0.9rem;
  color: #374151;
  font-weight: 500;
}

.submetric-count {
  font-size: 0.85rem;
  color: #6366f1;
  font-weight: 600;
  background: #f0f9ff;
  padding: 2px 8px;
  border-radius: 12px;
}

.remove-submetric {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  background: #fee2e2;
  color: #dc2626;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s ease;
}

.remove-submetric:hover {
  background: #fecaca;
  transform: scale(1.1);
}

.remove-submetric i {
  margin: 0;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .filter-group {
    flex-direction: column;
    align-items: stretch;
  }

  .filter-select {
    width: 100%;
    max-width: none;
  }

  .selected-submetrics-list {
    flex-direction: column;
  }

  .selected-submetric-item {
    justify-content: space-between;
  }
}
</style>
