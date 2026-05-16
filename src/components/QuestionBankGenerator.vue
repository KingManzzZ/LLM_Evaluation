<template>
  <div class="question-bank-generator">
    <h2 class="module-title">题库工厂</h2>
    <!-- 卡片式选择界面 -->
    <div v-if="!isGenerating" class="selection-card">
      <div class="method-grid">
        <div
          class="method-card"
          v-for="method in generationMethods"
          :key="method.key"
          @click="selectGenerationMethod(method.key)"
        >
          <div class="card-content">
            <div class="icon-wrapper">
              <div class="method-icon" :class="method.icon"></div>
            </div>
            <h3 class="method-title">{{ method.title }}</h3>
            <p class="method-desc">{{ method.description }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 输入关键词生成部分 -->
    <div v-if="isGenerating && generationMethod === 'keyword'" class="generation-panel">
      <div class="panel-header">
        <button @click="goBackToSelection" class="back-button">
          <i class="fas fa-arrow-left"></i>
        </button>
        <h3>智能生成模式</h3>
      </div>
      <div class="form-container">
        <!-- 筛选组布局参考DataSet.vue -->
        <div class="filter-group">
          <!-- 维度选择 -->
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

          <!-- 指标选择 -->
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

          <!-- 子指标选择 -->
          <el-select
            v-model="selectedSubMetric"
            placeholder="选择小指标"
            class="filter-select"
            :disabled="!selectedMetric"
          >
            <el-option
              v-for="sub in subMetricOptions"
              :key="sub.value"
              :label="sub.label"
              :value="sub.value"
            />
          </el-select>
          <el-select
            v-model="selectedQuestionTypes"
            multiple
            placeholder="选择题型"
            class="filter-select"
            style="width: 240px"
          >
            <el-option
              v-for="type in questionTypes"
              :key="type.value"
              :label="type.label"
              :value="type.value"
            />
          </el-select>
          <el-select
            v-model="selectedModel"
            placeholder="选择模型"
            class="filter-select"
            style="width: 240px"
          >
            <el-option
              v-for="model in modelOptions"
              :key="model.value"
              :label="model.label"
              :value="model.value"
            />
          </el-select>
        </div>

        <!-- 新增输入组 -->
        <div class="input-group">
          <el-input
            v-model="exampleText"
            placeholder="输入样例文本（可选）"
            type="textarea"
            :rows="3"
            class="modern-input"
          />

          <el-input-number
            v-model="generateCount"
            :min="1"
            :max="50"
            label="生成数量"
            class="count-input"
          />
        </div>

        <button @click="generateQuestions" class="submit-button">
          <i class="fas fa-magic"></i> 开始生成
        </button>
      </div>
    </div>

    <!-- 爬虫生成部分 -->
    <div v-if="isGenerating && generationMethod === 'crawler'" class="generation-panel">
      <div class="panel-header">
        <button @click="goBackToSelection" class="back-button">
          <i class="fas fa-arrow-left"></i>
        </button>
        <h3>网页抓取模式</h3>
      </div>
      <div class="form-container">
        <div class="form-group">
          <label class="form-label"> <i class="fas fa-link"></i> 目标网址 </label>
          <input
            v-model="crawlerUrl"
            type="url"
            placeholder="请输入题库采集地址（示例：https://example.com/exam）"
            class="modern-input"
          />
        </div>
        <button @click="generateCrawlerQuestions" class="submit-button">
          <i class="fas fa-spider"></i> 开始抓取
        </button>
        <!-- 生成结果展示区 -->
      </div>
    </div>

    <!-- 手动输入部分 -->
    <div v-if="isGenerating && generationMethod === 'manual'" class="generation-panel">
      <div class="panel-header">
        <button @click="goBackToSelection" class="back-button">
          <i class="fas fa-arrow-left"></i>
        </button>
        <h3>人工生成模式</h3>
      </div>
      <div class="form-container">
        <!-- 筛选组布局参考智能生成模式 -->
        <div class="filter-group">
          <!-- 维度选择 -->
          <div class="form-row">
            <label class="form-label required">维度</label>
            <el-select
              v-model="manualDimension"
              placeholder="请选择维度"
              class="filter-select"
              @change="handleManualDimensionChange"
              :clearable="true"
            >
              <el-option
                v-for="dim in dimensionOptions"
                :key="dim.value"
                :label="dim.label"
                :value="dim.value"
              />
            </el-select>
          </div>

          <!-- 指标选择 -->
          <div class="form-row">
            <label class="form-label required">分维度</label>
            <el-select
              v-model="manualMetric"
              placeholder="请选择分维度"
              class="filter-select"
              :disabled="!manualDimension"
              @change="handleManualMetricChange"
              :clearable="true"
            >
              <el-option
                v-for="metric in manualMetricOptions"
                :key="metric.value"
                :label="metric.label"
                :value="metric.value"
              />
            </el-select>
          </div>

          <!-- 子指标选择 -->
          <div class="form-row">
            <label class="form-label">小指标</label>
            <el-select
              v-model="manualSubMetric"
              placeholder="请选择小指标（可选）"
              class="filter-select"
              :disabled="!manualMetric"
              :clearable="true"
            >
              <el-option
                v-for="sub in manualSubMetricOptions"
                :key="sub.value"
                :label="sub.label"
                :value="sub.value"
              />
            </el-select>
          </div>

          <!-- 题型选择 -->
          <div class="form-row">
            <label class="form-label required">题型</label>
            <el-select
              v-model="manualQuestionType"
              placeholder="请选择题型"
              class="filter-select"
              @change="handleManualQuestionTypeChange"
              :clearable="true"
            >
              <el-option
                v-for="type in manualQuestionTypes"
                :key="type.value"
                :label="type.label"
                :value="type.value"
              />
            </el-select>
          </div>
        </div>

        <div class="form-group">
          <div class="form-row">
            <label for="manual-question" class="form-label required">
              <i class="fas fa-question-circle"></i> 题干
            </label>
            <el-input
              id="manual-question"
              v-model="manualQuestion"
              type="textarea"
              :rows="3"
              placeholder="请输入题目内容（示例：机器学习中的过拟合是指？）"
              class="modern-textarea"
              :maxlength="500"
              show-word-limit
            />
          </div>
        </div>

        <!-- 选项输入框（仅选择题显示） -->
        <div v-if="manualQuestionType === 'choice'" class="form-group">
          <div class="form-row">
            <label for="manual-options" class="form-label required">
              <i class="fas fa-list-ul"></i> 选项
            </label>
            <el-input
              id="manual-options"
              v-model="manualOptions"
              type="textarea"
              :rows="3"
              placeholder="请输入选项内容，多个选项用逗号分隔（示例：A.选项1,B.选项2,C.选项3）"
              class="modern-textarea"
              :maxlength="300"
              show-word-limit
            />
            <div class="input-tip">
              <i class="fas fa-info-circle"></i> 多个选项请用逗号分隔，如：A.选项1,B.选项2,C.选项3
            </div>
          </div>
        </div>

        <div class="form-group">
          <div class="form-row">
            <label for="manual-answer" class="form-label required">
              <i class="fas fa-check-circle"></i> 答案
            </label>
            <el-input
              id="manual-answer"
              v-model="manualAnswer"
              type="textarea"
              :rows="2"
              placeholder="请输入题目的答案"
              class="modern-textarea"
              :maxlength="200"
              show-word-limit
            />
          </div>
        </div>

        <div class="form-actions">
          <el-button
            @click="addManualQuestion"
            type="primary"
            :loading="isSubmitting"
            :disabled="!isFormValid"
            class="submit-button"
          >
            <i class="fas fa-plus-circle"></i> 添加题目
          </el-button>

          <el-button @click="resetForm" type="default" class="reset-button">
            <i class="fas fa-redo"></i> 重置表单
          </el-button>
        </div>

        <div v-if="hasGeneratedQuestions" class="success-message">
          <i class="fas fa-check-circle"></i> 题目已成功添加到列表
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import testData from '../../题库数据/test.json'
import hellaswagData from '../../题库数据/hellaswag.json'
import api from '@/api/index'

const router = useRouter()

const selectedDimension = ref('')
const selectedMetric = ref('')
const selectedSubMetric = ref('')
const selectedModel = ref('')
const exampleText = ref('')
const generateCount = ref(5)
const keyword = ref('')
const questions = ref<any[]>([])
const editIndex = ref<number | null>(null)
const manualQuestion = ref('')
const manualType = ref('选择题')
const manualScenario = ref('')
const manualAnswer = ref('') // 新增答案字段

const isGenerating = ref(false)
const generationMethod = ref('')
const hasGeneratedQuestions = ref(false) // 在setup部分新增响应式状态
const dimensionMap = {
  performance: '性能',
  reliability: '可靠性',
  safety: '安全性',
  fairness: '公平性',
}
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

const activeIndex = ref<number | null>(null)

const toggleActions = (index: number) => {
  activeIndex.value = activeIndex.value === index ? null : index
}

const selectGenerationMethod = (method: string) => {
  isGenerating.value = true
  generationMethod.value = method
}
const dimensionOptions = [
  { value: '', label: '全部' },
  ...Object.entries(dimensionMap).map(([value, label]) => ({ value, label })),
]

const metricOptions = ref<Array<{ value: string; label: string }>>([])
const subMetricOptions = ref<Array<{ value: string; label: string }>>([])

// 维度变化处理
const handleDimensionChange = () => {
  selectedMetric.value = ''
  metricOptions.value = selectedDimension.value
    ? metricOptionsMap[selectedDimension.value] || []
    : []
}

// 指标变化处理
const handleMetricChange = () => {
  selectedSubMetric.value = ''
  subMetricOptions.value = subMetricMap[selectedMetric.value] || []
}
// 修改返回方法
const goBackToSelection = () => {
  isGenerating.value = false
  generationMethod.value = ''
  // 重置所有生成相关状态
  questions.value = []
  hasGeneratedQuestions.value = false

  // 重置人工生成相关状态
  manualQuestion.value = ''
  manualOptions.value = ''
  manualAnswer.value = ''
  manualDimension.value = ''
  manualMetric.value = ''
  manualSubMetric.value = ''
  manualQuestionType.value = 'choice'
  manualMetricOptions.value = []
  manualSubMetricOptions.value = []
}
const selectedTypes = ref(['选择题', '判断题', '简答题', '仅问题', '问题对比组'])
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  only_question: '仅问题',
  compare_question: '问题对比组',
}
const confirmQuestions = () => {
  hasGeneratedQuestions.value = true
}
// 修改原有questionTypes定义
const questionTypes = ref(
  Object.entries(questionTypeMap).map(([value, label]) => ({ value, label })),
)
const selectedQuestionTypes = ref<string[]>([])

// 定义模型选项
const modelOptions = ref([
  { value: 'ERNIE-4.0-8K', label: 'ERNIE-4.0-8K' },
  { value: 'Qwen-max', label: 'Qwen-max' },
  { value: 'DeepSeek-V3', label: 'DeepSeek-V3' },
  { value: 'yi-lightning', label: 'yi-lightning' },
  { value: 'gpt-4o-mini', label: 'gpt-4o-mini' },
])

// 网络状态检测函数
const checkNetworkStatus = () => {
  return navigator.onLine
}

// 修改generateQuestions方法
const generateQuestions = async () => {
  try {
    // 网络状态检测
    if (!checkNetworkStatus()) {
      ElMessage.error('网络连接已断开，请检查网络设置')
      return
    }

    // 添加参数校验
    if (!selectedDimension.value || !selectedMetric.value) {
      ElMessage.warning('请选择主维度和分维度')
      return
    }

    const params = {
      modelName: selectedModel.value || undefined, // 新增modelName字段
      dimension: selectedDimension.value,
      metric: selectedMetric.value,
      subMetric: selectedSubMetric.value || undefined, // 允许空值
      questionType: selectedQuestionTypes.value,
      example: exampleText.value || undefined, // 允许空值
      count: Number(generateCount.value), // 确保数字类型
    }

    ElMessage.info('正在生成题目，请稍候...')

    // 使用统一API配置发送请求
    console.log('发送API请求，参数:', params)
    const response = await api.post('/dataInfo/generate/model', params, {
      timeout: 120000, // 保持超时时间为20秒
    })

    console.log('API Response:', response) // 调试日志
    console.log('API Response data:', response?.data) // 详细数据日志
    console.log('API Response code:', response?.code) // 状态码日志
    console.log('API Response 类型:', typeof response)
    console.log('API Response 完整结构:', JSON.stringify(response, null, 2))

    if (
      (response?.code === 0 || response?.code === 200) &&
      response?.data &&
      Array.isArray(response.data)
    ) {
      const processedQuestions = processGeneratedData(response.data)
      ElMessage.success(`成功生成${response.data.length}道题目`)

      console.log('准备跳转到生成题目页面')
      console.log('处理后的题目数据:', processedQuestions)

      // 保存数据到sessionStorage，避免URL长度限制
      sessionStorage.setItem('generatedQuestions', JSON.stringify(processedQuestions))
      console.log('数据已保存到sessionStorage')

      // 跳转到新页面展示生成的题目
      setTimeout(() => {
        console.log('执行路由跳转，目标路由:', 'GeneratedQuestions')
        console.log('当前路由:', router.currentRoute.value)

        try {
          router
            .push({
              name: 'GeneratedQuestions',
            })
            .then(() => {
              console.log('路由跳转成功')
            })
            .catch((error) => {
              console.error('路由跳转失败:', error)
              ElMessage.error('页面跳转失败，请检查路由配置')
            })
        } catch (error) {
          console.error('路由跳转异常:', error)
          ElMessage.error('页面跳转异常，请刷新页面重试')
        }
      }, 500)
    } else {
      const errorMsg = response?.msg || '未知错误'
      ElMessage.error(`服务端错误：${errorMsg}`)
    }
  } catch (error: any) {
    // 简化错误处理
    let errorMessage = '请求失败'
    if (error.response) {
      errorMessage = `服务端异常：${error.response.status || '未知状态码'}`
    } else if (error.request) {
      errorMessage = '请求未收到响应'
    } else {
      errorMessage = `请求错误：${error.message || '未知错误'}`
    }
    ElMessage.error(errorMessage)
    console.error('完整错误信息：', error)
  }
}

// 添加缺失的题型转换函数
const convertQuestionType = (type: string) => {
  // 基于之前定义的questionTypeMap对象，将英文题型转换为中文名称
  const questionTypeMap = {
    choice: '选择题',
    judgment: '判断题',
    short_answer: '简答题',
    only_question: '仅问题',
    compare_question: '问题对比组',
  }

  return questionTypeMap[type] || '未知题型'
}

// 增强数据处理容错性
const processGeneratedData = (responseData: any) => {
  // 修复：直接使用responseData，因为传入的已经是response.data
  if (!responseData || !Array.isArray(responseData)) return []

  return responseData.map((item: any) => ({
    dataId: item.dataId || null,
    modelId: item.modelId || null,
    metricId: item.metricId || null,
    question: item.question || '题目内容缺失',
    options: item.options || '',
    answer: item.answer || '参考答案未提供',
    dimension: item.dimension || 'performance',
    subMetricId: item.subMetricId || null,
    questionType: item.questionType || 'choice',
    dataSource: item.dataSource || 'generated',
    transformationType: item.transformationType || '',
    transformationDescription: item.transformationDescription || '',
    createTime: item.createTime || new Date().toISOString(),
    updateTime: item.updateTime || new Date().toISOString(),
    isDeleted: item.isDeleted || 0,
    isTransformed: item.isTransformed || 0,
    // 兼容原有组件使用的字段
    type: convertQuestionType(item.questionType),
    scenario: item.dimension ? [item.dimension] : ['通用场景'],
  }))
}

const showEditOptions = (index: number) => {
  editIndex.value = editIndex.value === index ? null : index
}

const modifyQuestion = (index: number) => {
  console.log('修改题目:', index)
  editIndex.value = null
}

const deleteQuestion = (index: number) => {
  questions.value.splice(index, 1)
  editIndex.value = null
}

// 修改手动添加方法，对接后端接口
const addManualQuestion = async () => {
  if (!isFormValid.value) {
    ElMessage.warning('请填写完整的表单信息')
    return
  }

  try {
    isSubmitting.value = true

    // 准备发送给后端的数据
    const requestData = {
      dimension: manualDimension.value,
      metric: manualMetric.value,
      subMetric: manualSubMetric.value || '',
      questionType: manualQuestionType.value,
      question: manualQuestion.value.trim(),
      options: manualQuestionType.value === 'choice' ? manualOptions.value.trim() : '',
      answer: manualAnswer.value.trim(),
    }

    ElMessage.info('正在提交题目到后端...')

    // 调用后端接口
    const response = await api.post('/dataInfo/generate/hand', requestData, {
      timeout: 15000, // 增加超时时间
    })

    console.log('后端接口响应:', response)

    if (response?.code === 200 || response?.code === 0) {
      ElMessage.success('题目添加成功！')

      // 将添加的题目数据保存到sessionStorage，以便跳转后展示
      const newQuestion = {
        dataId: response.data?.dataId || Date.now(),
        modelId: null,
        metricId: null,
        question: manualQuestion.value.trim(),
        options: manualQuestionType.value === 'choice' ? manualOptions.value.trim() : '',
        answer: manualAnswer.value.trim(),
        dimension: manualDimension.value,
        subMetricId: null,
        questionType: manualQuestionType.value,
        dataSource: 'manual',
        transformationType: '',
        transformationDescription: '',
        createTime: new Date().toISOString(),
        updateTime: new Date().toISOString(),
        isDeleted: 0,
        isTransformed: 0,
        type: convertQuestionType(manualQuestionType.value),
        scenario: [manualDimension.value],
      }

      // 保存到sessionStorage
      sessionStorage.setItem('generatedQuestions', JSON.stringify([newQuestion]))
      console.log('人工添加的题目数据已保存到sessionStorage:', newQuestion)

      // 清空表单
      resetForm()

      // 跳转到新页面展示生成的题目
      setTimeout(() => {
        try {
          router
            .push({
              name: 'GeneratedQuestions',
            })
            .then(() => {
              console.log('路由跳转成功')
            })
            .catch((error) => {
              console.error('路由跳转失败:', error)
              ElMessage.error('页面跳转失败，请检查路由配置')
            })
        } catch (error) {
          console.error('路由跳转异常:', error)
          ElMessage.error('页面跳转异常，请刷新页面重试')
        }
      }, 500)
    } else {
      const errorMsg = response?.msg || '未知错误'
      ElMessage.error(`添加失败：${errorMsg}`)
    }
  } catch (error: any) {
    console.error('添加题目失败:', error)
    let errorMessage = '网络请求失败'
    if (error.response) {
      errorMessage = `服务端异常：${error.response.status || '未知状态码'}`
    } else if (error.request) {
      errorMessage = '请求未收到响应，请检查网络连接'
    } else {
      errorMessage = `请求错误：${error.message || '未知错误'}`
    }
    ElMessage.error(errorMessage)
  } finally {
    isSubmitting.value = false
  }
}

const generationMethods = ref([
  {
    key: 'keyword',
    title: '智能生成',
    description: '通过关键词自动生成高质量题库',
    icon: 'fas fa-wand-magic-sparkles',
  },
  {
    key: 'crawler',
    title: '爬虫获取',
    description: '从指定网站抓取题目数据',
    icon: 'fas fa-spider',
  },
  {
    key: 'manual',
    title: '人工生成',
    description: '自定义添加题目到题库',
    icon: 'fas fa-hand-pointer',
  },
])

// 数据源映射
const dataSourceMap = {
  input: '输入',
  generated: '智能生成',
  crawler: '爬虫获取',
  manual: '人工添加',
}

// 在setup部分添加响应式数据
const crawlerUrl = ref('')

// 人工生成模块相关响应式数据
const manualDimension = ref('')
const manualMetric = ref('')
const manualSubMetric = ref('')
const manualQuestionType = ref('choice')
const manualOptions = ref('')
const manualMetricOptions = ref<Array<{ value: string; label: string }>>([])
const manualSubMetricOptions = ref<Array<{ value: string; label: string }>>([])

// 人工生成题型选项
const manualQuestionTypes = ref([
  { value: 'choice', label: '选择题' },
  { value: 'judgment', label: '判断题' },
  { value: 'short_answer', label: '简答题' },
  { value: 'only_question', label: '仅问题' },
  { value: 'compare_question', label: '问题对比组' },
])

// 人工生成维度变化处理
const handleManualDimensionChange = () => {
  manualMetric.value = ''
  manualMetricOptions.value = manualDimension.value
    ? metricOptionsMap[manualDimension.value] || []
    : []
}

// 人工生成指标变化处理
const handleManualMetricChange = () => {
  manualSubMetric.value = ''
  manualSubMetricOptions.value = subMetricMap[manualMetric.value] || []
}

// 人工生成题型变化处理
const handleManualQuestionTypeChange = () => {
  // 当题型不是选择题时，清空选项内容
  if (manualQuestionType.value !== 'choice') {
    manualOptions.value = ''
  }
}

// 新增响应式变量
const isSubmitting = ref(false)

// 表单验证计算属性
const isFormValid = computed(() => {
  return (
    manualDimension.value &&
    manualMetric.value &&
    manualQuestionType.value &&
    manualQuestion.value.trim() &&
    manualAnswer.value.trim() &&
    (manualQuestionType.value !== 'choice' || manualOptions.value.trim())
  )
})

// 重置表单函数
const resetForm = () => {
  manualQuestion.value = ''
  manualOptions.value = ''
  manualAnswer.value = ''
  manualDimension.value = ''
  manualMetric.value = ''
  manualSubMetric.value = ''
  manualQuestionType.value = 'choice'
  manualMetricOptions.value = []
  manualSubMetricOptions.value = []
  isSubmitting.value = false

  ElMessage.success('表单已重置')
}

// 修改爬虫生成方法
const generateCrawlerQuestions = () => {
  if (
    crawlerUrl.value === 'https://hf-mirror.com/datasets/Rowan/hellaswag/viewer/default/validation'
  ) {
    // 使用Map去重
    const questionMap = new Map()
    hellaswagData.forEach((item, index) => {
      const formattedQuestion = {
        question: `${item.question}\n选项：${item.options.join('、')}`,
        type: '选择题',
        scenario: ['可靠性'],
        answer: `正确答案：${item.answer}`,
      }
      questionMap.set(item.question, formattedQuestion)
    })

    // 转换为数组并排序
    const uniqueQuestions = Array.from(questionMap.values())
      .sort((a, b) => a.question.localeCompare(b.question))
      .slice(0, 20) // 取前20题

    // 保存数据到sessionStorage，避免URL长度限制
    sessionStorage.setItem('generatedQuestions', JSON.stringify(uniqueQuestions))

    // 跳转到新页面展示生成的题目
    setTimeout(() => {
      try {
        router
          .push({
            name: 'GeneratedQuestions',
          })
          .then(() => {
            console.log('路由跳转成功')
          })
          .catch((error) => {
            console.error('路由跳转失败:', error)
            ElMessage.error('页面跳转失败，请检查路由配置')
          })
      } catch (error) {
        console.error('路由跳转异常:', error)
        ElMessage.error('页面跳转异常，请刷新页面重试')
      }
    }, 500)
    return
  }
  // 示例逻辑，实际需要对接API
  console.log('开始抓取:', crawlerUrl.value)
  ElMessage.success('抓取请求已发送')
}
</script>

<style scoped>
/* 原有样式 */
.module-title {
  text-align: center;
  font-size: 2rem;
  color: #2c3e50;
  margin-bottom: 2rem;
}

.method-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.method-card {
  background: linear-gradient(145deg, #ffffff, #f8f9fa);
  border-radius: 15px;
  padding: 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.method-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
}

.icon-wrapper {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1.5rem;
}

.method-icon {
  font-size: 2rem;
  color: white;
}

.method-title {
  text-align: center;
  font-size: 1.25rem;
  color: #1f2937;
  margin-bottom: 0.5rem;
}

.method-desc {
  text-align: center;
  color: #6b7280;
  font-size: 0.9rem;
  line-height: 1.4;
}

.modern-input {
  flex: 1;
  padding: 12px 20px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.modern-input:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.generate-button {
  height: 46px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  padding: 12px 25px;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s ease;
  white-space: nowrap;
  min-width: 120px;
}

.generate-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
}

/* 修改原有样式保持统一 */
.button-container {
  gap: 2rem;
}

.button-container button {
  padding: 1.2rem 2rem;
  border-radius: 12px;
  font-size: 1.1rem;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
}

/* 新增手动录入样式 */
.generation-panel {
  max-width: 800px;
  margin: 20px auto;
  padding: 25px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}
.type-selector {
  width: 240px;
  :deep(.el-select__tags) {
    max-width: 180px;
  }
  :deep(.el-tag) {
    background-color: #eef2ff;
    color: #6366f1;
  }
}
.panel-header {
  display: flex;
  align-items: center;
  margin-bottom: 2rem;
}

.panel-header h3 {
  font-size: 1.5rem;
  color: #2c3e50;
  margin-left: 15px;
}

.form-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.filter-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.form-row {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label.required::after {
  content: '*';
  color: #f56565;
  margin-left: 4px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-weight: 500;
  color: #4b5563;
  display: flex;
  align-items: center;
  gap: 8px;
}

.modern-select {
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  transition: all 0.3s ease;
}

.modern-select:focus {
  border-color: #6366f1;
  outline: none;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.submit-button {
  align-self: flex-end;
  background: linear-gradient(135deg, #10b981, #34d399);
  padding: 12px 30px;
  border-radius: 8px;
  color: white;
  font-weight: 500;
  transition: all 0.3s ease;
}

.submit-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
}

/* 新增样式 */
.modern-textarea {
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.modern-textarea:hover {
  border-color: #d1d5db;
}

.modern-textarea:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.input-tip {
  font-size: 0.875rem;
  color: #6b7280;
  margin-top: 0.25rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.input-tip i {
  color: #6366f1;
}

.form-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  margin-top: 1.5rem;
}

.reset-button {
  background: #6b7280;
  border-color: #6b7280;
}

.reset-button:hover {
  background: #4b5563;
  border-color: #4b5563;
}

.success-message {
  background: #d1fae5;
  border: 1px solid #a7f3d0;
  color: #065f46;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.success-message i {
  color: #10b981;
}

.filter-select {
  width: 100%;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .filter-group {
    grid-template-columns: 1fr;
  }

  .form-actions {
    flex-direction: column;
  }

  .form-actions .el-button {
    width: 100%;
  }
}

/* 保留原有其他样式 */
.question-bank-generator {
  margin: 20px;
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.selection {
  margin-bottom: 20px;
  text-align: center;
}

.selection h3 {
  font-size: 20px;
  margin-bottom: 15px;
  color: #333;
}

.button-container {
  display: flex;
  justify-content: center;
  gap: 20px;
}

.button-container button {
  padding: 15px 30px;
  border: none;
  background-color: #007bff;
  color: white;
  cursor: pointer;
  border-radius: 8px;
  font-size: 16px;
  transition: background-color 0.3s ease;
}

.button-container button:hover {
  background-color: #0056b3;
}

.generation-method {
  margin-top: 20px;
}

.back-button {
  margin-bottom: 10px;
  padding: 8px 16px;
  border: none;
  background-color: #6c757d;
  color: white;
  cursor: pointer;
  border-radius: 4px;
  font-size: 14px;
}

.back-button:hover {
  background-color: #5a6268;
}

input {
  padding: 10px;
  margin-right: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

button {
  padding: 10px 20px;
  border: none;
  color: white;
  cursor: pointer;
}

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
  width: 100px;
  text-align: center;
  vertical-align: middle;
}

.modern-table th.date,
.modern-table td.date {
  width: 120px;
}

.source-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  border-radius: 16px;
  font-size: 0.85rem;
  background-color: #e0f2fe;
  color: #0369a1;
  height: 28px;
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

.action-container {
  display: flex;
  gap: 8px;
  align-items: center;
}
.question-text {
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
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
.icon-button {
  width: 32px;
  height: 32px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
}
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
/* 原代码中此处部分样式选择器有误，已修正 */
/* 以下继续保留原文件中未被删除的样式 */
.view-button {
  color: #64748b;
  margin-right: 8px;
  transition: color 0.2s;
}

.view-button:hover {
  color: #6366f1;
}

/* 同步数据集模块的模态框样式 */
.modal-mask {
  position: fixed;
  z-index: 999;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal-container {
  background: white;
  border-radius: 8px;
  width: 600px;
  padding: 20px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.2rem;
  color: #64748b;
}

.detail-item {
  margin: 1rem 0;
  padding: 0.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.question-text {
  color: #475569;
  line-height: 1.6;
}

.answer-content {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 6px;
  white-space: pre-wrap;
}

/* 在样式部分新增 */
.import-button {
  background: linear-gradient(135deg, #10b981, #3b82f6);
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
  margin-top: 1.5rem;
}

.import-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
}

.action-footer {
  display: flex;
  justify-content: flex-end;
  border-top: 1px solid #e2e8f0;
  padding-top: 1.5rem;
  margin-top: 2rem;
}

/* 新增爬虫生成模块样式 */
.result-container {
  margin-top: 1.5rem;
  padding: 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
}

.result-title {
  font-size: 1.2rem;
  color: #475569;
  margin-bottom: 0.5rem;
}

.result-stats {
  display: flex;
  gap: 1rem;
}

.stat-item {
  color: #64748b;
}
.keyword-input {
  width: 600px;
  max-width: 80%;
  min-width: 400px;
}

.input-group {
  display: flex;
  gap: 1rem;
  align-items: center;
  justify-content: center;
}

.type-badge {
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.85rem;
}

.type-badge-选择题 {
  background: #e3f2fd;
  color: #1976d2;
}

.type-badge-判断题 {
  background: #f3e5f5;
  color: #9c27b0;
}

.type-badge-简答题 {
  background: #f0f4c3;
  color: #827717;
}

.type-badge-仅问题 {
  background: #ffe0b2;
  color: #e65100;
}

.type-badge-问题对比组 {
  background: #c5cae9;
  color: #283593;
}

/* 新增删除按钮样式 */
.delete-button {
  color: #ef4444;
  margin-right: 8px;
  transition: color 0.2s;
}

.delete-button:hover {
  color: #b91c1c;
}
.filter-group {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.filter-select {
  flex: 1;
  min-width: 200px;
}

/* 调整多选标签间距 */
.filter-select.el-select--multiple {
  :deep(.el-select__tags) {
    gap: 4px;
    .el-tag {
      margin: 2px;
    }
  }
}
</style>
