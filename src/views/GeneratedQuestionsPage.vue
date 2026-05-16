<template>
  <div class="generated-questions-page">
    <div class="page-header">
      <h1>生成题目列表</h1>
      <button class="back-button" @click="goBack">
        <i class="fas fa-arrow-left"></i> 返回
      </button>
    </div>

    <div class="table-container">
      <table class="modern-table">
        <thead>
          <tr>
            <th class="index">序号</th>
            <th class="question">题干</th>
            <th class="type">类型</th>
            <th class="source">来源</th>
            <th class="date">创建日期</th>
            <th class="actions">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(question, index) in questions" :key="question.dataId || index">
            <td class="index">{{ index + 1 }}</td>
            <td class="question-content">{{ question.question || '题目内容缺失' }}</td>
            <td class="type">
              <span class="type-badge" :class="getTypeClass(question.questionType)">
                {{ getQuestionTypeName(question.questionType) }}
              </span>
            </td>
            <td class="source">
              <span class="source-badge" :class="getSourceClass(question.dataSource)">
                {{ getDataSourceName(question.dataSource) }}
              </span>
            </td>
            <td class="date">{{ formatDate(question.createTime) }}</td>
            <td class="actions">
              <div class="action-container">
                <button class="icon-button view-button" @click="showQuestionDetail(question)">
                  <i class="fas fa-eye"></i>
                </button>
                <div class="dropdown-container">
                  <button class="icon-button dropdown-button" @click="toggleDropdown(question.dataId || index)">
                    <i class="fas fa-ellipsis-v"></i>
                  </button>
                  <div v-if="activeDropdown === (question.dataId || index)" class="dropdown-menu">
                    <button class="dropdown-item" @click="editQuestion(question)">
                      <i class="fas fa-edit"></i> 编辑
                    </button>
                    <button class="dropdown-item delete-item" @click="deleteQuestion(question)">
                      <i class="fas fa-trash"></i> 删除
                    </button>
                  </div>
                </div>
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="questions.length === 0" class="empty-state">
        <p>暂无生成的题目</p>
      </div>
    </div>

    <!-- 导入题库按钮 -->
    <div class="import-section">
      <button class="import-button" @click="importToQuestionBank">
        <i class="fas fa-file-import"></i> 导入题库
      </button>
    </div>

    <!-- 题目详情模态框 -->
    <transition name="modal-fade">
      <div v-if="showDetailModal" class="modal-mask">
        <div class="modal-container">
          <div class="modal-header">
            <h3>题目详情</h3>
            <button @click="showDetailModal = false" class="modal-close">
              <i class="fas fa-times"></i>
            </button>
          </div>
          <div class="modal-content">
            <div class="detail-item">
              <label>题型：</label>
              <span class="type-badge" :class="getTypeClass(selectedQuestion.questionType)">
                {{ getQuestionTypeName(selectedQuestion.questionType) }}
              </span>
            </div>
            <div class="detail-item">
              <label>来源：</label>
              <span>{{ getDataSourceName(selectedQuestion.dataSource) }}</span>
            </div>
            <div class="detail-item">
              <label>日期：</label>
              <span>{{ formatDate(selectedQuestion.createTime) }}</span>
            </div>
            <div class="detail-item">
              <label>题干：</label>
              <p class="question-text">{{ selectedQuestion.question }}</p>
            </div>
            <div class="detail-item" v-if="selectedQuestion.options && selectedQuestion.questionType === 'choice'">
              <label>选项：</label>
              <div class="options-list">
                <div v-for="(option, idx) in parseOptions(selectedQuestion.options)" :key="idx">
                  {{ option }}
                </div>
              </div>
            </div>
            <div class="detail-item">
              <label>答案：</label>
              <pre class="answer-content">{{ selectedQuestion.answer }}</pre>
            </div>
            <div class="detail-item">
              <label>维度：</label>
              <span>{{ getDimensionName(selectedQuestion.dimension) }}</span>
            </div>
          </div>
          <div class="modal-footer">
            <button @click="showDetailModal = false" class="close-button">关闭</button>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// 响应式数据
const questions = ref<any[]>([])
const showDetailModal = ref(false)
const selectedQuestion = ref<any>(null)
const activeDropdown = ref<number | null>(null)

// 题型映射
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  only_question: '仅问题',
  compare_question: '问题对比组'
}

// 数据源映射
const dataSourceMap = {
  input: '输入',
  generated: '智能生成',
  crawler: '爬虫获取',
  manual: '人工添加'
}

// 维度映射
const dimensionMap = {
  performance: '性能',
  reliability: '可靠性',
  safety: '安全性',
  fairness: '公平性',
}

// 生命周期钩子
onMounted(() => {
  console.log('GeneratedQuestionsPage 组件已挂载')
  console.log('当前路由信息:', route)

  // 从sessionStorage中获取题目数据
  const questionsData = sessionStorage.getItem('generatedQuestions')
  console.log('从sessionStorage读取的数据:', questionsData)

  if (questionsData) {
    try {
      questions.value = JSON.parse(questionsData)
      console.log('成功从sessionStorage解析题目数据，数量:', questions.value.length)
      console.log('题目数据详情:', questions.value)
      // 数据读取后可以清除sessionStorage，避免数据残留
      sessionStorage.removeItem('generatedQuestions')
      console.log('sessionStorage已清理')
    } catch (error) {
      console.error('解析题目数据失败:', error)
      console.log('原始题目数据:', questionsData)
      questions.value = []
    }
  } else {
    console.log('未收到题目数据')
    // 也尝试从路由参数中获取，作为备选方案
    const routeQuestions = route.query.questions
    if (routeQuestions) {
      try {
        questions.value = JSON.parse(routeQuestions as string)
        console.log('从路由参数获取到题目数据')
      } catch (error) {
        console.error('从路由参数解析题目数据失败:', error)
        questions.value = []
      }
    }
  }
})

// 工具函数
const getQuestionTypeName = (type: string) => {
  return questionTypeMap[type as keyof typeof questionTypeMap] || '未知题型'
}

const getDataSourceName = (source: string) => {
  return dataSourceMap[source as keyof typeof dataSourceMap] || '未知来源'
}

const getDimensionName = (dimension: string) => {
  return dimensionMap[dimension as keyof typeof dimensionMap] || '通用维度'
}

const getTypeClass = (type: string) => {
  const typeName = getQuestionTypeName(type).replace(/\s/g, '-').toLowerCase()
  return `type-badge-${typeName}`
}

const getSourceClass = (source: string) => {
  const sourceName = getDataSourceName(source).replace(/\s/g, '-').toLowerCase()
  return `source-badge-${sourceName}`
}

const editQuestion = (question: any) => {
  console.log('编辑题目:', question)
  // 这里可以添加编辑题目的逻辑
  alert(`编辑题目: ${question.question}`)
}

const formatDate = (dateStr: string) => {
  if (!dateStr) return '无日期信息'
  try {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch (error) {
    return dateStr
  }
}

const parseOptions = (options: string) => {
  if (!options) return []

  // 如果是JSON格式的字符串，尝试解析
  if (options.startsWith('{') || options.startsWith('[')) {
    try {
      const parsed = JSON.parse(options)
      return Array.isArray(parsed) ? parsed : [options]
    } catch (e) {
      // 解析失败，尝试按逗号分割
      return options.split(',').map(opt => opt.trim())
    }
  }

  // 按逗号分割选项
  const optionList = options.split(',').map(opt => opt.trim())

  // 检查第一个选项是否已经包含字母前缀（如A.、B.等）
  const firstOption = optionList[0] || ''
  const hasLetterPrefix = /^[A-Z]\.\s*/.test(firstOption)

  // 如果选项已经包含字母前缀，直接返回原选项
  if (hasLetterPrefix) {
    return optionList
  }

  // 否则按逗号分割并返回
  return optionList
}

// 事件处理
const showQuestionDetail = (question: any) => {
  selectedQuestion.value = question
  showDetailModal.value = true
}

const toggleDropdown = (index: number) => {
  if (activeDropdown.value === index) {
    activeDropdown.value = null
  } else {
    activeDropdown.value = index
  }
}

const deleteQuestion = (question: any) => {
  if (confirm(`确定要删除题目"${question.question}"吗？`)) {
    const index = questions.value.findIndex(q => q.dataId === question.dataId)
    if (index !== -1) {
      questions.value.splice(index, 1)
      console.log('题目已删除:', question.question)
      // 关闭下拉菜单
      activeDropdown.value = null
    }
  }
}

const importToQuestionBank = () => {
  if (questions.value.length === 0) {
    alert('没有可导入的题目')
    return
  }

  // 这里可以添加导入题库的逻辑
  console.log('开始导入题库，题目数量:', questions.value.length)
  console.log('题目数据:', questions.value)

  // 模拟导入过程
  alert(`成功导入 ${questions.value.length} 道题目到题库！`)

  // 可以添加实际的API调用逻辑
  // 例如：调用后端接口将题目保存到题库
}

const goBack = () => {
  router.back()
}
</script>

<style scoped>
.generated-questions-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #e5e7eb;
}

.page-header h1 {
  margin: 0;
  color: #1f2937;
  font-size: 28px;
  font-weight: 600;
}

.back-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background-color: #6b7280;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

.back-button:hover {
  background-color: #4b5563;
}

.table-container {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

/* 复用DataSet.vue的modern-table样式 */
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
  vertical-align: middle;
}

.modern-table th.index,
.modern-table td.index {
  width: 60px;
  padding: 0 8px;
  text-align: center;
}

.modern-table th.question,
.modern-table td.question-content {
  width: 55%;
  max-width: 400px;
  white-space: normal;
  text-align: left;
}

.modern-table th.type,
.modern-table td.type {
  width: 120px;
  text-align: center;
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
  text-align: center;
}

.modern-table th.actions,
.modern-table td.actions {
  width: 100px;
  text-align: center;
}

.type-badge,
.source-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 60px;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  height: 32px;
}

.type-badge-选择题 {
  background-color: #e3f2fd;
  color: #1976d2;
}

.type-badge-判断题 {
  background-color: #f3e5f5;
  color: #9c27b0;
}

.type-badge-简答题 {
  background-color: #f0f4c3;
  color: #827717;
}

.type-badge-仅问题 {
  background-color: #ffe0b2;
  color: #e65100;
}

.type-badge-问题对比组 {
  background-color: #c5cae9;
  color: #283593;
}

.source-badge-智能生成 {
  background-color: #d1fae5;
  color: #065f46;
}

.source-badge-爬虫获取 {
  background-color: #fef3c7;
  color: #92400e;
}

.source-badge-人工添加 {
  background-color: #e0e7ff;
  color: #3730a3;
}

.source-badge-输入 {
  background-color: #f3e8ff;
  color: #6b21a8;
}

.action-container {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: center;
}

.icon-button {
  width: 32px;
  height: 32px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
}

.icon-button:hover {
  background: #f8fafc;
  border-color: #cbd5e1;
}

.view-button {
  color: #3b82f6;
}

.view-button:hover {
  background: #3b82f6;
  color: white;
}

.dropdown-button {
  color: #6b7280;
}

.dropdown-button:hover {
  background: #f8fafc;
  color: #374151;
}

/* 下拉菜单样式 */
.dropdown-container {
  position: relative;
  display: inline-block;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  right: 0;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  z-index: 100;
  min-width: 120px;
  margin-top: 4px;
}

.dropdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  color: #374151;
  transition: background-color 0.2s;
}

.dropdown-item:hover {
  background-color: #f8fafc;
}

.dropdown-item.delete-item {
  color: #ef4444;
}

.dropdown-item.delete-item:hover {
  background-color: #fef2f2;
}

.empty-state {
  padding: 60px 20px;
  text-align: center;
  color: #6b7280;
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

/* 更新后的详情项样式 */
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
  line-height: 1.6;
  color: #4b5563;
}

.options-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.answer-content {
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-all;
  border: 1px solid #e5e7eb;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  padding: 16px 24px;
  border-top: 1px solid #e5e7eb;
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

/* 导入按钮样式 */
.import-section {
  display: flex;
  justify-content: center;
  margin-top: 30px;
  padding: 20px 0;
  border-top: 1px solid #e5e7eb;
}

.import-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background-color: #10b981;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.3s;
}

.import-button:hover {
  background-color: #059669;
}

/* 动画 */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}
</style>
