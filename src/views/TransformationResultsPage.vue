<template>
  <div class="transformation-results-page">
    <div class="page-header">
      <h1>变形结果列表</h1>
      <button class="back-button" @click="goBack"><i class="fas fa-arrow-left"></i> 返回</button>
    </div>

    <div class="table-container">
      <table class="modern-table">
        <thead>
          <tr>
            <th class="index">序号</th>
            <th class="question">原题干</th>
            <th class="transformed-question">变形后题干</th>
            <th class="type">类型</th>
            <th class="source">来源</th>
            <th class="transformation-type">变形类型</th>
            <th class="date">创建日期</th>
            <th class="actions">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(result, index) in transformationResults" :key="result.id || index">
            <td class="index">{{ index + 1 }}</td>
            <td class="question-content">{{ result.originalQuestion || '原题内容缺失' }}</td>
            <td class="transformed-question-content">
              {{
                result.originalQuestionFromFrontend ||
                result.transformedQuestion ||
                '变形后题目内容缺失'
              }}
            </td>
            <td class="type">
              <span class="type-badge" :class="getTypeClass(result.questionType)">
                {{ getQuestionTypeName(result.questionType) }}
              </span>
            </td>
            <td class="source">
              <span class="source-badge" :class="getSourceClass(result.dataSource)">
                {{ getDataSourceName(result.dataSource) }}
              </span>
            </td>
            <td class="transformation-type">
              <span
                class="transformation-badge"
                :class="getTransformationClass(result.transformationType)"
              >
                {{ getTransformationTypeName(result.transformationType) }}
              </span>
            </td>
            <td class="date">{{ formatDate(result.createTime) }}</td>
            <td class="actions">
              <div class="action-container">
                <button class="icon-button view-button" @click="showResultDetail(result)">
                  <i class="fas fa-eye"></i>
                </button>
                <div class="dropdown-container">
                  <button
                    class="icon-button dropdown-button"
                    @click="toggleDropdown(result.id || index)"
                  >
                    <i class="fas fa-ellipsis-v"></i>
                  </button>
                  <div v-if="activeDropdown === (result.id || index)" class="dropdown-menu">
                    <button class="dropdown-item" @click="exportResult(result)">
                      <i class="fas fa-download"></i> 导出
                    </button>
                    <button class="dropdown-item delete-item" @click="deleteResult(result)">
                      <i class="fas fa-trash"></i> 删除
                    </button>
                  </div>
                </div>
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="transformationResults.length === 0" class="empty-state">
        <p>暂无变形结果</p>
      </div>
    </div>

    <!-- 导入题库按钮 -->
    <div class="import-section">
      <button class="import-button" @click="importToQuestionBank">
        <i class="fas fa-file-import"></i> 导入题库
      </button>
    </div>

    <!-- 变形结果详情模态框 -->
    <transition name="modal-fade">
      <div v-if="showDetailModal" class="modal-mask">
        <div class="modal-container">
          <div class="modal-header">
            <h3>变形结果详情</h3>
            <button @click="showDetailModal = false" class="modal-close">
              <i class="fas fa-times"></i>
            </button>
          </div>
          <div class="modal-content">
            <div class="detail-item">
              <label>原题：</label>
              <p class="question-text">{{ selectedResult.originalQuestion }}</p>
            </div>
            <div class="detail-item">
              <label>变形类型：</label>
              <span
                class="transformation-badge"
                :class="getTransformationClass(selectedResult.transformationType)"
              >
                {{ getTransformationTypeName(selectedResult.transformationType) }}
              </span>
            </div>
            <div class="detail-item">
              <label>变形后题目：</label>
              <p class="transformed-question-text">{{ selectedResult.transformedQuestion }}</p>
            </div>
            <div class="detail-item">
              <label>题型：</label>
              <span class="type-badge" :class="getTypeClass(selectedResult.questionType)">
                {{ getQuestionTypeName(selectedResult.questionType) }}
              </span>
            </div>
            <div class="detail-item">
              <label>来源：</label>
              <span>{{ getDataSourceName(selectedResult.dataSource) }}</span>
            </div>
            <div class="detail-item">
              <label>维度：</label>
              <span>{{ selectedResult.dimension || '未知' }}</span>
            </div>
            <div class="detail-item">
              <label>变形描述：</label>
              <span>{{ selectedResult.transformationDescription || '无描述' }}</span>
            </div>
            <div class="detail-item">
              <label>日期：</label>
              <span>{{ formatDate(selectedResult.createTime) }}</span>
            </div>
            <div
              class="detail-item"
              v-if="selectedResult.options && selectedResult.questionType === 'choice'"
            >
              <label>选项：</label>
              <div class="options-list">
                <div v-for="(option, idx) in parseOptions(selectedResult.options)" :key="idx">
                  {{ option }}
                </div>
              </div>
            </div>
            <div class="detail-item" v-if="selectedResult.answer">
              <label>答案：</label>
              <span>{{ selectedResult.answer }}</span>
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
const transformationResults = ref<any[]>([])
const showDetailModal = ref(false)
const selectedResult = ref<any>(null)
const activeDropdown = ref<number | null>(null)

// 题型映射
const questionTypeMap = {
  choice: '选择题',
  judgment: '判断题',
  short_answer: '简答题',
  only_question: '仅问题',
  compare_question: '问题对比组',
}

// 数据源映射
const dataSourceMap = {
  input: '输入',
  generated: '智能生成',
  crawler: '爬虫获取',
  manual: '人工添加',
}

// 变形类型映射
const transformationTypeMap = {
  rewrite: '问法改写',
  add_noise: '添加噪音',
  reverse_polarity: '反向极化',
  complicate: '表达复杂化',
  substitute: '同义替代',
}

// 生命周期钩子
onMounted(() => {
  console.log('TransformationResultsPage 组件已挂载')
  console.log('当前路由信息:', route)

  // 优先从路由参数中获取变形结果数据
  const routeResults = route.query.results
  if (routeResults) {
    try {
      transformationResults.value = JSON.parse(routeResults as string)
      console.log('从路由参数获取到变形结果数据，数量:', transformationResults.value.length)
      console.log('变形结果数据详情:', transformationResults.value)
    } catch (error) {
      console.error('从路由参数解析变形结果数据失败:', error)
      console.log('原始路由参数数据:', routeResults)
      transformationResults.value = []
    }
  } else {
    console.log('未从路由参数收到变形结果数据')
    // 备选方案：从sessionStorage中获取
    const resultsData = sessionStorage.getItem('transformationResults')
    console.log('从sessionStorage读取的变形结果数据:', resultsData)

    if (resultsData) {
      try {
        transformationResults.value = JSON.parse(resultsData)
        console.log(
          '成功从sessionStorage解析变形结果数据，数量:',
          transformationResults.value.length,
        )
        console.log('变形结果数据详情:', transformationResults.value)
        // 数据读取后可以清除sessionStorage，避免数据残留
        sessionStorage.removeItem('transformationResults')
        console.log('sessionStorage已清理')
      } catch (error) {
        console.error('解析变形结果数据失败:', error)
        console.log('原始变形结果数据:', resultsData)
        transformationResults.value = []
      }
    } else {
      console.log('未收到任何变形结果数据')
      transformationResults.value = []
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

const getTransformationTypeName = (type: string) => {
  return transformationTypeMap[type as keyof typeof transformationTypeMap] || '未知变形类型'
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
    'source-badge-人工添加': source.includes('人工添加'),
  }
}

const getTransformationClass = (type: string) => {
  return {
    'transformation-badge-rewrite': type === 'rewrite',
    'transformation-badge-add_noise': type === 'add_noise',
    'transformation-badge-reverse_polarity': type === 'reverse_polarity',
    'transformation-badge-complicate': type === 'complicate',
    'transformation-badge-substitute': type === 'substitute',
  }
}

const formatDate = (dateString: string) => {
  if (!dateString) return '未知'
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN')
}

const parseOptions = (optionsString: string) => {
  if (!optionsString) return []
  return optionsString.split('|').map((opt) => opt.trim())
}

// 操作方法
const goBack = () => {
  router.back()
}

const showResultDetail = (result: any) => {
  selectedResult.value = result
  showDetailModal.value = true
}

const toggleDropdown = (index: number) => {
  activeDropdown.value = activeDropdown.value === index ? null : index
}

const exportResult = (result: any) => {
  console.log('导出变形结果:', result)
  // 实现导出逻辑
}

const deleteResult = (result: any) => {
  console.log('删除变形结果:', result)
  // 实现删除逻辑
}

const importToQuestionBank = () => {
  console.log('导入题库')
  // 实现导入题库逻辑
}
</script>

<style scoped>
/* 复用GeneratedQuestionsPage的样式 */
.transformation-results-page {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-radius: 12px;
}

.page-header h1 {
  font-size: 1.75rem;
  color: #2c3e50;
  margin: 0;
}

.back-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: #64748b;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.back-button:hover {
  background: #475569;
}

.table-container {
  border-radius: 12px;
  overflow-x: auto;
  margin-bottom: 2rem;
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
  font-weight: 600;
}

.modern-table td {
  padding: 14px 20px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: middle;
}

.modern-table th.index,
.modern-table td.index {
  width: 60px;
  text-align: center;
}

.modern-table th.question,
.modern-table td.question-content {
  width: 25%;
  max-width: 300px;
  white-space: normal;
  text-align: left;
}

.modern-table th.transformed-question,
.modern-table td.transformed-question-content {
  width: 25%;
  max-width: 300px;
  white-space: normal;
  text-align: left;
}

.modern-table th.type,
.modern-table td.type {
  width: 100px;
  text-align: center;
}

.modern-table th.source,
.modern-table td.source {
  width: 100px;
  text-align: center;
}

.modern-table th.transformation-type,
.modern-table td.transformation-type {
  width: 120px;
  text-align: center;
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
.source-badge,
.transformation-badge {
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

.source-badge-输入 {
  background-color: #f3e8ff;
  color: #7c3aed;
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

.transformation-badge-rewrite {
  background-color: #e0f2fe;
  color: #0369a1;
}

.transformation-badge-add_noise {
  background-color: #fef7cd;
  color: #92400e;
}

.transformation-badge-reverse_polarity {
  background-color: #fce7f3;
  color: #be185d;
}

.transformation-badge-complicate {
  background-color: #f0fdf4;
  color: #166534;
}

.transformation-badge-substitute {
  background-color: #fefce8;
  color: #854d0e;
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
  border: none;
  cursor: pointer;
  transition: background 0.2s;
}

.view-button {
  background: #f1f5f9;
  color: #64748b;
}

.view-button:hover {
  background: #e2e8f0;
}

.dropdown-button {
  background: #f1f5f9;
  color: #64748b;
}

.dropdown-button:hover {
  background: #e2e8f0;
}

.dropdown-container {
  position: relative;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  right: 0;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  z-index: 10;
  min-width: 120px;
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
  text-align: left;
  font-size: 0.9rem;
}

.dropdown-item:hover {
  background: #f8fafc;
}

.delete-item {
  color: #ef4444;
}

.delete-item:hover {
  background: #fef2f2;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #64748b;
}

.import-section {
  display: flex;
  justify-content: center;
  margin-top: 2rem;
  padding: 2rem 0;
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
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.95rem;
}

.import-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
}

/* 模态框样式 */
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

.question-text,
.transformed-question-text {
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
</style>
