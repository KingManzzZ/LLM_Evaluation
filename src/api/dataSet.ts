import api from './index'

// 测试接口返回数据类型定义
// 更新以匹配新的接口响应格式
export interface TestResponseData {
  code: number
  msg: string
  data: {
    testId: number // 新增：测试ID
    finalScore: number
    singleScore: number[] | null
    metricScores: { [key: string]: number } | null // 修改：改为可为null
  }
  timestamp: number
}

// 测试详情接口返回数据类型定义
export interface TestDetailResponseData {
  code: number
  msg: string
  data: Array<{
    dataId: number // 新增：题目ID
    question: string
    answer: string | null
    modelOutput: string
    score: number
  }>
  timestamp: number
}

// 测试列表接口返回数据类型定义
export interface TestListResponseData {
  code: number
  msg: string
  data: {
    records: Array<{
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
    }>
    total: number
    size: number
    current: number
    pages: number
  }
  timestamp: number
}

export const DataSetService = {
  // 统一分页查询接口
  getQuestionsPaginated: (params: {
    pageNum: number
    questionType?: string
    dimension?: string
    metric?: string
    subMetric?: string
  }) => api.get('/dataInfo/select/pages', { params }),

  // 单题详情查询
  getQuestionDetail: (dataId: number) => api.get(`/dataInfo/${dataId}`),

  // 删除题目
  deleteQuestion: (dataId: number) => api.delete(`/dataInfo/delete/${dataId}`),

  // 编辑题目
  updateQuestion: (data: {
    dataId: number
    question: string
    options: string
    answer: string
    dimension: string
    questionType: string
    dataSource: string
  }) => api.put('/dataInfo/update', data),

  // 新建测试接口 - 性能维度下的系统响应效率和公平性维度
  createPerformanceTestSR: (data: {
    testName: string
    modelName: string
    dimension: string
    metric: string
    os: string
    cpu: string
    gpu: string
    questionList: number[]
  }) => api.post<TestResponseData>('/test/test1', data),

  // 新建测试接口 - 性能的复杂推理能力、长文本理解能力、可靠性、安全性
  createPerformanceTestOther: (data: {
    testName: string
    modelName: string
    dimension: string
    metric: string
    os: string
    cpu: string
    gpu: string
    questionList: number[]
  }) => api.post<TestResponseData>('/test/test2', data),

  // 查询测试详情接口
  getTestDetail: (testId: number) => api.get<TestDetailResponseData>(`/test/select/${testId}`),

  // 获取测试列表接口
  getTestList: (params: { pageNum: number }) =>
    api.get<TestListResponseData>('/test/select', { params }),
}
