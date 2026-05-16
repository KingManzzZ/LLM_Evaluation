import api from './index'

// 题目变形接口参数类型
export interface TransformationRequest {
  dataIds: number[]
  transformationType: string
}

// 变形结果类型
export interface TransformationResult {
  id?: number
  originalQuestion: string
  transformedQuestion: string
  questionType: string
  dataSource: string
  transformationType: string
  createTime?: string
  options?: string
  answer?: string
  dimension?: string
  transformationDescription?: string
}

// 使用新的接口地址，直接访问后端服务
const transformationApi = api.create({
  baseURL: '/api',
  timeout: 300000, // 增加超时时间到5分钟，避免处理复杂变形时超时
})

// 题目查询API - 根据dataId获取题目详情
const getQuestionById = async (dataId: number): Promise<any> => {
  try {
    console.log(`开始查询题目ID ${dataId} 的详情`)

    const response = await transformationApi.get(`/dataInfo/select/${dataId}`)

    if (response && response.data && response.data.code === 200) {
      console.log(`题目ID ${dataId} 查询成功:`, response.data.data)
      return response.data.data
    } else {
      console.warn(`题目ID ${dataId} 查询失败:`, response?.data)
      return null
    }
  } catch (error) {
    console.error(`题目ID ${dataId} 查询失败:`, error)
    return null
  }
}

// 题目变形API
export const transformQuestions = async (
  request: TransformationRequest,
): Promise<TransformationResult[]> => {
  try {
    console.log('开始调用题目变形API，参数:', request)

    // 构建请求体数据 - 根据新的接口规范
    const requestData = {
      dataIds: request.dataIds,
      transformationType: request.transformationType,
    }

    console.log('构建的请求体数据:', requestData)

    // 调用后端API - 使用POST方法，参数放在请求体中
    const response = await transformationApi.post('/dataInfo/transform', requestData, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    console.log('API响应状态:', response.status)
    console.log('API响应数据:', response.data)

    // 处理响应数据 - 根据后端返回的数据结构
    if (response && response.data) {
      const responseData = response.data

      // 检查响应码 - 根据实际返回的code值判断
      if (responseData.code === 200 && Array.isArray(responseData.data)) {
        console.log('API返回的原始数据:', responseData.data)

        // 处理每个变形结果，获取真正的原题内容
        const processedResults = []

        for (const item of responseData.data) {
          // 记录原始dataId，用于查找原题内容
          const originalDataId = item.originalDataId || item.dataId || request.dataIds[0] || 0

          // 获取真正的原题内容
          let originalQuestion = '原题内容缺失'
          if (originalDataId) {
            const originalQuestionData = await getQuestionById(originalDataId)
            if (originalQuestionData && originalQuestionData.question) {
              originalQuestion = originalQuestionData.question
            }
          }

          // 变形后的题目内容
          const transformedQuestion =
            item.question || item.transformedQuestion || '变形后题目内容缺失'

          processedResults.push({
            id: originalDataId, // 使用原始dataId作为唯一标识
            originalQuestion: originalQuestion,
            transformedQuestion: transformedQuestion,
            questionType: item.questionType || 'choice',
            dataSource: item.dataSource || 'input',
            transformationType: item.transformationType || request.transformationType,
            createTime: item.createTime || new Date().toISOString(),
            options: item.options || '',
            answer: item.answer || '',
            dimension: item.dimension || 'performance',
            transformationDescription: item.transformationDescription || '',
            originalDataId: originalDataId, // 保存原始dataId
          })
        }

        return processedResults
      } else {
        console.warn('API返回错误码或数据格式不正确:', responseData)
        throw new Error(responseData.msg || 'API返回数据格式错误')
      }
    }

    // 如果后端API暂时不可用，返回模拟数据
    console.log('后端API暂时不可用，返回模拟数据')
    return generateMockTransformationResults(request)
  } catch (error: any) {
    console.error('题目变形API调用失败:', error)

    // 详细错误信息
    if (error.response) {
      // 服务器响应了，但状态码不在2xx范围内
      console.error('服务器响应错误:', error.response.status, error.response.data)
    } else if (error.request) {
      // 请求已发送但没有收到响应
      console.error('网络错误，未收到响应:', error.request)
    } else {
      // 其他错误
      console.error('其他错误:', error.message)
    }

    // 如果API调用失败，返回模拟数据作为备选方案
    console.log('API调用失败，返回模拟数据')
    return generateMockTransformationResults(request)
  }
}

// 生成模拟变形结果数据
const generateMockTransformationResults = (
  request: TransformationRequest,
): TransformationResult[] => {
  const mockResults: TransformationResult[] = []
  const transformationTypeName = getTransformationTypeName(request.transformationType)

  request.dataIds.forEach((id, index) => {
    mockResults.push({
      id: id,
      originalQuestion: `这是第${id}个题目的原题内容，用于测试题目变形功能。`,
      transformedQuestion: `这是经过${transformationTypeName}变形后的第${id}个题目内容。`,
      questionType: 'choice',
      dataSource: 'generated',
      transformationType: request.transformationType,
      createTime: new Date().toISOString(),
      options: '选项A|选项B|选项C|选项D',
    })
  })

  return mockResults
}

// 获取变形类型名称
const getTransformationTypeName = (type: string): string => {
  const typeMap: { [key: string]: string } = {
    rewrite: '问法改写',
    add_noise: '添加噪音',
    reverse_polarity: '反向极化',
    complicate: '表达复杂化',
    substitute: '同义替代',
  }

  return typeMap[type] || '未知变形类型'
}

export default {
  transformQuestions,
}
