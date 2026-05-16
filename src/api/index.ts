import axios from 'axios'

const api = axios.create({
  baseURL: '/api', // 修改为相对路径
  timeout: 3000000, // 增加超时时间到3000000ms
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
api.interceptors.request.use((config) => {
  // 可在此处添加token等认证信息
  return config
})

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  },
)

export default api
