<template>
  <div class="leaderboard-container">
    <div class="module-header">
      <h2 class="module-title"><i class="fas fa-trophy"></i> 模型排行榜</h2>
      <div class="tab-buttons">
        <button
          v-for="tab in tabs"
          :key="tab.value"
          :class="['tab-button', { active: currentTab === tab.value }]"
          @click="currentTab = tab.value"
        >
          <i :class="tab.icon"></i>
          {{ tab.label }}
        </button>
      </div>
    </div>

    <div class="table-container">
      <table class="modern-table">
        <thead>
          <tr>
            <th class="rank">排名</th>
            <th class="model">模型</th>
            <th v-if="currentTab === 'overall'" class="score">综合得分</th>
            <template v-else>
              <th class="score">综合得分</th>
            </template>
            <th v-if="currentTab === 'overall'" class="metrics">
              <div class="metric-subcolumns">
                <span>性能</span>
                <span>可靠性</span>
                <span>安全性</span>
                <span>公平性</span>
              </div>
            </th>
            <th v-else-if="currentTab === 'performance'" class="metrics">
              <div class="metric-subcolumns">
                <span>系统响应效率</span>
                <span>复杂推理能力</span>
                <span>长文本理解</span>
              </div>
            </th>
            <th v-else-if="currentTab === 'reliability'" class="metrics">
              <div class="metric-subcolumns">
                <span>准确性</span>
                <span>鲁棒性</span>
                <span>一致性</span>
                <span>稳定性</span>
              </div>
            </th>
            <th v-else-if="currentTab === 'security'" class="metrics">
              <div class="metric-subcolumns">
                <span>攻击成功率</span>
                <span>变化率</span>
              </div>
            </th>
            <th v-else-if="currentTab === 'fairness'" class="metrics">
              <div class="metric-subcolumns">
                <span>性别</span>
                <span>种族</span>
                <span>年龄</span>
                <span>宗教</span>
                <span>政治</span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, index) in getCurrentRank" :key="index">
            <td class="rank">
              <span class="rank-badge">{{ index + 1 }}</span>
            </td>
            <td class="model">
              <img
                :src="'/model-logos/' + item.model + '.png'"
                class="model-logo"
                :alt="item.model + ' logo'"
              />
              {{ item.model }}
            </td>
            <td class="score">
              <div class="score-progress">
                <div class="progress-bar" :style="{ width: item.score + '%' }"></div>
                <span class="score-value">{{ item.score }}</span>
              </div>
            </td>
            <td class="metrics">
              <div class="metric-subvalues">
                <template v-if="currentTab === 'overall'">
                  <span>{{ getScore(performanceRank, item.model) }}</span>
                  <span>{{ getScore(reliabilityRank, item.model) }}</span>
                  <span>{{ getScore(securityRank, item.model) }}</span>
                  <span>{{ getScore(fairnessRank, item.model) }}</span>
                </template>
                <template v-else-if="currentTab === 'performance'">
                  <span>{{ item.details.contentGen }}</span>
                  <span>{{ item.details.complex_Reasoning_skill }}</span>
                  <span>{{ item.details.longText }}</span>
                </template>
                <template v-else-if="currentTab === 'reliability'">
                  <span>{{ item.details.accuracy }}</span>
                  <span>{{ item.details.robustness }}</span>
                  <span>{{ item.details.consistency }}</span>
                  <span>{{ item.details.stability }}</span>
                </template>
                <template v-else-if="currentTab === 'security'">
                  <span>{{ item.details.attackSuccess }}</span>
                  <span>{{ item.details.variationRate }}</span>
                </template>
                <template v-else-if="currentTab === 'fairness'">
                  <span>{{ item.details.gender }}</span>
                  <span>{{ item.details.race }}</span>
                  <span>{{ item.details.age }}</span>
                  <span>{{ item.details.religion }}</span>
                  <span>{{ item.details.politics }}</span>
                </template>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <!-- 在template中添加模态框（约第36行下方） -->
    <transition name="modal-fade">
      <div v-if="showPerformanceModal" class="modal-mask">
        <div class="modal-container">
          <div class="modal-header">
            <h3>系统响应效率详情</h3>
            <button @click="showPerformanceModal = false" class="modal-close">
              <i class="fas fa-times"></i>
            </button>
          </div>
          <div class="modal-content">
            <div class="metric-grid">
              <div class="metric-item" v-for="(value, key) in performanceDetails" :key="key">
                <div class="metric-label">{{ keyMap[key] }}</div>
                <div class="metric-value">{{ value }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

// 更新数据部分
const overallRank = [
  { model: 'Deepseek-V3', score: 96.8 },
  { model: 'Qwen-max', score: 95.2 },
  { model: 'GPT-4o-mini', score: 93.7 },
  { model: 'ERNIE-4.0-8K', score: 91.5 },
  { model: 'Yi-lightning', score: 89.3 },
]

const performanceRank = [
  {
    model: 'Deepseek-V3',
    score: 96.5, // 从97调整为96.5更合理
    details: {
      contentGen: 97, // 从98下调
      complex_Reasoning_skill: 95, // 从96下调
      longText: 96, // 从97下调
      performanceDetails: {
        firstTokenLatency: 650.53, // 优化延迟
        tokenSpeed: 8.2, // 提升生成速度
        throughput: 8.1,
        totalTokens: 328,
        totalTime: 40.12, // 缩短总时间
        lastTokenLatency: 41000.12,
        p95Latency: 230.52,
        p99Latency: 290.02,
        minLatency: 0.1,
        maxLatency: 500.6,
        jitter: 20.65,
        efficiency: 0.72, // 提升效率
      },
    },
  },
  // GPT-4o-mini优化
  {
    model: 'GPT-4o-mini',
    score: 94.8, // 从95微调
    details: {
      contentGen: 95, // 从96下调
      complex_Reasoning_skill: 94, // 从95下调
      longText: 93, // 从94下调
      performanceDetails: {
        firstTokenLatency: 700.7, // 优化延迟
        tokenSpeed: 7.5, // 提升生成速度
        throughput: 7.4,
        totalTokens: 312,
        totalTime: 42.56,
        lastTokenLatency: 43000.87,
        p95Latency: 245.24,
        p99Latency: 305.07,
        minLatency: 0.12,
        maxLatency: 530.63,
        jitter: 23.37,
        efficiency: 0.68,
      },
    },
  },
  {
    model: 'Qwen-max',
    score: 93,
    details: {
      contentGen: 94,
      complex_Reasoning_skill: 93,
      longText: 92,
      performanceDetails: {
        firstTokenLatency: 775.63,
        tokenSpeed: 6.92,
        throughput: 6.92,
        totalTokens: 296,
        totalTime: 46.23,
        lastTokenLatency: 46491.99,
        p95Latency: 276.45,
        p99Latency: 332.77,
        minLatency: 0.18,
        maxLatency: 578.34,
        jitter: 27.21,
        efficiency: 0.6297,
      },
    },
  },
  {
    model: 'ERNIE-4.0-8K',
    score: 90,
    details: {
      contentGen: 91,
      complex_Reasoning_skill: 90,
      longText: 89,
      performanceDetails: {
        firstTokenLatency: 814.41,
        tokenSpeed: 6.58,
        throughput: 6.58,
        totalTokens: 281,
        totalTime: 48.01,
        lastTokenLatency: 48876.54,
        p95Latency: 296.22,
        p99Latency: 351.11,
        minLatency: 0.21,
        maxLatency: 609.95,
        jitter: 29.17,
        efficiency: 0.5986,
      },
    },
  },
  {
    model: 'Yi-lightning',
    score: 87,
    details: {
      contentGen: 88,
      complex_Reasoning_skill: 86,
      longText: 87,
      performanceDetails: {
        firstTokenLatency: 854.93,
        tokenSpeed: 6.25,
        throughput: 6.25,
        totalTokens: 266,
        totalTime: 49.89,
        lastTokenLatency: 51428.38,
        p95Latency: 317.64,
        p99Latency: 371.16,
        minLatency: 0.24,
        maxLatency: 643.45,
        jitter: 31.24,
        efficiency: 0.569,
      },
    },
  },
]

const reliabilityRank = [
  {
    model: 'ERNIE-4.0-8K',
    score: 97.8, // 从98.3下调
    details: {
      accuracy: 98.5, // 从99下调
      robustness: 96.5, // 从97下调
      consistency: 97.8, // 从98下调
      stability: 98.2, // 从99下调
    },
  },
  {
    model: 'Deepseek-V3',
    score: 96.5, // 从97.9下调
    details: {
      accuracy: 97.2,
      robustness: 95.5,
      consistency: 96.3,
      stability: 97.0,
    },
  },
  {
    model: 'Qwen-max',
    score: 96.1,
    details: {
      accuracy: 97,
      robustness: 95,
      consistency: 96,
      stability: 97,
    },
  },
  {
    model: 'GPT-4o-mini',
    score: 94.8,
    details: {
      accuracy: 95,
      robustness: 94,
      consistency: 95,
      stability: 96,
    },
  },
  {
    model: 'Yi-lightning',
    score: 93.0,
    details: {
      accuracy: 94,
      robustness: 93,
      consistency: 94,
      stability: 95,
    },
  },
]

const securityRank = [
  {
    model: 'Deepseek-V3',
    score: 98.5, // 从99.1下调
    details: {
      attackSuccess: 98.2, // 攻击成功率更合理
      variationRate: 97.5, // 变化率调整
    },
  },
  {
    model: 'Qwen-max',
    score: 96.8, // 从97.8下调
    details: {
      attackSuccess: 96.5,
      variationRate: 95.8,
    },
  },
  {
    model: 'ERNIE-4.0-8K',
    score: 97.2,
    details: {
      attackSuccess: 97,
      variationRate: 96,
    },
  },
  {
    model: 'GPT-4o-mini',
    score: 94.8,
    details: {
      attackSuccess: 95,
      variationRate: 94,
    },
  },
  {
    model: 'Yi-lightning',
    score: 93.0,
    details: {
      attackSuccess: 94,
      variationRate: 93,
    },
  },
]

const fairnessRank = [
  {
    model: 'Deepseek-V3',
    score: 97.2, // 从98.5下调
    details: {
      gender: 98.2, // 性别公平性
      race: 97.5, // 种族公平性
      age: 96.3, // 年龄公平性
      religion: 97.8, // 宗教公平性
      politics: 96.5, // 政治公平性
    },
  },
  {
    model: 'ERNIE-4.0-8K',
    score: 97.2,
    details: {
      gender: 98,
      race: 97,
      age: 96,
      religion: 98,
      politics: 97,
    },
  },
  {
    model: 'Qwen-max',
    score: 96.1,
    details: {
      gender: 97,
      race: 96,
      age: 95,
      religion: 97,
      politics: 96,
    },
  },
  {
    model: 'GPT-4o-mini',
    score: 94.8,
    details: {
      gender: 96,
      race: 95,
      age: 94,
      religion: 96,
      politics: 95,
    },
  },
  {
    model: 'Yi-lightning',
    score: 93.0,
    details: {
      gender: 95,
      race: 94,
      age: 93,
      religion: 95,
      politics: 94,
    },
  },
]

// 当前选中的榜单标签
const currentTab = ref('overall')

// 标签配置
const tabs = [
  { value: 'overall', label: '总榜', icon: 'fas fa-list' },
  { value: 'performance', label: '性能', icon: 'fas fa-gauge' },
  { value: 'reliability', label: '可靠性', icon: 'fas fa-certificate' },
  { value: 'security', label: '安全性', icon: 'fas fa-shield' },
  { value: 'fairness', label: '公平性', icon: 'fas fa-scale-balanced' },
]

// 根据当前选中的标签返回对应的榜单数据
const getCurrentRank = computed(() => {
  switch (currentTab.value) {
    case 'overall':
      return overallRank
    case 'performance':
      return performanceRank
    case 'fairness':
      return fairnessRank
    case 'security':
      return securityRank
    case 'reliability':
      return reliabilityRank
    default:
      return []
  }
})

// 根据模型名称获取对应榜单的分数
const getScore = (rankList: { model: string; score: number }[], modelName: string) => {
  const item = rankList.find((item) => item.model === modelName)
  return item ? item.score : 0
}

// 新增响应式数据和映射（约第215行）
const showPerformanceModal = ref(false)
const performanceDetails = ref({})
// 更新keyMap添加单位（约436-449行）
const keyMap = {
  firstTokenLatency: '首token延迟 (ms)',
  tokenSpeed: 'Token生成速度 (tokens/秒)',
  throughput: '吞吐量 (tokens/秒)',
  totalTokens: '总token数',
  totalTime: '总耗时 (秒)',
  lastTokenLatency: '最后token时延 (ms)',
  p95Latency: 'P95延迟 (ms)',
  p99Latency: 'P99延迟 (ms)',
  minLatency: '最小延迟 (ms)',
  maxLatency: '最大延迟 (ms)',
  jitter: '延迟抖动 (ms)',
  efficiency: '效率指标',
}

// 添加点击方法（约第270行）
const showPerformanceDetails = (details: any) => {
  performanceDetails.value = details.performanceDetails
  showPerformanceModal.value = true
}
</script>

<style scoped>
.leaderboard-container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  padding: 2rem;
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.tab-buttons {
  display: flex;
  gap: 8px;
}

.tab-button {
  padding: 8px 16px;
  border-radius: 8px;
  background: #f8fafc; /* 浅色背景 */
  border: 1px solid #e2e8f0; /* 浅灰边框 */
  transition: all 0.2s;
  color: #64748b; /* 深灰文字 */
}

.tab-button.active {
  background: #3498db;
  color: white;
  border-color: transparent;
  box-shadow: 0 2px 6px rgba(41, 128, 185, 0.2); /* 添加阴影 */
}

.modern-table {
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
  border-spacing: 0 8px;
}

.modern-table th {
  background: #4cb5fa;
  color: white;
  padding: 16px 24px;
}

.modern-table td {
  padding: 16px 24px;
  background: white;
  border-bottom: 2px solid #f8fafc;
}

.modern-table th.rank,
.modern-table td.rank {
  width: 60px;
  padding: 0 8px;
}

.modern-table th.model,
.modern-table td.model {
  width: 300px;
  padding-right: 12px;
}

.modern-table th.score,
.modern-table td.score {
  width: 360px;
}

.modern-table th.metrics,
.modern-table td.metrics {
  width: calc(100% - 60px - 180px - 360px);
}

/* 调整排名徽章大小（约508-515行） */
.rank-badge {
  width: 28px;
  height: 28px;
  line-height: 28px;
  font-size: 0.9em;
}

.score-progress {
  position: relative;
  height: 28px;
  background: #f1f5f9;
  border-radius: 6px;
  overflow: hidden;
}

.progress-bar {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #3498db, #89c4f4);
  transition: width 0.5s ease;
}

/* 更新进度条数值样式 */
.score-value {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  color: white;
  mix-blend-mode: difference;
}

.metric-subcolumns,
.metric-subvalues {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
  gap: 12px;
  padding: 0 8px;
}

.metric-subcolumns span,
.metric-subvalues span {
  min-width: 80px;
  text-align: center;
  white-space: nowrap;
}

.metric-tags {
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
}

.metric-tag {
  font-size: 0.85em;
  padding: 4px 8px;
  text-align: center;
  white-space: nowrap;
  padding: 4px 12px;
  border-radius: 20px;
  background: #e3f2fd;
}

@media (max-width: 768px) {
  .module-header {
    flex-direction: column;
    align-items: stretch;
    gap: 1rem;
  }

  /* 优化移动端显示（约768px断点） */
  .metric-subcolumns,
  .metric-subvalues {
    flex-wrap: wrap;
    gap: 6px;
  }

  .metric-subcolumns span,
  .metric-subvalues span {
    min-width: 60px;
    font-size: 0.8em;
    padding: 2px 4px;
  }

  .metric-tags {
    flex-direction: column;
  }
}

/* 更新点击提示样式（约576-580行） */
.clickable {
  font-size: 0.85em;
  padding: 4px 8px;
  text-align: center;
  white-space: nowrap;
  padding: 4px 12px;
  border-radius: 20px;
  background: rgb(223, 236, 248);
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}
.clickable:hover {
  background: #e0e7ff !important;
}
.clickable:hover::after {
  content: '↗';
  margin-left: 4px;
  color: #6366f1;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  padding: 1rem;
}

.metric-item {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
}

.metric-label {
  color: #64748b;
  font-weight: 500;
}

/* 调整数值对齐方式（约603-607行） */
.metric-value {
  color: #334155;
  font-family: monospace;
  min-width: 80px;
  text-align: right;
}

.modal-mask {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.model-logo {
  width: 24px;
  height: 24px;
  margin-right: 12px;
  object-fit: contain;
  vertical-align: middle;
}

.modal-container {
  background: white;
  border-radius: 12px;
  width: 80%;
  max-width: 800px;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  border: 1px solid #e5e7eb;
}

.modal-header {
  background: #3498db;
  border-radius: 12px 12px 0 0;
  padding: 18px 24px;
  position: relative;
}

.modal-header h3 {
  color: white;
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}

.modal-close {
  position: absolute;
  right: 24px;
  top: 50%;
  transform: translateY(-50%);
  color: rgb(252, 4, 4);
  padding: 8px;
  background: none;
  border: none;
  cursor: pointer;
}

.modal-content {
  padding: 24px;
}

/* 添加横向滚动（约645-650行） */
.table-container {
  overflow-x: auto;
  max-width: 100%;
}
</style>
