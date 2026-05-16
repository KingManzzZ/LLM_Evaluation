<template>
  <nav>
    <div class="nav-content">
      <div class="brand-container">
        <div class="brand-glow"></div>
        <div class="brand">QuantumTrace</div>
      </div>
      <ul class="nav-menu">
        <li v-for="(item, index) in menuItems" :key="index">
          <template v-if="item.submenu">
            <el-dropdown trigger="hover">
              <div class="nav-link">
                {{ item.title }}
                <i class="fas fa-chevron-down dropdown-arrow"></i>
              </div>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item
                    v-for="(sub, subIndex) in item.submenu"
                    :key="subIndex"
                    :command="sub.path"
                  >
                    <router-link :to="sub.path" class="submenu-link">
                      {{ sub.title }}
                    </router-link>
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </template>
          <router-link v-else :to="item.path" class="nav-link">
            {{ item.title }}
          </router-link>
        </li>
      </ul>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref } from 'vue';

// 更新菜单项数据（约第42行）
const menuItems = ref([
  { title: '模型排行', path: '/' },
  {
    title: '数据集',
    submenu: [
      { title: '题库', path: '/dataset/manage' },
      { title: '题库生成', path: '/question-bank-generator' },
      { title: '题库变形', path: '/question-transformation' }
    ]
  },
  {
    title: '测试',
    submenu: [
      { title: '新建测试', path: '/testing/create' },
      { title: '测试列表', path: '/testing/list' }
    ]
  }
]);
</script>

<style scoped>
nav {
  position: fixed;
  top: 0;
  width: 100%;
  height: 64px;
  display: flex;
  align-items: center; /* 垂直居中 */
  padding: 0 2rem;
  background: rgba(15, 23, 42, 0.95);
  backdrop-filter: blur(12px);
  z-index: 1000;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* 调整品牌与菜单间距（约第76行） */
.nav-content {
  display: flex;
  align-items: center;
  gap: 0.05rem;
}

.brand-container {
  position: relative;
  flex-shrink: 0; /* 禁止品牌区域收缩 */
}

.brand {
  font-size: 1.3rem;
  white-space: nowrap;
  font-weight: 600;
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  position: relative;
  z-index: 2;
}

.brand-glow {
  position: absolute;
  width: 120%;
  height: 120%;
  background: radial-gradient(circle at 50% 50%,
    rgba(56, 189, 248, 0.2) 0%,
    transparent 60%);
  filter: blur(20px);
  z-index: 1;
  top: -10%;
  left: -10%;
}

/* 更新导航菜单容器样式 */
.nav-menu {
  display: flex;
  gap: 2rem;
  align-items: center;
  height: 100%;
  margin-left: 3rem;
}

/* 调整导航链接垂直对齐 */
.nav-link {
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 1rem;
}

.nav-link {
  font-size: 0.9rem;
  white-space: nowrap;
  color: #94a3b8;
  text-decoration: none;
  font-weight: 500;
  position: relative;
  transition: all 0.3s ease;
}

.nav-link:hover {
  color: #1577f7;
}

/* 更新激活状态样式（约第148-155行） */
.nav-link {
  &.active {
    color: #38bdf8;
    font-weight: 600;

    &::after {
      content: '';
      position: absolute;
      bottom: -12px;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 2px;
      animation: underline 0.3s ease-out;
    }
  }
}

@keyframes underline {
  from { width: 0 }
  to { width: 100% }
}

.link-underline {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 0;
  height: 2px;
  background: #38bdf8;
  transition: width 0.3s ease;
}

.nav-link:hover .link-underline,
.nav-link.active .link-underline {
  width: 100%;
}

/* 响应式调整 */
@media (max-width: 1280px) {
  .nav-content {
    gap: 1.5rem;
  }
  .brand {
    font-size: 1.1rem;
  }
}

@media (max-width: 1024px) {
  .nav-content {
    gap: 1rem;
  }
  .nav-menu {
    gap: 0.8rem;
  }
  .nav-link {
    font-size: 0.85rem;
    padding: 0.3rem;
  }
}

/* 新增下拉菜单样式 */
.dropdown-arrow {
  margin-left: 6px;
  font-size: 0.8em;
  transition: transform 0.2s;
}

.el-dropdown-menu {
  background: rgba(255, 255, 255, 0.98);
  border: none;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  padding: 0.5rem 0;
}

.el-dropdown-menu__item {
  padding: 8px 16px;
  color: #475569;
  &:hover {
    background: #f1f5f9;
  }
}

.submenu-link {
  color: inherit;
  text-decoration: none;
  display: block;
  width: 100%;
}

/* 新增子菜单激活状态（约第230行） */
.submenu-link {
  &.active {
    color: #38bdf8;
    font-weight: 500;
    position: relative;

    &::before {
      content: "•";
      position: absolute;
      left: -12px;
      color: #818cf8;
    }
  }
}

/* 优化下拉菜单对齐 */
.el-dropdown {
  display: flex;
  align-items: center;
}
</style>
