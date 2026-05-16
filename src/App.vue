<template>
  <div id="app">
    <NavBar />
    <main class="content-area">
      <router-view name="main"></router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import router from './router';
import NavBar from './components/NavBar.vue';

onMounted(() => {
  router.isReady().then(() => {
    router.push('/');
  });
});
</script>

<style>
.content-area {
  margin-top: 80px;
  padding: 2rem 1.5rem;
}

@media (max-width: 768px) {
  .content-area {
    padding: 1.5rem;
  }
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>

<style scoped>
header {
  line-height: 1.5;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }
}
</style>
