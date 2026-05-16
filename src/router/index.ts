import { createRouter, createWebHistory } from 'vue-router'
import NavBar from '../components/NavBar.vue'
import LeaderBoard from '../components/LeaderBoard.vue'
import QuestionBankGenerator from '../components/QuestionBankGenerator.vue'
import DataSet from '../components/DataSet.vue'
import QuestionTransformation from '../components/QuestionTransformation.vue'
import TestingView from '../components/TestingView.vue'
import TestList from '../components/TestList.vue'
import TestResultView from '../components/TestResultView.vue'
import TestReport from '../components/TestReport.vue'
import ReviewPage from '../components/ReviewPage.vue'
import GeneratedQuestionsPage from '../views/GeneratedQuestionsPage.vue'
import TransformationResultsPage from '../views/TransformationResultsPage.vue'

const routes = [
  {
    path: '/',
    components: {
      nav: NavBar,
      main: LeaderBoard,
    },
  },
  {
    path: '/dataset/manage',
    components: {
      nav: NavBar,
      main: DataSet,
    },
  },
  {
    path: '/question-bank-generator',
    components: {
      nav: NavBar,
      main: QuestionBankGenerator,
    },
  },
  {
    path: '/question-transformation',
    components: {
      nav: NavBar,
      main: QuestionTransformation,
    },
  },
  {
    path: '/testing',
    components: {
      nav: NavBar,
      main: TestingView,
    },
  },
  {
    path: '/testing/create',
    components: {
      nav: NavBar,
      main: TestingView,
    },
    props: { default: true, main: { mode: 'create' } },
  },
  {
    path: '/testing/list',
    components: {
      nav: NavBar,
      main: TestList,
    },
  },
  {
    path: '/test-result',
    name: 'TestResult',
    components: {
      nav: NavBar,
      main: TestResultView,
    },
    props: (route) => {
      const parsedData = JSON.parse(decodeURIComponent(route.query.data as string))
      return {
        testData: parsedData,
        apiResponseData: {
          testId: parsedData.testId,
          finalScore: parsedData.finalScore,
          singleScore: parsedData.singleScores || [],
        },
      }
    },
  },
  {
    path: '/test-report',
    name: 'TestReport',
    components: {
      nav: NavBar,
      main: TestReport,
    },
    props: (route) => ({
      reportId: route.query.reportId,
    }),
  },
  {
    path: '/generated-questions',
    name: 'GeneratedQuestions',
    components: {
      nav: NavBar,
      main: GeneratedQuestionsPage,
    },
  },
  {
    path: '/transformation-results',
    name: 'TransformationResults',
    components: {
      nav: NavBar,
      main: TransformationResultsPage,
    },
  },
  {
    path: '/testing/review',
    name: 'TestReview',
    components: {
      nav: NavBar,
      main: ReviewPage,
    },
    props: (route) => ({
      testData: route.query.testData,
    }),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
