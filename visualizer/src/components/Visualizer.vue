<template>
  <div>
    <el-row>
      <el-col :span="16" :offset="4">

        <el-row>
          <el-input placeholder="请输入验证集上结果所在的url" v-model="resultUrl">
            <el-button slot="append" icon="search" @click="fetchPredictFile"></el-button>
          </el-input>
        </el-row>

        <h2>验证集列表</h2>

        <el-row>
          <el-col :span="12">
            <el-select v-model="selectedVideo" placeholder="请选择视频" @change="onChangeVideo">
              <el-option
                v-for="item in validDataSet"
                :key="item.value"
                :label="item.label"
                :value="item.value">
              </el-option>
            </el-select>
          </el-col>
          <el-col :span="12">
            <span>预测结果</span>
            <el-select v-model="curPredict" placeholder="预测标签" @change="onChangePredictLabel">
              <el-option
                v-for="item in curPredicts.options"
                :key="item.value"
                :label="item.label"
                :value="item.value">
              </el-option>
            </el-select>
          </el-col>
        </el-row>

      </el-col>
    </el-row>

    <br/>

    <el-row>
      <!-- 视频播放框 -->
      <el-col :span="16" :offset="4">
        <el-col :span="12">
          <video ref="validVideoRef" width="400" controls>
            <source src="" type="video/mp4">
          </video>
        </el-col>
        <el-col :span="12">
          <video ref="predictVideoRef" width="400" controls>
            <source src="" type="video/mp4">
          </video>
        </el-col>
      </el-col>
    </el-row>
  </div>
</template>

<script>
  import { generateVideoList } from '../predictList'

  function generateOptionsFromPredict (predict) {
    return predict.split('\n').map(entryStr => {
      const items = entryStr.split(' ')
      const entry = {
        video: items[0],
        options: []
      }

      for (let i = 1; i < items.length; ++i) {
        const labelPair = items[i].split(':')
        entry.options.push({
          value: `class/${labelPair[1]}.mp4`,
          label: items[i]
        })
      }

      return entry
    })
  }

  export default {

    data () {
      return {
        resultUrl: '/result_top1_merge.txt',

        validDataSet: [],
        selectedVideo: '',

        predict: [],
        curPredicts: {video:'', options:[]},
        curPredict: ''
      }
    },
    created() {
      this.validDataSet = generateVideoList()
      this.fetchPredictFile()
    },
    methods: {
      fetchPredictFile () {
        this.$http.get(this.resultUrl)
          .then(data => {
            console.log(data.body)
            this.predict = generateOptionsFromPredict(data.body)
          })
      },
      onChangeVideo(url) {
        this.$refs.validVideoRef.src = `/${url}.M.mp4`
        this.$refs.validVideoRef.load();

        this.curPredicts = this.predict.find(item => {
          return item.video === url
        })
        this.curPredict = ''
      },

      onChangePredictLabel() {
        this.$refs.predictVideoRef.src = `/${this.curPredict}`
        this.$refs.predictVideoRef.load();
      }
    }
  }
</script>

<style scoped>
</style>
