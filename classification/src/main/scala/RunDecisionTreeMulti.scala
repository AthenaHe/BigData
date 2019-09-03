import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{DateTime, Duration}

/**
  * author: hehuan
  * date: 2019/9/2 20:51
  */
object RunDecisionTreeMulti {
  def main(args: Array[String]): Unit = {
    //不显示日志信息
    SetLog.SetLogger

    println("==============数据准备阶段============")
    val sc = new SparkContext(new SparkConf().setAppName("DecisionTreeMulti").setMaster("local[*]"))
    val (trainData,validationData,testData) = PrePareData(sc)
    trainData.persist(); validationData.persist();testData.persist()

    println("==============训练数据阶段============")
    println()
    println("是否需要进行参数校调(Y:是，N:否)？")
    if(readLine()=="Y"){
      val model = parametersTunning(trainData,validationData)

      println("==============测试评估阶段===============")
      val auc = evaluateModel(model,testData)
      println("使用testdata测试最佳模型，结果AUC："+auc)

      println("==============预测数据===============")
      PredictData(sc,model)
    }else{
      val model = trainEvaluation(trainData,validationData)

      println("==============测试评估阶段===============")
      val auc = evaluateModel(model,testData)
      println("使用testdata测试最佳模型，结果AUC："+auc)

      println("==============预测数据===============")
      PredictData(sc,model)
    }


    trainData.unpersist();validationData.unpersist();testData.unpersist()

  }


  //一、数据准备阶段
  def PrePareData(sc: SparkContext):(RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint])={
    //---------------1.导入/转换数据------------------
    println("开始导入数据。。。")
    val rawData = sc.textFile("/Users/hehuan/IdeaProjects/Classification/classification/src/main/resources/covtype.data")
   //读取每一行数据字段
    println("共计："+rawData.count.toString()+"条")
    //---------------2.创建评估所需要的数据RDD[LabeledPoint]-----------------
    println("准备训练数据。。。")

    val labelpointRDD = rawData.map {record=>
      val fields = record.split(',').map(_.toDouble)
      val label = fields.last-1
      LabeledPoint(label,Vectors.dense(fields.init))
    }
    println(labelpointRDD.first())

    //---------------3.以随机方式将数据分为3个部分返回-------------
    val Array(trainData,validationData,testData)=
      labelpointRDD.randomSplit(Array(0.8,0.1,0.1))
    println("将数据分成trainData:"+trainData.count()+"条，validationData:"+validationData.count()+"条，testData:"+testData.count())
    (trainData,validationData,testData)
  }

  //二、训练阶段
  def trainEvaluation(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]):DecisionTreeModel={
    println("开始训练。。。。。")
    val (model,time) = trainModel(trainData,"entropy",10,10)
    println("训练完成。。。。\n 所需时间："+time+"毫秒！")
    val AUC = evaluateModel(model,validationData)
    println("评估结果AUC="+AUC)
    (model)
  }
  //训练模型
  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int):(DecisionTreeModel,Double)={
    val startTime = new DateTime()
    //numclass=7代表label的种类有7种
    val model = DecisionTree.trainClassifier(trainData,7,Map[Int,Int](),impurity,maxDepth,maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime,endTime)
    (model,duration.getMillis()
    )
  }

  //测试评估
  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]):(Double)={
    val scoreAndLabels = validationData.map{ data=>
      var predict = model.predict(data.features)
      (predict,data.label)
    }
    val Metrics = new MulticlassMetrics(scoreAndLabels)
    val precision = Metrics.precision
    (precision)
  }

  //四、预测阶段
  def PredictData(sc: SparkContext, model: DecisionTreeModel)={
    //------------1.导入/转换数据--------------
    val rawData = sc.textFile("/Users/hehuan/IdeaProjects/Classification/classification/src/main/resources/covtype.data")

    println("testData共计："+rawData.count.toString()+"条")

    //------------2.创建训练评估所需的数据RDD[LabeledPoint]------------
    println("准备训练数据。。。")
    val Array(pData,oData) = rawData.randomSplit(Array(0.1,0.9))
    val data = pData.take(20).map{record=>
      val fields = record.split(',').map(_.toDouble)
      val features = Vectors.dense(fields.init)
      val label = fields.last-1
      val  predict = model.predict(features)
      val result = (if (label==predict) "正确" else "错误")
      println("土地条件：海拔："+features(0)+",方位："+features(1)+",斜率："+features(2)+",水源垂直距离："+features(3)+",水源水平距离："+
      features(4)+",9点时阴影："+features(5)+"。。。=》预测："+predict+" 实际："+label+"结果："+result)




    }
  }

  //参数校调
  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]):DecisionTreeModel={
    println("-----评估Impurity参数使用gini，entropy----------")
    evaluateParameter(trainData,validationData,"impurity",Array("gini","entropy"),Array(10),Array(10))
    println("------评估MaxDepth参数使用(3,5,10,15,20)----------")
    evaluateParameter(trainData,validationData,"maxDepth",Array("gini"),Array(3,5,10,15,20),Array(10))
    println("-----评估maxBins参数使用(3,5,10,50,100)-----------")
    evaluateParameter(trainData,validationData,"maxBins",Array("gini"),Array(10),Array(3,5,10,50,100,200))
    println("------所有参数交叉评估找出最优参数组合--------------")
    val bestModel = evaluateAllParameter(trainData,validationData,Array("gini","entropy"),Array(3,5,19,15,20),Array(3,5,10,50,100,200))
    (bestModel)
  }

  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], evaluateParameter: String, impurityArray: Array[String], maxDepthArray: Array[Int], maxBinsArray: Array[Int])={
    val dataBarChart = new DefaultCategoryDataset()
    val dataLineChart = new DefaultCategoryDataset()
    for (impurity<-impurityArray;maxDepth<-maxDepthArray;maxBins<-maxBinsArray){
      val (model,time) = trainModel(trainData,impurity,maxDepth,maxBins)
      val auc = evaluateModel(model,validationData)
      val parameterData =
        evaluateParameter match{
          case "impurity"=>impurity;
          case "maxDepth"=>maxDepth;
          case "maxBins"=>maxBins;
        }
      dataBarChart.addValue(auc,evaluateParameter,parameterData.toString())
      dataLineChart.addValue(time,"Time",parameterData.toString())
    }
    Chart.plotBarLineChart("DecisionTree evaluations "+evaluateParameter,evaluateParameter,"precision",0.6,1,"Time",dataBarChart,dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxDepthArray: Array[Int], maxBinsArray: Array[Int]):DecisionTreeModel={
    val evaluationsArray =
      for (impurity<-impurityArray;maxDepth<-maxDepthArray;maxBins<-maxBinsArray) yield {
        val (model,time) = trainModel(trainData,impurity,maxDepth,maxBins)
        val auc = evaluateModel(model,validationData)
        (impurity,maxDepth,maxBins,auc)
      }
    val BestEval =(evaluationsArray.sortBy(_._4).reverse)(0)
    println("调校后最佳参数：impurity:"+BestEval._1+",maxDepth:"+BestEval._2+",maxBins"+BestEval._3+",结果AUC="+BestEval._4)
    val (bestModel,time) = trainModel(trainData.union(validationData),BestEval._1,BestEval._2,BestEval._3)
    (bestModel)
  }
}
