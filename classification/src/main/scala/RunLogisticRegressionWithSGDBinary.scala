import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{DateTime, Duration}

/**
  * author: hehuan
  * date: 2019/9/2 15:37
  */
//预测准确度只有0.5,可能是程序逻辑有问题！后续再解决
object RunLogisticRegressionWithSGDBinary {
  def main(args: Array[String]): Unit = {
    //不显示日志信息
    SetLog.SetLogger

    println("==============数据准备阶段============")
    val sc = new SparkContext(new SparkConf().setAppName("LogisticRegressionWithSGDBinary").setMaster("local[*]"))
    val (trainData,validationData,testData,categoriesMap) = PrePareData(sc)
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
      PredictData(sc,model,categoriesMap)
    }else{
      val model = trainEvaluation(trainData,validationData)

      println("==============测试评估阶段===============")
      val auc = evaluateModel(model,testData)
      println("使用testdata测试最佳模型，结果AUC："+auc)

      println("==============预测数据===============")
      PredictData(sc,model,categoriesMap)
    }


    trainData.unpersist();validationData.unpersist();testData.unpersist()

  }
  //一、数据准备阶段
  def PrePareData(sc: SparkContext):(RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint],Map[String,Int])={
    //---------------1.导入/转换数据------------------
    println("开始导入数据。。。")
    val rawDataWithHeader = sc.textFile("/Users/hehuan/IdeaProjects/Classification/classification/src/main/resources/train.tsv")
    //train.scv第一行是字段名，需要删除第一行表头
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{
      (idx,iter)=>if(idx==0) iter.drop(1) else iter}
    //读取每一行数据字段
    val lines = rawData.map(_.split("\t"))
    println("共计："+lines.count().toString()+"条")


    //---------------2.创建训练评估所需的数据RDD[LabeledPoint]------------
    val categoriesMap = lines.map(fields=> fields(3))
      .distinct
      .collect
      .zipWithIndex
      .toMap

    val labelpointRDD = lines.map { fields =>
      val trFields = fields.map(_.replaceAll("\"",""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4,fields.size-1)
        .map(d=>if(d=="?") 0.0 else d.toDouble)
      val label = trFields(fields.size-1).toInt
      LabeledPoint(label,Vectors.dense(categoryFeaturesArray++numericalFeatures))

    }
    //进行数据标准化
      val featuresData = labelpointRDD.map(labelpoint=>labelpoint.features)
      println(featuresData.first())
    val stdScaler = new StandardScaler(true,true).fit(featuresData)
    val scaledRDD = labelpointRDD.map(labelpoint=>LabeledPoint(labelpoint.label,stdScaler.transform(labelpoint.features)))

    //---------------3.以随机方式将数据分为3个部分返回-------------
    val Array(trainData,validationData,testData)=
      labelpointRDD.randomSplit(Array(0.8,0.1,0.1))
    println("将数据分成trainData:"+trainData.count()+"条，validationData:"+validationData.count()+"条，testData:"+testData.count())
    (trainData,validationData,testData,categoriesMap)
  }

  //二、训练阶段
  def trainEvaluation(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]):LogisticRegressionModel={
    println("开始训练。。。。。")
    val (model,time) = trainModel(trainData,5,10,1)
    println("训练完成。。。。\n 所需时间："+time+"毫秒！")
    val AUC = evaluateModel(model,validationData)
    println("评估结果AUC="+AUC)
    (model)
  }
  //训练模型
  def trainModel(trainData: RDD[LabeledPoint], numIterations: Int, stepSize:Double,miniBatchFraction:Double):(LogisticRegressionModel,Double)={
  val startTime = new DateTime()
    val model = LogisticRegressionWithSGD.train(trainData,numIterations,stepSize,miniBatchFraction)
    val endTime = new DateTime()
    val duration = new Duration(startTime,endTime)
    (model,duration.getMillis()
    )
  }


  //测试评估
  def evaluateModel(model: LogisticRegressionModel, validationData: RDD[LabeledPoint]):(Double)={
    val scoreAndLabels = validationData.map{ data=>
      var predict = model.predict(data.features)
      (predict,data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC()
    (AUC)
  }

  //四、预测阶段
  def PredictData(sc: SparkContext, model: LogisticRegressionModel, categoriesMap: Map[String, Int])={
    //------------1.导入/转换数据--------------
    val rawDataWithHeader = sc.textFile("/Users/hehuan/IdeaProjects/Classification/classification/src/main/resources/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{
      (idx,iter)=>if(idx==0) iter.drop(1) else iter
    }
    val lines = rawData.map(_.split("\t"))
    println("testData共计："+lines.count().toString+"条")

    //------------2.创建训练评估所需的数据RDD[LabeledPoint]------------
    val dataRDD = lines.take(50).map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val label = 0


      //------------3.进行预测----------------
      val url = trFields(0)
      val Features = Vectors.dense(categoryFeaturesArray++numericalFeatures)
      val predict = model.predict(Features).toInt
      val predictDesc = {predict match{
        case 0=>"暂时性网页(ephemeral)";
        case 1=>"长青型网页(evergreen)";
      }}
      println("网址："+url+"==>预测："+predictDesc)

    }
  }

  //参数校调
  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]):LogisticRegressionModel={
    println("-----评估numIterations参数使用5，10，20，60，100----------")
    evaluateParameter(trainData,validationData,"numIterations",Array(5,15,20,60,100),Array(100),Array(1))
    println("------评估stepSize参数使用(10,50,100,200)----------")
    evaluateParameter(trainData,validationData,"stepSize",Array(100),Array(10,50,100,200),Array(1))
    println("-----评估miniBatchFraction参数使用(0.5,0.8,1)-----------")
    evaluateParameter(trainData,validationData,"miniBatchFraction",Array(100),Array(100),Array(0.5,0.8,1))
    println("------所有参数交叉评估找出最优参数组合--------------")
    val bestModel = evaluateAllParameter(trainData,validationData,Array(5,15,20,60,100),Array(10,50,100,200),Array(0.5,0.8,1))
    (bestModel)
  }

  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], evaluateParameter: String, numIterationsArray: Array[Int], stepSizeArray: Array[Double], miniBatchFractionArray: Array[Double])={
    val dataBarChart = new DefaultCategoryDataset()
    val dataLineChart = new DefaultCategoryDataset()
    for (numIterations<-numIterationsArray;stepSize<-stepSizeArray;miniBatchFraction<-miniBatchFractionArray){
      val (model,time) = trainModel(trainData,numIterations,stepSize,miniBatchFraction)
      val auc = evaluateModel(model,validationData)
      val parameterData =
        evaluateParameter match{
          case "numIterations"=>numIterations;
          case "stepSize"=>stepSize;
          case "miniBatchFraction"=>miniBatchFraction;
        }
      dataBarChart.addValue(auc,evaluateParameter,parameterData.toString())
      dataLineChart.addValue(time,"Time",parameterData.toString())
    }
    Chart.plotBarLineChart("LogisticRegressionWithSGDBinary evaluations "+evaluateParameter,evaluateParameter,"AUC",0.58,0.7,"Time",dataBarChart,dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], numIterationsArray: Array[Int], stepSizeArray: Array[Double], miniBatchFractionArray: Array[Double])={
    val evaluationsArray =
      for (numIterations<-numIterationsArray;stepSize<-stepSizeArray;miniBatchFraction<-miniBatchFractionArray) yield {
        val (model,time) = trainModel(trainData,numIterations,stepSize,miniBatchFraction)
        val auc = evaluateModel(model,validationData)
        (numIterations,stepSize,miniBatchFraction,auc)
      }
    val BestEval =(evaluationsArray.sortBy(_._4).reverse)(0)
    println("调校后最佳参数：numIterations:"+BestEval._1+",stepSize:"+BestEval._2+",miniBatchFraction:"+BestEval._3+",结果AUC="+BestEval._4)
    val (bestModel,time) = trainModel(trainData.union(validationData),BestEval._1,BestEval._2,BestEval._3)
    (bestModel)
  }

}
