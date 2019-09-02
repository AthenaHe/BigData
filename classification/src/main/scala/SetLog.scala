import org.apache.log4j.{Level, Logger}

/**
  * author: hehuan
  * date: 2019/9/1 20:32
  */
object SetLog {
  // 设置不显示log信息
  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("Spark.ui.showConsoleProgress","false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }
}
