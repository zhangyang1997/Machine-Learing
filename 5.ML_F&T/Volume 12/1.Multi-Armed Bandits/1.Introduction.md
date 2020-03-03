**1.简介**

**1.名词解释**

+ multi-armed bandits：多臂老虎机
+ action：动作
+ reward：奖赏
+ auxiliary：辅助的
+ feedback：反馈
+ IID ( independent identically distributed )：独立同分布
+ adversarial：对抗的
+ adversary：对手
+ constrained：约束的
+ stochastic：随机的
+ context：环境
+ dimensional：空间的
+ Bayesian priors：贝叶斯先验
+ domains：领域

**2.Multi-armed bandits问题举例**

+ 新闻网站：当一个新用户到达时，网站会选择一个文章标题来显示，观察用户是否点击了这个标题。网站的目标是使总点击量最大化。
  + action：展示文章标题。
  + arm：文章标题。
  + reward：如果被点击，reward为1，否则为0。
  + auxiliary feedback：没有。
  + other arms reward：其他arm(标题)没有reward，又称为bandits feedback。
  + context：用户位置和人口统计。

+ 动态定价：商店出售数字产品，如应用程序或歌曲。当一个新顾客到来时，商店会选择提供给这个顾客的价格。客户购买(或不购买)，然后永远离开。这家商店的目标是使总利润最大化。
  + action：规定价格。
  + arm：价格。
  + reward：如果客户购买，reward为$p$，否则为0。
  + auxiliary feedback：如果卖出，说明定价过低；如果没卖出，说明定价过高。
  + other arms reward：部分arm(价格)存在reward，又称为partial feedback。
  + context：客户的设备，位置，人口统计。
+ 投资：每天早上，你选择一只股票投资，然后投资1美元。在一天结束时，你观察每只股票的价值变化。目标是使总财富最大化。
  + action：投资股票。
  + arm：股票。
  + reward：根据股票价值确定。
  + auxiliray feedback：其他反馈都能获得。
  + other arms reward：所有arm(股票)都有奖赏，称为full feedback。
  + context：目前的经济形势。

**3.Multi-dimensional problem space**

+ **Auxiliary feedback**
+ **Rewards model**
  + IID rewards：每个arm下的rewards独立于一个固定的分布，与arm轮次无关。
  + Adversarial rewards：
  + Constrained adversary：由受约束的对手选择reward，例如每个arm的reword不能在一轮游戏中改变太多，或者每个arm的reward最多可以改变几次，或者rewards的总变化有上界。
  + Stochastic rewards(beyond IID)：rewards随时间的变化是一个随机过程。 
+ **Bayesian priors**：每个问题实例都来自一个已知的分布。
+ **Structured rewards**：rewards的结构可能已知。
+ **Global constraints**：全局约束，例如在动态定价中，可供销售的商品可能有限。
+ **Structured actions**：一个算法可能需要同时做出几个决定，例如一个新闻网站可能需要挑选一系列的文章，而一个卖家可能需要为所有的产品选择价格。

**4.Application domains**

+ 医学实验，action是开药，arm是药，reward是健康/不健康。
+ 网页设计，action是设计字体颜色或布局，reward是点击/不点击。
+ 网页内容，action是展示文章，reward是点击/不点击。
+ 网页搜索，action是返回查询结果，reward是用户满意/不满意。
+ 广告，action是展示广告，reward是广告收入。
+ 推荐系统，action是看电影，reward是接收推荐。
+ 销售/拍卖，action是定价，reward是收入。
+ 采购，action是用多少钱购买什么东西，reward是否购买。
+ 众包，action是给哪个工人分配任务分配多少工资，reward是完成任务。
+ 数据中心，action是将job路由到服务器，reward是时间。
+ 互联网，action是使用哪些TCP设置，reward是连接质量。
+ 机器人，action是给定任务的策略，reward是完成时间。