# TMD几道热门 SQL 面试题


### 背景
Sql中有一类函数叫聚合函数，比如count、sum、avg、min、max等，这些函数的可以将多行数据按照规整聚集为一行，一般聚集前的数据行要大于聚集后的数据行。而有时候我们不仅想要聚集前的数据，又想要聚集后的数据，这时候便引入了**窗口函数**。
下面通过几道TMD面试题介绍一下如何使用窗口函数。涉及知识点有用于排序的窗口函数、用于用户分组查询的窗口函数、用于偏移分析的窗口函数，每种会通过一道面试题背景题解答。

### 正文

#### 1、某顶尖外卖平台数据分析师面试题。

#### 现有交易数据表user_goods_table如下：

-  user_name    用户名 
-  goods_kind   用户订购的的外卖品类 

现在老板想知道每个用户购买的外卖品类偏好分布，并取出每个用户购买最多的外卖品类是哪个。
输出要求如下：

-  user_name    用户名 
-  goods_kind   该用户购买的最多外卖品类 

思路，利用窗口函数 row_number求得每个用户各自购买品类数量排行分布，并取出排行第一的品类即该用户购买最多的外卖品类。
**参考题解**：
```sql
select b.user_name,b.goods_kind from

(select 
 user_name,
 goods_kind,
 row_number() over(partition by user_name 
                  order by count(goods_kind) desc ) as rank 
 from user_goods_table) b where b.rank =1 
```


#### 2、某顶尖支付平台数据分析面试题。

#### 现有交易数据表user_sales_table如下：

-  user_name     用户名 
-  pay_amount   用户支付额度 

现在老板想知道支付金额在前20%的用户。
输出要求如下：

- user_name        用户名（前10%的用户）

思路，利用窗口函数 ntile将每个用户和对应的支付金额分成5组（这样每组就有1/5），取分组排名第一的用户组即前支付金额在前20%的用户。（注意这里是求前20%的用户而不是求支付排在前20的用户）
**参考题解**：
```sql
select b.user_name from 
(select 
 user_name,
 ntile(5) over(order by sum(pay_amount) desc) as level
 from user_sales_table group by user_name ) b 
where b.level = 1
```

#### 3、某顶尖小视频平台数据分析面试题。
现有用户登陆表user_login_table如下：

-  user_name     用户名 
-  date                用户登陆时间 

现在老板想知道连续7天都登陆平台的重要用户。
输出要求如下：

- user_name     用户名（连续7天都登陆的用户数）

思路，首先利用偏移窗口函数lead求得每个用户在每个登陆时间向后偏移7行的登陆时间，再计算每个用户在每个登陆时间滞后7天的登陆时间，如果每个用户向后偏移7行的登陆时间正好等于滞后7天的时间，说明该用户连续登陆了7天。
**参考题解**：
```sql
select b.user_name

(select user_name,
 date,lead(date,7) 
 over(partition by user_name order by date desc) as date_7
 from user_login_table) b 
where b.date is not null
and date_sub(cast(b.date as date,7)) = cast(b.date_7 as date)
```

### 总结：
本文分别从3家面试题了解了窗口函数的实际应用场景，当然假设是大家都已知道窗口函数的语法，窗口函数的使用也确实可以衡量作为数据分析师对sql能力的掌握程度，当然不管是学习何种用法都要结合实际应用背景思考为何需要这种分析函数。


> 原文: <https://www.yuque.com/lucky-bk3s1/sc1v5b/prwaefv2x705v3g7>