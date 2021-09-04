import time
import numpy as np
import pandas as pd
import os

def day_dif(df):
    """计算日期差"""
    return ((df['fdealday'] - df.shift(1)['fdealday']) / pd.Timedelta(1, 'D')).fillna(0).astype(int)

def cal_userate(di, useRate_dict, cate_rate):
    di['day_dif'] = day_dif(di).shift(-1).fillna(0)  ##后面的day_dif是函数
    di = di.drop(index=di.day_dif[di.day_dif == 0].index)
    di['rate'] = (di['day_contents'] / di['day_dif'])
    if di.shape[0] > 0:
        fsku_label = str(di.iloc[0]['fbuyer_id']) + '_on_' + str(di.iloc[0]['cata_5_name'])

        """如果包括需要预测的记录大于等于三行，则去除最大值，最大值很多情况下是连续购买，去掉最大值后求平均值；
            否则，去掉需要预测的记录的零行，取另外一个值"""

        if di['rate'].unique().size >= 3:
            ##去除最大值，最大值很多情况下是连续购买，去掉最大值后求平均值（也去掉还没计算出的0值）
            useRate_dict[fsku_label] = round(
                (di[(di['rate'] < di['rate'].max()) & (di['rate'] > di['rate'].min())]['rate']).mean(), 2)
        else:
            useRate_dict[fsku_label] = round(di['rate'].max(), 2)

        ## 计算按照类目区分的速率
        if di['rate'].unique().size >= 3:
            cate_rate[(di.iloc[0]['cata_5_name'])].append(
                round((di[(di['rate'] < di['rate'].max()) & (di['rate'] > di['rate'].min())]['rate']).mean(), 2))
        else:
            cate_rate[(di.iloc[0]['cata_5_name'])].append(round(di['rate'].max(), 2))

def cal_buytime(di, useRate_dict, buyTime_dict, maxprob_buysku):
    new_record = di.iloc[-1, :]

    ## 形成唯一标识：fuid+cata_5_name
    new_fsku_label = str(new_record['fbuyer_id']) + '_on_' + str(new_record['cata_5_name'])

    ## 读取已计算的速率
    new_fsku_rate = useRate_dict[new_fsku_label]

    ## 最可能购买的sku_id（即最近一次购买的sku）
    maxprob_buysku[new_fsku_label] = new_record['sku_id']

    ## 转化日期
    day_dif_stop = new_record['day_contents'] / new_fsku_rate

    ## 这边老报错，0.0/1.0计算总为inf,没找到原因，因此强行加一层转换
    if day_dif_stop == np.inf:
        day_dif_stop = 0.0

    day_dif_int = int(round(day_dif_stop, 2))  ##加一层判断是否为Inf或者nan
    #     day_dif_str=str(day_dif_int).replace('inf','0')

    ## 计算间隔购买时间
    new_day_dif = pd.to_timedelta(int(day_dif_int), unit='D')

    ## 一周的时间
    day_dif_7 = pd.to_timedelta(7, unit='D')

    ## 计算预测的购买时间
    new_buytime = new_record['fdealday'] + new_day_dif

    ## 如果已经过时的或者时间间隔超过一年，每隔7天（或一个周期）推荐一次
    if new_buytime < today_time or int(day_dif_int) > 365:
        new_buytime = today_time + day_dif_7

    buyTime_dict[new_fsku_label] = new_buytime.strftime('%Y-%m-%d')
    """还需要加一层和当前时间的比较已经过时的或者时间间隔超过一年，每隔7天（或一个周期）推荐一次"""


## 训练集主函数（主要计算得到不同的字典-哈希表）---------------------------------------------------------------------------------------##
"""输入得到的<fuid+五级品类+购买时间+sku_id+购买量>的dataframe"""
def prob_buytime(df, useRate_dict, buyTime_dict, maxprob_buysku, cate_rate):
    df['fdealday'] = pd.to_datetime(df['fdealday'])

    ##计算使用速率，需要传入useRate_dict(根据类目改变)
    df.groupby(['fbuyer_id', 'cata_5_name']).apply(lambda x: cal_userate(x, useRate_dict, cate_rate))

    ##预测使用时间，buyTime_dict(根据类目改变)
    # df.groupby(['fbuyer_id', 'cata_5_name']).apply(lambda x: cal_buytime(x, useRate_dict, buyTime_dict, maxprob_buysku))

## 预测测试集的购买时间
def predict_buytime(data_record, useRate_dict, cate_rate_avg, buyTime_dict, maxprob_buysku):
    #     print(data_record['latest_buytime'])
    ## 形成唯一标识：fuid+cata_5_name
    new_fsku_label = str(data_record['fbuyer_id']) + '_on_' + str(data_record['cata_5_name'])

    ## 最可能购买的sku_id（即最近一次购买的sku）
    maxprob_buysku[new_fsku_label] = data_record['sku_id']

    if new_fsku_label in useRate_dict:

        ## 形成速率标识：cata_5_name(取过平均)
        new_rate_label = str(data_record['cata_5_name'])

        ## 读取已计算的速率
        new_fsku_rate = cate_rate_avg[new_rate_label]

        ## 最可能购买的sku_id（即最近一次购买的sku）

        maxprob_buysku[new_fsku_label] = data_record['sku_id']

        ## 转化日期
        day_dif_stop = data_record['day_contents'] / new_fsku_rate

        ## 这边老报错，0.0/1.0计算总为inf,没找到原因，因此强行加一层转换
        if day_dif_stop == np.inf:
            day_dif_stop = 0.0

        day_dif_int = int(round(day_dif_stop, 2))  ##加一层判断是否为Inf或者nan
        #     day_dif_str=str(day_dif_int).replace('inf','0')

        ## 计算间隔购买时间
        new_day_dif = pd.to_timedelta(int(day_dif_int), unit='D')

        # ## 计算预测的购买时间
        # new_buytime = data_record['latest_buytime'] + new_day_dif

        ## 转化日期
        day_dif_stop = data_record['day_contents'] / new_fsku_rate

        ## 这边老报错，0.0/1.0计算总为inf,没找到原因，因此强行加一层转换
        if day_dif_stop == np.inf or new_fsku_rate <= 1:  # （多一些异常处理的情况）
            day_dif_stop = 0.0

        day_dif_int = int(round(day_dif_stop, 2))  ##加一层判断是否为Inf或者nan
        #     day_dif_str=str(day_dif_int).replace('inf','0')

        ## 计算间隔购买时间
        new_day_dif = pd.to_timedelta(int(day_dif_int), unit='D')

        ## 计算预测的购买时间
        new_buytime = data_record['latest_buytime'] + new_day_dif

    else:

        ## 形成速率标识：cata_5_name(取过平均)
        new_rate_label = str(data_record['cata_5_name'])

        ## 读取已计算的速率
        if new_rate_label in cate_rate_avg:

            new_fsku_rate = cate_rate_avg[new_rate_label]

        ## 最可能购买的sku_id（即最近一次购买的sku）

        maxprob_buysku[new_fsku_label] = data_record['sku_id']

        ## 转化日期
        day_dif_stop = data_record['day_contents'] / new_fsku_rate

        ## 这边老报错，0.0/1.0计算总为inf,没找到原因，因此强行加一层转换
        if day_dif_stop == np.inf:
            day_dif_stop = 0.0

        day_dif_int = int(round(day_dif_stop, 2))  ##加一层判断是否为Inf或者nan
        #     day_dif_str=str(day_dif_int).replace('inf','0')

        ## 计算间隔购买时间
        new_day_dif = pd.to_timedelta(int(day_dif_int), unit='D')

        ## 计算预测的购买时间
        new_buytime = data_record['latest_buytime'] + new_day_dif

    #     print(new_buytime)
    ## 如果已经过时的或者时间间隔超过一年，每隔7天（或一个周期）推荐一次

    ## 一周的时间
    day_dif_7 = pd.to_timedelta(7, unit='D')
    # if new_buytime <= today_time or int(day_dif_int) > 365:
    #     new_buytime = today_time + day_dif_7
    if int(day_dif_int) > 365:
        new_buytime = today_time + day_dif_7
    buyTime_dict[new_fsku_label] = new_buytime.strftime('%Y-%m-%d')
    """还需要加一层和当前时间的比较已经过时的或者时间间隔超过一年，每隔7天（或一个周期）推荐一次"""


if __name__ == '__main__':

    os.chdir('E:\数据分析\周期购')
    ## 当天时间，后期需要判断使用
    today_time = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()))

    ## 训练集即一年或者更多的数据来计算速率
    dt_train = pd.read_csv('tmp_try_train.csv', encoding='gbk')
    ## 测试集即近一个月的数据预测购买时间
    dt_test = pd.read_csv('tmp_try_test.csv', encoding='gbk')

    dt_test['latest_buytime'] = pd.to_datetime(dt_test['latest_buytime'])

    useRate_cate  = {}   ## 记录fuid+cate5的使用速率
    cate_rate_avg = {}   ## 记录cate5的平均使用速率
    buyTime_cate  = {}   ## 预测fuid+cate5的购买时间，后续的测试中可以不需要使用
    maxprob_buysku= {}   ## 从五级类目去关联最新购买的一次SKU，给每个fuid+五级类目标记上一个最新的SKU
    cate_rate = dict([(k, []) for k in dt_train['cata_5_name'].unique()])  ## 类目速率，用于计算类目下的平均速率和速率众数，用以计算平均情况

    buyTime_T = {}

    ## 用于创建结果表
    Fuid=[]
    Skuid=[]
    Prob_time=[]
    Cate_maxlike=[]

    prob_buytime(dt_train, useRate_cate, buyTime_cate, maxprob_buysku, cate_rate)

    ## 计算不同类目下的平均速率，这个cate_rate_avg需要传入到后期的计算中
    for i, j in cate_rate.items():
        cate_rate_avg[i] = round(np.mean(j), 2)

    ## 读取测试集的每一行，并进行预测
    for index, record in dt_test.iterrows():
        predict_buytime(record, useRate_cate, cate_rate_avg, buyTime_T, maxprob_buysku)

# --------------------------------------------------------结果表保存，但需要有波动，所以需要加±2天的数据
    ## 将测试集的每一条预测后的数据保存为结果表
    # for p_label, time in buyTime_T.items():
    #     if p_label in buyTime_T and p_label in maxprob_buysku:
    #         fuid,cate=p_label.split('_on_')
    #         Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append(time),Skuid.append(maxprob_buysku[p_label])
    #     else:
    #         print('暂时无法预测')
    # result_pd = pd.DataFrame({"Fuid": Fuid, "Cate_maxlike": Cate_maxlike, "Prob_time": Prob_time,"Skuid":Skuid})
    # result_pd.to_csv("predict_yynf.csv",encoding="utf_8_sig")
    # print(result_pd)
# --------------------------------------------------------结果表保存，但需要有波动，所以需要加±2天的数据

    # 将测试集的每一条预测后的数据保存为结果表
    for p_label, time in buyTime_T.items():
        if p_label in buyTime_T and p_label in maxprob_buysku:
            fuid,cate=p_label.split('_on_')
    # 需要取正负两天的数据并保存
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)-pd.to_timedelta(3, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)-pd.to_timedelta(2, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)-pd.to_timedelta(1, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)-pd.to_timedelta(0, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)+pd.to_timedelta(1, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)+pd.to_timedelta(2, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
            Fuid.append(fuid),Cate_maxlike.append(cate),Prob_time.append((pd.to_datetime(time)+pd.to_timedelta(3, unit='D')).strftime('%Y-%m-%d')),Skuid.append(maxprob_buysku[p_label])
        else:
            print('暂时无法预测')
    result_pd = pd.DataFrame({"Fuid": Fuid, "Cate_maxlike": Cate_maxlike, "Prob_time": Prob_time,"Skuid":Skuid})
    result_pd.to_csv("predict_yynf.csv",encoding="utf_8_sig")
    print(result_pd)
