import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas_cache import pandas_cache


def outliers_detect(dataset):
    z_score = []
    for i in dataset:
        z = (i - np.mean(dataset)) / np.std(dataset)
        if z > 3:
            z_score.append(100)
        elif z < -3:
            z_score.append(200)
        else:
            z_score.append(z)
    return z_score


def weighted_sum(dataset1):
    w_sum = []
    for i in dataset1:
        s = (i - np.sum(dataset1)) ** 2
        w_sum.append(s)
    return w_sum


def Fi(dataset1, x):
    Fi_list = []
    for i in dataset1:
        s = 50 * (1 + (i - np.sum(dataset1) / (3 * x)))
        Fi_list.append(s)
    return Fi_list


# 对数据进行离差标准化，让score落在0到1之间，再扩大100倍。
def Normalization(dataset):
    N_list = []
    for i in dataset:
        i_new = i
        Ni = (i_new - np.min(dataset)) / (np.max(dataset) - np.min(dataset)) * 100
        N_list.append(Ni)
    return N_list


# 需要调整这个参数，要根据Z score来定：
def rawX(dataset3):
    raw = []
    for i in dataset3:
        z = (i - np.mean(dataset3)) / np.std(dataset3)
        x = 30 * (i - z) + 150
        raw.append(x)
    return raw


def rawY(dataset, x, y):
    raw2 = []
    for i in dataset:
        f = 100 * (1 + (math.log(i) - math.log(x)) / (math.log(y) - math.log(x)))
        raw2.append(f)
    return raw2


def main():
    # Step 1: 将所有的股票分为大中小三组:

    # 导入准备好的e, b, r, c, d数据集
    eps = pd.read_excel('data_EPS.xlsx')
    book_value = pd.read_excel('data_BookValue.xlsx')
    revenue = pd.read_excel('data_Revenue.xlsx')
    cash = pd.read_excel('data_CashFlow.xlsx')
    dividend = pd.read_excel('data_Dividend.xlsx')

    # 将导入好的数据集做成dataframe:
    df_eps = pd.DataFrame(eps)
    df_book_value = pd.DataFrame(book_value)
    df_revenue = pd.DataFrame(revenue)
    df_cash = pd.DataFrame(cash)
    df_dividend = pd.DataFrame(dividend)

    # 市场上总资本数目:
    total_cap = df_eps['Total_Cap'].sum()
    cap_nominal = df_eps['Total_Cap']
    stock_num = pd.Series(df_eps['Stock #'])
    # 将每个股票的资本数目除以整个市场总资本可以得到每一个股票占市场总额的频数（总频数为1），进而转换为CDF
    # Divide whole columns by the integer of total cap.
    df_eps['Total_Cap'] = df_eps['Total_Cap'].div(int(total_cap)).round(20)
    df_book_value['Total_Capitalization'] = df_book_value['Total_Capitalization'].div(int(total_cap)).round(20)
    df_revenue['Total_Capitalization'] = df_revenue['Total_Capitalization'].div(int(total_cap)).round(20)
    df_cash['Total_Capitalization'] = df_cash['Total_Capitalization'].div(int(total_cap)).round(20)
    df_dividend['Total_Capitalization'] = df_dividend['Total_Capitalization'].div(int(total_cap)).round(20)
    # Total_Capitalization 这一行已经是频数了
    df_eps['cumulative'] = df_eps['Total_Cap'].cumsum()
    # 将频数图的column中，提取0.7，0.9，1.0的部分，并找到他们对应的行数，再根据这些行将数据拆分

    # Step 2: Calculate the Growth Score of each stocks in each group:
    df_eps.columns = ['Stock #', 'Stock_name', 'TC', 'e0', 'e_1', 'e_2', 'e_3', 'e_4', 'e_prospect', 'P', 'cumulative']
    df_book_value.columns = ['Stock #', 'Stock_name', 'TC', 'b0', 'b_1', 'b_2', 'b_3', 'b_4', 'b_prospect', 'P']
    df_revenue.columns = ['Stock #', 'Stock_name', 'TC', 'r0', 'r_1', 'r_2', 'r_3', 'r_4', 'r_prospect', 'total_share',
                          'P']
    df_cash.columns = ['Stock #', 'Stock_name', 'TC', 'c0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_prospect', 'P']
    df_dividend.columns = ['Stock #', 'Stock_name', 'TC', 'd0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_prospect', 'P']

    # 对eps数据的处理：
    df_eps.eval('g_1 = ((e0)/(e_1))**1-1', inplace=True)
    df_eps.eval('g_2 = ((e0)/(e_2))**(1/2)-1', inplace=True)
    df_eps.eval('g_3 = ((e0)/(e_3))**(1/3)-1', inplace=True)
    df_eps.eval('g_4 = ((e0)/(e_4))**(1/4)-1', inplace=True)
    df_eps.eval('g_average = (g_1+g_2+g_3+g_4)/4', inplace=True)
    df_eps.eval('e1_p = (e_prospect)/(P)', inplace=True)

    # df_eps.replace(np.inf, 0, inplace=True)
    # df_eps.replace(np.nan, 0, inplace=True)

    # 对book value 数据的处理：
    df_book_value.eval('g_1 = ((b0)/(b_1))**1-1', inplace=True)
    df_book_value.eval('g_2 = ((b0)/(b_2))**(1/2)-1', inplace=True)
    df_book_value.eval('g_3 = ((b0)/(b_3))**(1/3)-1', inplace=True)
    df_book_value.eval('g_4 = ((b0)/(b_4))**(1/4)-1', inplace=True)
    df_book_value.eval('g_average = (b_1+b_2+b_3+b_4)/4', inplace=True)
    df_book_value.eval('b1_p = (b_prospect)/(P)', inplace=True)

    # df_book_value.replace(np.inf, 0, inplace=True)
    # df_book_value.replace(np.nan, 0, inplace=True)

    # 对cash flow 数据的处理：
    df_cash.eval('g_1 = ((c0)/(c_1))**1-1', inplace=True)
    df_cash.eval('g_2 = ((c0)/(c_2))**(1/2)-1', inplace=True)
    df_cash.eval('g_3 = ((c0)/(c_3))**(1/3)-1', inplace=True)
    df_cash.eval('g_4 = ((c0)/(c_4))**(1/4)-1', inplace=True)
    df_cash.eval('g_average = (c_1+c_2+c_3+c_4)/4', inplace=True)
    df_cash.eval('c1_p = (c_prospect)/(P)', inplace=True)

    # df_cash.replace(np.inf, 0, inplace=True)
    # df_cash.replace(np.nan, 0, inplace=True)

    # 对revenue数据的处理：
    df_revenue.eval('g_1 = ((r0)/(r_1))**1-1', inplace=True)
    df_revenue.eval('g_2 = ((r0)/(r_2))**(1/2)-1', inplace=True)
    df_revenue.eval('g_3 = ((r0)/(r_3))**(1/3)-1', inplace=True)
    df_revenue.eval('g_4 = ((r0)/(r_4))**(1/4)-1', inplace=True)
    df_revenue.eval('g_average = (r_1+r_2+r_3+r_4)/4', inplace=True)
    df_revenue.eval('r1_p = (r_prospect)/(total_share * P)', inplace=True)

    # df_revenue.replace(np.inf, 0, inplace=True)
    # df_revenue.replace(np.nan, 0, inplace=True)

    # 对dividend数据的处理：
    df_dividend.eval('g_1 = ((d0)/(d_1))**1-1', inplace=True)
    df_dividend.eval('g_2 = ((d0)/(d_2))**(1/2)-1', inplace=True)
    df_dividend.eval('g_3 = ((d0)/(d_3))**(1/3)-1', inplace=True)
    df_dividend.eval('g_4 = ((d0)/(d_4))**(1/4)-1', inplace=True)
    df_dividend.eval('g_average = (d_1+d_2+d_3+d_4)/4', inplace=True)
    df_dividend.eval('d1_p = (d_prospect)/(P)', inplace=True)

    # df_dividend.replace(np.inf, 0, inplace=True)
    # df_dividend.replace(np.nan, 0, inplace=True)

    # 将每一个股票根据市值进行分组计算：
    # EPS 分组拆分与计算：
    giant_breakpoint1 = df_eps.loc[df_eps['cumulative'] <= 0.7]
    medium_breakpoint1 = df_eps.loc[df_eps['cumulative'] <= 0.9]
    small_breakpoint1 = df_eps.loc[df_eps['cumulative'] <= 1.0]

    g_df = pd.DataFrame(giant_breakpoint1)
    m_df = pd.DataFrame(medium_breakpoint1)
    s_df = pd.DataFrame(small_breakpoint1)

    g_df.sort_values(by='g_average', ascending=True)
    m_df.sort_values(by='g_average', ascending=True)
    s_df.sort_values(by='g_average', ascending=True)
    # 每一个股加权后的值：
    g_df.eval('weighted_average_g = g_average * TC', inplace=True)
    m_df.eval('weighted_average_g = g_average * TC', inplace=True)
    s_df.eval('weighted_average_g = g_average * TC', inplace=True)
    g_df.eval('weighted_e1_p = e1_p * TC', inplace=True)
    m_df.eval('weighted_e1_p = e1_p * TC', inplace=True)
    s_df.eval('weighted_e1_p = e1_p * TC', inplace=True)

    # Book Value的拆分,整理，与计算：
    giant_breakpoint2 = df_book_value.loc[df_book_value['TC'] <= 0.7]
    medium_breakpoint2 = df_book_value.loc[df_book_value['TC'] <= 0.9]
    small_breakpoint2 = df_book_value.loc[df_book_value['TC'] <= 1.0]

    g_df2 = pd.DataFrame(giant_breakpoint2)
    m_df2 = pd.DataFrame(medium_breakpoint2)
    s_df2 = pd.DataFrame(small_breakpoint2)

    g_df2.sort_values(by='g_average', ascending=True)
    m_df2.sort_values(by='g_average', ascending=True)
    s_df2.sort_values(by='g_average', ascending=True)
    # 每一个股加权后的值：
    g_df2.eval('weighted_average_g = g_average * TC', inplace=True)
    m_df2.eval('weighted_average_g = g_average * TC', inplace=True)
    s_df2.eval('weighted_average_g = g_average * TC', inplace=True)
    g_df2.eval('weighted_b1_p = b1_p * TC', inplace=True)
    m_df2.eval('weighted_b1_p = b1_p * TC', inplace=True)
    s_df2.eval('weighted_b1_p = b1_p * TC', inplace=True)

    # Revenue 的拆分，整理，与计算：
    giant_breakpoint3 = df_revenue.loc[df_revenue['TC'] <= 0.7]
    medium_breakpoint3 = df_revenue.loc[df_revenue['TC'] <= 0.9]
    small_breakpoint3 = df_revenue.loc[df_revenue['TC'] <= 1.0]

    g_df3 = pd.DataFrame(giant_breakpoint3)
    m_df3 = pd.DataFrame(medium_breakpoint3)
    s_df3 = pd.DataFrame(small_breakpoint3)

    g_df3.sort_values(by='g_average', ascending=True)
    m_df3.sort_values(by='g_average', ascending=True)
    s_df3.sort_values(by='g_average', ascending=True)
    # 每一个股加权后的值：
    g_df3.eval('weighted_average_g = g_average * TC', inplace=True)
    m_df3.eval('weighted_average_g = g_average * TC', inplace=True)
    s_df3.eval('weighted_average_g = g_average * TC', inplace=True)
    g_df3.eval('weighted_r1_p = r1_p * TC', inplace=True)
    m_df3.eval('weighted_r1_p = r1_p * TC', inplace=True)
    s_df3.eval('weighted_r1_p = r1_p * TC', inplace=True)

    # 对cash flow 的拆分，整理与计算：
    giant_breakpoint4 = df_cash.loc[df_cash['TC'] <= 0.7]
    medium_breakpoint4 = df_cash.loc[df_cash['TC'] <= 0.9]
    small_breakpoint4 = df_cash.loc[df_cash['TC'] <= 1.0]

    g_df4 = pd.DataFrame(giant_breakpoint4)
    m_df4 = pd.DataFrame(medium_breakpoint4)
    s_df4 = pd.DataFrame(small_breakpoint4)

    g_df4.sort_values(by='g_average', ascending=True)
    m_df4.sort_values(by='g_average', ascending=True)
    s_df4.sort_values(by='g_average', ascending=True)
    # 每一个股加权后的值：
    g_df4.eval('weighted_average_g = g_average * TC', inplace=True)
    m_df4.eval('weighted_average_g = g_average * TC', inplace=True)
    s_df4.eval('weighted_average_g = g_average * TC', inplace=True)
    g_df4.eval('weighted_c1_p = c1_p * TC', inplace=True)
    m_df4.eval('weighted_c1_p = c1_p * TC', inplace=True)
    s_df4.eval('weighted_c1_p = c1_p * TC', inplace=True)

    # 对dividend yields的拆分，整理与计算：
    giant_breakpoint5 = df_dividend.loc[df_dividend['TC'] <= 0.7]
    medium_breakpoint5 = df_dividend.loc[df_dividend['TC'] <= 0.9]
    small_breakpoint5 = df_dividend.loc[df_dividend['TC'] <= 1.0]

    g_df5 = pd.DataFrame(giant_breakpoint5)
    m_df5 = pd.DataFrame(medium_breakpoint5)
    s_df5 = pd.DataFrame(small_breakpoint5)

    g_df5.sort_values(by='g_average', ascending=True)
    m_df5.sort_values(by='g_average', ascending=True)
    s_df5.sort_values(by='g_average', ascending=True)
    # 每一个股加权后的值：
    g_df5.eval('weighted_average_g = g_average * TC', inplace=True)
    m_df5.eval('weighted_average_g = g_average * TC', inplace=True)
    s_df5.eval('weighted_average_g = g_average * TC', inplace=True)
    g_df5.eval('weighted_d1_p = d1_p * TC', inplace=True)
    m_df5.eval('weighted_d1_p = d1_p * TC', inplace=True)
    s_df5.eval('weighted_d1_p = d1_p * TC', inplace=True)

    # 将分别做了分组的计算的数据集再拼回原来的5个大数据集:
    result_eps = g_df.append([m_df, s_df])
    result_bv = g_df2.append([m_df2, s_df2])
    result_r = g_df3.append([m_df3, s_df3])
    result_c = g_df4.append([m_df4, s_df4])
    result_d = g_df5.append([m_df5, s_df5])

    # Output the data we analyzed and calculate the SD and Mean in the excel
    result_eps.to_excel(r'E:\jirui\Python\PyCharm Community Edition 2021.1.1\data after analysis\EPS.xlsx', index=False,
                        header=True)
    result_bv.to_excel(r'E:\jirui\Python\PyCharm Community Edition 2021.1.1\data after analysis\BookValue.xlsx',
                       index=False, header=True)
    result_r.to_excel(r'E:\jirui\Python\PyCharm Community Edition 2021.1.1\data after analysis\Revenue.xlsx',
                      index=False, header=True)
    result_c.to_excel(r'E:\jirui\Python\PyCharm Community Edition 2021.1.1\data after analysis\CashFlow.xlsx',
                      index=False, header=True)
    result_d.to_excel(r'E:\jirui\Python\PyCharm Community Edition 2021.1.1\data after analysis\Dividend.xlsx',
                      index=False, header=True)

    # 第二部分： 计算每一只股票的分数以及坐标值：
    # 导入上一个程序已经分析好的5个Excel文档：
    eps = pd.read_excel('P1_EPS.xlsx')
    bv = pd.read_excel('P1_BookValue.xlsx')
    revenue = pd.read_excel('P1_Revenue.xlsx')
    cf = pd.read_excel('P1_CashFlow.xlsx')
    dividend = pd.read_excel('P1_Dividend.xlsx')
    eps_original = pd.read_excel('data_EPS.xlsx')  # 这个导入是为了读取total capitalization

    # 将读取的数据集转换为dataframe：
    df_eps = pd.DataFrame(eps)
    df_bv = pd.DataFrame(bv)
    df_revenue = pd.DataFrame(revenue)
    df_cf = pd.DataFrame(cf)
    df_dividend = pd.DataFrame(dividend)
    df_eps_original = pd.DataFrame(eps_original)
    # 将那些总市值不清楚的数据drop掉：

    # 处理 eps:
    # 将数据集中的0都处理掉，不选择：
    df_eps.weighted_average_g.drop(0)
    df_eps.weighted_e1_p.drop(0)
    # 这组数据的平均值是 df_eps. mean(), 标准差是 df_eps. mean(), 排除三个标准差以外的数.
    # 运用Z Score 的定义来做 （函数上方已经定义）：
    z_eps1 = outliers_detect(df_eps['weighted_average_g'])
    z_eps2 = outliers_detect(df_eps['weighted_e1_p'])
    df_eps['Z_avg'] = pd.Series(z_eps1)
    df_eps['Z_ep'] = pd.Series(z_eps2)
    # 将两个变量中在区间范围内和不在区间范围的值都挑出来：(我们函数给定的条件是过大的都赋值100，所以要分别排除是100的情况）
    # 同时过小的数据赋值200，所以要排除两次，再将这里面的数分别辅以最高和最低的z-score.)
    df_eps_z1 = df_eps[df_eps.Z_avg < 100]
    df_eps_z1_big = df_eps[df_eps.Z_avg == 100]
    df_eps_z1_small = df_eps[df_eps.Z_avg == 200]
    df_eps_z2 = df_eps[df_eps.Z_ep < 100]
    df_eps_z2_big = df_eps[df_eps.Z_ep == 100]
    df_eps_z2_small = df_eps[df_eps.Z_ep == 200]
    # 留下来的新频数 g_average：
    df_eps_z1 = df_eps_z1.copy()  # 这一步是避免我们在一个slice上直接操作，copy一下他自己。
    # freq_e1 = df_eps_z1['TC'].sum()
    # df_eps_z1.loc[:, 'TC'] = df_eps_z1.loc[:, 'TC'].div(freq_e1, axis='rows').copy()
    # new weighted_g and weighted_e:
    df_eps_z1.eval('new_weighted_g = weighted_average_g * TC', inplace=True)
    e_ws1 = weighted_sum(df_eps_z1['new_weighted_g'])
    df_eps_z1['weighted_sum'] = pd.Series(e_ws1).values
    df_eps_z1.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    w_std = (df_eps_z1['weighted_std_single'].sum()) ** 0.5
    df_eps_z1['score'] = Fi(df_eps_z1['new_weighted_g'], w_std)
    df_eps_z1['score'] = Normalization((df_eps_z1['score']))
    e_score1 = pd.Series(df_eps_z1['score'])
    # 给踢出去的大数和小数分别附上df_eps_z1的最大和最小score：
    df_eps_z1_big = df_eps_z1_big.copy()
    df_eps_z1_small = df_eps_z1_small.copy()
    a1 = max(pd.Series(e_score1))
    a2 = min(pd.Series(e_score1))
    df_eps_z1_big['score'] = a1
    df_eps_z1_small['score'] = a2
    df_eps_score1 = pd.concat([df_eps_z1, df_eps_z1_big, df_eps_z1_small])
    df_eps_score1.sort_index(axis=0, ascending=True, inplace=True)
    # 对ep重复上述操作：
    df_eps_z2 = df_eps_z2.copy()
    df_eps_z2_big = df_eps_z2_big.copy()
    df_eps_z2_small = df_eps_z2_small.copy()
    # freq_e2 = df_eps_z2['TC'].sum()
    # df_eps_z2.loc[:, 'TC'] = df_eps_z2.loc[:, 'TC'].div(freq_e2, axis='rows').copy()
    df_eps_z2.eval('new_weighted_ep = weighted_e1_p * TC', inplace=True)
    e_ws2 = weighted_sum(df_eps_z2['new_weighted_ep'])
    df_eps_z2['weighted_sum'] = pd.Series(e_ws2).values
    df_eps_z2.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    w_std2 = (df_eps_z2['weighted_std_single'].sum()) ** 0.5
    e_score2 = Fi(df_eps_z2['new_weighted_ep'], w_std2)
    e_score2N = Normalization(e_score2)
    df_eps_z2['score'] = pd.Series(e_score2N).values
    a3 = max(pd.Series(e_score2N))
    a4 = min(pd.Series(e_score2N))
    df_eps_z2_big['score'] = a3
    df_eps_z2_small['score'] = a4
    df_eps_score2 = pd.concat([df_eps_z2, df_eps_z2_big, df_eps_z2_small])
    df_eps_score2.sort_index(axis=0, ascending=True, inplace=True)

    # 用同样的方法处理Book Value:
    # The average growth score of book_value:
    df_bv.weighted_average_g.drop(0)
    df_bv.weighted_b1_p.drop(0)
    z_bv1 = outliers_detect(df_bv['weighted_average_g'])
    z_bv2 = outliers_detect(df_bv['weighted_b1_p'])
    df_bv['Z_avg'] = pd.Series(z_bv1)
    df_bv['Z_bp'] = pd.Series(z_bv2)
    df_bv_z1 = df_bv[df_bv.Z_avg < 100]
    df_bv_z1_big = df_bv[df_bv.Z_avg == 100]
    df_bv_z1_small = df_bv[df_bv.Z_avg == 200]
    df_bv_z2 = df_bv[df_bv.Z_bp < 100]
    df_bv_z2_big = df_bv[df_bv.Z_bp == 100]
    df_bv_z2_small = df_bv[df_bv.Z_bp == 200]
    df_bv_z1 = df_bv_z1.copy()
    df_bv_z1_big = df_bv_z1_big.copy()
    df_bv_z1_small = df_bv_z1_small.copy()
    df_bv_z1.eval('new_weighted_g = weighted_average_g * TC', inplace=True)
    b_ws1 = weighted_sum(df_bv_z1['new_weighted_g'])
    df_bv_z1['weighted_sum'] = pd.Series(b_ws1)
    df_bv_z1.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    bv_w_std1 = (df_bv_z1['weighted_std_single'].sum()) ** 0.5
    b_score1 = Fi(df_bv_z1['new_weighted_g'], bv_w_std1)
    b_score1N = Normalization(b_score1)
    df_bv_z1['score'] = pd.Series(b_score1N).values
    b1 = max(pd.Series(b_score1N))
    b2 = min(pd.Series(b_score1N))
    df_bv_z1_big['score'] = b1
    df_bv_z1_small['score'] = b2
    df_bv_score1 = pd.concat([df_bv_z1, df_bv_z1_big, df_bv_z1_small])
    df_bv_score1.sort_index(axis=0, ascending=True, inplace=True)
    # The prospect e1/p score of the book_value:
    df_bv_z2 = df_bv_z2.copy()
    df_bv_z2_big = df_bv_z2_big.copy()
    df_bv_z2_small = df_bv_z2_small.copy()
    df_bv_z2.eval('new_weighted_b1p = weighted_b1_p * TC', inplace=True)
    b_ws2 = weighted_sum(df_bv_z2['new_weighted_b1p'])
    df_bv_z2['weighted_sum'] = pd.Series(b_ws2)
    df_bv_z2.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    bv_w_std2 = (df_bv_z2['weighted_std_single'].sum()) ** 0.5
    b_score2 = Fi(df_bv_z2['new_weighted_b1p'], bv_w_std2)
    b_score2N = Normalization(b_score2)
    df_bv_z2['score'] = pd.Series(b_score2N).values
    b3 = max(pd.Series(b_score2N))
    b4 = min(pd.Series(b_score2N))
    df_bv_z2_big['score'] = b3
    df_bv_z2_small['score'] = b4
    df_bv_score2 = pd.concat([df_bv_z2, df_bv_z2_big, df_bv_z2_small])
    df_bv_score2.sort_index(axis=0, ascending=True, inplace=True)

    # 相同的方法处理 revenue：
    # 处理 revenue growth score：
    df_revenue.weighted_average_g.drop(0)
    df_revenue.weighted_r1_p.drop(0)
    z_revenue1 = outliers_detect(df_revenue['weighted_average_g'])
    z_revenue2 = outliers_detect(df_revenue['weighted_r1_p'])
    df_revenue['Z_avg'] = pd.Series(z_revenue1)
    df_revenue['Z_rp'] = pd.Series(z_revenue2)
    df_revenue_z1 = df_revenue[df_revenue.Z_avg < 100]
    df_revenue_z1_big = df_revenue[df_revenue.Z_avg == 100]
    df_revenue_z1_small = df_revenue[df_revenue.Z_avg == 200]
    df_revenue_z2 = df_revenue[df_revenue.Z_rp < 100]
    df_revenue_z2_big = df_revenue[df_revenue.Z_rp == 100]
    df_revenue_z2_small = df_revenue[df_revenue.Z_rp == 200]
    df_revenue_z1 = df_revenue_z1.copy()
    df_revenue_z1_big = df_revenue_z1_big.copy()
    df_revenue_z1_small = df_revenue_z1_small.copy()
    df_revenue_z1.eval('new_weighted_g = weighted_average_g * TC', inplace=True)
    r_ws1 = weighted_sum(df_revenue_z1['new_weighted_g'])
    df_revenue_z1['weighted_sum'] = pd.Series(r_ws1)
    df_revenue_z1.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    r_std_1 = (df_revenue_z1['weighted_std_single'].sum()) ** 0.5
    r_score1 = Fi(df_revenue_z1['new_weighted_g'], r_std_1)
    r_score1N = Normalization(r_score1)
    df_revenue_z1['score'] = pd.Series(r_score1N).values
    r1 = max(pd.Series(r_score1N))
    r2 = min(pd.Series(r_score1N))
    df_revenue_z1_big['score'] = r1
    df_revenue_z1_small['score'] = r2
    df_revenue_score1 = pd.concat([df_revenue_z1, df_revenue_z1_big, df_revenue_z1_small])
    df_revenue_score1.sort_index(axis=0, ascending=True, inplace=True)
    # 处理 revenue 的 prospective yields：
    df_revenue_z2 = df_revenue_z2.copy()
    df_revenue_z2_big = df_revenue_z2_big.copy()
    df_revenue_z2_small = df_revenue_z2_small.copy()
    df_revenue_z2.eval('new_weighted_r1p = weighted_r1_p * TC', inplace=True)
    r_ws2 = weighted_sum(df_revenue_z2['new_weighted_r1p'])
    df_revenue_z2['weighted_sum'] = pd.Series(r_ws2)
    df_revenue_z2.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    r_std_2 = (df_revenue_z2['weighted_std_single'].sum()) ** 0.5
    r_score2 = Fi(df_revenue_z2['new_weighted_r1p'], r_std_2)
    r_score2N = Normalization((r_score2))
    df_revenue_z2['score'] = pd.Series(r_score2N).values
    r3 = max(pd.Series(r_score2N))
    r4 = min(pd.Series(r_score2N))
    df_revenue_z2_big['score'] = r3
    df_revenue_z2_small['score'] = r4
    df_revenue_score2 = pd.concat([df_revenue_z2, df_revenue_z2_big, df_revenue_z2_small])
    df_revenue_score2.sort_index(axis=0, ascending=True, inplace=True)

    # 相同的办法处理cash flow:
    # 处理 cash flow的 growth score:
    df_cf.weighted_average_g.drop(0)
    df_cf.weighted_c1_p.drop(0)
    z_cf1 = outliers_detect(df_cf['weighted_average_g'])
    z_cf2 = outliers_detect(df_cf['weighted_c1_p'])
    df_cf['Z_avg'] = pd.Series(z_cf1)
    df_cf['Z_cp'] = pd.Series(z_cf2)
    df_cf_z1 = df_cf[df_cf.Z_avg < 100]
    df_cf_z1_big = df_cf[df_cf.Z_avg == 100]
    df_cf_z1_small = df_cf[df_cf.Z_avg == 200]
    df_cf_z2 = df_cf[df_cf.Z_cp < 100]
    df_cf_z2_big = df_cf[df_cf.Z_cp == 100]
    df_cf_z2_small = df_cf[df_cf.Z_cp == 200]
    df_cf_z1 = df_cf_z1.copy()
    df_cf_z1_big = df_cf_z1_big.copy()
    df_cf_z1_small = df_cf_z1_small.copy()
    df_cf_z1.eval('new_weighted_g = weighted_average_g * TC', inplace=True)
    c_ws1 = weighted_sum(df_cf_z1['new_weighted_g'])
    df_cf_z1['weighted_sum'] = pd.Series(c_ws1)
    df_cf_z1.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    cf_std_1 = (df_cf_z1['weighted_std_single'].sum()) ** 0.5
    c_score1 = Fi(df_cf_z1['new_weighted_g'], cf_std_1)
    c_score1N = Normalization(c_score1)
    df_cf_z1['score'] = pd.Series(c_score1N).values
    c1 = max(pd.Series(c_score1N))
    c2 = min(pd.Series(c_score1N))
    df_cf_z1_big['score'] = c1
    df_cf_z1_small['score'] = c2
    df_cf_score1 = pd.concat([df_cf_z1, df_cf_z1_big, df_cf_z1_small])
    df_cf_score1.sort_index(axis=0, ascending=True, inplace=True)
    # 处理 cash flow 的 prospective yields：
    df_cf_z2 = df_cf_z2.copy()
    df_cf_z2_big = df_cf_z2_big.copy()
    df_cf_z2_small = df_cf_z2_small.copy()
    df_cf_z2.eval('new_weighted_c1p = weighted_c1_p * TC', inplace=True)
    c_ws2 = weighted_sum(df_cf_z2['new_weighted_c1p'])
    df_cf_z2['weighted_sum'] = pd.Series(c_ws2)
    df_cf_z2.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    cf_std_2 = (df_cf_z2['weighted_std_single'].sum()) ** 0.5
    c_score2 = Fi(df_cf_z2['new_weighted_c1p'], cf_std_2)
    c_score2N = Normalization(c_score2)
    df_cf_z2['score'] = pd.Series(c_score2N).values
    c3 = max(pd.Series(c_score2N))
    c4 = min(pd.Series(c_score2N))
    df_cf_z2_big['score'] = c3
    df_cf_z2_small['score'] = c4
    df_cf_score2 = pd.concat([df_cf_z2, df_cf_z2_big, df_cf_z2_small])
    df_cf_score2.sort_index(axis=0, ascending=True, inplace=True)

    # 相同方法处理 dividend 的数据：
    # 处理 dividend 的 growth score：
    df_dividend.weighted_average_g.drop(0)
    df_dividend.weighted_d1_p.drop(0)
    z_dividend1 = outliers_detect(df_dividend['weighted_average_g'])
    z_dividend2 = outliers_detect(df_dividend['weighted_d1_p'])
    df_dividend['Z_avg'] = pd.Series(z_dividend1)
    df_dividend['Z_dp'] = pd.Series(z_dividend2)
    df_dividend_z1 = df_dividend[df_dividend.Z_avg < 100]
    df_dividend_z1_big = df_dividend[df_dividend.Z_avg == 100]
    df_dividend_z1_small = df_dividend[df_dividend.Z_avg == 200]
    df_dividend_z2 = df_dividend[df_dividend.Z_dp < 100]
    df_dividend_z2_big = df_dividend[df_dividend.Z_dp == 100]
    df_dividend_z2_small = df_dividend[df_dividend.Z_dp == 200]
    df_dividend_z1 = df_dividend_z1.copy()
    df_dividend_z1_big = df_dividend_z1_big.copy()
    df_dividend_z1_small = df_dividend_z1_small.copy()
    df_dividend_z1.eval('new_weighted_g = weighted_average_g * TC', inplace=True)
    d_ws1 = weighted_sum(df_dividend_z1['new_weighted_g'])
    df_dividend_z1['weighted_sum'] = pd.Series(d_ws1)
    df_dividend_z1.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    d_std_1 = (df_dividend_z1['weighted_std_single'].sum()) ** 0.5
    d_score1 = Fi(df_dividend_z1['new_weighted_g'], d_std_1)
    d_score1N = Normalization(d_score1)
    df_dividend_z1['score'] = pd.Series(d_score1N).values
    d1 = max(pd.Series(d_score1N))
    d2 = min(pd.Series(d_score1N))
    df_dividend_z1_big['score'] = d1
    df_dividend_z1_small['score'] = d2
    df_dividend_score1 = pd.concat([df_dividend_z1, df_dividend_z1_big, df_dividend_z1_small])
    df_dividend_score1.sort_index(axis=0, ascending=True, inplace=True)
    df_dividend_z2 = df_dividend_z2.copy()
    df_dividend_z2_big = df_dividend_z2_big.copy()
    df_dividend_z2_small = df_dividend_z2_small.copy()
    df_dividend_z2.eval('new_weighted_d1p = weighted_d1_p * TC', inplace=True)
    d_ws2 = weighted_sum(df_dividend_z2['new_weighted_d1p'])
    df_dividend_z2['weighted_sum'] = pd.Series(d_ws2)
    df_dividend_z2.eval('weighted_std_single = -(weighted_sum/((TC-1)/TC))', inplace=True)
    d_std_2 = (df_dividend_z2['weighted_std_single'].sum()) ** 0.5
    d_score2 = Fi(df_dividend_z2['new_weighted_d1p'], d_std_2)
    d_score2N = Normalization(d_score2)
    df_dividend_z2['score'] = pd.Series(d_score2N).values
    d3 = max(pd.Series(d_score2N))
    d4 = min(pd.Series(d_score2N))
    df_dividend_z2_big['score'] = d3
    df_dividend_z2_small['score'] = d4
    df_dividend_score2 = pd.concat([df_dividend_z2, df_dividend_z2_big, df_dividend_z2_small])
    df_dividend_score2.sort_index(axis=0, ascending=True, inplace=True)

    # 将 score整理出来，单独做表：
    # 分别提取10组dataframe中的数据，growth的五项指标和value的五项指标：
    name = pd.Series(df_eps['Stock_name'])
    e_g = pd.Series(df_eps_score1['score'])
    b_g = pd.Series(df_bv_score1['score'])
    r_g = pd.Series(df_revenue_score1['score'])
    c_g = pd.Series(df_cf_score1['score'])
    d_g = pd.Series(df_dividend_score1['score'])

    name = pd.Series(df_eps['Stock_name'])
    e_value_score = pd.Series(df_eps_score2['score'])
    b_value_score = pd.Series(df_bv_score2['score'])
    r_value_score = pd.Series(df_revenue_score2['score'])
    c_value_score = pd.Series(df_cf_score1['score'])
    d_value_score = pd.Series(df_dividend_score2['score'])

    # 将得到的score提取出来做成新的dataframe，这样便于计算处理：
    # Stock_num 变量是一开始line 86导入的时候就已经提出来了：
    df_total_g = pd.DataFrame({'stock #': stock_num, 'stock name': name, 'e_growth_score': e_g, 'b_growth_score': b_g,
                               'r_growth_score': r_g, 'c_growth_score': c_g, 'd_growth_score': d_g})
    df_total_v = pd.DataFrame({'stock #': stock_num, 'stock name': name, 'e_v': e_value_score, 'b_v': b_value_score,
                               'r_v': r_value_score, 'c_v': c_value_score, 'd_v': d_value_score})
    # 通过特定的权重将每个个股的10项分数加权，再汇集到一个表格里面，得到两个最终分数：
    df_total_g.eval('final_score1 = e_growth_score * 0.5 + b_growth_score * 0.125 + r_growth_score * 0.125 + '
                    'c_growth_score * 0.125 + d_growth_score * 0.125', inplace=True)
    df_total_v.eval('final_score2 = e_v * 0.5 + b_v * 0.125 + r_v * 0.125 + c_v * 0.125 + d_v * 0.125', inplace=True)
    # 将两行final score提取出来单独做表：
    final_score1 = pd.Series(df_total_g['final_score1'])
    final_score2 = pd.Series(df_total_v['final_score2'])
    df_stock_score = pd.DataFrame({'Stock #': stock_num, 'stock name': name, 'G': final_score1, 'V': final_score2})

    # 计算 Raw X:
    # 用新定义的calculate rawX的函数计算出每一组函数的rawX。
    df_stock_score.eval('VCG = G-V', inplace=True)
    r_x = rawX(df_stock_score['VCG'])
    df_stock_score['raw_X'] = pd.Series(r_x)
    # 计算 raw Y:
    cap = cap_nominal
    cap_freq = df_eps['TC']
    # 重新加原始的资产规模和资产规模占总的频数，用于分组：
    df_stock_score['cap'] = pd.Series(cap)
    df_stock_score['freq'] = pd.Series(cap_freq)
    # total cap 数据中提取 0.7分位， 0.9分位，1.0分位:
    # 将slice 的 dataframe附表做成新的dataframe，然后提取最后一行倒数第二列的cap指。
    location2 = df_stock_score.loc[df_stock_score['freq'].cumsum() <= 0.7]
    location1 = df_stock_score.loc[df_stock_score['freq'].cumsum() <= 0.9]
    df_location1 = pd.DataFrame(location1)
    df_location2 = pd.DataFrame(location2)
    a = df_location2.iloc[-1, :]
    b = df_location1.iloc[-1, :]
    # 因为没有办法提取倒数第二列，只能提取最后一行的series的倒数最后两项变为list，再通过这个选择第0项（第一个）提取出cap。
    list_a = a.tail(2)
    list_b = b.tail(2)
    cap1 = list_b[0]
    cap2 = list_a[0]
    # 运用预先编译好的函数 rawY 算出 rawY 的值：
    raw_Y = pd.Series(rawY(cap, cap1, cap2))
    df_stock_score['raw_Y'] = raw_Y
    # 把末尾的几个市值不清的小股票drop掉：
    df_st_score_clean = df_stock_score.dropna()

    # Part III：根据不同基金持有的股票对对应的股票进行选择和绘图：
    # 将最后的score数据做表，因为我们检索只需要这个数据：
    df_Stock = pd.DataFrame([df_st_score_clean['stock name'], df_st_score_clean['raw_X'], df_st_score_clean['raw_Y']])
    stock_index = pd.Series(df_st_score_clean['Stock #'])
    # 将所有的股票代码单独拿出来做个list，这样便于提取他们的index然后进行检索：
    list_index = list(stock_index)
    df_stockT = df_Stock.transpose()  # 因为取出来的把列都变成了行，所以要再把它transpose一下。
    X_score = pd.Series(df_st_score_clean['raw_X'])
    Y_score = pd.Series(df_st_score_clean['raw_Y'])
    stock_name = pd.Series(df_st_score_clean['stock name'])
    # 编写一个用户友好界面，可以让我们输入股票代码，并自动形成我们的基金：
    df_score = "n"
    score_listX = []
    score_listY = []
    name_list = []
    num = input("Enter the Stock #: ")
    num.replace(" ", "")  # 将可能出现的输入空格剔除掉
    while num in list_index:
        n = list_index.index(num)
        scoreX = X_score[n]
        scoreY = Y_score[n]  # 从pandas series中定位我们输入的股票代码名称对应的索引，在通过索引在pandas series中找元素。
        name_st = stock_name[n]  # 需要的元素有 rawX， rawY， 和对应的股票名称。
        score_listX.append(scoreX)
        score_listY.append(scoreY)
        name_list.append(name_st)
        score_seriesX = pd.Series(score_listX)
        score_seriesY = pd.Series(score_listY)
        name_series = pd.Series(name_list)
        df_score = pd.DataFrame([name_series, score_seriesX, score_seriesY])
        num = input("Enter the Stock #: ")
    else:
        print("Loading the data...")
        print("The fund is shown as: ")

    df_scoreT = df_score.transpose()
    df_scoreT.columns = ['name', 'x_cord', 'y_cord']
    df_score_plot = df_scoreT.iloc[:, 1:3]  # 将名字切割掉，以便做sklearn的学习
    scaler = MinMaxScaler()
    df_score_plot = df_score_plot.copy()
    # Pandas DataFrame 的 normalization 的 method 需要对列出的columns做处理，不能直接fit——transform一整个df到一个variables。
    df_score_plot[['x_cord', 'y_cord']] = scaler.fit_transform(df_score_plot[['x_cord', 'y_cord']])
    df_score_plot.eval('x_cord_norm = x_cord * 500 - 100', inplace=True)
    df_score_plot.eval('y_cord_norm = y_cord * 500 - 100', inplace=True)
    X = df_score_plot['x_cord_norm']
    Y = df_score_plot['y_cord_norm']
    print(df_score_plot)

    plt.scatter(X, Y)
    plt.xlabel('G & V score')
    plt.xticks(range(0, 500, 100), ['Deep Value', 'Value', 'Blend', 'Growth', 'Deep Growth'])
    plt.yticks(range(0, 500, 100), ['Micro', 'Small', 'Median', 'Large', 'Giant'])
    plt.ylabel('Cap score')
    plt.title('Style Box of the fund')
    plt.style.use(['dark_background', 'seaborn-darkgrid'])

    plt.xlim(min(X), max(X))
    plt.ylim(min(Y), max(Y))

    plt.grid(color='green', linewidth=0.5)
    plt.show()


main()

# Try Except 函数用来检查算法哪里出错了。
# try:
#     g_average_result = g_average(eps_list1)
#     print(g_average_result)
# except:
#     print("error")
