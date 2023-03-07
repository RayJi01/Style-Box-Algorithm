import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


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
        f = 100 * [1 + (math.log(i)-math.log(x))/(math.log(y)-math.log(x))]
        raw2.append(f)
    return raw2


def main():
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
    # freq_c1 = df_cf_z1['TC'].sum()
    # df_cf_z1.loc[:, 'TC'] = df_cf_z1.loc[:, 'TC'].div(freq_c1, axis='rows').copy()
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
    df_total_g = pd.DataFrame({'stock name': name, 'e_growth_score': e_g, 'b_growth_score': b_g, 'r_growth_score': r_g,
                               'c_growth_score': c_g, 'd_growth_score': d_g})
    df_total_v = pd.DataFrame({'stock name': name, 'e_v': e_value_score, 'b_v': b_value_score, 'r_v': r_value_score,
                               'c_v': c_value_score, 'd_v': d_value_score})
    # 通过特定的权重将每个个股的10项分数加权，再汇集到一个表格里面，得到两个最终分数：
    df_total_g.eval('final_score1 = e_growth_score * 0.5 + b_growth_score * 0.125 + r_growth_score * 0.125 + '
                    'c_growth_score * 0.125 + d_growth_score * 0.125', inplace=True)
    df_total_v.eval('final_score2 = e_v * 0.5 + b_v * 0.125 + r_v * 0.125 + c_v * 0.125 + d_v * 0.125', inplace=True)
    # 将两行final score提取出来单独做表：
    final_score1 = pd.Series(df_total_g['final_score1'])
    final_score2 = pd.Series(df_total_v['final_score2'])
    df_stock_score = pd.DataFrame({'stock name': name, 'G': final_score1, 'V': final_score2})
    scaler = MinMaxScaler()
    df_trained = scaler.fit(df_stock_score.loc[:, 2: -1])
    df_scaled = scaler.transform(df_trained)
    print(df_scaled)

# 计算 Raw X:
# 用新定义的calculate rawX的函数计算出每一组函数的rawX。
    df_stock_score.eval('VCG = G-V', inplace=True)
    r_x = rawX(df_stock_score['VCG'])
    df_stock_score['raw_X'] = pd.Series(r_x)
# 计算 raw Y:
    cap = df_eps_original['Total_Capitalization']
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
    print(df_stock_score. head(20))




main()
