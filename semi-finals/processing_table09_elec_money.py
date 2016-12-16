# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-15'; 'last updated date: 2016-12-15'
    Email:   '383287471@qq.com'
    Describe: 处理表 9 的 电价 信息
"""
from __future__ import print_function
from data_util_func import *


def count_elec_money_type(x):
    """计算电价类型，电价，是否分时，第几档用户
        （1）合表 --- 固定的0.558，如果发现有一个就可以认定了，把那个用户的“是否合表”标为1。这类用户就直接不用标“是否分时”和“达到的档位”了
        （2）下面是非合表 用户：
            1）先看第一个非空的月是不是 0.538（并且电量低于2760，一般很少这么多，第一个月），如果不是，一般就是分时客户；反之，是0.538的话就是非分时客户。
    :param x:
    :return:
    """
    # print(x)
    # quit()
    x = x.sort_values('RCVBL_YM')
    x = x[(x['MONEY_PER_DEGREE'] != 0) & (x['MONEY_PER_DEGREE'] != np.inf)]
    # 是否合表，是否分时，是否中途转变
    result = ''
    # 到过的最高档
    if sum(x['RCVBL_AMT'] - x['MONEY_558']) == 0:
        # 一致都是558 ---> 合表用户
        result += '1,0,0'
        highest_degree = 0
        return result + ',%d' % highest_degree
    else:
        # 非合表用户
        result += '0'
    year_total_elec = x['T_PQ'].sum()

    if (x['RCVBL_AMT'] - x['MONEY_538']).iloc[0] != 0 and x['T_PQ'].iloc[0] <= 2760:
        # 第一个非空的月 != 0.538 && 电量低于2760 --- >分时用户
        result += ',1'
        if year_total_elec <= 2760:
            # 1档，
            highest_degree = 1
            if sum(x['MONEY_PER_DEGREE'].isin([0.538])) > 0:
                # ---> 有中途转变
                result += ',1'
            else:
                result += ',0'
        elif 2760 < year_total_elec <= 4800:
            # 2档，
            highest_degree = 2
            if sum(x['MONEY_PER_DEGREE'].isin([0.538, 0.588])) > 0:
                # ---> 有中途转变
                result += ',1'
            else:
                result += ',0'
        else:
            # 3档，
            highest_degree = 3
            if sum(x['MONEY_PER_DEGREE'].isin([0.538, 0.588, 0.838])) > 0:
                # ---> 有中途转变
                result += ',1'
            else:
                result += ',0'
    else:
        # --- >非分时用户
        result += ',0'
        if sum(x['RCVBL_AMT'] < x['MONEY_538']) > 0:
            result += ',1'
        else:
            result += ',0'

        if year_total_elec <= 2760:
            highest_degree = 1
        elif 2760 < year_total_elec <= 4800:
            highest_degree = 2
        else:
            highest_degree = 3

    return result + ',%d' % highest_degree


def main(type='train'):
    if type=='train':
        train_data09_resident_df = load_data(
            'train_data09_resident_df.csv',
            index_col=0,
            header=0,
            encoding='utf8',
            converters={
                'CONS_NO': unicode,
            }
        )

        # print(train_data09_resident_df.head())

        train_data09_resident_cons_elec_money_type_series = train_data09_resident_df.groupby('CONS_NO').apply(
            count_elec_money_type)
        print(train_data09_resident_cons_elec_money_type_series.head())

        save_data(
            train_data09_resident_cons_elec_money_type_series,
            'train_data09_resident_cons_elec_money_type_series.csv',
            index=True,
        )
    elif type == 'test':
        test_data09_resident_df = load_data(
            'test_data09_resident_df.csv',
            index_col=0,
            header=0,
            encoding='utf8',
            converters={
                'CONS_NO': unicode,
            }
        )

        print(test_data09_resident_df.head())

        test_data09_resident_cons_elec_money_type_series = test_data09_resident_df[:100].groupby('CONS_NO').apply(
            count_elec_money_type)
        print(test_data09_resident_cons_elec_money_type_series.head())

        save_data(
            test_data09_resident_cons_elec_money_type_series,
            'test_data09_resident_cons_elec_money_type_series.csv',
            index=True,
        )
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # main(type='train')
    main(type='test')
