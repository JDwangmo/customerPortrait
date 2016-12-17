# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-30'; 'last updated date: 2016-11-30'
    Email:   '383287471@qq.com'
    Describe: 预定义数据类处理函数
                - load_data(file_name, header=0): 加载数据
                - def show_attribute_detail( data=None, attribute_name=None): 显示数据某个属性的详情
                - show_df_info(data=None): 显示数据某个属性的详情
                - get_accept_content_type(x,return_first1 = False):  获取 ACCEPT_CONTENT 字段中 的工单类型
"""
from __future__ import print_function
from feature_encoder import FeatureEncoder
import pandas as pd
import numpy as np
import re
from collections import Counter

ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE = 0.01

ACCEPT_CONTENT_TYPE_EXTRA_PARAM = '_tag1ge5'
RF_N_ESTIMATORS = 100
print('RF_N_ESTIMATORS:%d' % RF_N_ESTIMATORS)

# 设置数据 的特征工具类
data_feature_encoder = FeatureEncoder(
    num_of_worker=True,
    num_of_search_action=True,
    max_num_month_search_action=True,
    # num_of_used_pay_mode=True,
    num_of_in_season4=True,
    # 违约次数
    num_of_rcvbl_penalty=True,
    normal_num_of_rcvbl_penalty=False,

    num_of_sensitive_workers=True,

    average_rcvbl_amt=True,
    normal_average_rcvbl_amt=True,
    # money_per_degree_std=True,
    month_pq_std=True,

    # is_connect_to_06table=True,
    # is_connect_to_07table=True,
    # is_connect_to_08table=True,
    # is_connect_to_09table=False,
    # 是否违约
    is_penalty=True,
    is_exceeding_rcvbl_ym_ge_1mon=True,

    # 是否包含缴费方式 020311
    is_pay_mode_contains_020311=True,
    # 是否包含缴费方式 010101
    is_pay_mode_contains_010101=True,
    # 是否包含缴费方式 020261
    is_pay_mode_contains_020261=True,

    is_rcvbl_amt_t_pq_equal0=True,
    is_rcvbl_amt_equal0_t_pq_grater0=True,
    is_rcvbl_amt_lower0=True,

    handle_year=False,
    handle_month=True,
    handle_day=True,
    handle_hour=True,

    # cust_no_3bit=True,
    # busi_type_code=True,
    urban_rural_flag=True,
    # city_org_no=False,
    elec_type=True,
    accept_content_type=True,
    multi_accept_content_type=True,
    accept_content_type_version='%s%s' % (ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE, ACCEPT_CONTENT_TYPE_EXTRA_PARAM),

    # cont_type_06table=False,
    # status_06table=False,

    cons_status_07table=True,
    rca_flag_07table=True,

    # org_no_7bit_09table=True,
    org_no_9bit_09table=True,

    # 用户最后一个月的缴费方式
    last_month_pay_mode_09table=True,
    last_month_pay_mode_4bit_09table=True,
    # 用户缴费方式 的 转变路线
    # pay_mode_change_clue_09table=True
    # pay_mode_4bit_change_clue_09table=True

    elec_degree_09table=True,

    is_hebiao_user_09table=True,
    is_elec_eq_zero_09table=False,
    is_seperate_time_09table=True,
    is_mid_change_09table=True,

)


def load_data(file_name, header=0, encoding='gbk', index_col=None, converters=None):
    """ 加载数据

    :param file_name: str
        文件名
    :param header:
        是否有头文件
    :return:
    """
    data = pd.read_csv(file_name,
                       sep='\t',
                       encoding=encoding,
                       header=header,
                       quoting=3,
                       converters=converters,
                       index_col=index_col,
                       )
    return data


def save_data(data, file_name, header=True, index=False):
    """ 保存数据

    :param index: bool
        是否保存索引
    :param data: pd.DataFrame()
    :param file_name: str
    :param header:
    :return:
    """
    data.to_csv(file_name, sep='\t',
                index=index,
                header=header,
                encoding='utf8')


def show_df_info(data=None):
    """显示数据某个属性的详情：

    :param data: pd.DataFrame()
        数据框
    """
    pass
    print(data.head())
    print(data.info())


def show_attribute_detail(
        data=None,
        attribute_name=None,
        split_by_tag=False,
        tag_name=None,
        show_pic=True,
        multi_items=False,
        multi_item_split_char='+',
):
    """显示数据某个属性的详情：
        - 画图

    :param multi_items: bool
        是否多项统计，即 属性值 中会 出现 '2户无电+3户无电' 等 ，以 '+'字符分割
    :param multi_item_split_char: str
        多项 分割符
    :param show_pic: bool
        是否画图
    :param data: pd.DataFrame()
        数据框
    :param attribute_name: str
        属性名
    :param split_by_tag: bool
        是否 根据标签来分别统计 ，要求在 tag_name中指定 TAG 属性名
    :param tag_name: str
        TAG 属性名
    """
    # region 检验参数
    assert (data is not None) or (attribute_name is not None), 'data is None or attribute_name is None!'
    assert attribute_name in data.columns, 'attribute_name not in data!'
    if multi_items:
        assert split_by_tag == True, 'set split_by_tag == True!'
    if split_by_tag:
        assert tag_name in data.columns, 'tag_name not in data!'
    # endregion


    # region 单项统计，即 每个属性值 都是只有 一项
    if not multi_items:
        # 不按tag_name 分开统计的情况下（即全部一起）
        total = data[attribute_name].value_counts().sort_index()

        if not split_by_tag:
            print(total.shape)
            if show_pic:
                total.plot(kind='bar')
            return total

        # 依次按不同的 tag_value 统计
        # tag_list = {}
        temp = pd.DataFrame(data={'total': total})
        for tag_value in data[tag_name].unique():
            tag1 = data.loc[data[tag_name] == tag_value, attribute_name].value_counts().sort_index()
            temp['tag%d' % tag_value] = tag1
            temp['tag%d/total' % tag_value] = tag1 / total
            # print(tag_value)
            # print('-' * 100)
        temp = temp.fillna(0)

    # endregion



    # region 多项统计，即每个属性值有 多项 ，需要切割
    else:
        total_type_dict = {}
        tag1_type_dict = {}
        tag0_type_dict = {}
        for item in data[attribute_name, tag_name].values:
            #     print item[-1],item[-2]
            if item[0] is None:
                continue
            for i in item[0].split('+'):
                if len(i) == 0:
                    continue
                i = i.strip()
                total_type_dict[i] = total_type_dict.get(i, 0) + 1
                if item[1] == 1:
                    tag1_type_dict[i] = tag1_type_dict.get(i, 0) + 1
                else:
                    tag0_type_dict[i] = tag0_type_dict.get(i, 0) + 1

        temp = pd.DataFrame(
            data={'key': total_type_dict.keys()}
        )

        temp['total'] = temp['accept_content_type'].map(total_type_dict)
        temp['tag1'] = temp['accept_content_type'].map(tag1_type_dict)
        temp['tag0'] = temp['accept_content_type'].map(tag0_type_dict)

        temp['tag0'] = temp['tag0'].fillna(0)
        temp['tag1'] = temp['tag1'].fillna(0)
        temp[['total', 'tag1', 'tag0']] = temp[['total', 'tag1', 'tag0']].astype(int)
        temp['tag1/total'] = temp['tag1'] / temp['total']

    # endregion
    print(temp.shape)

    if show_pic:
        if tag_name == 'TAG':
            temp['tag1/total'].plot(kind='bar')
        else:
            temp.plot(kind='bar', stacked=True)
    return temp


def get_highest_sensitive_workerid(iters, tag1_rate_dict):
    """根据 tag1_rate_dict 的 分数 来 返回最大的 APP_NO
    生成代表工单需要
    :param iters:
    :param tag1_rate_dict:
    :return:
    """
    highest_score = -2
    higest_x = None
    for item in iters:
        if item[-1] is None:
            score = -1
        else:
            score = tag1_rate_dict.get(item[-1], 0)

        if score > highest_score:
            highest_score = score
            higest_x = item
    return higest_x[0]


def get_accept_content_type(x, return_first1=False):
    """ 获取 ACCEPT_CONTENT 字段中 的工单类型
#     有两类，用规则找出：
#      1、【[^【]*?】     :    【】
#      2、^[^【]*?】$)   :     ....】    --- 注意这里前面要求无【
    :param return_first1: bool
        返回 第一个 匹配到的类型就好，默认 返回所有，并 + 连接
    :param x: str
        ACCEPT_CONTENT 记录
    :return:
    """
    result = re.findall(u'(【[^【]*?】|^[^【]*?】$)', x)
    # 有 出现【】但是中间是空内容的情况，在这里过略掉
    result = [item.strip(u'【】').strip() for item in result if len(item.strip(u'【】').strip()) > 0]

    if len(result) == 0:
        return None
    else:
        if return_first1:
            return result[0]
        else:
            result = list(set(result))
            return u'+'.join([item for item in result])


def truncate_org_no(x, truncate_len=7):
    """ 截断 表9 供电单位编号 ORGORG_NO 的长度

    :param x:
    :param truncate_len:
    :return:
    """
    # 截断
    org_no_list_after_truncate = [unicode(item)[:truncate_len] for item in x]
    if len(set(org_no_list_after_truncate)) == 1:
        return org_no_list_after_truncate[0]
    else:
        # >1
        count_dict = Counter(org_no_list_after_truncate)

        return count_dict.keys()[np.argmax(count_dict.values())]


def seperate_data_to_classifier(data, type='train', plan_type=1):
    """  将数据 根据条件进行细分 ，有两个方案
        1. 方案1
        2. 方案2

    :param plan_type: 选择哪种过滤规则
    :param type: 数据的类型
    :param data:
    :return:
    """
    # total shape
    print('total: %d' % data.shape[0])

    if plan_type == 1:
        data_sensitive, data_toclassify = seperate_data_to_classifier_plan1(data, type)

    elif plan_type == 2:
        data_sensitive, data_toclassify = seperate_data_to_classifier_plan2(data, type)

    elif plan_type == 3:
        data_sensitive, data_toclassify = seperate_data_to_classifier_plan3(data, type)

    else:
        raise NotImplementedError
    return data_toclassify, data_sensitive


def seperate_data_to_classifier_plan2(data, type):
    """ 数据细分 方案2

    :param data:
    :param type:
    :return:
    """
    # region 1. 过滤规则1 ： 根据 用户是否连接上表9来做过略
    # is_connect_to_09table == 1 则保留
    data_connect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 1]
    print('is_connect_to_09table == 1 : %d' % data_connect_to_09table.shape[0])
    # is_connect_to_09table == 0 则直接判为非敏感，过滤掉
    data_notconnect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 0]
    print('is_connect_to_09table == 0 : %d' % data_notconnect_to_09table.shape[0])
    # endregion

    # region 2. 过滤规则2 ： 根据 工单类型敏感度 tag1/total
    # 直接判为敏感
    print('data_connect_to_09table 进一步根据tag1/total过滤...')
    data_sensitive = data_connect_to_09table[data_connect_to_09table['tag1/total'] == 1]
    if type == 'train':
        print('tag1/total == 1 : %d --->直接判为敏感(敏感：%d,不敏感：%d)' % (data_sensitive.shape[0],
                                                                 sum(data_sensitive['TAG'] == 1),
                                                                 sum(data_sensitive['TAG'] == 0))
              )
    else:
        print('tag1/total == 1 : %d --->直接判为敏感' % data_sensitive.shape[0])

    # 直接判为非敏感
    data_nonsensitive = data_connect_to_09table[data_connect_to_09table['tag1/total'] == 0]
    if type == 'train':
        print('tag1/total == 0 : %d --->直接判为非敏感(敏感：%d,不敏感：%d)' % (data_nonsensitive.shape[0],
                                                                  sum(data_nonsensitive['TAG'] == 1),
                                                                  sum(data_nonsensitive['TAG'] == 0))
              )
    else:
        print('tag1/total == 0 : %d --->直接判为非敏感' % data_nonsensitive.shape[0])

    # 剩下送给分类器
    data_toclassify = data_connect_to_09table[data_connect_to_09table['tag1/total'] < 1]
    data_toclassify = data_toclassify[data_toclassify['tag1/total'] > 0]
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])

    # endregion
    return data_sensitive, data_toclassify


def seperate_data_to_classifier_plan3(data, type):
    """ 数据细分 方案3
        （1）is_connect_to_09table 1/0
        （2）ELEC_TYPE_IS_NONSENSITIVE
        （3）CONS_STATUS==3
        （4）CONT_TYPE==2
        （5）tag1/total == 1
        （6）tag1/total < 0.001判为非敏感
    :param data:
    :param type:
    :return:
    """
    data_sensitive = pd.DataFrame()

    # region 1. 过滤规则1 ： 根据 用户是否连接上表9来做过略
    # is_connect_to_09table == 1 则保留
    data_connect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 1]
    print('is_connect_to_09table == 1 : %d' % data_connect_to_09table.shape[0])
    # is_connect_to_09table == 0 则直接判为非敏感，过滤掉
    data_notconnect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 0]
    print('is_connect_to_09table == 0 : %d' % data_notconnect_to_09table.shape[0])

    data_toclassify = data_connect_to_09table
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])

    # endregion

    # region 2. 过滤规则2 ： 根据 用电类别 ELEC_TYPE 进行过滤
    print('根据 用电类别 ELEC_TYPE 进行过滤')
    # 不敏感用电类型表
    nonsensitive_elec_type_list = [504.0, 503.0, 500.0, 0.0, 404.0, 102.0, 101.0, 900.0]
    nonsensitive_elec_type_dict = {item: 1 for item in nonsensitive_elec_type_list}

    elec_type_is_nonsensitive = data_toclassify['ELEC_TYPE'].map(nonsensitive_elec_type_dict)
    data_elec_type_is_nonsensitive = data_toclassify[elec_type_is_nonsensitive.notnull()]

    if type == 'train':
        print('ELEC_TYPE_IS_NONSENSITIVE: %d(敏感：%d,不敏感：%d)' % (data_elec_type_is_nonsensitive.shape[0],
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 1),
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('ELEC_TYPE_IS_NONSENSITIVE: %d' % (data_elec_type_is_nonsensitive.shape[0]))
    data_toclassify = data_toclassify[elec_type_is_nonsensitive.isnull()]

    print('剩下: %d' % data_toclassify.shape[0])
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    # region 3. 过滤规则3 ： 根据 CONS_STATUS 和 CONT_TYPE 进行过滤
    # CONS_STATUS==3  ——> 规则直接判为非敏感
    data_cons_status_is_nonsensitive = data_toclassify[data_toclassify['CONS_STATUS'] == 3]
    if type == 'train':
        print('CONS_STATUS==3: %d ——> 规则直接判为非敏感(敏感：%d,不敏感：%d)' % (data_cons_status_is_nonsensitive.shape[0],
                                                                  sum(data_cons_status_is_nonsensitive['TAG'] == 1),
                                                                  sum(data_cons_status_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('CONS_STATUS==3: %d ——> 规则直接判为非敏感' % data_cons_status_is_nonsensitive.shape[0])

    # CONT_TYPE==2  ——> 规则直接判为非敏感
    data_cont_type_is_nonsensitive = data_toclassify[data_toclassify['CONT_TYPE'] == 2]
    if type == 'train':
        print('CONT_TYPE==2: %d ——> 规则直接判为非敏感(敏感：%d,不敏感：%d)' % (data_cont_type_is_nonsensitive.shape[0],
                                                                sum(data_cont_type_is_nonsensitive['TAG'] == 1),
                                                                sum(data_cont_type_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('CONT_TYPE==2: %d ——> 规则直接判为非敏感' % data_cont_type_is_nonsensitive.shape[0])

    data_toclassify = data_toclassify.loc[
        (data_toclassify['CONS_STATUS'] != 3) & (data_toclassify['CONT_TYPE'] != 2)]

    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    # region 4. 过滤规则4 ： 根据 工单类型敏感度 total 和tag1/total
    print('data_connect_to_09table 进一步根据total和tag1/total过滤...')

    #  'tag1/total' == 1 ——> 规则直接判为敏感
    data_tag1rate_eq1_sensitive = \
        data_toclassify.loc[data_toclassify['tag1/total_after09table'] == 1]
    data_sensitive = pd.concat([data_sensitive, data_tag1rate_eq1_sensitive], axis=0)
    if type == 'train':
        print('tag1/total == 1 ——> 规则直接判为敏感: %d(敏感：%d,不敏感：%d)' % (
            data_tag1rate_eq1_sensitive.shape[0],
            sum(data_tag1rate_eq1_sensitive['TAG'] == 1),
            sum(data_tag1rate_eq1_sensitive['TAG'] == 0))
              )
    else:
        print('tag1/total == 1 ——> 规则直接判为敏感: %d' % data_tag1rate_eq1_sensitive.shape[0])

    # 'tag1/total' < ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE ——> 规则直接判为不敏感
    data_lowtag1rate_nonsensitive = \
        data_toclassify.loc[
            data_toclassify['tag1/total_after09table'] < ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE]

    if type == 'train':
        print('tag1/total < %s ——> 规则直接判为非敏感: %d(敏感：%d,不敏感：%d)' % (ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE,
                                                                   data_lowtag1rate_nonsensitive.shape[0],
                                                                   sum(data_lowtag1rate_nonsensitive['TAG'] == 1),
                                                                   sum(data_lowtag1rate_nonsensitive['TAG'] == 0))
              )
    else:
        print('tag1/total < %s ——> 规则直接判为非敏感: %d' % (
            ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE, data_lowtag1rate_nonsensitive.shape[0]))

    data_toclassify = data_toclassify.loc[(data_toclassify['tag1/total_after09table'] != 1) & (
        data_toclassify['tag1/total_after09table'] >= ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE)]

    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])

    # endregion

    # region 5. 过滤规则5 ： 根据 所属市（区）公司供电单位编码 CITY_ORG_NO 进行过滤(弃用)
    # print('根据 所属市（区）公司供电单位编码 CITY_ORG_NO 进行过滤')
    # nonsensitive_city_org_no_list = [33409, 33410, 33411]
    # nonsensitive_city_org_no_dict = {item: 1 for item in nonsensitive_city_org_no_list}
    # data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'] = data_toclassify['CITY_ORG_NO'].map(
    #     nonsensitive_city_org_no_dict)
    # data_city_org_no_is_nonsensitive = data_toclassify[data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'].notnull()]
    # if type == 'train':
    #     print('CITY_ORG_NO_IS_NONSENSITIVE: %d(敏感：%d,不敏感：%d)' % (data_city_org_no_is_nonsensitive.shape[0],
    #                                                              sum(data_city_org_no_is_nonsensitive['TAG'] == 1),
    #                                                              sum(data_city_org_no_is_nonsensitive['TAG'] == 0))
    #           )
    # else:
    #     print('CITY_ORG_NO_IS_NONSENSITIVE: %d' % data_city_org_no_is_nonsensitive.shape[0])
    # data_toclassify = data_toclassify[data_toclassify['ELEC_TYPE_IS_NONSENSITIVE'].isnull()]
    # data_toclassify = data_toclassify[data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'].isnull()]
    #
    # if type == 'train':
    #     print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
    #                                           sum(data_toclassify['TAG'] == 1),
    #                                           sum(data_toclassify['TAG'] == 0),
    #                                           sum(data_toclassify['TAG'] == 0) / float(
    #                                               sum(data_toclassify['TAG'] == 1)))
    #           )
    # else:
    #     print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    return data_sensitive, data_toclassify


def seperate_data_to_classifier_plan4(data, type):
    """ 数据细分 方案3
        （1）is_connect_to_09table 1/0
        （2）ELEC_TYPE_IS_NONSENSITIVE
        （3）CONS_STATUS==3
        （4）


        （4）CONT_TYPE==2
        （5）tag1/total == 1
        （6）tag1/total < 0.001判为非敏感
    :param data:
    :param type:
    :return:
    """
    data_sensitive = pd.DataFrame()

    # region 1. 过滤规则1 ： 根据 用户是否连接上表9来做过略
    # is_connect_to_09table == 1 则保留
    data_connect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 1]
    print('is_connect_to_09table == 1 : %d' % data_connect_to_09table.shape[0])
    # is_connect_to_09table == 0 则直接判为非敏感，过滤掉
    data_notconnect_to_09table = data.loc[data['IS_CONNECT_TO_09TABLE'] == 0]
    print('is_connect_to_09table == 0 : %d' % data_notconnect_to_09table.shape[0])

    data_toclassify = data_connect_to_09table
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])

    # endregion

    # region 2. 过滤规则2 ： 根据 用电类别 ELEC_TYPE 进行过滤
    print('根据 用电类别 ELEC_TYPE 进行过滤')
    # 不敏感用电类型表
    nonsensitive_elec_type_list = [504.0, 503.0, 500.0, 0.0, 404.0, 102.0, 101.0, 900.0]
    nonsensitive_elec_type_dict = {item: 1 for item in nonsensitive_elec_type_list}

    elec_type_is_nonsensitive = data_toclassify['ELEC_TYPE'].map(nonsensitive_elec_type_dict)
    data_elec_type_is_nonsensitive = data_toclassify[elec_type_is_nonsensitive.notnull()]

    if type == 'train':
        print('ELEC_TYPE_IS_NONSENSITIVE: %d(敏感：%d,不敏感：%d)' % (data_elec_type_is_nonsensitive.shape[0],
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 1),
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('ELEC_TYPE_IS_NONSENSITIVE: %d' % (data_elec_type_is_nonsensitive.shape[0]))
    data_toclassify = data_toclassify[elec_type_is_nonsensitive.isnull()]

    print('剩下: %d' % data_toclassify.shape[0])
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    # region 3. 过滤规则3 ： 根据 CONS_STATUS 和 CONT_TYPE 进行过滤
    # CONS_STATUS==3  ——> 规则直接判为非敏感
    data_cons_status_is_nonsensitive = data_toclassify[data_toclassify['CONS_STATUS'] == 3]
    if type == 'train':
        print('CONS_STATUS==3: %d ——> 规则直接判为非敏感(敏感：%d,不敏感：%d)' % (data_cons_status_is_nonsensitive.shape[0],
                                                                  sum(data_cons_status_is_nonsensitive['TAG'] == 1),
                                                                  sum(data_cons_status_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('CONS_STATUS==3: %d ——> 规则直接判为非敏感' % data_cons_status_is_nonsensitive.shape[0])

    # CONT_TYPE==2  ——> 规则直接判为非敏感
    data_cont_type_is_nonsensitive = data_toclassify[data_toclassify['CONT_TYPE'] == 2]
    if type == 'train':
        print('CONT_TYPE==2: %d ——> 规则直接判为非敏感(敏感：%d,不敏感：%d)' % (data_cont_type_is_nonsensitive.shape[0],
                                                                sum(data_cont_type_is_nonsensitive['TAG'] == 1),
                                                                sum(data_cont_type_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('CONT_TYPE==2: %d ——> 规则直接判为非敏感' % data_cont_type_is_nonsensitive.shape[0])

    data_toclassify = data_toclassify.loc[
        (data_toclassify['CONS_STATUS'] != 3) & (data_toclassify['CONT_TYPE'] != 2)]

    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    # region 4. 过滤规则4 ： 根据 工单类型敏感度 total 和tag1/total
    print('data_connect_to_09table 进一步根据total和tag1/total过滤...')

    #  'tag1/total' == 1 ——> 规则直接判为敏感
    data_tag1rate_eq1_sensitive = \
        data_toclassify.loc[data_toclassify['tag1/total_after09table'] == 1]
    data_sensitive = pd.concat([data_sensitive, data_tag1rate_eq1_sensitive], axis=0)
    if type == 'train':
        print('tag1/total == 1 ——> 规则直接判为敏感: %d(敏感：%d,不敏感：%d)' % (
            data_tag1rate_eq1_sensitive.shape[0],
            sum(data_tag1rate_eq1_sensitive['TAG'] == 1),
            sum(data_tag1rate_eq1_sensitive['TAG'] == 0))
              )
    else:
        print('tag1/total == 1 ——> 规则直接判为敏感: %d' % data_tag1rate_eq1_sensitive.shape[0])

    # 'tag1/total' < ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE ——> 规则直接判为不敏感
    data_lowtag1rate_nonsensitive = \
        data_toclassify.loc[
            data_toclassify['tag1/total_after09table'] < ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE]

    if type == 'train':
        print('tag1/total < %s ——> 规则直接判为非敏感: %d(敏感：%d,不敏感：%d)' % (ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE,
                                                                   data_lowtag1rate_nonsensitive.shape[0],
                                                                   sum(data_lowtag1rate_nonsensitive['TAG'] == 1),
                                                                   sum(data_lowtag1rate_nonsensitive['TAG'] == 0))
              )
    else:
        print('tag1/total < %s ——> 规则直接判为非敏感: %d' % (
            ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE, data_lowtag1rate_nonsensitive.shape[0]))

    data_toclassify = data_toclassify.loc[(data_toclassify['tag1/total_after09table'] != 1) & (
        data_toclassify['tag1/total_after09table'] >= ACCEPT_CONTENT_TYPE_NONSENSITIVE_RATE)]

    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])

    # endregion

    return data_sensitive, data_toclassify


def seperate_data_to_classifier_plan1(data, type):
    """ 将数据 根据条件进行细分的版本1 ，规则如下：
        1. 过滤规则1 ： 根据 小工单类型（ACCEPT_CONTENT_TYPE） 的 total 和 tag1/total
            - 训练集数量<300的，直接按照敏感比例（也即是tag1/total）是否大于 0.122全部判为敏感或非敏感
            - 训练集数量>=300，并且敏感比例（也即是tag1/total）小于0.03的全部判为非敏感

    :param data:
    :param type:
    :return:
    """
    # region 1. 过滤规则1 ： 根据 小工单类型（ACCEPT_CONTENT_TYPE） 的 total 和 tag1/total
    # total >= 300
    data_ge300 = data.loc[data['total'] >= 300]
    print('total >= 300: %d' % data_ge300.shape[0])
    # total < 300
    data_lt300 = data.loc[data['total'] < 300]
    print('total < 300: %d' % data_lt300.shape[0])
    # total < 300  &&  'tag1/total' > 0.122 ——> 规则直接判为敏感
    data_lt300_sensitive = \
        data_lt300.loc[data_lt300['tag1/total'] >= 0.122]
    print('total < 300  &&  tag1/total > 0.122 ——> 规则直接判为敏感: %d' % data_lt300_sensitive.shape[0])
    # total < 300  &&  'tag1/total' < 0.122 ——> 规则直接判为非敏感
    data_lt300_nonsensitive = data_lt300.loc[data_lt300['tag1/total'] < 0.122]
    print('total < 300  &&  tag1/total < 0.122 ——> 规则直接判为非敏感: %d' % data_lt300_nonsensitive.shape[0])
    # total >= 300  &&  'tag1/total' < 0.03 ——> 规则直接判为非敏感
    data_ge300_nonsensitive = data_ge300.loc[data_ge300['tag1/total'] < 0.03]
    print('total >= 300  &&  tag1/total < 0.03 ——> 规则直接判为非敏感: %d' % data_ge300_nonsensitive.shape[0])
    # total >= 300  &&  'tag1/total' >= 0.03 ——> 留给 分类器 解决
    data_ge300_toclassify = data_ge300.loc[data_ge300['tag1/total'] >= 0.03]
    print('total >= 300  &&  tag1/total >= 0.03 ——> 留给 分类器 解决: %d' % data_ge300_toclassify.shape[0])
    # 另外，空 小工单类型 数据 也留给 分类器解决
    data_contenttype_isnull = data.loc[data['ACCEPT_CONTENT_TYPE'].isnull()]
    print('另外，空 小工单类型 数据  ——> 也留给 分类器 解决: %d' % data_contenttype_isnull.shape[0])
    # 需要进一步处理的数据
    data_toclassify = pd.concat([data_ge300_toclassify, data_contenttype_isnull], axis=0)
    print('留给 分类器 解决: %d' % data_toclassify.shape[0])
    # endregion
    # region 2. 过滤规则2 ： 根据 用电类别 ELEC_TYPE 进行过滤
    print('根据 用电类别 ELEC_TYPE 进行过滤')
    # 不敏感用电类型表
    nonsensitive_elec_type_list = [504.0, 503.0, 500.0, 0.0, 404.0, 102.0, 101.0, 900.0]
    nonsensitive_elec_type_dict = {item: 1 for item in nonsensitive_elec_type_list}
    data_toclassify['ELEC_TYPE_IS_NONSENSITIVE'] = data_toclassify['ELEC_TYPE'].map(nonsensitive_elec_type_dict)
    data_elec_type_is_nonsensitive = data_toclassify[data_toclassify['ELEC_TYPE_IS_NONSENSITIVE'].notnull()]
    if type == 'train':
        print('ELEC_TYPE_IS_NONSENSITIVE: %d(敏感：%d,不敏感：%d)' % (data_elec_type_is_nonsensitive.shape[0],
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 1),
                                                               sum(data_elec_type_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('ELEC_TYPE_IS_NONSENSITIVE: %d' % (data_elec_type_is_nonsensitive.shape[0]))
    data_toclassify = data_toclassify[data_toclassify['ELEC_TYPE_IS_NONSENSITIVE'].isnull()]
    print('剩下: %d' % data_toclassify.shape[0])
    # endregion
    # region 3. 过滤规则3 ： 根据 所属市（区）公司供电单位编码 CITY_ORG_NO 进行过滤
    print('根据 所属市（区）公司供电单位编码 CITY_ORG_NO 进行过滤')
    nonsensitive_city_org_no_list = [33409, 33410, 33411]
    nonsensitive_city_org_no_dict = {item: 1 for item in nonsensitive_city_org_no_list}
    data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'] = data_toclassify['CITY_ORG_NO'].map(nonsensitive_city_org_no_dict)
    data_city_org_no_is_nonsensitive = data_toclassify[data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'].notnull()]
    if type == 'train':
        print('CITY_ORG_NO_IS_NONSENSITIVE: %d(敏感：%d,不敏感：%d)' % (data_city_org_no_is_nonsensitive.shape[0],
                                                                 sum(data_city_org_no_is_nonsensitive['TAG'] == 1),
                                                                 sum(data_city_org_no_is_nonsensitive['TAG'] == 0))
              )
    else:
        print('CITY_ORG_NO_IS_NONSENSITIVE: %d' % data_city_org_no_is_nonsensitive.shape[0])
    data_toclassify = data_toclassify[data_toclassify['ELEC_TYPE_IS_NONSENSITIVE'].isnull()]
    data_toclassify = data_toclassify[data_toclassify['CITY_ORG_NO_IS_NONSENSITIVE'].isnull()]
    if type == 'train':
        print('剩下: %d(敏感：%d,不敏感：%d，比例：%f)' % (data_toclassify.shape[0],
                                              sum(data_toclassify['TAG'] == 1),
                                              sum(data_toclassify['TAG'] == 0),
                                              sum(data_toclassify['TAG'] == 0) / float(
                                                  sum(data_toclassify['TAG'] == 1)))
              )
    else:
        print('剩下: %d' % data_toclassify.shape[0])
    # endregion

    return data_lt300_sensitive, data_toclassify


def train_test_split(data, split_rate=0.7):
    np.random.RandomState(0).permutation(data.loc[data['TAG'] == 1].as_matrix())


def extend_train_data(X_train, y_train, n):
    '''将数据的正例 放大 n倍

    :param X_train:
    :param y_train:
    :param n:
    :return:
    '''
    X_mul7 = np.concatenate([X_train[y_train == 1]] * n + [X_train[y_train == 0]])
    y_mul7 = np.concatenate([y_train[y_train == 1]] * n + [y_train[y_train == 0]])

    X_mul7_1 = np.random.RandomState(1).permutation(X_mul7)
    y_mul7_1 = np.random.RandomState(1).permutation(y_mul7)
    return X_mul7_1, y_mul7_1


def model_train(model_name='rf',
                train_X=None,
                train_y=None
                ):
    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=10, random_state=0)

    elif model_name == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='lbfgs', tol=1e-5, max_iter=100, verbose=1, C=5)
    else:
        raise NotImplementedError

    model.fit(train_X, train_y)

    return model


def model_predict(model, test_X, test_y, verbose=2):
    """
        模型预测
    :param model:
    :param test_X:
    :param test_y:
    :return:
    """
    y_predict = model.predict(test_X)
    TP, FP, TN, FN = get_metrics(list(test_y), y_predict, verbose)
    return y_predict, TP, FP, TN, FN


def get_metrics(true_y, predict_y, verbose=2):
    """
        结果度量
    :param true_y:
    :param predict_y:
    :return:
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    TP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 1))
    FP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 0))
    TN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 0))
    FN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 1))
    if verbose > 1:
        print('total:%d' % len(true_y))
        print('TP:%d,FP:%d,TN:%d,FN:%d' % (TP, FP, TN, FN))
        print('f1_score:%f' % f1_score(true_y, predict_y))
        print('precision_score:%f' % precision_score(true_y, predict_y))
        print('recall_score:%f' % recall_score(true_y, predict_y))
        print('accu:%f' % np.mean(true_y == predict_y))

        print('预测为敏感：%d，不敏感：%d' % (sum(predict_y == 1), sum(predict_y == 0)))

    return TP, FP, TN, FN


def get_metrics2(true_y, predict_y):
    TP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 1))
    FP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 0))
    TN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 0))
    FN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 1))
    return (TP, FP, TN, FN)
