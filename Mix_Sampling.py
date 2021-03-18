import numpy as np
import pandas as pd
import cnn_tool as tool


def mix_sampling(df_data, train_ratio, test_ratio):
    """
    # 학습에 사용될 Data를 정탐과 오탐으로 조건 샘플링 + 각 row 평균 or 표준편차 균등 샘플링
    # Date 20.02.14
    # @param df_data 학습에 사용될 dataframe 형식 데이터
    # @param train_ratio 학습 데이터 비율
    # @param test_ratio 테스트 데이터 비율
    # @param mix_type 균등 샘플링 대상 인자값
    # @return 샘플링된 train_set, test_set
    """
    # column = len(df_data.columns)


    # 상품
    class1_index = df_data[df_data['label'] == 0]
    class1_index = class1_index['label']

    # 뉴스
    class2_index = df_data[df_data['label'] == 1]
    class2_index = class2_index['label']

    """
    # 평균
    class1_index = class1_index.mean(1).sort_values().index
    """

    total_train_count = int(len(df_data) * train_ratio)
    total_test_count = int(len(df_data) * test_ratio)

    train_result1, test_result1 = divide(class1_index, train_ratio, df_data)
    train_result2, test_result2 = divide(class2_index, train_ratio, df_data)
    train_result = pd.concat([train_result1, train_result2])
    test_result = pd.concat([test_result1, test_result2])

    # x_train y_train x_test y_test 로 나누기
    y_train = train_result['label']
    y_train = y_train.values
    y_train = tool.make_output(y_train, threshold=1)

    x_train = train_result.drop(columns='label')
    x_train = x_train.values


    y_test = test_result['label']
    y_test = y_test.values
    y_test = tool.make_output(y_test, threshold=1)

    x_test = test_result.drop(columns='label')
    x_test = x_test.values
    return x_train, x_test, y_train, y_test


# target을 train_ratio, test_ratio 로 나눠서 traint_list, test_list 에 append 하는 함수
def divide(target, train_ratio, df_data):
    train_list = []
    test_list = []

    train_count = int(len(target) * train_ratio)

    for i in target.index:
        if train_count > len(train_list):
            train_list.append(df_data[df_data.index == i])
        else:
            test_list.append(df_data[df_data.index == i])
    train_result = pd.concat(train_list)
    test_result = pd.concat(test_list)

    return train_result, test_result
