import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper


def data_preprocess(train_df,test_df):
    li = [i for i in range(15)]
    # drop happiness under zero training set
    train_df = train_df[(train_df['happiness'].isin(li[1:6]))]
    test_df['minor_child'] = test_df['minor_child'].fillna(0.0)
    # deal with inc_exp and income
    combine = [train_df, test_df]
    for dataset in combine:
        dataset.loc[dataset['income'] < 0, 'income'] = dataset.loc[
            dataset['income'] >= 0, 'income'].mean()
        dataset.loc[dataset['inc_exp'] < 0, 'inc_exp'] = dataset.loc[
            dataset['inc_exp'] >= 0, 'inc_exp'].mean()
    train_df['diff'] = train_df['inc_exp'] - train_df['income']
    test_df['diff'] = test_df['inc_exp'] - test_df['income']

    combine = [train_df, test_df]
    for datasets in combine:
        datasets.loc[(datasets['diff'] <= 0), 'diff'] = 0
        datasets.loc[(datasets['diff'] > 0) & (datasets['diff'] <= 10000), 'diff'] = 1
        datasets.loc[(datasets['diff'] > 10000) & (datasets['diff'] <= 30000), 'diff'] = 2
        datasets.loc[(datasets['diff'] > 30000) & (datasets['diff'] <= 90000), 'diff'] = 3
        datasets.loc[(datasets['diff'] > 90000), 'diff'] = 4

    train_df = train_df.drop('inc_exp', axis=1)
    test_df = test_df.drop('inc_exp', axis=1)

    # set ageband
    train_df['Age'] = 2015 - train_df['birth']
    test_df['Age'] = 2015 - test_df['birth']
    train_df.loc[train_df['Age'] <= 41, 'Age'] = 0
    train_df.loc[(train_df['Age'] > 41) & (train_df['Age'] <= 60), 'Age'] = 1
    train_df.loc[(train_df['Age'] > 60) & (train_df['Age'] <= 79), 'Age'] = 2
    train_df.loc[(train_df['Age'] > 79), 'Age'] = 3
    test_df.loc[test_df['Age'] <= 41, 'Age'] = 0
    test_df.loc[(test_df['Age'] > 41) & (test_df['Age'] <= 60), 'Age'] = 1
    test_df.loc[(test_df['Age'] > 60) & (test_df['Age'] <= 79), 'Age'] = 2
    test_df.loc[(test_df['Age'] > 79), 'Age'] = 3

    train_df = train_df.drop(['id', 'city', 'county', 'survey_time', 'birth',
                              'edu_other','edu_yr','property_other','invest_other',
                              'join_party','s_birth','marital_now','s_edu',
                              's_political','f_birth','f_edu','f_political','m_birth','m_edu','m_political',
                              'marital_1st'], axis=1)
    test_df = test_df.drop(['id', 'city', 'county', 'survey_time', 'birth',
                            'edu_other', 'edu_yr', 'property_other', 'invest_other',
                            'join_party','s_birth','marital_now','s_edu',
                            's_political','f_birth','f_edu','f_political','m_birth','m_edu','m_political',
                            'marital_1st'], axis=1)

    # use common number to fill in negative number
    combine = [train_df, test_df]
    for dataset in combine:
        dataset.loc[dataset['depression'].isin(li[1:6]) == False, 'depression'] = dataset['depression'].mode().iloc[0]
        dataset.loc[dataset['socialize'].isin(li[1:6]) == False, 'socialize'] = dataset['socialize'].mode().iloc[0]
        dataset.loc[dataset['relax'].isin(li[1:6]) == False, 'relax'] = dataset['relax'].mode().iloc[0]
        dataset.loc[dataset['learn'].isin(li[1:6]) == False, 'learn'] = dataset['learn'].mode().iloc[0]
        dataset.loc[dataset['equity'].isin(li[1:6]) == False, 'equity'] = dataset['equity'].mode().iloc[0]
        dataset.loc[dataset['class'].isin(li[1:11]) == False, 'class'] = dataset['class'].mode().iloc[0]
        dataset.loc[dataset['family_m'] < 0, 'family_m'] = dataset['family_m'].mode().iloc[0]
        dataset.loc[dataset['family_status'].isin(li[1:6]) == False, 'family_status'] = \
        dataset['family_status'].mode().iloc[0]
        dataset.loc[dataset['house'] < 0, 'house'] = dataset['house'].mode().iloc[0]
        dataset.loc[dataset['car'].isin(li[1:3]) == False, 'car'] = dataset['car'].mode().iloc[0]
        dataset.loc[dataset['status_peer'].isin(li[1:4]) == False, 'status_peer'] = \
                                                    dataset['status_peer'].mode().iloc[0]
        dataset.loc[dataset['status_3_before'].isin(li[1:4]) == False, 'status_3_before'] = \
        dataset['status_3_before'].mode().iloc[0]
        dataset.loc[dataset['view'].isin(li[1:6]) == False, 'view'] = dataset['view'].mode().iloc[0]
        dataset.loc[dataset['inc_ability'].isin(li[1:5]) == False, 'inc_ability'] = \
            dataset['inc_ability'].mode().iloc[0]
        dataset.loc[dataset['nationality'].isin(li[1:9]) == False, 'nationality'] = \
            dataset['nationality'].mode().iloc[0]
        dataset.loc[dataset['religion'].isin([0, 1]) == False, 'religion'] = \
            dataset['religion'].mode().iloc[0]
        dataset.loc[dataset['religion_freq'].isin(li[1:10]) == False, 'religion_freq'] = \
            dataset['religion_freq'].mode().iloc[0]
        dataset.loc[dataset['edu'].isin(li[1:]) == False, 'edu'] = \
            dataset['edu'].mode().iloc[0]
        dataset.loc[dataset['political'].isin(li[1:5]) == False, 'political'] = \
            dataset['political'].mode().iloc[0]
        dataset.loc[dataset['health'].isin(li[1:6]) == False, 'health'] = \
            dataset['health'].mode().iloc[0]
        dataset.loc[dataset['health_problem'].isin(li[1:6]) == False, 'health_problem'] = \
            dataset['health_problem'].mode().iloc[0]
        dataset.loc[dataset['edu_status'].isin(li[1:5]) == False, 'edu_status'] = dataset['edu_status'].mode().iloc[0]
        dataset.loc[dataset['media_1'].isin(li[1:6]) == False, 'media_1'] = dataset['media_1'].mode().iloc[0]
        dataset.loc[dataset['media_2'].isin(li[1:6]) == False, 'media_2'] = dataset['media_2'].mode().iloc[0]
        dataset.loc[dataset['media_3'].isin(li[1:6]) == False, 'media_3'] = dataset['media_3'].mode().iloc[0]
        dataset.loc[dataset['media_4'].isin(li[1:6]) == False, 'media_4'] = dataset['media_4'].mode().iloc[0]
        dataset.loc[dataset['media_5'].isin(li[1:6]) == False, 'media_5'] = dataset['media_5'].mode().iloc[0]
        dataset.loc[dataset['media_6'].isin(li[1:6]) == False, 'media_6'] = dataset['media_6'].mode().iloc[0]
        dataset.loc[dataset['leisure_1'].isin(li[1:6]) == False, 'leisure_1'] = dataset['leisure_1'].mode().iloc[0]
        dataset.loc[dataset['leisure_2'].isin(li[1:6]) == False, 'leisure_2'] = dataset['leisure_2'].mode().iloc[0]
        dataset.loc[dataset['leisure_3'].isin(li[1:6]) == False, 'leisure_3'] = dataset['leisure_3'].mode().iloc[0]
        dataset.loc[dataset['leisure_4'].isin(li[1:6]) == False, 'leisure_4'] = dataset['leisure_4'].mode().iloc[0]
        dataset.loc[dataset['leisure_5'].isin(li[1:6]) == False, 'leisure_5'] = dataset['leisure_5'].mode().iloc[0]
        dataset.loc[dataset['leisure_6'].isin(li[1:6]) == False, 'leisure_6'] = dataset['leisure_6'].mode().iloc[0]
        dataset.loc[dataset['leisure_7'].isin(li[1:6]) == False, 'leisure_7'] = dataset['leisure_7'].mode().iloc[0]
        dataset.loc[dataset['leisure_8'].isin(li[1:6]) == False, 'leisure_8'] = dataset['leisure_8'].mode().iloc[0]
        dataset.loc[dataset['leisure_9'].isin(li[1:6]) == False, 'leisure_9'] = dataset['leisure_9'].mode().iloc[0]
        dataset.loc[dataset['leisure_10'].isin(li[1:6]) == False, 'leisure_10'] = dataset['leisure_10'].mode().iloc[0]
        dataset.loc[dataset['leisure_11'].isin(li[1:6]) == False, 'leisure_11'] = dataset['leisure_11'].mode().iloc[0]
        dataset.loc[dataset['leisure_12'].isin(li[1:6]) == False, 'leisure_12'] = dataset['leisure_12'].mode().iloc[0]
        dataset.loc[dataset['social_neighbor'].isin(li[1:8]) == False, 'social_neighbor'] = \
        dataset['social_neighbor'].mode().iloc[0]
        dataset.loc[dataset['social_friend'].isin(li[1:8]) == False, 'social_friend'] = \
        dataset['social_friend'].mode().iloc[0]
        dataset.loc[dataset['socia_outing'].isin(li[1:7]) == False, 'socia_outing'] = \
        dataset['socia_outing'].mode().iloc[0]
        dataset.loc[dataset['class_10_before'].isin(li[1:11]) == False, 'class_10_before'] = \
        dataset['class_10_before'].mode().iloc[0]
        dataset.loc[dataset['class_10_after'].isin(li[1:11]) == False, 'class_10_after'] = \
        dataset['class_10_after'].mode().iloc[0]
        dataset.loc[dataset['class_14'].isin(li[1:11]) == False, 'class_14'] = dataset['class_14'].mode().iloc[0]
        dataset.loc[dataset['insur_1'].isin(li[1:3]) == False, 'insur_1'] = dataset['insur_1'].mode().iloc[0]
        dataset.loc[dataset['insur_2'].isin(li[1:3]) == False, 'insur_2'] = dataset['insur_2'].mode().iloc[0]
        dataset.loc[dataset['insur_3'].isin(li[1:3]) == False, 'insur_3'] = dataset['insur_3'].mode().iloc[0]
        dataset.loc[dataset['insur_4'].isin(li[1:3]) == False, 'insur_4'] = dataset['insur_4'].mode().iloc[0]
        dataset.loc[dataset['trust_1'].isin(li[1:6]) == False, 'trust_1'] = dataset['trust_1'].mode().iloc[0]
        dataset.loc[dataset['trust_2'].isin(li[1:6]) == False, 'trust_2'] = dataset['trust_2'].mode().iloc[0]
        dataset.loc[dataset['trust_3'].isin(li[1:6]) == False, 'trust_3'] = dataset['trust_3'].mode().iloc[0]
        dataset.loc[dataset['trust_4'].isin(li[1:6]) == False, 'trust_4'] = dataset['trust_4'].mode().iloc[0]
        dataset.loc[dataset['trust_5'].isin(li[1:6]) == False, 'trust_5'] = dataset['trust_5'].mode().iloc[0]
        dataset.loc[dataset['trust_6'].isin(li[1:6]) == False, 'trust_6'] = dataset['trust_6'].mode().iloc[0]
        dataset.loc[dataset['trust_7'].isin(li[1:6]) == False, 'trust_7'] = dataset['trust_7'].mode().iloc[0]
        dataset.loc[dataset['trust_8'].isin(li[1:6]) == False, 'trust_8'] = dataset['trust_8'].mode().iloc[0]
        dataset.loc[dataset['trust_9'].isin(li[1:6]) == False, 'trust_9'] = dataset['trust_9'].mode().iloc[0]
        dataset.loc[dataset['trust_10'].isin(li[1:6]) == False, 'trust_10'] = dataset['trust_10'].mode().iloc[0]
        dataset.loc[dataset['trust_11'].isin(li[1:6]) == False, 'trust_11'] = 3
        dataset.loc[dataset['trust_12'].isin(li[1:6]) == False, 'trust_12'] = 3
        dataset.loc[dataset['trust_13'].isin(li[1:6]) == False, 'trust_13'] = dataset['trust_13'].mode().iloc[0]
        dataset.loc[dataset['neighbor_familiarity'].isin(li[1:6]) == False, 'neighbor_familiarity'] = \
        dataset['neighbor_familiarity'].mode().iloc[0]
        dataset.loc[dataset['son'] < 0, 'son'] = 1
        dataset.loc[dataset['daughter'] < 0, 'daughter'] = 1
        dataset.loc[dataset['minor_child'] < 0, 'minor_child'] = 0
        dataset.loc[dataset['s_hukou'].isin(li[1:9]) == False, 's_hukou'] = dataset['s_hukou'].mode().iloc[0]
        dataset.loc[dataset['s_work_exper'].isin(li[1:7]) == False, 's_work_exper'] = \
        dataset['s_work_exper'].mode().iloc[0]
        dataset.loc[dataset['s_work_status'].isin(li[1:10]) == False, 's_work_status'] = \
        dataset['s_work_status'].mode().iloc[0]
        dataset.loc[dataset['s_work_type'].isin(li[1:3]) == False, 's_work_type'] = dataset['s_work_type'].mode().iloc[
            0]
        dataset.loc[dataset['f_work_14'].isin(li[1:18]) == False, 'f_work_14'] = dataset['f_work_14'].mode().iloc[0]
        dataset.loc[dataset['m_work_14'].isin(li[1:18]) == False, 'm_work_14'] = dataset['m_work_14'].mode().iloc[0]
        dataset.loc[dataset['public_service_1'] < 0, 'public_service_1'] = dataset.loc[
            dataset['public_service_1'] >= 0, 'public_service_1'].mean()
        dataset.loc[dataset['public_service_2'] < 0, 'public_service_2'] = dataset.loc[
            dataset['public_service_2'] >= 0, 'public_service_2'].mean()
        dataset.loc[dataset['public_service_3'] < 0, 'public_service_3'] = dataset.loc[
            dataset['public_service_3'] >= 0, 'public_service_3'].mean()
        dataset.loc[dataset['public_service_4'] < 0, 'public_service_4'] = dataset.loc[
            dataset['public_service_4'] >= 0, 'public_service_4'].mean()
        dataset.loc[dataset['public_service_5'] < 0, 'public_service_5'] = dataset.loc[
            dataset['public_service_5'] >= 0, 'public_service_5'].mean()
        dataset.loc[dataset['public_service_6'] < 0, 'public_service_6'] = dataset.loc[
            dataset['public_service_6'] >= 0, 'public_service_6'].mean()
        dataset.loc[dataset['public_service_7'] < 0, 'public_service_7'] = dataset.loc[
            dataset['public_service_7'] >= 0, 'public_service_7'].mean()
        dataset.loc[dataset['public_service_8'] < 0, 'public_service_8'] = dataset.loc[
            dataset['public_service_8'] >= 0, 'public_service_8'].mean()
        dataset.loc[dataset['public_service_9'] < 0, 'public_service_9'] = dataset.loc[
            dataset['public_service_9'] >= 0, 'public_service_9'].mean()
        dataset.loc[dataset['s_income'] < 0, 's_income'] = dataset.loc[dataset['s_income'] >= 0, 's_income'].mean()

    # cut floor area
    combine = [train_df, test_df]
    for datasets in combine:
        datasets.loc[(datasets['floor_area'] <= 63), 'floor_area'] = 0
        datasets.loc[(datasets['floor_area'] > 63) & (datasets['floor_area'] <= 95), 'floor_area'] = 1
        datasets.loc[(datasets['floor_area'] > 95) & (datasets['floor_area'] <= 130), 'floor_area'] = 2
        datasets.loc[(datasets['floor_area'] > 130) & (datasets['floor_area'] <= 180), 'floor_area'] = 3
        datasets.loc[(datasets['floor_area'] > 180), 'floor_area'] = 4

    # use bmi to represent height and weight
    train_df['bmi'] = (train_df['weight_jin'] / 2) / ((train_df['height_cm'] / 100) ** 2)
    test_df['bmi'] = (test_df['weight_jin'] / 2) / ((test_df['height_cm'] / 100) ** 2)

    # use international standard to cut bmi
    combine = [train_df, test_df]
    for datasets in combine:
        datasets.loc[(datasets['bmi'] <= 18.5), 'bmi'] = 0
        datasets.loc[(datasets['bmi'] > 18.5) & (datasets['bmi'] <= 22.9), 'bmi'] = 1
        datasets.loc[(datasets['bmi'] > 22.9) & (datasets['bmi'] <= 24.9), 'bmi'] = 2
        datasets.loc[(datasets['bmi'] > 24.9) & (datasets['bmi'] <= 30), 'bmi'] = 3
        datasets.loc[(datasets['bmi'] > 30), 'bmi'] = 4

    # use family status to estimate family income
    guess_income = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    combine = [train_df, test_df]
    for dataset in combine:
        for i in range(1, 6):
            guess_df = dataset[(dataset['family_status'] == i) & (dataset['family_income'] >= 0)][
                'family_income'].dropna()
            guess_money = guess_df.mean()
            guess_income[i] = int(guess_money / 0.5 + 0.5) * 0.5

        for i in range(1, 6):
            dataset.loc[(dataset['family_income'] < 0) & (dataset['family_status'] == i), 'family_income'] = \
            guess_income[i]

    # drop work related because it's not complete and has work_expr to represent work
    train_df = train_df.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1)
    test_df = test_df.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1)

    # cut income and family income
    combine = [train_df, test_df]
    for datasets in combine:
        datasets.loc[(datasets['income'] <= 3000), 'income'] = 0
        datasets.loc[(datasets['income'] > 3000) & (datasets['income'] <= 20000), 'income'] = 1
        datasets.loc[(datasets['income'] > 20000) & (datasets['income'] <= 36000), 'income'] = 2
        datasets.loc[(datasets['income'] > 36000), 'income'] = 3

        datasets.loc[(datasets['family_income'] <= 20000), 'family_income'] = 0
        datasets.loc[(datasets['family_income'] > 20000) & (datasets['family_income'] <= 44367), 'family_income'] = 1
        datasets.loc[(datasets['family_income'] > 44367) & (datasets['family_income'] <= 79987.5), 'family_income'] = 2
        datasets.loc[(datasets['family_income'] > 79987.5), 'family_income'] = 3

        datasets.loc[(datasets['height_cm'] <= 155), 'height_cm'] = 0
        datasets.loc[(datasets['height_cm'] > 155) & (datasets['height_cm'] <= 158), 'height_cm'] = 1
        datasets.loc[(datasets['height_cm'] > 158) & (datasets['height_cm'] <= 164), 'height_cm'] = 2
        datasets.loc[(datasets['height_cm'] > 164) & (datasets['height_cm'] <= 170), 'height_cm'] = 3
        datasets.loc[(datasets['height_cm'] > 170) & (datasets['height_cm'] <= 175), 'height_cm'] = 4
        datasets.loc[(datasets['height_cm'] > 175), 'height_cm'] = 5

        datasets.loc[datasets['son'] <= 2, 'son'] = 0
        datasets.loc[datasets['son'] > 2, 'son'] = 1
        datasets.loc[datasets['daughter'] <= 2, 'daughter'] = 0
        datasets.loc[datasets['daughter'] > 2, 'daughter'] = 1
        datasets.loc[datasets['minor_child'] <= 2, 'minor_child'] = 0
        datasets.loc[datasets['minor_child'] > 2, 'minor_child'] = 1

        datasets.loc[(datasets['s_income'] <= 3000), 's_income'] = 0
        datasets.loc[(datasets['s_income'] > 3000) & (datasets['s_income'] <= 20000), 's_income'] = 1
        datasets.loc[(datasets['s_income'] > 20000) & (datasets['s_income'] <= 30880.136), 's_income'] = 2
        datasets.loc[(datasets['s_income'] > 30880.136), 's_income'] = 3

    train_df['s_income'] = train_df['s_income'].fillna(0.0)
    test_df['s_income'] = test_df['s_income'].fillna(0.0)

    for i in range(1, 10):
        train_df['public_service_' + str(i)],bins = pd.cut(train_df['public_service_' + str(i)], 4, retbins=True)
        test_df['public_service_' + str(i)] = pd.cut(test_df['public_service_' + str(i)], bins)

    mapper = DataFrameMapper([('public_service_1', LabelEncoder()),
                              ('public_service_2', LabelEncoder()),
                              ('public_service_3', LabelEncoder()),
                              ('public_service_4', LabelEncoder()),
                              ('public_service_5', LabelEncoder()),
                              ('public_service_6', LabelEncoder()),
                              ('public_service_7', LabelEncoder()),
                              ('public_service_8', LabelEncoder()),
                              ('public_service_9', LabelEncoder())], df_out=True, default=None)

    train_df = mapper.fit_transform(train_df.copy())
    test_df = mapper.fit_transform(test_df.copy())

    # drop weight and height for now
    train_df = train_df.drop(['weight_jin'], axis=1)
    test_df = test_df.drop(['weight_jin'], axis=1)

    train_df['hukou_loc'] = train_df['hukou_loc'].fillna(1.0)
    # drop one nan number in train set
    train_df = train_df.dropna(axis=0, how='any')
    return train_df, test_df


def predict_model(X_train, y_train):
    #clf = tree.DecisionTreeClassifier()
    #  1.07
    #clf = RandomForestClassifier(50, max_depth=5, random_state=0) 0.65
    clf = xgb.XGBClassifier(max_depth=4, min_child_weight=6, learning_rate=0.1,
                            n_estimators=160, silent=True, objective='multi:softmax')
    #parameters={'learning_rate'}
    #clf1 = GridSearchCV(clf,parameters,cv=5)
    #0.62 can only use on server
    clf.fit(X_train, y_train)

    return clf


def calc_error(y_predict, y_test):
    mean_squared_error = (1 / len(y_test)) * np.sum(np.square(y_predict - y_test))
    return mean_squared_error


def write_to_csv(path, df):
    df.to_csv(path, sep=",", index = 0)


if __name__ == '__main__':
    train_df = pd.read_csv('happiness_train_complete.csv',encoding='gbk')
    test_df = pd.read_csv('happiness_test_complete.csv',encoding='gbk')
    test_id = test_df['id']
    train_df, test_df = data_preprocess(train_df, test_df)
    pd.set_option('max_columns', 300)
    print(train_df.describe())
    print(test_df.head())
    pd.set_option('max_columns', 100)
    # print(test_df.describe())
    X = train_df.drop('happiness', axis=1)
    Y = train_df['happiness']
    # print(X.shape, Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    model = predict_model(X_train, y_train)

    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    cols = list(X_train.columns)
    cols = [cols[x] for x in indices]
    print(cols)
    """
    y_predict = model.predict(X_test)
    print(calc_error(y_predict, y_test))

    final_predict = model.predict(test_df)
    final_df = pd.DataFrame({ 'id':test_id,'happiness':final_predict})
    final_df.to_csv('happiness_submit.csv',sep=',',columns=['id','happiness'],index=False)

