import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def data_preprocess(train_df,test_df):
    li = [i for i in range(15)]
    # drop happiness under zero training set
    train_df = train_df[(train_df['happiness'].isin(li[1:6]))]
    # set ageband
    train_df['Age'] = 2019 - train_df['birth']
    test_df['Age'] = 2019 - test_df['birth']
    train_df.loc[train_df['Age'] <= 41, 'Age'] = 0
    train_df.loc[(train_df['Age'] > 41) & (train_df['Age'] <= 60), 'Age'] = 1
    train_df.loc[(train_df['Age'] > 60) & (train_df['Age'] <= 79), 'Age'] = 2
    train_df.loc[(train_df['Age'] > 79), 'Age'] = 3
    test_df.loc[test_df['Age'] <= 41, 'Age'] = 0
    test_df.loc[(test_df['Age'] > 41) & (test_df['Age'] <= 60), 'Age'] = 1
    test_df.loc[(test_df['Age'] > 60) & (test_df['Age'] <= 79), 'Age'] = 2
    test_df.loc[(test_df['Age'] > 79), 'Age'] = 3

    train_df = train_df.drop(['id', 'city', 'county', 'survey_time', 'birth'], axis=1)
    test_df = test_df.drop(['id', 'city', 'county', 'survey_time', 'birth'], axis=1)

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
    # drop weight and height for now
    train_df = train_df.drop(['height_cm', 'weight_jin'], axis=1)
    test_df = test_df.drop(['height_cm', 'weight_jin'], axis=1)

    # drop one nan number in train set
    train_df = train_df.dropna(axis=0, how='any')
    return train_df, test_df


def predict_model(X_train, y_train):
    # clf = tree.DecisionTreeClassifier() 1.07
    # clf = RandomForestClassifier(50, max_depth=5, random_state=0) 0.65
    clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,
                            n_estimators=160, silent=True, objective='multi:softmax')
    #0.62 can only use on server
    clf = clf.fit(X_train, y_train)

    return clf


def calc_error(y_predict, y_test):
    mean_squared_error = (1 / len(y_test)) * np.sum(np.square(y_predict - y_test))
    return mean_squared_error


def write_to_csv(path, df):
    df.to_csv(path, sep=",", index = 0)


if __name__ == '__main__':
    train_df = pd.read_csv('happiness_train_abbr.csv')
    test_df = pd.read_csv('happiness_test_abbr.csv')
    test_id = test_df['id']
    train_df, test_df = data_preprocess(train_df, test_df)
    pd.set_option('max_columns', 100)
    # print(test_df.describe())
    X = train_df.drop('happiness', axis=1)
    Y = train_df['happiness']
    # print(X.shape, Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    model = predict_model(X_train, y_train)
    y_predict = model.predict(X_test)
    print(calc_error(y_predict, y_test))
    
    final_predict = model.predict(test_df)
    final_df = pd.DataFrame({ 'id':test_id,'happiness':final_predict})
    final_df.to_csv('happiness_submit.csv',sep=',',columns=['id','happiness'],index=False)
