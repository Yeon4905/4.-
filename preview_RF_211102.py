import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:\\HYUSTL\\5. 수업\\2기\\포장관리체계\\1. 연구주제\\0. 연구계획서\\0. 데이터\\팬오션 선바운항 자료(초단위)\\PanOcean_데이터_20170101-20170331_수정\\")

if __name__ == "__main__":
    scaler = RobustScaler()
    df = pd.read_csv("activation_en_route_211102.csv")
    df = df.astype(float)
    df = df.dropna(axis=0)
    df = df.reset_index()
    y = df['Main Engine Fuel Oil Consumption Per Hourly']
    X = df.drop(['Main Engine Fuel Oil Consumption Per Hourly'], axis=1)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=0)
    scaled_train_X = scaler.fit_transform(train_X)
    scaled_test_X = scaler.transform(test_X)
    regressor = RandomForestRegressor(max_depth = 10, min_samples_leaf = 18, min_samples_split=12, n_estimators = 100, random_state=0)
    regressor.fit(scaled_train_X, train_y)
    y_pred = regressor.predict(scaled_test_X)

    # X_grid = np.arange(min(scaled_train_X), max(scaled_train_X), 0.01)
    # X_grid = X_grid.reshape((len(X_grid),1))

    # plt.scatter(scaled_train_X, train_y, color = 'blue')
    # plt.plot(X_grid, regressor.predict(X_grid), color='green')
    # plt.title('RF Regressor')
    # plt.xlabel('X variables')
    # plt.ylabel('Fuel Consumption Per Hourly')
    # plt.show()

    r2 = r2_score(test_y, y_pred)
    print(r2)

    score = {'r2':[r2]}
    result = pd.DataFrame(score)
    df.to_csv("result_preview_RF_211101_1.csv", encoding='utf-8')
    # 0.9035015246585907

    ax1 = sns.distplot(test_y, hist = False,label = 'actual_y')
    ax2 = sns.distplot(y_pred, hist = False,label = 'pred_y')
    plt.legend(title="Random Forest")
    plt.show()
