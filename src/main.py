import pandas as pd
df = pd.read_csv("./Data/USD_JPY.csv")


import talib as ta
import numpy as np

#以降全ての計算でレート終値を使う
close = np.array(df["終値"])

#特徴量を入れるための空のdataframeを作成
df_feature = pd.DataFrame(index=range(len(df)),columns=["SMA5/current", "SMA20/current","RSI","MACD","BBANDS+2σ","BBANDS-2σ"])

#以下、talibを用いてテクニカル指標（今回の学習で用いる特徴量）を算出しdf_feature入れる

#単純移動平均は、単純移動平均値とその日の終値の比を特徴量として用いる
df_feature["SMA5/current"]= ta.SMA(close, timeperiod=5) / close
df_feature["SMA20/current"]= ta.SMA(close, timeperiod=20) / close

#RSI
df_feature["RSI"] = ta.RSI(close, timeperiod=14)

#MACD
df_feature["MACD"], _ , _= ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

#ボリンジャーバンド 
upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=3, nbdevdn=3)
df_feature["BBANDS+2σ"] = upper / close
df_feature["BBANDS-2σ"] = lower / close


def classify(x):
#前日比が-0.2%以下ならグループ０
    if x <= -0.2:
        return 0
#前日比が0.2%<x<0.2%ならグループ1
    elif -0.2 < x < 0.2:
        return 1
#前日比が0.2%以上ならグループ２
    elif 0.2 <= x:
        return 2


df["前日比_float"] = df["前日比 %"].apply(lambda x: float(x.replace("%", "")))

#前日比%の分類の仕方。できるだけ各クラスのサンプルが等しいようにわける
def classify(x):
    if x <= -0.2:
        return 0
    elif -0.2 < x < 0.2:
        return 1
    elif 0.2 <= x:
        return 2


df["前日比_classified"] = df["前日比_float"].apply(lambda x: classify(x))

#教師にしたいデータを一日ずつずらす（意味を考えればわかると思います）
df_y = df["前日比_classified"].shift()

df_xy = pd.concat([df_feature, df_y], axis=1)
df_xy = df_xy.dropna(how="any")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

X_train, X_test, Y_train, Y_test = train_test_split(df_xy[["SMA5/current", "SMA20/current","RSI","MACD","BBANDS+2σ","BBANDS-2σ"]],df_xy["前日比_classified"], train_size=0.8)

def objective(trial):
    min_samples_split = trial.suggest_int("min_samples_split", 2,16)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4,64,4))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 50,500,50))
    max_depth = trial.suggest_int("max_depth", 3,10)
    clf = RandomForestClassifier(random_state=1, n_estimators = n_estimators, max_leaf_nodes = max_leaf_nodes, max_depth=max_depth, max_features=None,criterion=criterion,min_samples_split=min_samples_split)
    clf.fit(X_train, Y_train)
    return 1 - accuracy_score(Y_test, clf.predict(X_test))

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(1-study.best_value)
print(study.best_params)
