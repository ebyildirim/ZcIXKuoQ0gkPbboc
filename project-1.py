import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


X=pd.read_csv("term-deposit-marketing-2020.csv")
y=X['y']


deletedFeatures = ["y","contact","day","month","duration","campaign"]
#change balance and age with intervals
X.drop(labels=deletedFeatures, axis=1, inplace=True)

age=X['age'].tolist()  # 0-30 / 31-35 / 36-40 / 41-45  / 46-50 / 51-55 / +56
for i in range (0,40000):
    if age[i] < 31:
        age[i]=1
    elif age[i] < 36:
        age[i]=2
    elif age[i] < 41:
        age[i]=3
    elif age[i] < 46:
        age[i]=4
    elif age[i] < 51:
        age[i]=5
    elif age[i] < 56:
        age[i]=6
    else:
        age[i]=7

X['age']=age


balance=X['balance'].tolist()  # <0 / 0-1000 / 1001-2000 / 2001-3000 / 3001-4000 / 4001-5000/ +5000
for i in range (0,40000):
    if balance[i] < 0:
        balance[i]=0;

    elif balance[i] < 1001:
        balance[i]=1;

    elif balance[i] < 2001:
        balance[i]=2;

    elif balance[i] < 3001:
        balance[i]=3;

    elif balance[i] < 4001:
        balance[i]=4;

    elif balance[i] < 5001:
        balance[i]=5;

    elif balance[i] > 5000:
        balance[i]=6;


X['balance']=balance


le = preprocessing.LabelEncoder()



X['job']=le.fit_transform(X['job'])
X['marital']=le.fit_transform(X['marital'])
X['education']=le.fit_transform(X['education'])
X['default']=le.fit_transform(X['default'])
X['housing']=le.fit_transform(X['housing'])
X['loan']=le.fit_transform(X['loan'])





"""
for i in X.columns.values:
    unique_value = X[i].unique()
    print(unique_value)
    size = len(unique_value)
    print(i,size)
"""



tree=RandomForestClassifier(n_estimators=200,n_jobs=-1)

tree.fit(X,y)

scores = cross_val_score(tree, X, y, cv=5) #[0.922625 0.9195   0.917875 0.913375 0.922125]

print(scores)

