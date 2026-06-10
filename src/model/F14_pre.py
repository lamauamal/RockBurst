"""model已保存好配置和各种权重"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

with open('../../static/models/F-14_run20260607_145549/42.1110.0.0.model', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('../../static/merge.csv')
X = data[['σθ', 'σt', 'Wet', 'B1']]
y = data['MR']
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42,
        stratify=y,
        shuffle=True
    )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
report = classification_report(
        y_test, y_pred,
        digits=3
    )
print(report)

