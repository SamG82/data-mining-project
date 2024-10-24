import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd


dataset = "breast-cancer.csv"

names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

df = pd.read_csv(dataset, names=names)


label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print(df.head())

X = df.drop('class', axis=1)

y = df['class']

n_estimators = []
accuracies = []

for i in range(1, 200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=i, random_state=43)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    n_estimators.append(i)
    accuracies.append(accuracy)

print(max(accuracies))
print(n_estimators[accuracies.index(max(accuracies))])
plt.plot(n_estimators, accuracies)
plt.xlabel('N_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest n_estimators compared to accuracy')
plt.show()