import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# import dataset and setup dataframe 
dataset = "breast-cancer.csv"
names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
df = pd.read_csv(dataset, names=names)

# label encode all of the categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# show dataframe
print(df.head())

# set regular x to all features
X = df.drop('class', axis=1)

# drop the less important features based on importance of model 
X_limited = df.drop(['class', 'irradiat', 'menopause', 'breast', 'node-caps', 'inv-nodes'], axis=1)

# set y as class target
y = df['class']

# store list of accuracies from trained models
accuracies = []

# iteratively train model 500 times
for i in range(500):
    # split data set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # train model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # get predictions
    y_pred = clf.predict(X_test)

    # store accuracy in the accuracies list
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    importances = clf.feature_importances_
    feature_names = X.columns

print(f"Mean Accuracy: {sum(accuracies) / len(accuracies)}")

# show histogram of model accuracies
plt.hist(accuracies, bins=20)
plt.xlabel("Decision Tree accuracy")
plt.ylabel("Instance Count")
plt.show()

# show bar graph of feature importance 
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Decision Tree Classifier Feature Importances')
plt.show()