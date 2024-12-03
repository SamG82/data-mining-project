import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# read the recurrence dataset and return a preprocessed X, X_lmited, y tuple
def preprocess_recurrence_dataset(csv_path):

    # import dataset and setup dataframe 
    names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    df = pd.read_csv(csv_path, names=names)

    # label encode all of the categorical variables
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # set regular x to all features
    X = df.drop('class', axis=1)

    # doesn't include the less important features based on decisiontree importance
    X_limited = df.drop(['class', 'irradiat', 'menopause', 'breast', 'node-caps', 'inv-nodes', 'breast-quad'], axis=1)

    # set y as class target
    y = df['class']

    return (X, X_limited, y)

# runs model n_iterations, returns a list of accuracies, tuple of y_test/y_pred values, and the most accurate model
def iterative_model_performance(Model, n_iterations, X_y_data, **kwargs):
    X, y = X_y_data
    accuracies = []

    highest_accuracy = 0
    best_model = None
    result_y_test = None
    result_y_pred = None

    # iteratively train new models with randomized train_test_split
    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        new_model = Model(**kwargs)
        new_model.fit(X_train, y_train)
        y_pred = new_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        # set new results if the accuracy_score was higher
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            result_y_test = y_test
            result_y_pred = y_pred
            best_model = new_model

    return accuracies, (result_y_test, result_y_pred), best_model

# plot two lists of accuracy values on a histogram to compare performance
def plot_comparative_accuracies(accuracy_tuple, labels, legend, title):
    flattened = []
    for result in accuracy_tuple:
        for accuracy in result:
            flattened.append(accuracy)

    # generate bins based on the min and max from both lists
    bins = numpy.linspace(min(flattened), max(flattened), 15)

    # setup plot
    plt.hist(accuracy_tuple, bins, label=labels)
    plt.legend(loc='upper right')
    
    xlabel, ylabel = legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


# go through n_estimators to return a tuple of the estimators used and accuracies to find best RF performance
def tune_random_forest_hyperparameters(n_estimators_limit, X_y_data):
    estimators = []
    accuracies = []
    current_best_accuracy = 0
    best_estimator = 1

    X, y = X_y_data
    for i in range(1, n_estimators_limit):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # train the model with the current iteration of n_estimators
        clf = RandomForestClassifier(n_estimators=i, random_state=43)
        clf.fit(X_train, y_train)
        
        # predict and score
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > current_best_accuracy:
            current_best_accuracy = accuracy
            best_estimator = i

        # add information to corresponding lists
        estimators.append(i)
        accuracies.append(accuracy)
    
    return estimators, accuracies, best_estimator

def analyze_recurrence_dataset():
    # get X, X with limited features, and y from the data set
    X, X_limited, y = preprocess_recurrence_dataset('./breast-cancer.csv')
    
    model_iterations = 250
    # train and test 2 different decision trees using each different X 500 times each
    regular_accuracies, reg_prediction_results, reg_tree = iterative_model_performance(DecisionTreeClassifier, model_iterations, (X, y))
    limited_accuracies, lim_prediction_results, lim_tree = iterative_model_performance(DecisionTreeClassifier, model_iterations, (X_limited, y))
    
    
    estimators, estimator_accuracies, best_estimator = tune_random_forest_hyperparameters(150, (X, y))

    # train random forest using best n_estimators
    rf_accuracies, rf_prediction_results, rf_clf = iterative_model_performance(
        RandomForestClassifier,
        model_iterations,
        (X, y),
        **{
            'n_estimators': best_estimator,
            'random_state': 42
        }
    )

    classifiers = [
        ('reg_dt', reg_tree),
        ('random_forest', rf_clf)
    ]

    voting_accuracies, voting_prediction_results, vote_clf = iterative_model_performance(
        VotingClassifier,
        model_iterations,
        (X, y),
        **{
            'estimators': classifiers,
            'voting': 'soft'
        }
    )

    # show a plot of estimators compared to their achieved accuracy for random forest
    plt.plot(estimators, estimator_accuracies)
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest number of estimators compared to accuracy')
    plt.show()

    # show bar graph of feature importances using the best regular decision tree
    bar_width = 0.2
    bar_x_axis = numpy.arange(len(X.columns))

    plt.bar(bar_x_axis - bar_width, reg_tree.feature_importances_, 0.3, label='Decision Tree feature importances (all features)', color='skyblue')
    plt.bar(bar_x_axis + bar_width, rf_clf.feature_importances_, 0.3, label='Random forest feature importances', color='green')
    plt.xticks(bar_x_axis, X.columns)
    plt.legend()
    plt.xlabel('Feature Name')
    plt.ylabel('Feature Importance Score (Gini)')
    plt.title('Comparison of feature importances between Decision Tree and Random Forest')
    plt.show()

    # plot the comparison of accuracies between the regular and limited decision tree
    plot_comparative_accuracies(
        (regular_accuracies, limited_accuracies, rf_accuracies, voting_accuracies),
        ('All features decision tree', 'Limited features decision tree',
         f'Random forest using n_estimators={best_estimator}', 'Voting Classifier using Decision Tree and Random Forest'),
        ('Model Accuracy', 'Instance count'),
        'Comparison of models on predicting breast-cancer recurrence'
    )

    targets = ('Recurrence', 'No recurrence')

    print("Classification report for decision tree using all features: ")
    reg_y_test, reg_y_pred = reg_prediction_results
    print(classification_report(reg_y_test, reg_y_pred, target_names=targets))    

    print("Classification report for decision tree using limited features: ")
    lim_y_test, lim_y_pred = lim_prediction_results
    print(classification_report(lim_y_test, lim_y_pred, target_names=targets))

    print("Classification report for random forest: ")
    rf_y_test, rf_y_pred = rf_prediction_results
    print(classification_report(rf_y_test, rf_y_pred, target_names=targets))

    print("Classification report for Voting Classifier using regular decision tree and random forest: ")
    vote_y_test, vote_y_pred = voting_prediction_results
    print(classification_report(vote_y_test, vote_y_pred, target_names=targets))

# preprocesses the diagnosis dataset and returns the main X, y for the dataset, along with more specific dataset of perimeter and texture
def preprocess_diagnosis_dataset(csv_path):
    # all individual features
    feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

    # either M for malignant or B for benign
    target_name = 'Diagnosis'

    prediction_data = pd.read_csv(csv_path)

    # isolate target variable from features
    target = prediction_data[target_name]

    # remove the ID from meaningful features
    features = prediction_data.drop(['ID', target_name], axis=1)

    # create a new dataframe to hold the magnitudes
    feature_magnitudes = pd.DataFrame()

    # get the combined magnitudes from the different x, y, z and points of each feature's vector
    for feature in feature_names:
        # ['radius1', 'radius2', 'radius3'], etc
        feature_dimensions = [f"{feature}{i}" for i in range(1, 4)]

        # calculate and store magnitude in the df
        total = sum(features[dimension]**2 for dimension in feature_dimensions)
        feature_magnitudes[feature] = numpy.sqrt(total)

    # combine target with magnitudes for visualization
    target_with_magnitudes = pd.concat([target, feature_magnitudes], axis=1)

    # output csv for separate orange analysis
    target_with_magnitudes.to_csv('target_with_magnitudes.csv', index=False)

    # only using perimeter/texture features from here because they were the most significant based on Orange analysis of scatter plot
    # drop all features except for perimeter and texture
    target_perim_texture = target_with_magnitudes.drop([feature for feature in feature_names if feature not in ('perimeter', 'texture')], axis=1)

    target = target_perim_texture['Diagnosis']
    features = target_perim_texture.drop('Diagnosis', axis=1)

    # scale the features using standardscaler
    features_scaled = StandardScaler().fit_transform(features)
    features = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

    return features, target

def scatter_with_svm_boundary(svm_clf, target_perim_texture):
    # create the svm line
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # create the X range using a linsapce of the perimeter attributej
    svm_x = numpy.linspace(target_perim_texture['perimeter'].min(), target_perim_texture['perimeter'].max(), 100) 

    # solve for y using the coef and intercept from the SVM to create y points
    svm_y = -(w[0] / w[1]) * svm_x - b / w[1] 

    # red for malignant, blue for benign
    colors = {
        'M': 'red',
        'B': 'blue'
    }

    # labels for legend
    labels = {
        'M': 'Malignant',
        'B': 'Benign'
    }

    # plot the svm line
    plt.plot(svm_x, svm_y, c='black', label='SVM Decision Boundary')

    # scatter the points based on perimeter and texture, color by malignancy
    for label in target_perim_texture['Diagnosis'].unique():
        subset = target_perim_texture[target_perim_texture['Diagnosis'] == label]
        plt.scatter(subset['perimeter'], subset['texture'], color=colors[label], label=labels[label])

    # legend and labels
    plt.xlabel('Perimeter')
    plt.ylabel('Texture')
    plt.title('Predicting breast cancer malignancy using a SVM on the perimeter and texture of masses')
    plt.legend()

    plt.show()

def analyze_diagnosis_dataset():
    # preprocess and split the dataset
    X, y = preprocess_diagnosis_dataset('./wdbc.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train svm and get classification report
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    scatter_with_svm_boundary(clf, pd.concat([X, y], axis=1))

    print("Classification report for predicting diagnosis using SVM")
    targets = ('Malignant', 'Benign')
    print(classification_report(y_test, y_pred, target_names=targets))
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})
    analyze_recurrence_dataset()
    analyze_diagnosis_dataset()