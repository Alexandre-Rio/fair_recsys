import pandas as pd
import sklearn as skl
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':
    # Parameters
    include_disc = True
    model_type = 'SVM'  # 'Logistic Regression' or 'DecisionTree' or 'SVC'

    # Read data and select relevant features
    data = pd.read_csv('../recruitmentdataset-2022-1.3.csv')  # Download dataset at https://www.kaggle.com/datasets/ictinstitute/utrecht-fairness-recruitment-dataset
    disc_features = ['gender', 'age', 'nationality']
    target_features = ['decision']
    predictive_features = ['ind-university_grade', 'ind-debateclub', 'ind-programming_exp', 'ind-international_exp', 'ind-entrepeneur_exp', 'ind-languages', 'ind-exact_study', 'ind-degree']

    # Split by company
    data_A = data[data['company'] == 'A']
    data_B = data[data['company'] == 'B']
    data_C = data[data['company'] == 'C']
    data_D = data[data['company'] == 'D']
    X_A = pd.get_dummies(data_A[include_disc * disc_features + predictive_features]).replace({True: 1, False: 0})
    X_B = pd.get_dummies(data_B[include_disc * disc_features + predictive_features]).replace({True: 1, False: 0})
    X_C = pd.get_dummies(data_C[include_disc * disc_features + predictive_features]).replace({True: 1, False: 0})
    X_D = pd.get_dummies(data_D[include_disc * disc_features + predictive_features]).replace({True: 1, False: 0})
    y_A = data_A[target_features].replace({True: 1, False: 0}).to_numpy().ravel()
    y_B = data_B[target_features].replace({True: 1, False: 0}).to_numpy().ravel()
    y_C = data_C[target_features].replace({True: 1, False: 0}).to_numpy().ravel()
    y_D = data_D[target_features].replace({True: 1, False: 0}).to_numpy().ravel()

    # Create regression objects
    if model_type == 'LogisticRegression':
        reg_A = LogisticRegressionCV(max_iter=1000)
        reg_B = LogisticRegressionCV(max_iter=1000)
        reg_C = LogisticRegressionCV(max_iter=1000)
        reg_D = LogisticRegressionCV(max_iter=1000)
    elif model_type == 'DecisionTree':
        reg_A = DecisionTreeClassifier()
        reg_B = DecisionTreeClassifier()
        reg_C = DecisionTreeClassifier()
        reg_D = DecisionTreeClassifier()
    elif model_type == 'SVM':
        kernel = 'linear'
        reg_A = SVC(kernel=kernel)
        reg_B = SVC(kernel=kernel)
        reg_C = SVC(kernel=kernel)
        reg_D = SVC(kernel=kernel)

    # Split datasets
    XA_train, XA_test, yA_train, yA_test = train_test_split(X_A, y_A, test_size=0.5)
    XB_train, XB_test, yB_train, yB_test = train_test_split(X_B, y_B, test_size=0.5)
    XC_train, XC_test, yC_train, yC_test = train_test_split(X_C, y_C, test_size=0.5)
    XD_train, XD_test, yD_train, yD_test = train_test_split(X_D, y_D, test_size=0.5)

    # Fit regression models
    reg_A.fit(XA_train, yA_train)
    reg_B.fit(XB_train, yB_train)
    reg_C.fit(XC_train, yC_train)
    reg_D.fit(XD_train, yD_train)

    # Compute accuracy scores
    yA_pred = reg_A.predict(XA_test)
    yB_pred = reg_B.predict(XB_test)
    yC_pred = reg_C.predict(XC_test)
    yD_pred = reg_D.predict(XD_test)
    acc_A = accuracy_score(yA_test, yA_pred)
    f1_A = f1_score(yA_test, yA_pred)
    acc_B = accuracy_score(yB_test, yB_pred)
    f1_B = f1_score(yB_test, yB_pred)
    acc_C = accuracy_score(yC_test, yC_pred)
    f1_C = f1_score(yC_test, yC_pred)
    acc_D = accuracy_score(yD_test, yD_pred)
    f1_D = f1_score(yD_test, yD_pred)
    print(f"Accuracy Score: A:{acc_A} \t B:{acc_B} \t C:{acc_C} \t D:{acc_D}")
    print(f"F1 Score: A:{f1_A} \t B:{f1_B} \t C:{f1_C} \t D:{f1_D}")

    # Get decision scores on test set (which will serve as relevance scores)
    X = pd.concat([XA_test, XB_test, XC_test, XD_test])
    def get_confidence(model, X, model_type):
        if model_type in ['LogisticRegression', 'SVM'] :
            return model.decision_function(X)
        elif model_type == 'DecisionTree':
            return model.predict_proba(X)[:, 1]

    dec_func_A = get_confidence(reg_A, X, model_type)
    dec_func_B = get_confidence(reg_B, X, model_type)
    dec_func_C = get_confidence(reg_C, X, model_type)
    dec_func_D = get_confidence(reg_D, X, model_type)
    score_A = 5 * (dec_func_A - dec_func_A.min()) / (dec_func_A.max() - dec_func_A.min())
    score_B = 5 * (dec_func_B - dec_func_B.min()) / (dec_func_B.max() - dec_func_B.min())
    score_C = 5 * (dec_func_C - dec_func_C.min()) / (dec_func_C.max() - dec_func_C.min())
    score_D = 5 * (dec_func_D - dec_func_D.min()) / (dec_func_D.max() - dec_func_D.min())

    # Format final data
    format_data = data[disc_features + target_features + predictive_features + ["company"]]
    format_data = format_data.iloc[X.index]
    format_data = pd.get_dummies(format_data, columns=['ind-degree'])
    format_data = format_data * 1
    format_data['score_A'] = score_A
    format_data['score_B'] = score_B
    format_data['score_C'] = score_C
    format_data['score_D'] = score_D
    format_data['score'] = (score_A + score_B + score_C + score_D) / 4
    format_data = format_data.sort_index()

    format_data.index.name = 'applicantId'
    format_data["disc_gender"] = np.where(format_data["gender"] == "male", 1, 0)
    format_data["disc_age"] = np.where(format_data["age"] >= 25, 1, 0)
    format_data["disc_nationality"] = np.where(format_data["nationality"] != "Dutch", 1, 0)

    format_data.to_csv('../hiring_format3' + include_disc * '_disc' + f'{model_type}' '.csv')

    end = True