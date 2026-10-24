import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = {
    "Tenure": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15,
               18, 20, 22, 24, 30, 32, 36, 40, 42, 48],

    "MonthlyCharges": [90, 85, 80, 95, 70, 75, 65, 60, 55, 50,
                       60, 45, 50, 40, 55, 35, 45, 30, 40, 25],

    "SupportCalls": [5, 6, 4, 7, 3, 5, 2, 3, 2, 1,
                     4, 2, 3, 1, 2, 1, 2, 0, 1, 0],

    "ContractType": [
        "Monthly", "Monthly", "Monthly", "Monthly", "Yearly",
        "Monthly", "Yearly", "Yearly", "Yearly", "Yearly",
        "Monthly", "Yearly", "Yearly", "Yearly", "Monthly",
        "Yearly", "Monthly", "Yearly", "Monthly", "Yearly"
    ],

    "Churn": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

print("Data Size:",df.shape)
print("Class Distribution:\n",df["Churn"].value_counts())
print("Missing Value:\n",df.isnull().sum())

X = df.drop("Churn",axis=1)
y = df["Churn"]

numerical_feature = ["Tenure","MonthlyCharges","SupportCalls"]
categorical_feature = ["ContractType"]

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
    
categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    
preprocessor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_feature),
    ("cat",categorical_pipeline,categorical_feature)
    ])
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

baseline_model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
    ])

baseline_model.fit(X_train,y_train)
baseline_prediction = baseline_model.predict(X_test)

print("Baseline Prediction  :",baseline_prediction)
print("Actual Value         :",y_test.to_numpy())


model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
    ])
    
param_distributions = {
    "classifier__C" : [0.01,0.1,1,10,100],
    "classifier__class_weight" : [None,"balanced"]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
    )

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    cv=cv,
    n_iter=6,
    scoring="f1",
    random_state=42,
    refit=True
    )

search.fit(X_train,y_train)

tuned_model = search.best_estimator_
tuned_prediction = tuned_model.predict(X_test)

print("Best Parameter:",search.best_params_)
print("Best Cross Validation F1:",search.best_score_)

print("Tuned Prediction:",tuned_prediction)
print("Actual Value    :",y_test.to_numpy())

def evaluate_model(model,actual,predict):
    print(f"Model = {model}")
    print("Accuracy  :",np.round(accuracy_score(actual,predict),2))
    print("Precision :",np.round(precision_score(actual,predict,zero_division=0),2))
    print("Recall    :",np.round(recall_score(actual,predict,zero_division=0),2))
    print("F1        :",np.round(f1_score(actual,predict,zero_division=0),2))
    print("Confusion Matrix:\n",confusion_matrix(actual,predict))
    print("\n")

evaluate_model("Baseline Model",y_test,baseline_prediction)
evaluate_model("Tuned Model",y_test,tuned_prediction)



print("Tuned Model Classification Report:\n",classification_report(y_test,tuned_prediction,zero_division=0))

joblib.dump(tuned_model,"best_churn_model.joblib")
print("Model Saved Successfully!")
