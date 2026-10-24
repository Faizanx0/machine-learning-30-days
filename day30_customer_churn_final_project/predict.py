import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_churn_model.joblib")

customer = pd.DataFrame([{
    "Tenure": 7,
    "MonthlyCharges": 80,
    "SupportCalls": 4,
    "ContractType": "Monthly"
}])

prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

print("Prediction:",prediction)
print(f"Estimated Churn Probability: {np.round(probability,2)*100}%")

if prediction==1 :
    print("The model predicts that the customer will churn.")
else:
    print("The model predicts that the customer will stay.")
