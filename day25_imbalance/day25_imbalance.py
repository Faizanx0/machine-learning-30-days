"""
1. Build your Pipeline
Use numerical imputation and scaling, categorical imputation and one-hot encoding, and Logistic Regression.

2. Create a parameter grid
Test C values [0.01, 0.1, 1, 10, 100].

3. Create GridSearchCV
Use 5-fold StratifiedKFold and scoring='f1'.

4. Fit the search
Call grid_search.fit(X, y) for this practice exercise.

5. Print the results
Display best_params_, best_score_, and best_estimator_.

6. Explain the concepts
Explain what C controls and why we use classifier__C. - C is a Hyperparameter which tunes the model for best result, we use classifier__C becoz it have all the values of Hyperparameter for tuning model why it wrtten as classifier__C becoz the LogisticRegression is label as classifier
"""

