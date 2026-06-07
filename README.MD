# Telecom Churn Prediction

## Problem and Dataset

This project predicts whether a telecom customer is likely to leave the company. This is called customer churn.

The dataset used is the Telco Customer Churn dataset from Kaggle/IBM. It contains customer information such as contract type, tenure, monthly charges, total charges, internet service, payment method, and whether the customer has churned.

Dataset source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

This is a classification problem because the target column, `Churn`, has two possible categories: `Yes` or `No`.

## Data Preparation

The dataset was loaded and inspected with Pandas. I checked the shape of the dataset, the first rows, column information, missing values, and the distribution of the target column.

The column `TotalCharges` looked numeric but was stored as text. I converted it to numeric values with `pd.to_numeric()`. Some rows became missing after conversion, so I removed rows where `TotalCharges` was missing.

I selected customer and service-related features, including:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `InternetService`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

Categorical features were converted with one-hot encoding. Numeric features were standardized before training.

## Visualization

I created a chart showing customer churn by contract type.

The chart shows that churn is more common among month-to-month customers than among customers with one-year or two-year contracts. This makes sense because month-to-month customers can leave more easily.

The visualization is saved in:

```text
outputs/churn_by_contract.png
```

## Models

I first used logistic regression as a simple baseline model. Logistic regression is suitable for binary classification problems, where the model predicts one of two categories.

The data was split into training and test data with `train_test_split`. I used stratification so that the distribution of `Churn` stayed similar in both the training and test data.

The model used a Scikit-learn pipeline with:

- standardization for numeric columns
- one-hot encoding for categorical columns
- logistic regression as the classifier

To compare the baseline model with another method, I also trained a Random Forest model. After that, I used a small `GridSearchCV` to tune the Random Forest model with different values for `n_estimators`, `max_depth`, and `min_samples_leaf`.

## Evaluation

The models were evaluated on the test data using accuracy, precision, recall, F1-score, and a confusion matrix.

These metrics are relevant because this is a classification problem. Accuracy gives an overall result, while precision, recall, and F1-score give more detailed information about how well the model predicts each class.

The confusion matrix is saved in:

```text
outputs/confusion_matrix.png
```

The prediction results are saved in:

```text
outputs/churn_predictions.csv
```

The model comparison is saved in:

```text
outputs/model_comparison.csv
```

## Results

The model comparison gave these results:

| Model | Accuracy | Precision Yes | Recall Yes | F1 Yes |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.656 | 0.552 | 0.600 |
| Random Forest | 0.782 | 0.609 | 0.501 | 0.550 |
| Tuned Random Forest | 0.796 | 0.653 | 0.499 | 0.566 |

The tuned Random Forest improved compared with the first Random Forest model, but Logistic Regression still had the best accuracy and the best F1-score for the `Yes` class.

This shows that a more complex model is not automatically better. In this project, the simpler logistic regression baseline performed best with the selected features and preprocessing.

## Analysis and Possible Improvements

The overall accuracy was about 80%, which is a useful first result. However, the recall for the `Yes` class was lower than for the `No` class. This means the model missed some customers who actually churned.

In a real churn problem, recall for customers who churn can be especially important. If the company misses customers who are likely to leave, it may lose the chance to contact them or improve their experience.

Possible improvements include:

- testing more models
- using more hyperparameter tuning
- adjusting the decision threshold to focus more on churn recall
- adding more feature engineering
- checking whether the model performs differently for different customer groups

For example, contract type, tenure, and monthly charges may be combined or analyzed more deeply because they seem connected to churn.

## Limitations and Ethics

This model should not be used as the only basis for business decisions. It is trained on historical customer data, so it may repeat patterns or bias that already exist in the data.

There may also be customer groups that the model predicts better or worse for. A company should therefore check whether the model is fair across different types of customers.

The model could also be misused if the company treats customers differently only because a model predicts that they might leave. A better use would be decision support, for example identifying customers who may need better service or clearer information.

In a real business situation, the model should be tested more carefully before use. Customers should be treated fairly, and the company should be careful with personal customer data.