# Customer Churn Analysis & Prediction

**Tools:** SQL Server · Power BI · Python · scikit-learn  
**Domain:** Customer Analytics · ETL Pipeline · Predictive Modeling · Business Intelligence  
**Dataset:** ~7,000+ customers · 30+ features · Sourced from Kaggle

## Project Summary

This is a **full end-to-end data analytics project** covering every stage of the analytics workflow — from raw data ingestion in SQL Server, through Power BI dashboard design, to a machine learning model that predicts *future* churners before they leave.

**Key outcomes:**
- Built a complete **ETL pipeline** in SQL Server — staging, cleaning, and production tables
- Designed an interactive **Power BI dashboard** across 3 pages: Summary, Churn Reasons, and Predictions
- Trained a **Random Forest Classifier (~80% accuracy)** to identify at-risk customers
- Surfaced predictions back into Power BI for a live **Churn Prediction page**
- Identified competitor offers as the #1 churn driver, with month-to-month contracts and fiber optic service as highest-risk segments


## The Business Problem

Customer churn is one of the costliest challenges in the telecom industry. Every lost customer represents not just lost monthly revenue, but the full lifetime value of that relationship.

This project set out to answer six core business questions:

1. What is the overall churn rate and its revenue impact?
2. Which customer segments are most likely to churn?
3. How do contract type, internet service, and payment method influence churn?
4. What are the most common reasons customers leave?
5. What demographic factors (age, gender, location) correlate with churn?
6. Can we predict which customers are at risk *before* they churn?


##  Project Architecture

Raw CSV File
     ↓
[STEP 1] SQL Server — ETL (Staging → Cleaning → Production)
     ↓
[STEP 2] Power BI — Data Transformation (Power Query)
     ↓
[STEP 3] Power BI — DAX Measures
     ↓
[STEP 4] Power BI — Dashboard (Summary + Churn Reasons pages)
     ↓
[STEP 5] Python — Random Forest Churn Prediction Model
     ↓
[STEP 6] Power BI — Churn Prediction Page (from model output)
```

## Dataset Overview

| Field | Description |
|---|---|
| `Customer_ID`, `Gender`, `Age`, `State`, `Married` | Customer demographics |
| `Tenure_in_Months`, `Contract`, `Number_of_Referrals` | Account history |
| `Internet_Type`, `Phone_Service`, `Streaming_*` | Services subscribed |
| `Monthly_Charge`, `Total_Revenue`, `Total_Refunds` | Billing & revenue |
| `Customer_Status`, `Churn_Category`, `Churn_Reason` | Churn outcome & reason |

## STEP 1 — ETL Process in SQL Server

### Create Database & Import Data
```sql
CREATE DATABASE db_Churn
```
Data was imported from CSV into a staging table (`stg_Churn`) using the SQL Server Import Wizard. All columns were set to allow nulls during import to prevent load errors.

### Data Exploration — Check Distributions
```sql
-- Contract type distribution
SELECT Contract, Count(Contract) as TotalCount,
Count(Contract) * 1.0 / (Select Count(*) from stg_Churn) as Percentage
FROM stg_Churn
GROUP BY Contract

-- Revenue contribution by customer status
SELECT Customer_Status, Count(Customer_Status) as TotalCount,
Sum(Total_Revenue) as TotalRev,
Sum(Total_Revenue) / (Select sum(Total_Revenue) from stg_Churn) * 100 as RevPercentage
FROM stg_Churn
GROUP BY Customer_Status
```

### Null Check Across All Columns
```sql
SELECT
    SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count,
    SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) AS Gender_Null_Count,
    SUM(CASE WHEN Value_Deal IS NULL THEN 1 ELSE 0 END) AS Value_Deal_Null_Count,
    SUM(CASE WHEN Internet_Type IS NULL THEN 1 ELSE 0 END) AS Internet_Type_Null_Count,
    SUM(CASE WHEN Churn_Category IS NULL THEN 1 ELSE 0 END) AS Churn_Category_Null_Count,
    SUM(CASE WHEN Churn_Reason IS NULL THEN 1 ELSE 0 END) AS Churn_Reason_Null_Count
    -- All 30+ columns checked
FROM stg_Churn;
```

### Clean & Load to Production Table
Nulls replaced with meaningful defaults, then loaded into `prod_Churn`:

```sql
SELECT
    Customer_ID, Gender, Age, Married, State,
    ISNULL(Value_Deal, 'None') AS Value_Deal,
    ISNULL(Multiple_Lines, 'No') AS Multiple_Lines,
    ISNULL(Internet_Type, 'None') AS Internet_Type,
    ISNULL(Online_Security, 'No') AS Online_Security,
    ISNULL(Device_Protection_Plan, 'No') AS Device_Protection_Plan,
    ISNULL(Churn_Category, 'Others') AS Churn_Category,
    ISNULL(Churn_Reason, 'Others') AS Churn_Reason
    -- All columns included
INTO [db_Churn].[dbo].[prod_Churn]
FROM [db_Churn].[dbo].[stg_Churn];
```

### Create Views for Power BI
```sql
-- Historical data for dashboard analysis
CREATE VIEW vw_ChurnData AS
    SELECT * FROM prod_Churn
    WHERE Customer_Status IN ('Churned', 'Stayed')

-- New joiners for churn prediction
CREATE VIEW vw_JoinData AS
    SELECT * FROM prod_Churn
    WHERE Customer_Status = 'Joined'
```


## STEP 2 — Power BI Transformation (Power Query)

New calculated columns added in Power Query:

```
Churn Status = if [Customer_Status] = "Churned" then 1 else 0

Monthly Charge Range =
  if [Monthly_Charge] < 20 then "< 20"
  else if [Monthly_Charge] < 50 then "20-50"
  else if [Monthly_Charge] < 100 then "50-100"
  else "> 100"

Age Group =
  if [Age] < 20 then "< 20"
  else if [Age] < 36 then "20-35"
  else if [Age] < 51 then "36-50"
  else "> 50"

Tenure Group =
  if [Tenure_in_Months] < 6 then "< 6 Months"
  else if [Tenure_in_Months] < 12 then "6-12 Months"
  else if [Tenure_in_Months] < 24 then "12-24 Months"
  else ">= 24 Months"
```

A **services mapping table** was created by unpivoting all service columns into `Services` and `Status` — enabling a single visual to display churn rate across every service type.


## STEP 3 _ DAX Measures

```dax
Total Customers = COUNT(prod_Churn[Customer_ID])

New Joiners = CALCULATE(COUNT(prod_Churn[Customer_ID]),
              prod_Churn[Customer_Status] = "Joined")

Total Churn = SUM(prod_Churn[Churn Status])

Churn Rate = [Total Churn] / [Total Customers]

Count Predicted Churners = COUNT(Predictions[Customer_ID]) + 0

Title Predicted Churners =
  "COUNT OF PREDICTED CHURNERS : " & COUNT(Predictions[Customer_ID])
```


## STEP 4 — Power BI Dashboard

### Page 1 — Summary

| Section | Visuals |
|---|---|
| **KPI Cards** | Total Customers · New Joiners · Total Churn · Churn Rate % |
| **Demographics** | Churn Rate by Gender · Churn Rate & Volume by Age Group |
| **Account Info** | Churn Rate by Payment Method · Contract Type · Tenure Group |
| **Geography** | Top 5 States by Churn Rate |
| **Churn Distribution** | Churn by Category with tooltip showing detailed reasons |
| **Services** | Churn Rate by Internet Type · All services unpivoted view |

*Add dashboard summary screenshot here:*
![Dashboard Summary](assets/dashboard-summary.png)

### Page 2 — Churn Reasons (Tooltip)
A tooltip page showing granular churn reasons — appears on hover over the Churn Category visual on the Summary page.

*Add churn reasons screenshot here:*
![Churn Reasons](assets/churn-reasons.png)


## STEP 5 — Churn Prediction Model (Python)

### Libraries
```python
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Data Preprocessing
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Drop non-predictive columns
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)

# Label encode categorical variables
columns_to_encode = [
    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service',
    'Multiple_Lines', 'Internet_Service', 'Internet_Type',
    'Online_Security', 'Online_Backup', 'Device_Protection_Plan',
    'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract',
    'Paperless_Billing', 'Payment_Method'
]

label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Encode target variable
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Train & Evaluate the Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Results:**

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| Top Predictors | Contract Type · Tenure · Monthly Charge · Payment Method · Internet Type |

### Feature Importance
```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.show()
```

*Add feature importance chart here:*
![Feature Importance](assets/feature-importance.png)

### Predict on New Customers & Export
```python
# Run model on new joiners
new_predictions = rf_model.predict(new_data)
original_data['Customer_Status_Predicted'] = new_predictions

# Keep only predicted churners
original_data = original_data[original_data['Customer_Status_Predicted'] == 1]
original_data.to_csv("Predictions.csv", index=False)
```

## STEP 6 — Power BI Churn Prediction Page

The model output (`Predictions.csv`) was loaded back into Power BI for a dedicated prediction page:

### Prediction Page Layout

| Section | Visuals |
|---|---|
| **KPI** | Count of Predicted Churners |
| **Customer Grid** | Customer ID · Monthly Charge · Total Revenue · Refunds · Referrals |
| **Demographics** | Predicted Churn by Gender · Age Group · Marital Status |
| **Account Info** | By Payment Method · Contract Type · Tenure Group |
| **Geography** | Predicted Churn Count by State |

*Add prediction page screenshot here:*
![Churn Prediction Page](assets/churn-prediction.png)


## Key Insights

| Finding | Business Implication |
|---|---|
| Month-to-month customers churn at the highest rate | Long-term contracts are the strongest retention lever |
| Fiber optic users churn more than cable users | Service reliability investment is needed |
| Electronic check payers are the highest-risk group | Auto-pay promotion can reduce friction and churn |
| Competitor offers are the #1 churn reason | Pricing and bundle competitiveness is critical |
| Customers with tenure under 6 months are most at risk | Early engagement programs are essential |


## Skills Demonstrated

| Area | Detail |
|---|---|
| ETL & SQL | Staging tables, null handling with ISNULL, production loads, SQL views |
| Power Query | Calculated columns, mapping tables, unpivoting, data type management |
| DAX | Churn Rate, filtered counts, dynamic card titles |
| Power BI Design | 3-page dashboard, tooltip pages, map visuals, KPI cards |
| Python | pandas, LabelEncoder, train/test split, model evaluation, feature importance |
| Machine Learning | Random Forest Classifier, churn scoring on unseen data, CSV export |
| Business Thinking | Translated findings into 5 prioritized, actionable retention recommendations |


## Key Insights

| Finding | Implication |
|---|---|
| Month-to-month customers churn at the highest rate | Long-term contracts drive retention |
| Fiber optic users churn more than cable users | Service quality issues need addressing |
| Electronic check payers are the highest-risk payment group | Auto-pay promotion could reduce churn |
| Competitor offers are the #1 churn reason | Pricing and bundle competitiveness is critical |
| Customers with shorter tenure are most at risk | Early engagement programs are essential |



## Recommendations

**1. Contract Strategy**
Offer discounts or perks (free months, device upgrades) to incentivize customers to move from month-to-month to annual contracts — the single highest-impact retention lever found in this analysis.

**2. Service Quality**
Invest in fiber optic reliability improvements and device support. Dissatisfaction with devices was the second most common churn reason.

**3. Payment Method**
Actively promote credit card or bank transfer auto-pay. Electronic check users show disproportionately high churn — friction in payment may signal lower engagement overall.

**4. Early Retention Programs**
New customers (0–12 months tenure) are the most vulnerable. A structured onboarding and loyalty program in the first year could significantly reduce early churn.

**5. Competitive Intelligence**
Competitor offers are the #1 reason customers leave. Regular benchmarking of competitor pricing and bundle offerings should inform product strategy.


## Machine Learning — Churn Prediction (Python)

A **Random Forest Classifier** was trained to identify customers at risk of churning before they leave.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Results:**

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| Top Predictors | Contract Type, Tenure, Payment Method, Internet Type, Monthly Charges |

The model outputs a **churn probability score per customer**, enabling the retention team to prioritize outreach to highest-risk accounts before they cancel.

*Add feature importance chart here:*
![Feature Importance](assets/feature-importance.png)


## Conclusion

This project demonstrates the full data analytics workflow - from messy raw data to boardroom-ready recommendations. By combining **SQL for data engineering**, **Python for predictive modeling**, and **Power BI for business storytelling**, it shows how data-driven thinking can directly support customer retention strategy in a competitive industry.


*Dataset sourced from [Kaggle]
