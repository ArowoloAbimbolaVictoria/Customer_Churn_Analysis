# Customer Churn Analysis 

**Tools:** SQL · Python · Power BI · scikit-learn  
**Domain:** Customer Analytics · Predictive Modeling · Business Intelligence  
**Dataset:** ~7,000+ customers · 30+ features · Sourced from Kaggle


## Project Summary

Telecom companies lose significant revenue every time a customer churns. This project delivers a **full-stack churn analysis** - from raw data cleaning in SQL, through exploratory analysis and machine learning in Python, to an interactive Power BI dashboard - to help business stakeholders understand *who* is churning, *why*, and *what to do about it*.

**Key outcomes:**
- Identified that **month-to-month contract customers churn significantly more** than long-term contract holders
- Found **fiber optic users and electronic check payers** as highest-risk segments
- Built a **Random Forest model with ~80% accuracy** to flag at-risk customers before they leave
- Delivered actionable retention recommendations across contract, payment, and service dimensions


## The Business Problem

Customer churn is one of the costliest challenges in the telecom industry. Every lost customer represents not just lost monthly revenue, but the full lifetime value of that relationship.

This project set out to answer six core business questions:

1. What is the overall churn rate and its revenue impact?
2. Which customer segments are most likely to churn?
3. How do contract type, internet service, and payment method influence churn?
4. What are the most common reasons customers leave?
5. What demographic factors (age, gender, location) correlate with churn?
6. Can we predict which customers are at risk *before* they churn?


## Dataset Overview

Sourced from **Kaggle** - one main file containing customer demographics, account details, subscribed services, payment methods, revenue metrics, and churn labels.

| Field | Description |
|---|---|
| `Customer_ID`, `Gender`, `Age`, `State` | Customer demographics |
| `Tenure_in_Months`, `Contract` | Account history and commitment level |
| `Internet_Type`, `Phone_Service`, `Streaming_Services` | Services subscribed |
| `Monthly_Charge`, `Total_Revenue` | Billing information |
| `Customer_Status`, `Churn_Category`, `Churn_Reason` | Churn outcome and reason |


## Data Cleaning (SQL & Python)

Raw data was imported into **SQL Server** for initial cleaning, then passed to **Python** for further preparation:

- Removed duplicates and resolved missing values
- Removed invalid records (e.g., negative charges)
- Standardized categorical variables (`Yes/No` → `1/0`)
- Engineered new features for better segmentation:
  - **Tenure Group** — bucketed into 0–12, 13–24, 25–36, 36+ months
  - **Monthly Charge Band**  low, medium, high tiers
  - **Churn Flag** - binary indicator (1 = churned, 0 = retained)


## Data Modeling

**In SQL**, data was normalized and structured into a **star schema** - a central fact table for customer records linked to dimension tables for services, demographics, and billing.

**In Power BI**, relationships were built between tables and the following DAX measures were created:

```dax
Churn Rate = DIVIDE([Total Churned Customers], [Total Customers])

Avg Monthly Revenue (Churned) =
    AVERAGEX(
        FILTER(Customers, Customers[Status] = "Churned"),
        Customers[Monthly_Charge]
    )
```


## Power BI Dashboard

The interactive dashboard was structured across six views:

| Page | Content |
|---|---|
| **Overview** | Total Customers, Churn Rate, Revenue at Risk |
| **Demographics** | Churn by Gender, Age Group, Marital Status |
| **Geography** | Churn rate by State (map visual) |
| **Services** | Churn by Internet Type, Phone & Streaming Services |
| **Account Details** | Churn by Contract Type and Payment Method |
| **Churn Reasons** | Top churn categories and granular reasons |



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


##  Skills Demonstrated

| Area | Detail |
|---|---|
| Data Cleaning | SQL Server, Python (pandas), handling nulls, type standardization |
| Data Modeling | Star schema design, Power BI relationships, DAX measures |
| Visualization | Interactive Power BI dashboard across 6 analytical views |
| Exploratory Analysis | Python (matplotlib, seaborn), segment profiling |
| Machine Learning | Random Forest, feature engineering, scikit-learn pipeline |
| Business Thinking | Translated findings into 5 concrete retention recommendations |


## Conclusion

This project demonstrates the full data analytics workflow - from messy raw data to boardroom-ready recommendations. By combining **SQL for data engineering**, **Python for predictive modeling**, and **Power BI for business storytelling**, it shows how data-driven thinking can directly support customer retention strategy in a competitive industry.


*Dataset sourced from [Kaggle]
