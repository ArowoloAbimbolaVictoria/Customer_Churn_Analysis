# Customer_Churn_Analysis
# 📊 Customer Churn Analysis  

## 📝 Introduction  
Customer churn is one of the biggest challenges for telecom companies, as it directly impacts revenue and customer growth. This project leverages **SQL, Python, and Power BI** to analyze churn patterns, uncover the key drivers of customer attrition, and provide data-driven recommendations for customer retention.  

By analyzing customer demographics, services subscribed, billing methods, and churn reasons, this project supports strategic decision-making aimed at reducing churn and increasing customer loyalty.  
  

## 🛠 Skills Demonstrated  
- Data Cleaning and Preparation (**SQL & Python**)  
- Data Modeling (**SQL & Power BI**)  
- DAX Calculations and KPIs (**Power BI**)  
- Interactive Dashboard Design (**Power BI**)  
- Exploratory Data Analysis (**Python**)  
- Machine Learning (**Python, scikit-learn**)  
- Storytelling with Data  
- Critical Thinking and Business Insight Generation  

---

## 📂 Data Sourcing  
The dataset was sourced from **Kaggle** and contains one main file:  

- **Churn Data**: Customer demographics, account details, subscribed services, payment methods, revenue metrics, and churn labels.  

**Size:** ~7,000+ rows × 30+ columns  

**Key Fields:**  
- Customer_ID, Gender, Age, State  
- Tenure_in_Months, Contract, Payment_Method  
- Internet_Type, Phone_Service, Streaming_Services  
- Monthly_Charge, Total_Revenue  
- Customer_Status, Churn_Category, Churn_Reason  


## ❓ Problem Statement  
This analysis sought to address the following questions:  
1. What is the overall churn rate and revenue impact?  
2. Which customer segments are most likely to churn?  
3. How do contract type, internet service, and payment method influence churn?  
4. What are the most common churn categories and reasons?  
5. What demographic factors (age, gender, marital status, state) correlate with churn?  
6. Can we build a predictive model to identify customers at risk of churn?  


## 🧹 Data Cleaning  
After importing the dataset into **SQL Server** and **Python**:  
- Removed duplicates and handled missing values.  
- Fixed invalid records (e.g., negative charges).  
- Standardized categorical variables (Yes/No → 1/0).  
- Created derived fields for better segmentation:  
  - **Tenure Group** (0–12 months, 13–24, etc.)  
  - **Monthly Charge Band**  
  - **Churn Flag** (binary indicator for churn).  

This ensured the data was consistent and ready for modeling and visualization.  


## 🗂 Data Modeling  
- In **SQL**, tables were normalized and transformed into a **star schema** for easier querying.  
- In **Power BI**, relationships were built between customer details, services, and revenue metrics.  
- Calculated columns and DAX measures were created, including:  

```DAX
Churn Rate = DIVIDE([Total Churned Customers], [Total Customers])

Avg Monthly Revenue (Churned) =
    AVERAGEX(
        FILTER(Customers, Customers[Status] = "Churned"),
        Customers[Monthly_Charge]
    )

## 📊 Data Visualization (Power BI Dashboard)

The **Power BI dashboard** was designed with the following sections:

- **Overview Page**: Total Customers, Churned Customers, Churn Rate, Total Revenue.  
- **Demographics**: Churn by Gender, Age Group, Marital Status.  
- **Geography**: Churn by State.  
- **Services**: Churn by Internet Type, Phone Service, and Streaming Services.  
- **Account Details**: Churn by Contract Type and Payment Method.  
- **Churn Reasons**: Top categories and detailed reasons for churn.  

📷 *Dashboard screenshots are available in the repository.*  



## 🔑 Insights  

Key findings from the analysis include:  

- Customers on **month-to-month contracts** churned significantly more than those on yearly contracts.  
- **Fiber optic customers** had a higher churn rate compared to cable users.  
- Customers paying via **electronic check** were more likely to churn compared to those using credit cards or bank transfers.  
- **Competitor offers** were the top churn reason, followed by dissatisfaction with devices and service issues.  
- Customers with **longer tenure showed greater loyalty**, while newer customers were more likely to leave.  



## 💡 Recommendations  

Based on the analysis, the following actions are recommended:  

- **Contract Strategy**: Incentivize long-term contracts with discounts or perks.  
- **Service Quality**: Improve fiber optic reliability and device support.  
- **Payment Options**: Promote credit card and bank transfer payments.  
- **Customer Retention Programs**: Launch loyalty rewards for new customers within their first year.  
- **Competitive Benchmarking**: Monitor competitor offers closely and adjust bundles/pricing.  



## 🤖 Machine Learning (Python)  

A **Random Forest Classifier** was trained to predict churn likelihood.  

- **Accuracy**: ~80%  
- **Key Predictors**: Contract Type, Tenure, Payment Method, Internet Type, Monthly Charges.  

The model highlights **at-risk customers**, enabling proactive retention campaigns.  



## ✅ Conclusion  

This churn analysis project provides actionable insights to **reduce customer attrition** and improve retention strategies.  

By combining **SQL for data cleaning**, **Python for predictive analytics**, and **Power BI for interactive dashboards**, the project demonstrates the full potential of **data-driven decision-making in the telecom industry**.  


