# Telecom Customer Churn Prediction

A complete end-to-end project for predicting customer churn in a telecom company.  
I built this to practice the full machine learning workflow — from raw data cleaning and exploratory analysis to feature engineering, model training, evaluation, and a simple interactive dashboard.

---

## چرا این پروژه رو ساختم؟ / Why this project?

ریزش مشتری یکی از مشکلات واقعی و پرهزینه‌ی شرکت‌های مخابراتی و سرویس‌های اشتراکی هست.  
هدفم این بود که هم تحلیل کسب‌وکاری انجام بدم و هم مدل‌های مختلف رو با هم مقایسه کنم تا ببینم کدوم‌شون برای این مسئله بهتر کار می‌کنه.

This is a classic business problem with clear ROI. I wanted something that goes beyond just “train a model and report accuracy”.

---

## Dataset

- Source: IBM Telco Customer Churn dataset
- ~7000 customers
- Features include: tenure, contract type, internet service, payment method, monthly charges, additional services, etc.
- Target: Churn (Yes/No) → ~26.6% positive class

---

## Project Structure

```
telecom-churn-prediction/
├── data/
│   ├── raw/               # original csv
│   └── processed/
├── notebooks/
│   └── 01_eda_and_insights.py
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/                # saved models + encoders
├── reports/figures/       # plots from EDA
├── app/
│   └── streamlit_app.py   # simple dashboard
├── requirements.txt
└── README.md
```

---

## Key Findings from EDA (تحلیل اکتشافی)

After looking at the data, a few patterns stood out clearly:

1. **Contract type is the strongest signal**  
   - Month-to-month → 42.7% churn  
   - One year → 11.3%  
   - Two year → only 2.8%

2. **New customers churn a lot more**  
   - First year (0-12 months): ~47.7%  
   - After 4 years: drops to ~9.5%

3. **Fiber optic customers** have higher churn than DSL (41.9% vs 19.0%).

4. **Electronic check** payment method is associated with much higher churn (45.3%).

5. Customers with **more additional services** (tech support, online security, etc.) tend to stay longer.

These insights already give the business team actionable ideas (e.g. push longer contracts, improve fiber experience, encourage automatic payments).

---

## Modeling Approach

I trained four models:

| Model               | ROC-AUC | PR-AUC | Notes                          |
|---------------------|---------|--------|--------------------------------|
| Logistic Regression | 0.817   | 0.615  | Solid baseline, interpretable  |
| Random Forest       | 0.819   | 0.609  | Good all-rounder               |
| XGBoost             | 0.815   | 0.614  | Strong performance             |
| LightGBM            | 0.811   | 0.608  | Fast and competitive           |

Because the classes are imbalanced, I focused more on **Precision-Recall AUC** and recall for the positive class rather than plain accuracy.

I also applied SMOTE on the training set to help the models see more minority examples.

The final saved model is the one with the highest PR-AUC on the hold-out set.

---

## How to run

```bash
# 1. Create environment (recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA (optional)
cd notebooks
PYTHONPATH=../src python 01_eda_and_insights.py

# 4. Train models
cd ../src
python train_model.py

# 5. Launch the dashboard
cd ../app
streamlit run streamlit_app.py
```

---

## Business Recommendations (پیشنهادهای کسب‌وکاری)

Based on the analysis:

- Focus retention efforts on **month-to-month** customers, especially in their first 12 months.
- Customers on **fiber optic** + **electronic check** are a high-risk group.
- Offering discounts or free months for switching to annual/two-year contracts can significantly reduce churn.
- Customers with tech support and online security have lower churn — consider bundling these services.

---

## What I learned / چیزایی که یاد گرفتم

- Looking at the data carefully before jumping into modeling saves a lot of time.
- For imbalanced problems, PR-AUC and the confusion matrix tell a more honest story than accuracy.
- Feature engineering based on domain knowledge (tenure groups, number of services) helped more than I expected.
- A simple Streamlit app makes the results much easier to show to non-technical people.

---

## Future improvements

- Add SHAP values for model explanation
- Try survival analysis (time-to-churn)
- Deploy the model with FastAPI + Docker
- Collect more recent data and retrain periodically

---


