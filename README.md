# Customer Segmentation Analysis

A comprehensive customer segmentation project using K-Means clustering to identify distinct customer groups based on purchasing behavior, demographics, and engagement patterns.

## 📊 Project Overview

This project analyzes customer data to segment customers into meaningful groups for targeted marketing strategies. Using machine learning techniques, we identify 6 distinct customer segments with unique characteristics and behaviors.

## 📁 Dataset Information

- **Original Dataset**: `customer_segmentation.csv` (2,240 rows)
- **Enhanced Dataset**: `SDV-generated-data.csv` (156,000 rows)
  - Generated using Synthetic Data Vault (SDV) from the original dataset
  - Created in `sdv.ipynb` to provide more robust training data

## 🛠️ Technologies Used

- **Python** 3.11.5
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Data Scaling**: StandardScaler
- **Clustering**: K-Means
- **Dimensionality Reduction**: Principal Component Analysis(PCA)
- **Model Persistence**: joblib

## 📋 Features Analyzed

### Original Features
- `Income`: Customer annual income
- `Age`: Customer age (derived from Year_Birth)
- `Recency`: Days since last purchase
- `Education`: Education level
- `Marital_Status`: Marital status
- `NumWebPurchases`: Number of web purchases
- `NumStorePurchases`: Number of store purchases
- `NumWebVisitsMonth`: Monthly web visits

### Engineered Features
- `Age`: Calculated as 2025 - Year_Birth
- `Total_children`: Sum of Kidhome + Teenhome
- `Total_spending`: Sum of all product spending categories
- `Customer_since`: Days since customer registration
- `AcceptedAny`: Binary indicator for campaign acceptance
- `Age_Group`: Categorical age groups (18-29, 30-39, etc.)

## 🔍 Analysis Pipeline

### 1. Data Preprocessing
- Load and explore the dataset
- Handle missing values with dropna()
- Convert date columns to datetime format
- Create derived features

### 2. Exploratory Data Analysis (EDA)
- **Age Distribution**: Histogram with KDE
- **Income Distribution**: Histogram with KDE  
- **Spending Distribution**: Histogram with KDE
- **Education vs Income**: Box plots
- **Marital Status vs Spending**: Box plots
- **Correlation Matrix**: Heatmap of numerical features

### 3. Advanced Analysis
- **Pivot Analysis**: Income by Education and Marital Status
- **Group Analysis**: Spending patterns by education level
- **Campaign Analysis**: Acceptance rates by marital status
- **Age Group Analysis**: Income patterns across age groups

### 4. Machine Learning Pipeline
- **Feature Selection**: 7 key numerical features
- **Data Scaling**: StandardScaler for normalization
- **Optimal Clusters**: Elbow method (k=2 to k=9)
- **K-Means Clustering**: 6 clusters identified
- **PCA Visualization**: 2D representation of clusters

## 🎯 Customer Segments Identified

| Cluster | Description | Characteristics |
|---------|-------------|-----------------|
| 0 | **Premium Customers** | High income, high spending |
| 1 | **Value Seekers** | Medium income, high spending |
| 2 | **Digital Buyers** | High web purchases, low store purchases |
| 3 | **Thrifty Customers** | Low income, medium spending |
| 4 | **Potential Loyalists** | Young, medium income |
| 5 | **Dormant Customers** | Low recency, inactive |

## 📈 Key Insights

1. **Income-Education Correlation**: Higher education levels correlate with higher incomes
2. **Spending Patterns**: Marital status significantly influences spending behavior
3. **Digital Preference**: Clear distinction between online and offline shoppers
4. **Age Groups**: Different income patterns across age demographics
5. **Campaign Response**: Varying acceptance rates by marital status

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Analysis
1. Ensure you have the dataset files:
   - `customer_segmentation.csv` (original)
   - `SDV-generated-data.csv` (enhanced)

2. Run the main analysis:
   ```bash
   jupyter notebook main.ipynb
   ```

3. For synthetic data generation:
   ```bash
   jupyter notebook sdv.ipynb
   ```

## 📁 File Structure
```
├── customer_segmentation.csv    # Original dataset
├── SDV-generated-data.csv      # Enhanced dataset
├── main.ipynb                  # Main analysis notebook
├── sdv.ipynb                   # Synthetic data generation
├── kmeans_model.pkl            # Trained K-Means model
├── scaler.pkl                  # Fitted StandardScaler
├── segmentation.py             # Python script version
└── README.md                   # Project documentation
```

## 🔮 Model Deployment

The trained models are saved for future predictions:
- `kmeans_model.pkl`: Trained K-Means clustering model
- `scaler.pkl`: Fitted StandardScaler for data preprocessing

### Loading and Using the Model
```python
import joblib
import pandas as pd

# Load the models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare new data
new_data = [[age, income, total_spending, web_purchases, store_purchases, web_visits, recency]]
new_data_scaled = scaler.transform(new_data)

# Predict cluster
cluster = kmeans.predict(new_data_scaled)
```

## 📊 Visualizations Generated

1. **Distribution Plots**: Age, Income, and Spending distributions
2. **Box Plots**: Categorical analysis of Education and Marital Status
3. **Correlation Heatmap**: Feature relationships
4. **Bar Charts**: Group analyses and campaign acceptance rates
5. **PCA Scatter Plot**: 2D visualization of customer segments
6. **Elbow Plot**: Optimal cluster selection

## 💡 Business Applications

- **Targeted Marketing**: Customize campaigns for each segment
- **Product Development**: Align offerings with segment preferences
- **Customer Retention**: Identify and re-engage dormant customers
- **Pricing Strategy**: Optimize pricing for different segments
- **Channel Strategy**: Focus digital vs. physical presence

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements or additional analysis.

---
*This analysis provides actionable insights for customer-centric business strategies through data-driven segmentation.*
