# ================================
# Final Assessment – Practical
# Comprehensive Stock Price Research
# ================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D

# ================================
# 2. Load Datasets
# Replace with your actual CSV paths from GitHub repo
# ================================
adobe = pd.read_csv("Adobe.csv", parse_dates=['Date'], index_col='Date')
microsoft = pd.read_csv("Microsoft.csv", parse_dates=['Date'], index_col='Date')
oracle = pd.read_csv("Oracle.csv", parse_dates=['Date'], index_col='Date')
salesforce = pd.read_csv("Salesforce.csv", parse_dates=['Date'], index_col='Date')

# ================================
# 3. Descriptive Statistics
# ================================
print(adobe.describe())
print(microsoft.describe())

# ================================
# 4. NumPy Research Examples
# ================================
returns = np.log(adobe['Close'] / adobe['Close'].shift(1))
annual_volatility = returns.std() * np.sqrt(252)

# Drawdown
cum_return = np.cumprod(1 + returns.fillna(0))
running_max = np.maximum.accumulate(cum_return)
drawdown = (cum_return - running_max) / running_max

# ================================
# 5. Pandas Feature Engineering
# ================================
adobe['Year'] = adobe.index.year
adobe['MA20'] = adobe['Close'].rolling(20).mean()
adobe['Mom5'] = adobe['Close'] / adobe['Close'].shift(5)
adobe['Vol20'] = returns.rolling(20).std()

# ================================
# 6. Seaborn Visualization
# ================================
sns.histplot(returns, kde=True)
plt.title("Return Distribution")
plt.show()

sns.heatmap(adobe.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 7. Scikit-Learn Models
# ================================
# Example: Predict tomorrow up/down
adobe['Target'] = (adobe['Close'].shift(-1) > adobe['Close']).astype(int)

X = adobe[['MA20','Mom5','Vol20']].dropna()
y = adobe['Target'].dropna()

tscv = TimeSeriesSplit(n_splits=5)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

for train_idx, test_idx in tscv.split(X):
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = pipeline.predict(X.iloc[test_idx])
    print("Fold accuracy:", (preds == y.iloc[test_idx]).mean())

# ================================
# 8. Clustering & PCA
# ================================
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(X)
adobe['Cluster'] = clusters

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
plt.scatter(pca_result[:,0], pca_result[:,1], c=clusters)
plt.title("Market Regime Clusters")
plt.show()

# ================================
# 9. TensorFlow/Keras Deep Learning
# ================================
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Example LSTM setup
lstm_model = Sequential([
    LSTM(128, input_shape=(60,1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# ================================
# 10. Backtesting & Metrics
# ================================
# Example Sharpe Ratio
sharpe = returns.mean() / returns.std()
print("Sharpe Ratio:", sharpe)

# ================================
# 11. Research Journal (Manual Step)
# ================================
# For each experiment:
# - Hypothesis
# - Dataset
# - Features
# - Model
# - Parameters
# - Metrics
# - Results
# - Conclusions
