# Exploratory Data Analysis (EDA)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/processed_data/processed_traffic_data.csv")

# Visualize traffic trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="date_time", y="traffic_volume")
plt.title("Traffic Volume Over Time")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()