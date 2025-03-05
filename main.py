import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

# Load the assignment data file
data = pd.read_csv("assignment1dataset.csv")

"""
    The following is an extra section for Exploratory Data Analysis
    I know that the data is already clean and ready for modeling so this is an extra section
"""
# Display the first few rows
print(data.head(), "\n \n")

# Overview of the dataset
print(data.info(), "\n \n")

# Checking for missing values
print(data.isnull().sum(), "\n \n")

# Statistical Summary
print(data.describe(), "\n \n")

# Calculate correlations of RevenuePerDay with other columns
correlations = data.corr()[['RevenuePerDay']].sort_values(by='RevenuePerDay', ascending=False)

# Create a figure with subplots
plt.figure(figsize=(14, 6))
plt.suptitle('Exploratory Data Analysis', fontsize=24, fontweight='bold')

# First Subplot: Heatmap of correlations with RevenuePerDay
plt.subplot(1, 2, 1)
sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation with RevenuePerDay')

# Second Subplot: Revenue Per Day Distribution
plt.subplot(1, 2, 2)
sns.histplot(data['RevenuePerDay'], kde=True, color='purple')
plt.title('Revenue Per Day Distribution')
plt.xlabel('Revenue Per Day')
plt.ylabel('Frequency')

# Display the plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# A function for applying linear regression
"""
    I used default parameters in case i didn't pass the learning rate or the epochs
"""


def LinearRegression(x, y, lr=0.0000001, e=100):
    n = float(len(x))  # Number of elements in X
    # m and c are zeros initially as required
    m = 0
    c = 0
    for i in range(e):
        Y_pred = m * x + c  # The current predicted value of Y
        D_m = (-2 / n) * sum((y - Y_pred) * x)  # Derivative wrt m
        D_c = (-2 / n) * sum(y - Y_pred)  # Derivative wrt c
        m = m - lr * D_m  # Update m
        c = c - lr * D_c  # Update c
    prediction = m * x + c
    return prediction


# Applying simple linear regression to predict the revenue per day of the cafe

# 1. Applying the model between Y = RevenuePerDay and X = NCustomersPerDay
Y = data["RevenuePerDay"]
X = data["NCustomersPerDay"]
L = 0.0000001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent
Y1 = LinearRegression(X, Y, L, epochs)

# 2. Applying the model between Y = RevenuePerDay and X = AverageOrderValue
Y = data["RevenuePerDay"]
X = data["AverageOrderValue"]
L = 0.00001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent
Y2 = LinearRegression(X, Y, L, epochs)

# 3. Applying the model between Y = RevenuePerDay and X = MarketingSpendPerDay
Y = data["RevenuePerDay"]
X = data["MarketingSpendPerDay"]
L = 0.00001  # The learning Rate
epochs = 5000  # The number of iterations to perform gradient descent
Y3 = LinearRegression(X, Y, L, epochs)

# 4. Applying the model between Y = RevenuePerDay and X = LocationFootTraffic
Y = data["RevenuePerDay"]
X = data["LocationFootTraffic"]
L = 0.0000001  # The learning Rate
epochs = 20000  # The number of iterations to perform gradient descent
Y4 = LinearRegression(X, Y, L, epochs)

# Creating two calculated columns or features
data['MarketingEfficiency'] = data['NCustomersPerDay'] / data['MarketingSpendPerDay']
data['EmployeeProductivity'] = data['NCustomersPerDay'] / data['NEmployees']

# 5. Applying the model between Y = RevenuePerDay and X = MarketingEfficiency
Y = data["RevenuePerDay"]
X = data["MarketingEfficiency"]
L = 0.000001  # The learning Rate
epochs = 15000  # The number of iterations to perform gradient descent
Y5 = LinearRegression(X, Y, L, epochs)

# 6. Applying the model between Y = RevenuePerDay and X = MarketingEfficiency
Y = data["RevenuePerDay"]
X = data["EmployeeProductivity"]
L = 0.0000001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent
Y6 = LinearRegression(X, Y, L, epochs)

# Create a 3x2 grid for the six plots
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Linear Regression Models', fontsize=24, fontweight='bold')

# List of X variables, predictions, and labels
models = [
    ('NCustomersPerDay', Y1, 'Y = RevenuePerDay vs X = NCustomersPerDay'),
    ('AverageOrderValue', Y2, 'Y = RevenuePerDay vs X = AverageOrderValue'),
    ('MarketingSpendPerDay', Y3, 'Y = RevenuePerDay vs X = MarketingSpendPerDay'),
    ('LocationFootTraffic', Y4, 'Y = RevenuePerDay vs X = LocationFootTraffic'),
    ('MarketingEfficiency', Y5, 'Y = RevenuePerDay vs X = MarketingEfficiency'),
    ('EmployeeProductivity', Y6, 'Y = RevenuePerDay vs X = EmployeeProductivity')
]

# Loop through the models and add them to the subplots
for i, (x_label, y_pred, title) in enumerate(models):
    row, col = divmod(i, 2)  # Determine row and column position
    axes[row, col].scatter(data[x_label], data['RevenuePerDay'], color='blue', alpha=0.6)
    axes[row, col].plot(data[x_label], y_pred, color='red', linewidth=2)
    # axes[row, col].set_title(title, fontsize=14)
    axes[row, col].set_xlabel(x_label, fontsize=12)
    axes[row, col].set_ylabel('RevenuePerDay', fontsize=12)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Data for each model (Model Name, [(lr, epoch, mse), ...])
results = {
    'NCustomersPerDay': [
        (1e-07, 1000, 464122.89), (1e-07, 5000, 464114.67), (1e-07, 10000, 464104.39), (1e-07, 20000, 464083.85),
        (1e-06, 1000, 464104.39), (1e-06, 5000, 464022.32), (1e-06, 10000, 463920.05), (1e-06, 20000, 463716.64),
        (1e-05, 1000, 463920.05), (1e-05, 5000, 463115.23), (1e-05, 10000, 462141.60), (1e-05, 20000, 460297.56)
    ],
    'AverageOrderValue': [
        (1e-07, 1000, 4558706.59), (1e-07, 5000, 4291205.08), (1e-07, 10000, 3982712.65), (1e-07, 20000, 3442824.29),
        (1e-06, 1000, 3982688.86), (1e-06, 5000, 2299054.19), (1e-06, 10000, 1348695.37), (1e-06, 20000, 802799.21),
        (1e-05, 1000, 1348221.49), (1e-05, 5000, 694142.74), (1e-05, 10000, 693339.81), (1e-05, 20000, 692753.49)
    ],
    'MarketingSpendPerDay': [
        (1e-07, 1000, 1405919.58), (1e-07, 5000, 1405723.57), (1e-07, 10000, 1405478.67), (1e-07, 20000, 1404989.20),
        (1e-06, 1000, 1405478.67), (1e-06, 5000, 1403523.59), (1e-06, 10000, 1401090.17), (1e-06, 20000, 1396257.93),
        (1e-05, 1000, 1401090.16), (1e-05, 5000, 1382033.90), (1e-05, 10000, 1359210.88), (1e-05, 20000, 1316695.75)
    ],
    'MarketingEfficiency': [
        (1e-07, 1000, 4619712.56), (1e-07, 5000, 4584577.70), (1e-07, 10000, 4541871.85), (1e-07, 20000, 4460334.72),
        (1e-06, 1000, 4541870.68), (1e-06, 5000, 4243782.97), (1e-06, 10000, 3959691.41), (1e-06, 20000, 3587157.28),
        (1e-05, 1000, 3959627.17), (1e-05, 5000, 3122340.10), (1e-05, 10000, 2800086.44), (1e-05, 20000, 2341975.44)
    ],
    'EmployeeProductivity': [
        (1e-07, 1000, 2364323.60), (1e-07, 5000, 1837471.85), (1e-07, 10000, 1835887.82), (1e-07, 20000, 1834059.03),
        (1e-06, 1000, 1835887.81), (1e-06, 5000, 1828592.79), (1e-06, 10000, 1819547.57), (1e-06, 20000, 1801698.90),
        (1e-05, 1000, 1819547.50), (1e-05, 5000, 1750035.59), (1e-05, 10000, 1669879.90), (1e-05, 20000, 1529623.44)
    ],
    'LocationFootTraffic': [
        (1e-06, 15000, 1676644.3418139352), (1e-06, 20000, 1673681.140185226), (1e-06, 25000, 1670730.0610188749),
        (1e-07, 15000, 1684705.9284421033), (1e-07, 20000, 1684405.7573226432), (1e-07, 25000, 1684105.7092298674),
        (1e-08, 15000, 1685517.0053794126), (1e-08, 20000, 1685486.949481843), (1e-08, 25000, 1685456.8948163567)
    ]
}


# Function to extract and group data by learning rate and epochs
def extract_data_by_lr_epochs(results):
    lr_data, epoch_data = {}, {}

    for model, values in results.items():
        lr_mse = {}
        epoch_mse = {}

        for lr, epoch, mse in values:
            if lr not in lr_mse:
                lr_mse[lr] = []
            if epoch not in epoch_mse:
                epoch_mse[epoch] = []

            lr_mse[lr].append(mse)
            epoch_mse[epoch].append(mse)

        # Average MSE for each learning rate and epoch
        lr_data[model] = (list(lr_mse.keys()), [sum(v)/len(v) for v in lr_mse.values()])
        epoch_data[model] = (list(epoch_mse.keys()), [sum(v)/len(v) for v in epoch_mse.values()])

    return lr_data, epoch_data


# Extracted data
lr_data, epoch_data = extract_data_by_lr_epochs(results)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Learning Rate vs. MSE
for model, (lrs, mse) in lr_data.items():
    axes[0].plot(lrs, mse, marker='o', label=model)

axes[0].set_xscale('log')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Learning Rate vs. MSE')
axes[0].legend()
axes[0].grid(True)

# Plot Epochs vs. MSE
for model, (epochs, mse) in epoch_data.items():
    axes[1].plot(epochs, mse, marker='o', label=model)

axes[1].set_xscale('log')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Epochs vs. MSE')
axes[1].legend()
axes[1].grid(True)

# Show both plots
plt.tight_layout()
plt.show()
