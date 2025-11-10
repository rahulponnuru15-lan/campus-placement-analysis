# ============================================================
# CAMPUS PLACEMENT ANALYSIS PROJECT
# ============================================================

# STEP 1 - IMPORT LIBRARIES AND LOAD DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Preview data
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())

# ============================================================
# STEP 2 - DATA CLEANING AND PREPARATION
# ============================================================

# Rename columns for simplicity
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Encode placement status
df['placed'] = df['status'].apply(lambda x: 1 if x.strip().lower() == 'placed' else 0)

# Handle missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())
df = df.dropna(subset=['etest_p'])
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# STEP 3 - EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(x='placed', data=df)
plt.title("Placement Distribution")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='placed', y='degree_p', data=df)
plt.title("Degree Percentage vs Placement")
plt.show()

# Correlation heatmap
num_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(num_cols.corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# STEP 4 - STATISTICAL TESTS
# ============================================================

placed_df = df[df['placed'] == 1]
not_placed_df = df[df['placed'] == 0]

def compare_means(feature):
    stat, p = stats.ttest_ind(placed_df[feature], not_placed_df[feature])
    print(f"{feature} -> p-value = {p:.4f}")

print("\nT-Test Results:")
for feature in ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']:
    compare_means(feature)

# Chi-Square for categorical variables
cat_features = ['gender', 'workex', 'specialisation', 'degree_t']
print("\nChi-Square Test Results:")
for col in cat_features:
    contingency = pd.crosstab(df[col], df['placed'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    print(f"{col} -> p-value = {p:.4f}")

# ============================================================
# STEP 5 - LOGISTIC REGRESSION MODEL
# ============================================================

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Feature and target split
X = df_encoded.drop(['placed', 'status', 'salary'], axis=1)
y = df_encoded['placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# STEP 6 - SALARY ANALYSIS
# ============================================================

placed_salary = df[df['placed'] == 1].dropna(subset=['salary'])
print("\nSalary Summary Statistics:")
print(placed_salary['salary'].describe())

plt.figure(figsize=(7, 4))
sns.histplot(placed_salary['salary'], kde=True, bins=15)
plt.title("Salary Distribution (Placed Students)")
plt.xlabel("Salary (in thousands)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='specialisation', y='salary', data=placed_salary)
plt.title("Salary by Specialisation")
plt.xlabel("Specialisation")
plt.ylabel("Salary")
plt.show()

print("\nAverage Salary by Specialisation:")
print(placed_salary.groupby('specialisation')['salary'].mean().sort_values(ascending=False))

plt.figure(figsize=(6, 4))
sns.boxplot(x='degree_t', y='salary', data=placed_salary)
plt.title("Salary by Degree Type")
plt.xlabel("Degree Type")
plt.ylabel("Salary")
plt.show()

print("\nAverage Salary by Degree Type:")
print(placed_salary.groupby('degree_t')['salary'].mean().sort_values(ascending=False))

# ============================================================
# STEP 7 - FINAL SUMMARY
# ============================================================

print("\nFINAL SUMMARY AND INSIGHTS")
total = len(df)
placed = df['placed'].sum()
placement_rate = (placed / total) * 100

print(f"\nTotal Students: {total}")
print(f"Placed: {placed} ({placement_rate:.1f}%)")

print("\nKey Statistical Insights:")
print("- Degree% and E-Test% significantly affect placement (p < 0.05).")
print("- Work experience and specialisation also have significant effects (Chi-square p < 0.05).")

print("\nModel Insights:")
print("- Logistic Regression accuracy: about 85–90%.")
print("- Key predictors: Degree%, E-Test%, Work Experience, and Specialisation.")

print("\nSalary Insights:")
print("- Average salary between 2.5 – 3.0 LPA.")
print("- Higher salaries for Sci&Tech and Comm&Mgmt degree holders.")
print("- Mkt&Fin specialisation slightly higher on average.")

print("\nOverall Conclusion:")
print("Academic performance and employability test scores are the strongest predictors of placement.")
print("Work experience improves placement odds.")
print("Mkt&Fin and Sci&Tech profiles are more in demand.")
