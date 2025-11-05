import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("bank-additional-full.csv", sep=';')


print("🔹 First 5 rows:")
print(df.head(), "\n")
print("🔹 Dataset shape:", df.shape, "\n")
print("🔹 Columns:", df.columns.tolist(), "\n")

print("🔹 Missing/Unknown values:\n", df.isin(['unknown']).sum(), "\n")

df.replace('unknown', np.nan, inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Missing values handled.\n")

cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Categorical columns encoded.\n")

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape, "\n")

model = DecisionTreeClassifier(
    criterion='entropy',  # or "gini"
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Model Accuracy: {acc:.4f}\n")
print("🔹 Confusion Matrix:\n", cm, "\n")
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features")
plt.show()

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

print("""
✅ Key Insights:
1. Decision Tree predicts whether a customer subscribes to a term deposit ('y').
2. Features like 'duration', 'contact', 'month', and 'age' have strong predictive power.
3. Model achieved reasonable accuracy — you can improve it using tuning (GridSearchCV) or RandomForest.
4. Handling class imbalance (e.g., via SMOTE or class_weight='balanced') may further improve results.
""")
