import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Set title of the Streamlit app
st.title("Health Data Analysis and Classification")

# Load data
food_data = pd.read_csv("./health.csv")

# Display data
st.dataframe(food_data.head(10))

# Display column names and missing values count
st.write("Column names:", food_data.columns)
missing_values_count = food_data.isnull().sum()
st.write("Missing values count:", missing_values_count)

# Calculate percentage of missing values
total_cells = np.product(food_data.shape)
total_missing = missing_values_count.sum()
total_missing_percentage = (total_missing / total_cells) * 100
st.write("Percentage of missing values:", "%.2f%%" % total_missing_percentage)

# Age grouping
age_bins = [0, 18, 35, 50, 65, 100]
age_labels = ['0-18', '19-35', '36-50', '51-65', '66+']
food_data['age_group'] = pd.cut(food_data['age'], bins=age_bins, labels=age_labels, right=False)

# Categorical columns encoding
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
food_data_encoded = pd.get_dummies(food_data, columns=categorical_columns, drop_first=True)

# Heatmap
heatmap_data = food_data.pivot_table(index='age_group', columns='gender', values='bmi', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(data=heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f",
            linewidths=0.5, linecolor='gray', cbar=True, cbar_kws={'label': 'Average BMI'})
plt.xlabel('Gender')
plt.ylabel('Age Group')
plt.title('Heatmap: Average BMI by Age Group and Gender')
st.pyplot(plt)
plt.close()

# Correlation matrix for encoded categorical columns
correlation_matrix = food_data_encoded.drop(columns=['age_group']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f",
            linewidths=0.5, linecolor='gray', cbar=True, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix')
st.pyplot(plt)
plt.close()

# Clean data and prepare for modeling
food_data['bmi'] = pd.to_numeric(food_data['bmi'], errors='coerce')
food_data.dropna(subset=['bmi'], inplace=True)

X = food_data.drop(columns=['stroke'])
y = food_data['stroke']

categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display model information
st.write("Models are ready for predictions.")

# Feature selection and preparation
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
X_train_numeric = X_train[numerical_columns]
X_test_numeric = X_test[numerical_columns]

# Initialize accuracy variables
accuracy_gb = 0.0
accuracy_rf = 0.0
accuracy_logreg = 0.0

# Create a dropdown to select a model
model_selection = st.selectbox("Pick a model", ["Gradient Boosting", "Random Forest", "Logistic Regression"])

if model_selection == "Gradient Boosting":
    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(random_state=42)
    gb_classifier.fit(X_train_numeric, y_train)

    # Predictions and evaluation for Gradient Boosting Classifier
    y_pred_gb = gb_classifier.predict(X_test_numeric)
    st.write("Gradient Boosting Classifier:")
    classification_rep_gb = classification_report(y_test, y_pred_gb, zero_division=1)
    classification_rep_gb = classification_rep_gb.replace('\n', '\n\n')  # Adjust formatting
    st.text(classification_rep_gb)

    conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
    st.write("Confusion matrix:")
    st.write(conf_matrix_gb)

    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    st.write("Gradient Boosting Classifier Accuracy:", accuracy_gb)

elif model_selection == "Random Forest":
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_numeric, y_train)

    # Predictions and evaluation for Random Forest Classifier
    y_pred_rf = rf_classifier.predict(X_test_numeric)
    st.write("Random Forest Classifier:")
    classification_rep_rf = classification_report(y_test, y_pred_rf, zero_division=1)
    classification_rep_rf = classification_rep_rf.replace('\n', '\n\n')  # Adjust formatting
    st.text(classification_rep_rf)

    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    st.write("Confusion matrix:")
    st.write(conf_matrix_rf)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write("Random Forest Classifier Accuracy:", accuracy_rf)

elif model_selection == "Logistic Regression":
    # Logistic Regression Classifier
    logreg = LogisticRegression(max_iter=10000, class_weight='balanced')
    logreg.fit(X_train_numeric, y_train)

    # Predictions and evaluation for Logistic Regression Classifier
    y_pred_logreg = logreg.predict(X_test_numeric)
    st.write("Logistic Regression Classifier:")
    classification_rep_logreg = classification_report(y_test, y_pred_logreg, zero_division=1)
    classification_rep_logreg = classification_rep_logreg.replace('\n', '\n\n')  # Adjust formatting
    st.text(classification_rep_logreg)

    conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
    st.write("Confusion matrix:")
    st.write(conf_matrix_logreg)

    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    st.write("Logistic Regression Classifier Accuracy:", accuracy_logreg)



# Display confusion matrix heatmaps (for the selected model)
if model_selection == "Gradient Boosting":
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_gb, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Gradient Boosting Classifier")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(plt)
    plt.close()

elif model_selection == "Random Forest":
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest Classifier")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(plt)
    plt.close()

elif model_selection == "Logistic Regression":
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_logreg, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression Classifier")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()  # Ensures heatmaps fit in one line
    st.pyplot(plt)
    plt.close()


# Create a bar graph to compare classifier accuracies
classifier_names = ["Gradient Boosting", "Random Forest", "Logistic Regression"]
classifier_accuracies = [accuracy_gb, accuracy_rf, accuracy_logreg]

plt.figure(figsize=(10, 6))
bars = plt.bar(classifier_names, classifier_accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison')

# Display accuracy values on the bars
for bar in bars:
    yval = round(bar.get_height(), 3)
    plt.text(bar.get_x() + bar.get_width()/2, yval, yval, ha='center', va='bottom', color='black', fontweight='bold')

st.pyplot(plt)
plt.close()

# Determine best method based on accuracy
best_accuracy = max(accuracy_gb, accuracy_rf, accuracy_logreg)
best_method = None
if best_accuracy == accuracy_gb:
    best_method = "Gradient Boosting Classifier"
elif best_accuracy == accuracy_rf:
    best_method = "Random Forest Classifier"
elif best_accuracy == accuracy_logreg:
    best_method = "Logistic Regression Classifier"

# Display all accuracy values
st.write("Gradient Booster Classifier Accuracy:", accuracy_gb)
st.write("Random Forest Classifier Accuracy:", accuracy_rf)
st.write("Logistic Regression Classifier Accuracy:", accuracy_logreg)

