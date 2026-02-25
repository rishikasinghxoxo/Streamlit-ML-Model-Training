import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

st.title("Machine Learning Model Trainer")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # -------------------------
    # Select Features & Target
    # -------------------------
    X_cols = st.multiselect("Select Feature Columns (X)", columns)
    y_col = st.selectbox("Select Target Column (y)", columns)

    # -------------------------
    # Train-Test Split
    # -------------------------
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    # -------------------------
    # Model Selection
    # -------------------------
    model_name = st.selectbox(
        "Select Model",
        ["Logistic Regression",
         "Decision Tree",
         "Random Forest",
         "Gaussian Naive Bayes"]
    )

    # -------------------------
    # Train Button
    # -------------------------
    if st.button("Train Model"):

        if len(X_cols) == 0:
            st.error("Please select at least one feature column.")
        else:
            X = df[X_cols]
            y = df[y_col]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )

            # Model selection
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()

            elif model_name == "Random Forest":
                model = RandomForestClassifier()

            else:
                model = GaussianNB()

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Accuracy
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            st.success("Model Trained Successfully")

            st.write(f"### Training Accuracy: {train_acc:.4f}")
            st.write(f"### Testing Accuracy: {test_acc:.4f}")

            # Confusion Matrix (Test)
            cm = confusion_matrix(y_test, y_test_pred)

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)