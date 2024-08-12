import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_app import render_sidebar
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Display sidebar
render_sidebar()

st.subheader("Learning Data")

# Display raw data
with st.expander("Raw data"):
    st.write("**Raw data**")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    )
    df

    st.write("**X (Input value)**")
    X_raw = df.drop("species", axis=1)
    X_raw

    st.write("**y (Target value)**")
    y_raw = df.species
    y_raw

# Data visualization
with st.expander("Data visualization"):
    st.write("**Pairplot of Features**")
    fig = sns.pairplot(df, hue="species", markers=["o", "s", "D"])
    st.pyplot(fig)

# Data encoding
encode = ["island", "sex"]
X_encoded = pd.get_dummies(X_raw, prefix=encode)

target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
y_encoded = y_raw.map(target_mapper)

with st.expander("Data encoding"):
    st.write("**Encoded X (Input value)**")
    X_encoded
    st.write("**Encoded y (Target value)**")
    y_encoded

# Load the model
clf = joblib.load("penguin_classifier_model.pkl")

# Split data for cross-validation
x_train, x_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=1
)

with st.expander("Model evaluation"):
    st.write("**Model Accuracy**")
    train_acc = accuracy_score(y_train, clf.predict(x_train))
    test_acc = accuracy_score(y_test, clf.predict(x_test))

    # Create two columns
    col1, col2 = st.columns(2)
    # Display the metrics in each column
    with col1:
        st.metric(label="Training Accuracy",
                  value=f"{accuracy_score(y_train, clf.predict(x_train)):.2%}")
    with col2:
        st.metric(label="Testing Accuracy",
                  value=f"{accuracy_score(y_test, clf.predict(x_test)):.2%}")

    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, clf.predict(x_test))
    fig, ax = plt.subplots()

    # Font size setting
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=ax, annot_kws={"size": 6})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("Actual", fontsize=6)
    ax.tick_params(axis="both", which="major", labelsize=6)
    st.pyplot(fig)

    st.write("**Feature Importance**")
    feature_importances = pd.Series(
        clf.feature_importances_, index=X_encoded.columns)
    st.bar_chart(feature_importances.sort_values(ascending=False))

# Custom CSS for expanding label font size
st.markdown(
    """
    <style>
    h2, h3 {
        font-size: 1.25rem !important;
    }
    div[data-testid="stExpander"] summary p {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
