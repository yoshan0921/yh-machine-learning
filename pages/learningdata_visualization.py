import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_app import render_sidebar
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Display sidebar
render_sidebar()

st.subheader("Preparation process")

with st.expander("Data"):
    st.write("**Raw data**")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
    df

    st.write("**X**")
    X_raw = df.drop("species", axis=1)
    X_raw

    st.write("**y**")
    y_raw = df.species
    y_raw

with st.expander("Data visualization"):
    st.write("**Pairplot of Features**")
    fig = sns.pairplot(df, hue="species", markers=["o", "s", "D"])
    st.pyplot(fig)

# Data encoding
# Encode X
encode = ["island", "sex"]
df_penguins = pd.get_dummies(X_raw, prefix=encode)

X = df_penguins

# Encode y
target_mapper = {"Adelie": 0,
                 "Chinstrap": 1,
                 "Gentoo": 2}


def target_encode(val):
    return target_mapper[val]


y = y_raw.apply(target_encode)

with st.expander("Data encoding"):
    st.write("**Encoded X (input penguin)**")
    df_penguins
    st.write("**Encoded y**")
    y

with st.expander("Model evaluation"):
    # Load the model
    clf = joblib.load("penguin_classifier_model.pkl")

    # Split data for cross-validation
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    st.write(f'Train_acc：{accuracy_score(y_train, clf.predict(x_train))}')
    st.write(f'Test_acc ：{accuracy_score(y_test, clf.predict(x_test))}')

    st.write("**Feature Importance**")
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
    st.bar_chart(feature_importances.sort_values(ascending=False))

    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, clf.predict(x_test))
    fig, ax = plt.subplots()

    # Font size setting
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax, annot_kws={"size": 6})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel('Predicted', fontsize=6)
    ax.set_ylabel('Actual', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    st.pyplot(fig)
