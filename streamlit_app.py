import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

# Function for rendering the sidebar


def render_sidebar():
    with st.sidebar:
        st.image("./images/PenguinClassifier_transparent.png")
        st.info('The purpose of this application is to experience the process of creating predictive models easily in Python and scikit-learn.')

        st.page_link(
            "streamlit_app.py",
            label="Predict",
            icon=":material/smart_toy:"
        )
        st.page_link(
            "pages/learningdata_visualization.py",
            label="Learning Data",
            icon=":material/database:"
        )


# Display sidebar
render_sidebar()

# Define container
container = st.container(border=True)

# Set parameters
container.header("Input features")
sex = container.selectbox("Sex", ("male", "female"))
island = container.selectbox("Island", ("Biscoe", "Dream", "Torgersen",))
bill_length_mm = container.slider("Bill length (mm)", 32.1, 59.6, 43.9)
bill_depth_mm = container.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
flipper_length_mm = container.slider(
    "Flipper length (mm)", 172.0, 231.0, 201.0)
body_mass_g = container.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)

# Execute prediction
container.button("Predict", use_container_width=True)

st.write("## 📊  Prediction Results")

# Create Dataframe for the input features
data = {"island": island,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "sex": sex}

# Values specified in sidebar
input_df = pd.DataFrame(data, index=[0])


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

# Values specified in sidebar + CSV data
input_penguins = pd.concat([input_df, X_raw], axis=0)

# Data encoding
# Encode X
encode = ["island", "sex"]
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

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

# Model training
# Split data for cross-validation
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Train the ML model
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
clf

st.write(f'Train_acc：{accuracy_score(y_train, clf.predict(x_train))}')
st.write(f'Test_acc ：{accuracy_score(y_test, clf.predict(x_test))}')

with st.expander("Model evaluation"):
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

# Apply model to make predictions

# Load the model
clf = joblib.load("penguin_classifier_model.pkl")

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ["Adelie", "Chinstrap", "Gentoo"]
df_prediction_proba.rename(columns={0: "Adelie",
                                    1: "Chinstrap",
                                    2: "Gentoo"})

# Display predicted result
st.subheader("Preficted species")

st.write("**Input penguin**")
input_df

st.write("**Prediction result**")
st.dataframe(df_prediction_proba,
             column_config={
                 "Adelie": st.column_config.ProgressColumn(
                     "Adelie",
                     format="%f",
                     width="medium",
                     min_value=0,
                     max_value=1
                 ),
                 "Chinstrap": st.column_config.ProgressColumn(
                     "Chinstrap",
                     format="%f",
                     width="medium",
                     min_value=0,
                     max_value=1
                 ),
                 "Gentoo": st.column_config.ProgressColumn(
                     "Gentoo",
                     format="%f",
                     width="medium",
                     min_value=0,
                     max_value=1
                 ),
             }, hide_index=True)

penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.success(str(penguins_species[prediction][0]))

# Custom CSS
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FFFFFF;
        color: #F63365;
    }
    div.stButton > button:hover {
        background-color: #F63365;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
