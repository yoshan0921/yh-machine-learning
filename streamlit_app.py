import streamlit as st
import numpy as np
import pandas as pd
import joblib


def render_sidebar():
    with st.sidebar:
        st.image("./images/PenguinClassifier_transparent.png")
        st.info(
            "The purpose of this application is to provide a simple experience of the process of creating an ML model and releasing a web application that uses that model."
        )

        st.page_link("streamlit_app.py", label="Predict", icon=":material/smart_toy:")
        st.page_link(
            "pages/learningdata_visualization.py",
            label="Learning Data",
            icon=":material/database:",
        )


# Display sidebar
render_sidebar()

# Create container
container = st.container(border=True)

# Input parameters fileds
container.header("Input features")
sex = container.selectbox("Sex", ("male", "female"))
island = container.selectbox(
    "Island",
    (
        "Biscoe",
        "Dream",
        "Torgersen",
    ),
)
bill_length_mm = container.slider("Bill length (mm)", 32.1, 59.6, 43.9)
bill_depth_mm = container.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
flipper_length_mm = container.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
body_mass_g = container.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)

# Create Dataframe for the input features
data = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": sex,
}
input_df = pd.DataFrame(data, index=[0])

# Data encoding for category variables
encode = ["island", "sex"]
input_encoded_df = pd.get_dummies(input_df, prefix=encode)

# Ensure all dummy variables used during model training are present in this order
expected_columns = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island_Biscoe",
    "island_Dream",
    "island_Torgersen",
    "sex_female",
    "sex_male",
]

# Add missing category variables as columns with 0 value
for col in expected_columns:
    if col not in input_encoded_df.columns:
        input_encoded_df[col] = False

# Reorder df_penguins in line with expected_columns
input_encoded_df = input_encoded_df[expected_columns]

# Load the model
clf = joblib.load("penguin_classifier_model.pkl")

# Execute prediction
prediction = clf.predict(input_encoded_df)
prediction_proba = clf.predict_proba(input_encoded_df)
prediction_proba = [n * 100 for n in prediction_proba]

# Display prediction result
st.write("## 🐧Prediction results")
penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.success(str(penguins_species[prediction][0]))

# Display prediction probability
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ["Adelie", "Chinstrap", "Gentoo"]
df_prediction_proba.rename(columns={0: "Adelie", 1: "Chinstrap", 2: "Gentoo"})
st.dataframe(
    df_prediction_proba,
    column_config={
        "Adelie": st.column_config.ProgressColumn(
            "Adelie", format="%d %%", min_value=0, max_value=100
        ),
        "Chinstrap": st.column_config.ProgressColumn(
            "Chinstrap", format="%d %%", min_value=0, max_value=100
        ),
        "Gentoo": st.column_config.ProgressColumn(
            "Gentoo", format="%d %%", min_value=0, max_value=100
        ),
    },
    hide_index=True,
    width=704,
)

# Custom CSS for expanding label font size
st.markdown(
    """
    <style>
    h2, h3 {
        font-size: 1.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
