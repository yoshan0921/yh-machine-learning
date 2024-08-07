import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('The purpose of this application is to experience the process of creating predictive models easily in Python and scikit-learn.')

with st.expander("Data"):
  st.write("**Raw data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("**X**")
  X = df.drop("species", axis=1)
  X
  
  st.write("**Y**")
  y = df.species
  y


with st.expander("Data visualization"):
  st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color="species")

# Data preparations
with st.sidebar:
  st.header("Input features")
  island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen",))
  gender = st.selectbox("Gendar", ("Male", "Female"))
  bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
  body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)

  # Create Dataframe for the input features
  data = {"island": island,
          "bill_length": bill_length_mm,
          "bill_depth": bill_depth_mm,
          "flipper_length_mm": flipper_length_mm,
          "body_mass_g": body_mass_g,
          "gender": gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X], axis=0)
  
input_penguins
  
  
