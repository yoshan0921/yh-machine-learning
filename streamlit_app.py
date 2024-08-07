import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title('🤖 Machine Learning App')

st.info('The purpose of this application is to experience the process of creating predictive models easily in Python and scikit-learn.')

# タブの作成
tab1, tab2, tab3 = st.tabs(["元データ", "データ可視化", "モデル評価と推測結果"])

# 元データタブ
with tab1:
    with st.expander("Data"):
        st.write("**Raw data**")
        df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
        st.dataframe(df)

        st.write("**X**")
        X_raw = df.drop("species", axis=1)
        st.dataframe(X_raw)
  
        st.write("**Y**")
        y_raw = df.species
        st.dataframe(y_raw)

# データ可視化タブ
with tab2:
    with st.expander("Data visualization1"):
        st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color="species")

    with st.expander("Data visualization2"):
        st.write("**Distribution of Features**")
        for feature in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
            st.write(f"**{feature} by species**")
            st.bar_chart(data=df, x=feature, y="species")

    with st.expander("Data visualization3"):
        st.write("**Pairplot of Features**")
        fig = sns.pairplot(df, hue="species", markers=["o", "s", "D"])
        st.pyplot(fig)

# モデル評価と推測結果タブ
with tab3:
    with st.sidebar:
        st.header("Input features")
        island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen",))
        bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        sex = st.selectbox("Sex", ("Male", "Female"))

        # Create Dataframe for the input features
        data = {"island": island,
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_depth_mm,
                "flipper_length_mm": flipper_length_mm,
                "body_mass_g": body_mass_g,
                "sex": sex}
        input_df = pd.DataFrame(data, index=[0])
        input_penguins = pd.concat([input_df, X_raw], axis=0)

    with st.expander("Input features"):
        st.write("**Input penguin**")
        st.dataframe(input_df)
        st.write("**Combined penguins data**")
        st.dataframe(input_penguins)

    # Data preparation
    ## Encode X
    encode = ["island", "sex"]
    df_penguins = pd.get_dummies(input_penguins, prefix=encode)

    X = df_penguins[1:]
    input_row = df_penguins[:1]

    ## Encode y
    target_mapper = {"Adelie": 0,
                    "Chinstrap": 1,
                    "Gentoo": 2}
    y = y_raw.apply(lambda val: target_mapper[val])
    
    with st.expander("Data preparation"):
        st.write("**Encoded X (input penguin)**")
        st.dataframe(input_row)
        st.write("**Encoded y**")
        st.dataframe(y)

    # Model training
    ## Train the ML model
    clf = RandomForestClassifier()
    clf.fit(X, y)

    with st.expander("Feature Importance"):
        feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
        st.bar_chart(feature_importances.sort_values(ascending=False))

    ## Apply model to make predictions
    prediction = clf.predict(input_row)
    prediction_proba = clf.predict_proba(input_row)

    df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Adelie", "Chinstrap", "Gentoo"])

    # Display predicted species
    st.subheader("Predicted species")
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
    st.success(f"Predicted species: {penguins_species[prediction][0]}")

    with st.expander("Classification Report"):
        report = classification_report(y, clf.predict(X), target_names=["Adelie", "Chinstrap", "Gentoo"])
        st.text(report)

    with st.expander("Confusion Matrix"):
        cm = confusion_matrix(y, clf.predict(X))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)





# import streamlit as st
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# st.title('🤖 Machine Learning App')

# st.info('The purpose of this application is to experience the process of creating predictive models easily in Python and scikit-learn.')

# with st.expander("Data"):
#   st.write("**Raw data**")
#   df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
#   df

#   st.write("**X**")
#   X_raw = df.drop("species", axis=1)
#   X_raw
  
#   st.write("**Y**")
#   y_raw = df.species
#   y_raw


# with st.expander("Data visualization1"):
#   st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color="species")

# with st.expander("Data visualization2"):
#   st.write("**Distribution of Features**")
#   for feature in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
#     st.write(f"**{feature} by species**")
#     st.bar_chart(data=df, x=feature, y="species")

# with st.expander("Data visualization3"):
#   st.write("**Pairplot of Features**")
#   fig = sns.pairplot(df, hue="species", markers=["o", "s", "D"])
#   st.pyplot(fig)
  
# # Input features
# with st.sidebar:
#   st.header("Input features")
#   island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen",))
#   bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
#   bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
#   flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
#   body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
#   sex = st.selectbox("Sex", ("Male", "Female"))

#   # Create Dataframe for the input features
#   data = {"island": island,
#           "bill_length_mm": bill_length_mm,
#           "bill_depth_mm": bill_depth_mm,
#           "flipper_length_mm": flipper_length_mm,
#           "body_mass_g": body_mass_g,
#           "sex": sex}
#   input_df = pd.DataFrame(data, index=[0])
#   input_penguins = pd.concat([input_df, X_raw], axis=0)

# with st.expander("Input features"):
#   st.write("**Input penguin**")
#   input_df
#   st.write("**Combined penguins data**")
#   input_penguins

# # Data preparation
# ## Encode X
# encode = ["island", "sex"]
# df_penguins = pd.get_dummies(input_penguins, prefix=encode)

# X = df_penguins[1:]
# input_row = df_penguins[:1]

# ## Encode y
# target_mapper = {"Adelie": 0,
#                 "Chinstrap": 1,
#                 "Gentoo": 2}
# def target_encode(val):
#     return target_mapper[val]
# y = y_raw.apply(target_encode)
    
# with st.expander("Data preparation"):
#   st.write("**Encoded X (input penguin)**")
#   input_row
#   st.write("**Encoded y**")
#   y

# # Model training
# ## Train the ML model
# clf = RandomForestClassifier()
# clf.fit(X, y)

# with st.expander("Feature Importance"):
#   feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
#   st.bar_chart(feature_importances.sort_values(ascending=False))

# ## Apply model to make predictions
# prediction = clf.predict(input_row)
# prediction_proba = clf.predict_proba(input_row)

# df_prediction_proba = pd.DataFrame(prediction_proba)
# df_prediction_proba.columns = ["Adelie", "Chinstrap", "Gentoo"]
# df_prediction_proba.rename(columns={0: "Adelie",
#                                     1: "Chinstrap",
#                                     2: "Gentoo"})

# # Display predicted species
# st.subheader("Preficted species")
# st.dataframe(df_prediction_proba,
#              column_config={
#                "Adelie": st.column_config.ProgressColumn(
#                  "Adelie",
#                  format="%f",
#                  width="medium",
#                  min_value=0,
#                  max_value=1
#                ),
#                "Chinstrap": st.column_config.ProgressColumn(
#                  "Chinstrap",
#                  format="%f",
#                  width="medium",
#                  min_value=0,
#                  max_value=1
#                ),
#                "Gentoo": st.column_config.ProgressColumn(
#                  "Gentoo",
#                  format="%f",
#                  width="medium",
#                  min_value=0,
#                  max_value=1
#                ),
#              },hide_index=True)

# penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
# st.success(str(penguins_species[prediction][0]))

# with st.expander("Classification Report"):
#     report = classification_report(y, clf.predict(X), target_names=["Adelie", "Chinstrap", "Gentoo"])
#     st.text(report)

# with st.expander("Confusion Matrix"):
#     cm = confusion_matrix(y, clf.predict(X))
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('Actual')
#     st.pyplot(fig)
