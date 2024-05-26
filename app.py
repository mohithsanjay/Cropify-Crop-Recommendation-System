import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import joblib


pd.options.display.max_colwidth = 2000
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-color:#D1FFBD;

}}
[data-testid="stSidebar"] {{
background-color:#61B136;

}}
[data-testid="stHeader"] {{
background-color:#D1FFBD;
}}
[data-testid="stToolbar"] {{
background-color:#85FFB9 ;

}}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

def load_bootstrap():
        return st.markdown("""<link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">""", unsafe_allow_html=True)

with st.sidebar:
    
    load_bootstrap()

    image = Image.open('logo.png')
    

    st.image(image, width=300)
    st.markdown("<h1 style='text-align: center;'>Crop Recommendation System</h1>", unsafe_allow_html= True)
    st.markdown("""
        <h4 style='text-align: center;'>
        This simple crop recommender system was trained using Random Forest Algorithm in giving
        recommendations to farmers the best and suitable crop based on an Indian Crop 
        Recommendation <a style='text-align: center; color: blue;' 
        href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset">Dataset</a>.
        By inputing N, P, K, and pH values based on soil conditions, 
        weather conditions such as temperature, humidity, and rainfall, the system can
        recommend what the best and most suitable crop to plant.
        </h4>
        """, unsafe_allow_html=True)

    df_desc = pd.read_csv('Dataset/Crop_Desc.csv', sep = ';', encoding = 'utf-8')

    df = pd.read_csv('Dataset/Crop_recommendation.csv')

    rdf_clf = joblib.load('RDF_model.pkl')

    X = df.drop('label', axis = 1)
    y = df['label']

st.markdown("<h3 style='text-align: center;'>Please input the feature values to predict the best crop to plant.</h3><br>", unsafe_allow_html=True)


col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

with col3:
    n_input = st.number_input('Insert N (kg/ha) value:',  help = 'Insert here the Nitrogen density (kg/ha) from 0 to 140.')
    p_input = st.number_input('Insert P (kg/ha) value:',  help = 'Insert here the Phosphorus density (kg/ha) from 5 to 145.')
    k_input = st.number_input('Insert K (kg/ha) value:',  help = 'Insert here the Potassium density (kg/ha) from 5 to 205.')
    temp_input = st.number_input('Insert Avg Temperature (ºC) value:', step = 1., format="%.2f", help = 'Insert here the Avg Temperature (ºC) from 9 to 43.')

with col5:
    hum_input = st.number_input('Insert Avg Humidity (%) value:',  step = 1., format="%.2f", help = 'Insert here the Avg Humidity (%) from 15 to 99.')
    ph_input = st.number_input('Insert pH value:', step = 0.1, format="%.2f", help = 'Insert here the pH from 3.6 to 9.9')
    rain_input = st.number_input('Insert Avg Rainfall (mm) value:', step = 0.1, format="%.2f", help = 'Insert here the Avg Rainfall (mm) from 21 to 298')
      
with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button('Recommend Crop',)
    st.markdown("<br>", unsafe_allow_html=True)

    predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,1,0,0,0,0,0]]
    
if predict_btn:
    rdf_predicted_value = rdf_clf.predict(predict_inputs)

    if(n_input and p_input and k_input and temp_input and hum_input and ph_input and rain_input):
        
        st.markdown(f"<br><h5 style='text-align: center;'>Best Crop to Plant </h5> <h3 style='text-align: center; line-height: 0'> <b>{rdf_predicted_value[0]} </b></h3>", unsafe_allow_html=True)
        df_desc = df_desc.astype({'label':str,'image':str})
        df_desc['label'] = df_desc['label'].str.strip()
        df_desc['image'] = df_desc['image'].str.strip()
        df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
        df_image = df_pred_image['image'].item()
        st.markdown(f"""<h5 style = 'text-align: center ; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)

    else:
        st.markdown("Please fill all input fields") 

    

    # st.markdown(f"""<h5 style='text-align: center;'>Statistics Summary about NPK and Weather Conditions values for <b> {rdf_predicted_value[0]} 
    #     </b></h5>""", unsafe_allow_html=True)
    # df_pred = df[df['label'] == rdf_predicted_value[0]]
    # st.dataframe(df_pred.describe(), use_container_width = True)        





