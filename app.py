import pandas as pd
import pickle
import streamlit as st

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species
""")

st.sidebar.header('User Inputs')
st.sidebar.markdown('''
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
''')

uploaded_file =st.sidebar.file_uploader("Upload file in csv format",type=["csv"])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

data=input_df

enc_is = pickle.load(open('penguins_enc_is.pkl', 'rb'))
enc_sp = pickle.load(open('penguins_enc_sp.pkl', 'rb'))
enc_sx = pickle.load(open('penguins_enc_sx.pkl', 'rb'))
model = pickle.load(open('penguins_model.pkl', 'rb'))
data['island']=enc_is.transform(data['island'])
data['sex'] =enc_sx.transform(data['sex'])

prediction=(enc_sp.inverse_transform([model.predict(data)[0]])[0])

st.subheader('Prediction')
st.write(prediction)


st.write("""
**Built with :heart:**
""")
