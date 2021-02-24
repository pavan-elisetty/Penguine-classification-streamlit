import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_cleaned.csv')
df=data.copy()

encoder_species=LabelEncoder()
encoder_island=LabelEncoder()
encoder_sex=LabelEncoder()
df['species']=encoder_species.fit_transform(df['species'])
df['island']=encoder_island.fit_transform(df['island'])
df['sex']=encoder_sex.fit_transform(df['sex'])

def encoder_save():
    pickle.dump(encoder_species, open('penguins_enc_sp.pkl', 'wb'))
    pickle.dump(encoder_island, open('penguins_enc_is.pkl', 'wb'))
    pickle.dump(encoder_sex, open('penguins_enc_sx.pkl', 'wb'))
encoder_save()
x=df.drop('species',axis=1)
y=df['species']
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x,y)
pickle.dump(model, open('penguins_model.pkl', 'wb'))
print('finished')