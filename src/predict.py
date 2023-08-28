#!/usr/bin/env python
# coding: utf-8

# # Predict

# ***

# ## Libraries

# In[19]:


import os
import pandas as pd
import json
import warnings
from joblib import load
warnings.filterwarnings('ignore')


# In[20]:


# DO NOT CHANGE THESE LINES.
ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR,  'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
OUTPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "outputs")
INPUT_SCHEMA_DIR = os.path.join(INPUT_DIR, "schema")
DATA_DIR = os.path.join(INPUT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")
MODEL_PATH = os.path.join(MODEL_INPUTS_OUTPUTS, "model")
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
OHE_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'ohe.joblib')
LABEL_ENCODER_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'label_encoder.joblib')
PREDICTOR_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "predictor")
PREDICTOR_FILE_PATH = os.path.join(PREDICTOR_DIR_PATH, "predictor.joblib")
IMPUTATION_FILE = os.path.join(MODEL_ARTIFACTS_PATH, 'imputation.joblib')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)


# ## Load Data

# In[21]:


file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('.json')][0]
schema_path = os.path.join(INPUT_SCHEMA_DIR, file_name)
with open(schema_path, "r", encoding="utf-8") as file:
    schema = json.load(file)
features = schema['features']

numeric_features = []
categorical_features = []
for f in features:
    if f['dataType'] == 'CATEGORICAL':
        categorical_features.append(f['name'])
    else:
        numeric_features.append(f['name'])

id_feature = schema['id']['name']
target_feature = schema['target']['name']


# In[22]:


#load testing data
file_name = [f for f in os.listdir(TEST_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TEST_DIR, file_name)
df = pd.read_csv(file_path)
ids = df[id_feature]
df = df.dropna()
df.head(3)


# ## Process Data

# In[23]:


def bucketize_age(age):
    if age < 20:
        return "Under 20"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60 and above"

def bucketize_death(deaths):
    if deaths > 1:
        return 1
    else:
        return 0
    
def bucketize_hour(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'
    
# Apply the bucketizing
df['age_bucket'] = df['age'].apply(bucketize_age)
df['death_bucket'] = df['deaths'].apply(bucketize_death)
df['hour_bucket'] = df['hour'].apply(bucketize_hour)
categorical_features.extend(["age_bucket","death_bucket","hour_bucket"])


# ## Encoding

# In[24]:


#Clean DF
if 'a_ct' in categorical_features:
    categorical_features.remove('a_ct')

if 'a_hr' in categorical_features:
    categorical_features.remove('a_hr')


# In[25]:

print(categorical_features)
print(len(categorical_features))

# Encoding
encoder = load(OHE_ENCODER_FILE)

categorical = df[categorical_features]  
categorical_encoded = encoder.transform(categorical)


# ## Make Prediction

# In[26]:


model = load(PREDICTOR_FILE_PATH)
predictions = model.predict_proba(categorical_encoded)
predictions


# In[27]:


encoder = load(LABEL_ENCODER_FILE)

class_names = encoder.inverse_transform([0, 1, 2])

predictions = pd.DataFrame(predictions, columns=class_names)
predictions.insert(0, 'u_id', ids)
predictions.to_csv(PREDICTIONS_FILE)
predictions

print('Done')
