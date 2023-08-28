#!/usr/bin/env python
# coding: utf-8

# # Training

# ***

# In[5]:


#Libraries
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2_contingency
from feature_engine.encoding import OneHotEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ## Enviorment

# In[6]:


#Enviorment Tree
ROOT_DIR = os.path.dirname(os.getcwd())
MODEL_INPUTS_OUTPUTS = os.path.join(ROOT_DIR, 'model_inputs_outputs/')
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
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

if not os.path.exists(MODEL_ARTIFACTS_PATH):
    os.makedirs(MODEL_ARTIFACTS_PATH)
if not os.path.exists(PREDICTOR_DIR_PATH):
    os.makedirs(PREDICTOR_DIR_PATH)


# # Load Data

# In[7]:


# Reading schema and clasifying
file_name = [f for f in os.listdir(INPUT_SCHEMA_DIR) if f.endswith('json')][0]
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


# In[8]:


#Load data to Dataframe
file_name = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')][0]
file_path = os.path.join(TRAIN_DIR, file_name)
df = pd.read_csv(file_path)


# ## Data Quality Report

# In[9]:


def Data_Quality_Report(df):

    # Initial table
    freqDF = pd.DataFrame(columns=['Feature',
                                   'Mode',
                                   'Mode Freq.',
                                   'Mode %',
                                   '2nd Mode',
                                   '2nd Mode Freq.',
                                   '2nd Mode %'])
    for col in df.columns:
        freq = df[col].value_counts()
        freqdf = freq.to_frame()
        fRow = freqdf.iloc[0]
        secRow = freqdf.iloc[1] if len(freqdf) > 1 else pd.Series([0, 0], index=['index', col])
        fPrct = fRow[0] / len(df[col])
        secPrct = secRow[0] / len(df[col]) if len(freqdf) > 1 else 0
        mode1 = fRow.name
        mode2 = secRow.name
        new_row = {'Feature': col,
                   'Mode': mode1,
                   'Mode Freq.': fRow[0],
                   'Mode %': fPrct,
                   '2nd Mode': mode2,
                   '2nd Mode Freq.': secRow[0],
                   '2nd Mode %': secPrct}
        freqDF = pd.concat([freqDF, pd.DataFrame([new_row])], ignore_index=True)
        
    freqDF = freqDF.set_index('Feature')

    # Nulls, Counts, Cardinality
    NUllFeatures = (df.isnull().sum() / df.shape[0]).round(4).sort_values(ascending=False)
    Count = df.count()
    uni = df.nunique()

    # Formatting
    NUllFeatures = NUllFeatures.to_frame(name="% Miss.")
    Count = Count.to_frame(name="Count")
    uni = uni.to_frame(name="Card.")
    result = pd.concat([Count, NUllFeatures, uni, freqDF], axis=1)
    result = result.style.format({'% Miss.': "{:.1%}",
                                  'Mode %': "{:.0%}",
                                  '2nd Mode %': "{:.0%}",
                                  'Count': "{:,}",
                                  'Card.': "{:,}",
                                  'Mode Freq.': "{:,}",
                                  '2nd Mode Freq.': "{:,}"})
    return result


# In[10]:


# View Data Quality Report
DQR = Data_Quality_Report(df)
DQR


# In[11]:


#Deop fields woth high MODE
#df = df.drop(columns=['minute','ve_forms','a_ct','a_polpur'])

# Drop Null Fields
df = df.dropna()
DQR = Data_Quality_Report(df)
DQR


# ## Process Data

# #### Age processing

# In[12]:


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


# In[13]:


def bucketize_death(deaths):
    if deaths > 1:
        return 1
    else:
        return 0


# In[14]:


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


# In[15]:


# Apply the bucketizing
df['age_bucket'] = df['age'].apply(bucketize_age)
df['death_bucket'] = df['deaths'].apply(bucketize_death)
df['hour_bucket'] = df['hour'].apply(bucketize_hour)
categorical_features.extend(["age_bucket","death_bucket","hour_bucket"])


# ### Enconding Data

# #### Features

# In[16]:


#items to drop
#items_to_remove = ['a_ct', 'a_polpur', 've_forms']
#for item in items_to_remove:
    #categorical_features.remove(item)

# Encoding the features
encoder = OneHotEncoder(top_categories=10)

for cat in categorical_features:
    df[cat] = df[cat].astype(str)

categorical = df[categorical_features]    
encoder.fit(categorical)
categorical_encoded = encoder.transform(categorical)


# #### Target

# In[17]:


target = df[target_feature]
encoder = LabelEncoder()
y = encoder.fit_transform(target.values.reshape(-1, 1))
dump(encoder, LABEL_ENCODER_FILE)


# #### Numeric

# In[18]:


#items to drop
"""
items_to_remove = ['minute', 'age', 'mod_year']
for item in items_to_remove:
    numeric_features.remove(item)
"""
numeric = df[numeric_features]
numeric


# ## Feature Selection

# In[19]:


# Apply Chi-Squared test to each categorical feature
chi2_results = {}
for cat_feature in categorical_features:
    contingency_table = pd.crosstab(df[cat_feature], df[target_feature])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    chi2_results[cat_feature] = {"chi2": chi2, "p_value": p_value}


# In[20]:


#Sort the features based on their p-values
sorted_features = sorted(chi2_results.keys(), key=lambda x: chi2_results[x]["p_value"])

# Filter Chi-Squared test results for p-values above 0.05
unsignificant_results = {feature: result for feature, result in chi2_results.items() if result["p_value"] > 0.05}

# Extract unsignificant feature names to a list
unsignificant_results_list = list(unsignificant_results.keys())


# In[21]:


unsignificant_results_list


# #### Updated catagorical features

# In[22]:


#items to drop
items_to_remove = unsignificant_results_list
for item in items_to_remove:
    try:
        categorical_features.remove(item)
    except:
        pass


# Encoding the features
encoder = OneHotEncoder(top_categories=10)

for cat in categorical_features:
    df[cat] = df[cat].astype(str)

categorical = df[categorical_features]    
encoder.fit(categorical)
categorical_encoded = encoder.transform(categorical)
categorical_encoded

# Saving the encoder to use it on the testing dataset
path = dump(encoder, OHE_ENCODER_FILE)


# In[23]:


categorical.info()


# ## Modeling

# In[24]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(categorical_encoded, y, test_size=0.2, random_state=18)


# In[25]:


# List of classifier models
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=19)),
    ('Gradient Boosting',GradientBoostingClassifier(learning_rate=0.2,max_depth=4,n_estimators=150)),
    ('LogisticRegression',LogisticRegression(C=100, penalty='l2', solver='liblinear'))
    #('Gradient Boosting', GradientBoostingClassifier(learning_rate: 0.2, max_depth: 4, n_estimators: 150))
    #('SVM', SVC(kernel='linear', C=1.0, random_state=19)),
    ##('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
    #('MLP', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=19)),
    #('Naive Bayes', GaussianNB())
]



# Train and evaluate each model
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{model_name} - Macro-Averaged F1 Score: {macro_f1:.4f}")


# ## Save Model

# In[26]:


model = GradientBoostingClassifier(learning_rate=0.2,max_depth=4,n_estimators=150)
model.fit(X_train, y_train)

# Saving the model to use it for predictions
path = dump(model, PREDICTOR_FILE_PATH)

