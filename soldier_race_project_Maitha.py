#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# In this project, you must apply EDA processes for the development of predictive models. Handling outliers, domain knowledge and feature engineering will be challenges.
# 
# Also, this project aims to improve your ability to implement algorithms for Multi-Class Classification. Thus, you will have the opportunity to implement many algorithms commonly used for Multi-Class Classification problems.
# 
# Before diving into the project, please take a look at the determines and tasks.

# # Determines

# The 2012 US Army Anthropometric Survey (ANSUR II) was executed by the Natick Soldier Research, Development and Engineering Center (NSRDEC) from October 2010 to April 2012 and is comprised of personnel representing the total US Army force to include the US Army Active Duty, Reserves, and National Guard. In addition to the anthropometric and demographic data described below, the ANSUR II database also consists of 3D whole body, foot, and head scans of Soldier participants. These 3D data are not publicly available out of respect for the privacy of ANSUR II participants. The data from this survey are used for a wide range of equipment design, sizing, and tariffing applications within the military and has many potential commercial, industrial, and academic applications.
# 
# The ANSUR II working databases contain 93 anthropometric measurements which were directly measured, and 15 demographic/administrative variables explained below. The ANSUR II Male working database contains a total sample of 4,082 subjects. The ANSUR II Female working database contains a total sample of 1,986 subjects.
# 
# 
# DATA DICT:
# https://data.world/datamil/ansur-ii-data-dictionary/workspace/file?filename=ANSUR+II+Databases+Overview.pdf
# 
# ---
# 
# To achieve high prediction success, you must understand the data well and develop different approaches that can affect the dependent variable.
# 
# Firstly, try to understand the dataset column by column using pandas module. Do research within the scope of domain (body scales, and race characteristics) knowledge on the internet to get to know the data set in the fastest way.
# 
# You will implement ***Logistic Regression, Support Vector Machine, XGBoost, Random Forest*** algorithms. Also, evaluate the success of your models with appropriate performance metrics.
# 
# At the end of the project, choose the most successful model and try to enhance the scores with ***SMOTE*** make it ready to deploy. Furthermore, use ***SHAP*** to explain how the best model you choose works.

# # Tasks

# #### 1. Exploratory Data Analysis (EDA)
# - Import Libraries, Load Dataset, Exploring Data
# 
#     *i. Import Libraries*
#     
#     *ii. Ingest Data *
#     
#     *iii. Explore Data*
#     
#     *iv. Outlier Detection*
#     
#     *v.  Drop unnecessary features*
# 
# #### 2. Data Preprocessing
# - Scale (if needed)
# - Separete the data frame for evaluation purposes
# 
# #### 3. Multi-class Classification
# - Import libraries
# - Implement SVM Classifer
# - Implement Decision Tree Classifier
# - Implement Random Forest Classifer
# - Implement XGBoost Classifer
# - Compare The Models
# 
# 

# # EDA
# - Drop unnecessary colums
# - Drop DODRace class if value count below 500 (we assume that our data model can't learn if it is below 500)

# ## Import Libraries
# Besides Numpy and Pandas, you need to import the necessary modules for data visualization, data preprocessing, Model building and tuning.
# 
# *Note: Check out the course materials.*

# In[107]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV


get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[108]:


get_ipython().system('pip install -U xgboost')


# In[109]:


pip install scikit-plot


# In[110]:


pip install --upgrade pip


# ## Ingest Data from links below and make a dataframe
# - Soldiers Male : https://query.data.world/s/h3pbhckz5ck4rc7qmt2wlknlnn7esr
# - Soldiers Female : https://query.data.world/s/sq27zz4hawg32yfxksqwijxmpwmynq

# In[111]:


dff = pd.read_csv('ANSUR II FEMALE Public.csv')


# In[112]:


dfm = pd.read_csv('ANSUR II MALE Public.csv', encoding='latin-1')


# ## Explore Data

# In[113]:


print(dff.shape)
print(dfm.shape)


# In[114]:


dff.head()


# In[115]:


dfm.head()


# In[116]:


dff.rename(columns= {"SubjectId": "subjectid"}, inplace= True)


# In[117]:


df = pd.concat([dfm, dff], axis=0, ignore_index=True)


# In[118]:


df.shape


# In[119]:


df.duplicated().sum()


# In[120]:


df.describe().T


# In[121]:


df.info()


# In[122]:


print(df.isna().sum().to_string())


# In[123]:


df = df.drop(columns='Ethnicity', axis=1)


# In[124]:


df.isnull().sum().any()


# In[125]:


df.DODRace.value_counts(dropna = False)


# In[126]:


df = df[df["DODRace"].isin([1,2,3])]
df.DODRace.value_counts(dropna = False)


# In[127]:


df.shape


# In[128]:


print(df["DODRace"].value_counts());
df["DODRace"].value_counts().plot(kind="bar",figsize=(10,10));


# In[129]:


df.head()


# In[130]:


categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)


# In[131]:


unique_values = {col: df[col].unique() for col in categorical_columns}

# Print the unique values for each categorical column
for col, values in unique_values.items():
    print(f"Unique values for column '{col}': {len(values)}")


# In[132]:


df.SubjectsBirthLocation.value_counts()


# In[133]:


states_by_region = {
    "Alabama": "South",
    "Alaska": "West",
    "Arizona": "West",
    "Arkansas": "South",
    "California": "West",
    "Colorado": "West",
    "Connecticut": "Northeast",
    "Delaware": "Mid Atlantic",
    "Florida": "South",
    "Georgia": "South",
    "Hawaii": "West",
    "Idaho": "West",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Kentucky": "South",
    "Louisiana": "South",
    "Maine": "Northeast",
    "Maryland": "Mid Atlantic",
    "Massachusetts": "Northeast",
    "Michigan": "Midwest",
    "Minnesota": "Midwest",
    "Mississippi": "South",
    "Missouri": "Midwest",
    "Montana": "West",
    "Nebraska": "Midwest",
    "Nevada": "West",
    "New Hampshire": "Northeast",
    "New Jersey": "Mid Atlantic",
    "New Mexico": "West",
    "New York": "Northeast",
    "North Carolina": "South",
    "North Dakota": "Midwest",
    "Ohio": "Midwest",
    "Oklahoma": "South",
    "Oregon": "West",
    "Pennsylvania": "Mid Atlantic",
    "Rhode Island": "Northeast",
    "South Carolina": "South",
    "South Dakota": "Midwest",
    "Tennessee": "South",
    "Texas": "South",
    "Utah": "West",
    "Vermont": "Northeast",
    "Virginia": "Mid Atlantic",
    "Washington": "West",
    "West Virginia": "South",
    "Wisconsin": "Midwest",
    "Wyoming": "West",
    "District of Columbia": "Mid Atlantic"
}


# In[134]:


countries_by_continent = {
    "Afghanistan": "Asia",
    "Albania": "Europe",
    "Algeria": "Africa",
    "American Samoa": "Oceania",
    "Andorra": "Europe",
    "Angola": "Africa",
    "Antigua and Barbuda": "North America",
    "Argentina": "South America",
    "Armenia": "Asia",
    "Australia": "Oceania",
    "Austria": "Europe",
    "Azerbaijan": "Asia",
    "Bahamas": "North America",
    "Bahrain": "Asia",
    "Bangladesh": "Asia",
    "Barbados": "North America",
    "Belarus": "Europe",
    "Belgium": "Europe",
    "Belize": "North America",
    "Benin": "Africa",
    "Bhutan": "Asia",
    "Bolivia": "South America",
    "Bosnia and Herzegovina": "Europe",
    "Botswana": "Africa",
    "Brazil": "South America",
    "British Virgin Islands": "North America",
    "Brunei": "Asia",
    "Bulgaria": "Europe",
    "Burkina Faso": "Africa",
    "Burundi": "Africa",
    "Cambodia": "Asia",
    "Cameroon": "Africa",
    "Canada": "North America",
    "Cape Verde": "Africa",
    "Central African Republic": "Africa",
    "Chad": "Africa",
    "Chile": "South America",
    "China": "Asia",
    "Colombia": "South America",
    "Comoros": "Africa",
    "Congo, Democratic Republic of the": "Africa",
    "Congo, Republic of the": "Africa",
    "Costa Rica": "North America",
    "Côte d'Ivoire": "Africa",
    "Croatia": "Europe",
    "Cuba": "North America",
    "Cyprus": "Europe",
    "Czech Republic": "Europe",
    "Denmark": "Europe",
    "Djibouti": "Africa",
    "Dominica": "North America",
    "Dominican Republic": "North America",
    "East Timor": "Asia",
    "Ecuador": "South America",
    "Egypt": "Africa",
    "El Salvador": "North America",
    "England": "Europe",
    "Equatorial Guinea": "Africa",
    "Eritrea": "Africa",
    "Estonia": "Europe",
    "Eswatini": "Africa",
    "Ethiopia": "Africa",
    "Federated States of Micronesia": "Oceania",
    "Fiji": "Oceania",
    "Finland": "Europe",
    "France": "Europe",
    "French Guiana": "South America",
    "Gabon": "Africa",
    "Gambia": "Africa",
    "Georgia": "Asia",
    "Germany": "Europe",
    "Ghana": "Africa",
    "Greece": "Europe",
    "Grenada": "North America",
    "Guam": "Oceania",
    "Guadalupe" : "North America",
    "Guatemala": "North America",
    "Guinea": "Africa",
    "Guinea-Bissau": "Africa",
    "Guyana": "South America",
    "Haiti": "North America",
    "Honduras": "North America",
    "Hungary": "Europe",
    "Iceland": "Europe",
    "India": "Asia",
    "Indonesia": "Asia",
    "Iran": "Asia",
    "Iraq": "Asia",
    "Ireland": "Europe",
    "Israel": "Asia",
    "Italy": "Europe",
    "Ivory Coast": "Africa",
    "Jamaica": "North America",
    "Japan": "Asia",
    "Jordan": "Asia",
    "Kazakhstan": "Asia",
    "Kenya": "Africa",
    "Kiribati": "Oceania",
    "Kosovo": "Europe",
    "Kuwait": "Asia",
    "Kyrgyzstan": "Asia",
    "Laos": "Asia",
    "Latvia": "Europe",
    "Lebanon": "Asia",
    "Lesotho": "Africa",
    "Liberia": "Africa",
    "Libya": "Africa",
    "Liechtenstein": "Europe",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Macedonia, Republic of": "Europe",
    "Madagascar": "Africa",
    "Malawi": "Africa",
    "Malaysia": "Asia",
    "Maldives": "Asia",
    "Mali": "Africa",
    "Malta": "Europe",
    "Marshall Islands": "Oceania",
    "Mauritania": "Africa",
    "Mauritius": "Africa",
    "Mexico": "North America",
    "Micronesia": "Oceania",
    "Moldova": "Europe",
    "Monaco": "Europe",
    "Mongolia": "Asia",
    "Montenegro": "Europe",
    "Morocco": "Africa",
    "Mozambique": "Africa",
    "Myanmar (Burma)": "Asia",
    "Namibia": "Africa",
    "Nauru": "Oceania",
    "Nepal": "Asia",
    "Netherlands": "Europe",
    "New Zealand": "Oceania",
    "Nicaragua": "North America",
    "Niger": "Africa",
    "Nigeria": "Africa",
    "North Korea": "Asia",
    "Norway": "Europe",
    "Oman": "Asia",
    "Pakistan": "Asia",
    "Palau": "Oceania",
    "Palestine": "Asia",
    "Panama": "North America",
    "Papua New Guinea": "Oceania",
    "Paraguay": "South America",
    "Peru": "South America",
    "Philippines": "Asia",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Puerto Rico": "North America",
    "Qatar": "Asia",
    "Romania": "Europe",
    "Russia": "Europe",
    "Rwanda": "Africa",
    "Saint Kitts and Nevis": "North America",
    "Saint Lucia": "North America",
    "Saint Vincent and the Grenadines": "North America",
    "Samoa": "Oceania",
    "San Marino": "Europe",
    "Sao Tome and Principe": "Africa",
    "Saudi Arabia": "Asia",
    "Scotland": "Europe",
    "Senegal": "Africa",
    "Serbia": "Europe",
    "Seychelles": "Africa",
    "Sierra Leone": "Africa",
    "Singapore": "Asia",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Solomon Islands": "Oceania",
    "Somalia": "Africa",
    "South Africa": "Africa",
    "South Korea": "Asia",
    "South Sudan": "Africa",
    "Spain": "Europe",
    "Sri Lanka": "Asia",
    "Sudan": "Africa",
    "Suriname": "South America",
    "Swaziland": "Africa",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Syria": "Asia",
    "Taiwan": "Asia",
    "Tajikistan": "Asia",
    "Tanzania": "Africa",
    "Thailand": "Asia",
    "Togo": "Africa",
    "Tonga": "Oceania",
    "Trinidad and Tobago": "North America",
    "Tunisia": "Africa",
    "Turkey": "Europe",
    "Turkmenistan": "Asia",
    "Tuvalu": "Oceania",
    "Uganda": "Africa",
    "Ukraine": "Europe",
    "United Arab Emirates": "Asia",
    "United Kingdom": "Europe",
    "United States": "North America",
    "Uruguay": "South America",
    "US Virgin Islands": "North America",
    "Uzbekistan": "Asia",
    "Vanuatu": "Oceania",
    "Vatican City": "Europe",
    "Venezuela": "South America",
    "Vietnam": "Asia",
    "Wales": "Europe",
    "Yemen": "Asia",
    "Zambia": "Africa",
    "Zimbabwe": "Africa",
    "South America": "South America",
    "Burma": "Asia",
    "Korea": "Asia",
    "Northern Mariana Islands": "Oceania",
    "Bermuda": "North America",
}


# In[135]:


df["SubjectsBirthLocation"] = [i if i in states_by_region else countries_by_continent[i] for i in df["SubjectsBirthLocation"].values ]
df["SubjectsBirthLocation"].value_counts()


# In[136]:


df["SubjectsBirthLocation"].nunique()


# In[137]:


df["DODRace"] = df.DODRace.map({1 : "White", 2 : "Black", 3 : "Hispanic"})
df.DODRace.value_counts()


# In[138]:


df.groupby(["Component"])["DODRace"].value_counts(normalize=True)


# In[139]:


ct = pd.crosstab( df.Component,df.DODRace, margins=True, margins_name="Total", normalize='index')
ct


# In[140]:


ct.plot(kind='bar')
plt.title('DODRace vs Component')
plt.xlabel('Component')
plt.ylabel('Race Ratio')
plt.show()


# In[141]:


df.groupby(["Component", "Branch"])["DODRace"].value_counts(normalize=True)


# In[142]:


ct = pd.crosstab( df.DODRace, [df.Component, df.Branch],  margins=True, margins_name="Total", normalize='columns')
ct


# In[143]:


df.groupby(["Component"])["DODRace"].value_counts().plot(kind="barh", figsize=(7, 7))


# In[144]:


df.SubjectNumericRace.value_counts()


# In[145]:


drop_list_nonnumeric = ["Date", "Installation", "Component","Branch","Weightlbs","Heightin","SubjectNumericRace","PrimaryMOS","subjectid"]
df.drop(drop_list_nonnumeric, axis=1, inplace=True)


# In[146]:


df.shape


# In[147]:


df.plot(by ='DODRace', kind="box", subplots=True, layout=(32, 3), figsize=(20, 40), vert=False, sharex=False, sharey=False)
plt.tight_layout();


# In[148]:


df.sample(5)


# In[149]:


cat_onehot = ['Gender','SubjectsBirthLocation', 'WritingPreference']


# In[150]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

column_trans = make_column_transformer(
                        (OneHotEncoder(handle_unknown="ignore", sparse=False), cat_onehot),
                      
                         remainder='passthrough',
                         verbose_feature_names_out=False)

column_trans=column_trans.set_output(transform="pandas")


# # DATA Preprocessing
# - In this step we divide our data to X(Features) and y(Target) then ,
# - To train and evaluation purposes we create train and test sets,
# - Lastly, scale our data if features not in same scale. Why?

# In[151]:


X= df.drop("DODRace",axis=1)
y= df.DODRace


# In[152]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[153]:


X_train = column_trans.fit_transform(X_train)
X_test = column_trans.transform(X_test)


# In[154]:


print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


# In[155]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Modelling
# - Fit the model with train dataset
# - Get predict from vanilla model on both train and test sets to examine if there is over/underfitting   
# - Apply GridseachCV for both hyperparemeter tuning and sanity test of our model.
# - Use hyperparameters that you find from gridsearch and make final prediction and evaluate the result according to chosen metric.

# ## 1. Logistic model

# ### Vanilla Logistic Model

# In[156]:


logistic_model = LogisticRegression(class_weight='balanced',max_iter=10000,random_state=42)
logistic_model.fit(X_train,y_train)


# In[157]:


y_pred = logistic_model.predict(X_test)


# In[158]:


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


# In[159]:


eval_metric(logistic_model, X_train, y_train, X_test, y_test)


# In[160]:


ConfusionMatrixDisplay.from_estimator(logistic_model, X_test, y_test);


# In[161]:


scoring = {
    "precision_Hispanic": make_scorer(
        precision_score, average=None, labels=["Hispanic"]
    ),
    "recall_Hispanic": make_scorer(recall_score, average=None, labels=["Hispanic"]),
    "f1_Hispanic": make_scorer(f1_score, average=None, labels=["Hispanic"]),
}


# In[162]:


model = LogisticRegression(class_weight='balanced',max_iter=5000,random_state=42)

scores = cross_validate(model, X_train, y_train, scoring = scoring , cv = 10, return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# ### Logistic Model GridsearchCV

# In[163]:


recall_Hispanic = make_scorer(recall_score, average=None, labels=["Hispanic"])


# In[164]:


CVmodel = LogisticRegression(class_weight='balanced',max_iter=10000,random_state=42) 


# In[165]:


penalty = ["l1", "l2"]
C = [0.01, 0.1, 1, 5, 16, 19, 22, 25]
log__solver = ["liblinear", "lbfgs"]

param_grid = {
    "penalty": penalty,
    "C": C,
    "solver": log__solver
}


# In[166]:


grid_model = GridSearchCV(CVmodel, param_grid=param_grid, scoring=recall_Hispanic, cv=10, n_jobs=-1, return_train_score=True)
grid_model.fit(X_train, y_train)


# In[167]:


grid_model.best_params_


# In[168]:


pd.DataFrame(grid_model.cv_results_).loc[grid_model.best_index_, ["mean_test_score", "mean_train_score"]]


# In[169]:


ConfusionMatrixDisplay.from_estimator(grid_model, X_test, y_test);


# In[170]:


eval_metric(grid_model, X_train, y_train, X_test, y_test)


# In[171]:


from scikitplot.metrics import plot_roc, plot_precision_recall


model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test, y_pred_proba)
plt.show();


# ## 2. SVC

# ### Vanilla SVC model

# In[172]:


SVCmodel = SVC(class_weight="balanced", random_state=101)


# In[173]:


SVCmodel.fit(X_train, y_train)
eval_metric(SVCmodel, X_train, y_train, X_test, y_test)


# In[174]:


model = SVC(class_weight="balanced", random_state=101)

scores = cross_validate(model, X_train, y_train, scoring=recall_Hispanic, cv = 10, n_jobs=-1, return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# ###  SVC Model GridsearchCV

# In[175]:


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1],
}


# In[176]:


model = SVC(class_weight="balanced", random_state=101)

svm_model_grid = GridSearchCV(
    model,
    param_grid,
    scoring=recall_Hispanic,
    cv=10,
    n_jobs=-1,
    return_train_score=True,
)


# In[177]:


svm_model_grid.fit(X_train, y_train)


# In[178]:


svm_model_grid.best_params_


# In[179]:


svm_model_grid.best_estimator_


# In[180]:


pd.DataFrame(svm_model_grid.cv_results_).loc[svm_model_grid.best_index_, ["mean_test_score", "mean_train_score"]]


# In[181]:


y_pred = svm_model_grid.predict(X_test)
y_pred


# In[182]:


ConfusionMatrixDisplay.from_estimator(svm_model_grid, X_test, y_test);


# In[183]:


eval_metric(svm_model_grid, X_train, y_train, X_test, y_test)


# In[184]:


from scikitplot.metrics import plot_roc, plot_precision_recall


model = SVC(C=1, class_weight='balanced', random_state=101)

model.fit(X_train, y_train)
decision_function = model.decision_function(X_test)

# y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test, decision_function)
plt.show();


# ## 3. RF

# ### Vanilla RF Model

# In[185]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=101, class_weight='balanced')
rf_model.fit(X_train,y_train)


# In[186]:


y_pred = rf_model.predict(X_test)


# In[187]:


eval_metric(rf_model, X_train, y_train, X_test, y_test)


# In[188]:


ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test);


# In[189]:


model = RandomForestClassifier(class_weight="balanced", random_state=101)

scores = cross_validate(
    model, X_train, y_train, scoring=scoring, cv=5, n_jobs=-1, return_train_score=True
)
df_scores = pd.DataFrame(scores, index=range(1, 6))
df_scores.mean()[2:]


# ### RF Model GridsearchCV

# In[190]:


model = RandomForestClassifier(class_weight='balanced', random_state=101)


# In[191]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}


# In[192]:


grid_search = GridSearchCV(model, param_grid,scoring=recall_Hispanic, n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)


# In[193]:


grid_search.best_params_


# In[194]:


grid_search.best_estimator_


# In[195]:


grid_search.best_estimator_


# In[196]:


eval_metric(grid_search, X_train, y_train, X_test, y_test)


# In[197]:


model = RandomForestClassifier(class_weight="balanced", max_depth=2, n_estimators=400, random_state=101)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test, y_pred_proba)
plt.show();


# ## 4. XGBoost

# ### Vanilla XGBoost Model

# In[202]:


import xgboost as xgb


# In[203]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[200]:


XGBmodel = xgb.XGBClassifier(random_state=101)


# In[204]:


XGBmodel.fit(X_train, y_train_encoded)


# In[205]:


eval_metric(XGBmodel, X_train, y_train_encoded, X_test, y_test_encoded)


# In[206]:


from sklearn.utils import class_weight

classes_weights = class_weight.compute_sample_weight(
    class_weight="balanced", y=y_train_encoded
)
classes_weights


# In[207]:


my_dict = {"weights": classes_weights, "label": y_train_encoded}

comp = pd.DataFrame(my_dict)

comp.head()


# In[208]:


comp.groupby("label").value_counts()


# In[209]:


scoring_xgb = {
    "precision_Hispanic": make_scorer(precision_score, average=None, labels=[1]),
    "recall_Hispanic": make_scorer(recall_score, average=None, labels=[1]),
    "f1_Hispanic": make_scorer(f1_score, average=None, labels=[1]),
}


# In[210]:


model = xgb.XGBClassifier(random_state=101)

scores = cross_validate(
    model,
    X_train,
    y_train_encoded,
    scoring=scoring_xgb,
    cv=5,
    n_jobs=-1,
    return_train_score=True
)
df_scores = pd.DataFrame(scores, index=range(1, 6))
df_scores.mean()[2:]


# ### XGBoost Model GridsearchCV

# In[211]:


param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300]
}


# In[212]:


XGBmodel = XGBClassifier(random_state=101)


# In[213]:


grid_model = GridSearchCV(
    XGBmodel, param_grid=param_grid, scoring=recall_Hispanic, cv=10, n_jobs=-1, return_train_score=True
)


# In[214]:


grid_model.fit(X_train, y_train_encoded)


# In[215]:


grid_model.best_estimator_


# In[216]:


grid_model.best_params_


# In[217]:


eval_metric(grid_model, X_train, y_train_encoded, X_test, y_test_encoded)


# In[218]:


model = XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.05,
            max_depth=2,
            n_estimators=20,
            subsample=0.8,
            random_state=101,
        )
model.fit(X_train, y_train_encoded, sample_weight=classes_weights)

y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test_encoded, y_pred_proba)
plt.show()


# ---
# Final Model
# ---

# ---
# ---

# In[219]:


operations_final = [
    ("OneHotEncoder", column_trans),
    (
        "log",
        LogisticRegression(class_weight="balanced", max_iter=10000, random_state=101),
    ),
]


# In[220]:


finalModel = Pipeline(steps=operations_final)


# In[221]:


finalModel.fit(X, y)


# In[222]:


X.shape


# In[223]:


X_test.shape


# In[224]:


X[X.Gender == "Female"].describe()


# In[226]:


newFemale = X[X.Gender == "Female"].describe(include="all").loc["mean"]


# In[227]:


newFemale = X[X.Gender == "Female"].describe(include="all").loc["mean"]
newFemale


# In[228]:


newFemale["Gender"] = "Female"
newFemale["SubjectsBirthLocation"] = "New York"
newFemale["WritingPreference"] = "Left hand"


# In[229]:


finalModel.predict(pd.DataFrame(newFemale).T)


# In[230]:


from sklearn.metrics import matthews_corrcoef

matthews_corrcoef(y_test, y_pred)


# In[231]:


from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test, y_pred)


# ---
# ---

# # SMOTE
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

# ##  Smote implement

# ## Logistic Regression Over/ Under Sampling

# In[233]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# # Over Sampling

# In[234]:


smote = SMOTE()

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution
print(y_train_resampled.value_counts())


# In[235]:


model = LogisticRegression(class_weight='balanced',max_iter=10000,random_state=101)
model.fit(X_train_resampled, y_train_resampled)


# In[236]:


y_pred = model.predict(X_test)


# In[237]:


eval_metric(model, X_train_resampled, y_train_resampled, X_test, y_test)


# In[238]:


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# # Under Sampling

# In[239]:


under = RandomUnderSampler(sampling_strategy='auto')


# In[240]:


X_train_resampled, y_train_resampled = under.fit_resample(X_train, y_train)

# Check the class distribution
print(y_train_resampled.value_counts())


# In[241]:


model = LogisticRegression(class_weight='balanced',max_iter=10000,random_state=101)
model.fit(X_train_resampled, y_train_resampled)


# In[242]:


y_pred = model.predict(X_test)


# In[243]:


eval_metric(model, X_train_resampled, y_train_resampled, X_test, y_test)


# In[244]:


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# # Custom Sampling Ratios

# In[245]:


over = SMOTE(sampling_strategy={"Hispanic": 1000})
under = RandomUnderSampler(sampling_strategy={"White": 2500})


# In[246]:


X_resampled_over, y_resampled_over = over.fit_resample(X_train, y_train)


# In[248]:


y_resampled_over.value_counts()


# In[249]:


X_resampled_under, y_resampled_under = under.fit_resample(X_train, y_train)


# In[250]:


y_resampled_under.value_counts()


# In[251]:


model = LogisticRegression(class_weight='balanced',max_iter=10000,random_state=101)
model.fit(X_resampled_over, y_resampled_over)


# In[252]:


y_pred = model.predict(X_test)


# In[253]:


eval_metric(model, X_train_resampled, y_train_resampled, X_test, y_test)


# In[254]:


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# In[255]:


model = LogisticRegression(class_weight='balanced',max_iter=1000,random_state=101)
model.fit(X_resampled_under, y_resampled_under)


# In[256]:


y_pred = model.predict(X_test)


# In[257]:


eval_metric(model, X_train_resampled, y_train_resampled, X_test, y_test)


# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
