
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm


# ## Helper functions and variables

# #### I will be using three more features apart from text feature i.e. like, share and owner type

# In[2]:

features1 = ['owner_type']
features2 = ['nb_like', 'nb_share']
features3 = ['nb_like', 'nb_share', 'owner_type']
categ_dict = {'OK': 0, 'Reseller': 1, 'Non': 2}
owner_dict = {'user': 0, 'page': 1}

def string_to_int(row, some_dict):
    if row in some_dict:
        row = some_dict[row]
    return row

vectorizer = TfidfVectorizer(min_df=1, stop_words={'english'})

def vectorize_text_tfidf(dataframe, type_dataframe):
    if type_dataframe == "train":
        return vectorizer.fit_transform(dataframe['text'].values)
    elif type_dataframe == "test":
        return vectorizer.transform(dataframe['text'].values)
    else:
        return "Please enter train or test"
    
def concat_features(dataframe, tfidf_text_features, features_list):
    for x in features_list:
        features = hstack((tfidf_text_features, dataframe[x].values.reshape(tfidf_text_features.shape[0], 1)))
    return features

# TODO - stemming
def text_preprocessing(text):
    # for removing html tags from text
    text = BeautifulSoup(text, 'html5lib')
    # to keep only english letters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())
    # to remove spaces, tabs and new lines
    text = re.sub( '\s+', ' ', text).strip().lower()
    return text


# ## 1. Reading the data and data wrangling

# In[3]:

data_df = pd.read_csv("Facebook_Sellers_Challenge.csv", delimiter="\t")


# ### 1.1 Converting target label and other features to numeric form

# In[4]:

# Convert target labels to numeric
data_df['INDEX New'] = data_df['INDEX New'].apply(string_to_int, args=(categ_dict,))
data_df['owner_type'] = data_df['owner_type'].apply(string_to_int, args=(owner_dict,))


# ### 1.2 Classifying data with minimal cleaning using all features
# #### Let's test out data with raw data in tfidf format

# In[5]:

# drop rows where description is Nan 
raw_data_df = data_df.dropna(subset=["description", "owner_type"])
# rename columns for more readable columns name
raw_data_df = raw_data_df.rename(columns = {'description':'text', 'INDEX New': 'label'}).reset_index()


# #### Data points in each class

# In[6]:

raw_data_df.groupby(['label']).agg(['count'])['index']


# #### Split into train and test

# In[7]:

raw_train_df, raw_test_df = train_test_split(raw_data_df, test_size = 0.2, random_state=42)


# #### Training the model

# In[8]:

raw_train_tfidf = concat_features(raw_train_df, vectorize_text_tfidf(raw_train_df, "train"), features3)
# these parameters were learned from grid search below
raw_forest = RandomForestClassifier(n_estimators = 200, oob_score=True, n_jobs=-1, max_features="auto")
raw_forest = raw_forest.fit(raw_train_tfidf, raw_train_df['label'].values)


# #### Testing the model

# In[9]:

raw_test_tfidf = concat_features(raw_test_df, vectorizer.transform(raw_test_df['text'].values), features3)

predicted = raw_forest.predict(raw_test_tfidf)
print("Accuracy : %0.2f" % (np.mean(predicted == raw_test_df['label'].values) * 100))


# So we get around 86% accuracy

# ### But this is not very good approach as we need to do cleaning and get just words for good features.

# ### 1.3 Data cleaning

# In[10]:

# drop rows where description is Nan 
data_df = data_df.dropna(subset=["description", "owner_type"])
# drop duplicates based on multiple features
data_df = data_df.drop_duplicates(subset=["description", "nb_like", "nb_share", "owner_type"]).reset_index()
# rename columns for more readable columns name
data_df = data_df.rename(columns = {'description':'text', 'INDEX New': 'label'}).reset_index()


# #### Cleaning text for good and clean text

# #### I lost around 12,000 data points in this whole process but I felt it is important because there were lot or rows with no description, duplicates and also empty description after this text processing step 

# In[11]:

data_df['text'] = data_df['text'].apply(text_preprocessing)
data_df = data_df.drop(data_df[data_df.text == ""].index)


# #### Data distribution for three categories.
# #### 0 - 'OK', 1 - 'Reseller', 2 - 'Non'

# In[12]:

data_df.groupby(['label']).agg(['count'])['index']


# ## Splitting data into train and test

# In[13]:

train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state=42)


# # 2. Models

# ## 2.1 Random Forest
# ### 2.1.1 Grid Search of Random forest Algorithm to find the best parameters.

# #### Convert all data into Tf-Idf vector form

# In[14]:

# Convert text into tfidf vector form
vectorizer = TfidfVectorizer(min_df=1, stop_words={'english'})
text_data_tfidf = vectorizer.fit_transform(data_df['text'].values)


# #### Model Initilization with initial parameters.

# In[15]:

# Initialize model with parameters
rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True)
# parameters to optimize
# I think these two parameters (n_estimators, and max_features) are very important to optimize.
params = { 
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2']
}


# #### Initialize the grid object

# In[172]:

# Initialize the GridSearchCV object with parameters
# run a cross validation for 5 sets of data
CV_rfc = GridSearchCV(estimator=rfc, param_grid=params, cv= 5)


# #### Fitting the model with our data

# In[173]:

# Fit the data to find the optimal parameters
CV_rfc.fit(text_data_tfidf, data_df['label'].values)


# #### Find the best parameters

# In[182]:

# best_params_ attribute gives the best parameters to be used for the model for our data.
print(CV_rfc.best_params_)


# ### 2.1.2 Classification using learned parameters of Random Forest Model

# #### Fitting the model

# In[401]:

# Creating the pipeline for our classifcation task to make it more readable
# n_jobs = -1 will use all the processors. You can limit it to one by setting n_jobs = 1
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(stop_words={'english'})),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators = 200, max_features="auto", n_jobs=-1, oob_score=True))])


# #### Fitting our pipeline with our data

# In[402]:

pipeline.fit(train_df['text'].values, train_df['label'].values)


# #### Testing our dataset with test dataset

# In[403]:

predictions = pipeline.predict(test_df['text'].values)
mean_accuracy = np.mean(predictions == test_df['label'].values) * 100
print("Mean accuracy: %.2f" % mean_accuracy)


# #### Different metrics of our classifcation

# In[423]:

target_names = ['OK', 'Reseller', 'Non']
print(classification_report(test_df['label'].values, predictions, target_names=target_names))


# #### Confusion matrices
# #### 0 - 'OK', 1 - 'Reseller', 2 - 'Non'

# In[405]:

# confusion matrix with number of false and true classifcations
pd.crosstab(test_df['label'].values, predictions, rownames=['True'], colnames=['Predicted'], margins=True)


# #### With percentages

# In[406]:

lookup = {0 : 'OK', 1: 'Reseller', 2: 'Non'}
y_true = pd.Series([lookup[_] for _ in test_df['label'].values])
y_pred = pd.Series([lookup[_] for _ in predictions])
pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())


# ## 2.2 Combining models with VotingClassifier

# #### Getting tf-idf vector of data.

# In[45]:

text_train_tfidf = concat_features(train_df, vectorize_text_tfidf(train_df, "train"), features3)


# #### Combining three models: Multinomial Naive Bayes, Random Forest and Support Vector Machine

# In[48]:

model1 = MultinomialNB()
model2 = RandomForestClassifier(n_estimators = 200, oob_score=True, n_jobs=-1, max_features="auto")
model3 = svm.SVC(kernel='rbf', probability=True)
ensemble = VotingClassifier(estimators=[('np', model1), ('forest', model2), 
                                        ('svc', model3)], voting='soft', weights=[2,3,1], n_jobs=-1)


# In[49]:

ensemble = ensemble.fit(text_train_tfidf,train_df['label'].values)


# #### Testing

# In[50]:

text_test_tfidf = concat_features(test_df, vectorizer.transform(test_df['text'].values), features3)

predicted = ensemble.predict(text_test_tfidf)
print("Accuracy : %0.2f " % (np.mean(predicted == test_df['label'].values) * 100))


# # 3. Models with Multiple features

# ## 3.1 Random Forest

# ### Training
# #### Using all features becasue not much difference with using less or more features

# In[18]:

text_train_tfidf = concat_features(train_df, vectorize_text_tfidf(train_df, "train"), features3)

forest2 = RandomForestClassifier(n_estimators = 200, oob_score=True, n_jobs=-1, max_features="auto")
forest2 = forest2.fit(text_train_tfidf, train_df['label'].values)


# ### Test the model

# In[19]:

text_test_tfidf = concat_features(test_df, vectorizer.transform(test_df['text'].values), features3)

predicted = forest2.predict(text_test_tfidf)
print("Accuracy : %0.2f " % (np.mean(predicted == test_df['label'].values) * 100))


# ## 3.2 Multinomial Naive Bayes

# ### Training
# #### Here the best results came up with using all features

# In[38]:

text_train_tfidf = concat_features(train_df, vectorize_text_tfidf(train_df, "train"), features3)


# In[39]:

mnb = MultinomialNB()
mnb = mnb.fit(text_train_tfidf, train_df['label'].values)


# ### Testing

# In[40]:

text_test_tfidf = concat_features(test_df, vectorizer.transform(test_df['text'].values), features3)

predicted = mnb.predict(text_test_tfidf)
print("Accuracy : %0.2f" % (np.mean(predicted == test_df['label'].values) * 100))


# ## 3.3 SVM

# ### Training

# In[23]:

text_train_tfidf = concat_features(train_df, vectorize_text_tfidf(train_df, "train"), features3)


# In[24]:

svm_model = svm.LinearSVC()
svm_model = svm_model.fit(text_train_tfidf, train_df['label'].values)


# ### Testing

# In[25]:

text_test_tfidf = concat_features(test_df, vectorizer.transform(test_df['text'].values), features3)

predicted = svm_model.predict(text_test_tfidf)
print("Accuracy : %0.2f" % (np.mean(predicted == test_df['label'].values) * 100))


# ## 3.4 OneVsRestClassifier and Random forest

# ### Training

# In[26]:

text_train_tfidf = concat_features(train_df, vectorize_text_tfidf(train_df, "train"), features3)


# In[27]:

onevsone_model = OneVsRestClassifier(RandomForestClassifier(n_estimators = 200, oob_score=True, 
                            max_features="auto"), n_jobs=-1).fit(text_train_tfidf, train_df['label'].values)


# ### Testing

# In[28]:

text_test_tfidf = concat_features(test_df, vectorize_text_tfidf(test_df, "test"), features3)

predicted = onevsone_model.predict(text_test_tfidf)
print("Accuracy: %0.2f" % (np.mean(predicted == test_df['label'].values) * 100))


# In[35]:




# In[ ]:




# In[ ]:




# In[ ]:



