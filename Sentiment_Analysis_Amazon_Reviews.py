#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis for Amazon Reviews

#  #Importing all the necessary libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow
import seaborn as sns
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re
from tensorflow.python.keras import models, layers, optimizers
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


# #Passing the zip file in the function 

# In[2]:


def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
    return np.array(labels), texts
train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('test.ft.txt.bz2')


# # General overview of data

# In[3]:


train_labels[0]


# In[4]:


train_texts[0:5]


# In[5]:


df = pd.DataFrame({'Text':train_texts,'Label':train_labels})
df.head(20)


# In[6]:


df['Label'][:10000].value_counts()

## 5097/4903 = 1.04
## here we see that for a sample of 10,000 comments the ratio is in favour of negative reviews at 1.04


# In[7]:


## so we're going to try to take a larger sample to see if the trend remains the same


# In[8]:


df['Label'][:20000].value_counts()

## 10257/9743 = 1.05
## we see that for a sample of 20,000 comments, it's the opposite, the ratio is in favour of positive reviews at 1.05


# In[9]:


## So we're going to take a much larger sample to get an idea


# In[10]:


df['Label'][:100000].value_counts()

## 51267/48733 = 1.05
## for a sample really much larger, the ratio is the same as for a sample of 20,000 comments
## we can perform our model on the basis of the sample of 20,000 comments


# #Spliting the data

# In[11]:


train_labels = train_labels[0:20000]


# In[12]:


train_texts = train_texts[0:20000]


# #Text pre-processing

# In[13]:


train_texts[0:5]


# # Data Cleaning

# In[14]:


# Verify if there are any Duplicate lines
duplicates = df.duplicated()

# print the number of lines in double 
print("Number of duplicate lines in dataset : ", duplicates.sum())


# In[15]:


# Verify if there are any missing values
missing = df.isna().sum()

# Print the number of missing values 
print(missing)


# In[16]:


df['Label'].value_counts()


# In[17]:


# Print the number of labels 
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique,counts)))


# # Data Visualisation

# In[18]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Calculate the length of each reviews
review_lengths = [len(x) for x in train_texts[:20000]]
# Print the average length of reviews
print("Average review length:", np.mean(review_lengths))


# In[19]:


# Draw a Histogram of length of reviews
plt.hist(review_lengths, bins=30)
plt.xlabel('Length of the Reviews')
plt.ylabel('Number of Reviews')
plt.show()


# In[20]:


# Creating Wordcloud for Positive and Negative Reviews


# In[21]:


# If label is 1 it is a Positive Review and if label is 0 it is a Negative Review


# In[22]:


positive_reviews = ' '.join([review for review, label in zip(train_texts, train_labels) if label == 1])


# In[23]:


negative_reviews = ' '.join([review for review, label in zip(train_texts, train_labels) if label == 0])


# In[ ]:


consolidated=' '.join(word for word in df['Text'][df['Label']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


# Wordcloud for Positive reviews
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(positive_reviews[:20000])
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[ ]:


# Wordcloud for Negative Reviews
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(negative_reviews[:20000])
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# # Transform the database in order to perform Machine learning model

# In[33]:


import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalise_texts(texts):
    normalised_texts=[]
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r' ', no_punctuation)
        normalised_texts.append(no_non_ascii)
    return normalised_texts

train_texts = normalise_texts(train_texts)
test_texts = normalise_texts(test_texts)


# In[35]:


train_texts[0:5]


# # Converting the string reviews into Binary 

# In[37]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)


# In[39]:


X_test


# #Applying Logistic Regression model to our Dataset

# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X_train,X_val,y_train,y_val = train_test_split(X, train_labels,train_size=0.75)

for c in [0.01, 0.05 , 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train,y_train)
    print ("Accuracy for C=%s: %s" %(c, accuracy_score(y_val, lr.predict(X_val))))


# In[43]:


#Checking the accuracy


# In[45]:


lr.predict(X_test[29])


# In[47]:


test_labels[29]


# In[49]:


test_texts[29]


# In[51]:


from sklearn.metrics import classification_report


# In[53]:


predictions= lr.predict(X_val)
report = classification_report(y_val,predictions, output_dict=True)

df_report = pd.DataFrame(report).transpose().round(2)

#visualising
cm = sns.light_palette("green", as_cmap=True)
df_report.style.background_gradient(cmap=cm)


# In[55]:


# Visualizing Confusion Matrix 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ax= plt.subplot()
cm= confusion_matrix(y_val,predictions)

sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Greens');  


# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['neg', 'pos']); ax.yaxis.set_ticklabels(['neg', 'pos']);


# In[ ]:





# # Now to apply the Random Forrest
# 

# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Transforming Text into vectors 
tfidf_vectorizer = TfidfVectorizer(max_features= 1432)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)







# In[58]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, train_labels)


# In[59]:


rf_predictions = rf_model.predict(X_test_tfidf)

# Model Evaluating
rf_accuracy = accuracy_score(test_labels, rf_predictions)
rf_classification_report = classification_report(test_labels, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Classification Report:\n", rf_classification_report)


# In[60]:


#  rf_predictions are model predictions and test_labels are the true labels
cm = confusion_matrix(test_labels, rf_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizing the confusion matrix

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[61]:


# Ploting Precision-Recall Curve

from sklearn.metrics import precision_recall_curve, auc

rf_probs = rf_model.predict_proba(X_test_tfidf)[:, 1]

precision, recall, thresholds = precision_recall_curve(test_labels, rf_probs)
auc_score = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (area = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


# In[62]:


# Comparing two models
rf_accuracy = 0.84  
rf_f1 = 0.84  

lr_accuracy = 0.87  
lr_f1 = 0.87  

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Random Forest": [rf_accuracy, rf_f1],
    "Logistic Regrr": [lr_accuracy, lr_f1]  
})

comparison_df


# In[ ]:





# # Conclusion
# 
# In this sentiment analysis project, we explored two different machine learning models: Random Forest and Logistic Regression. 
#     
# Our analysis revealed that while both models performed admirably, Logistic Regression is slightly more in terms of accuracy and F1-score.
# 
# 

# In[ ]:





# In[ ]:




