
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json('train.json', orient='columns')
df['IngredientString'] = df['ingredients'].astype('str')
df['IngredientString'] = df['IngredientString'].str.strip('[').str.strip(']').str.replace(',',' ').str.replace(' ','').str.replace('\'',' ')

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(binary='true')
train_documents = [line for line in df['IngredientString']]
train_documents = count_vectorizer.fit_transform(train_documents)
train_labels = [line for line in df['cuisine']]

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(train_labels)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_documents, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB().fit(X_train,y_train)


#y_pred = classifier.predict(count_vectorizer.transform(["romainelettuce  blackolives  grapetomatoes  garlic  pepper  purpleonion  seasoning  garbanzobeans  fetacheesecrumbles"]))
y_pred = classifier.predict(X_test)
label = labelencoder_y.inverse_transform(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

df_test = pd.read_json('test.json', orient='columns')
df_test['IngredientString'] = df_test['ingredients'].astype('str')
df_test['IngredientString'] = df_test['IngredientString'].str.strip('[').str.strip(']').str.replace(',',' ').str.replace(' ','').str.replace('\'',' ')
test_documents = [line for line in df_test['IngredientString']]
test_documents = count_vectorizer.transform(test_documents)

y_test_submission = classifier.predict(test_documents)
test_labels = labelencoder_y.inverse_transform(y_test_submission)

sub = pd.read_csv('sample_submission.csv')
sub['cuisine'] = ''
sub['cuisine'] = test_labels[sub.index.values]
    
sub.to_csv('sample_submission.csv',index = False)

