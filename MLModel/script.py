import pandas as pd
import regex as re
import nltk
from sklearn.metrics import confusion_matrix,accuracy_score
from nltk.corpus import stopwords
import pickle
import numpy as np
import seaborn as sns
#nltk.download('punkt')
#   nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

data=pd.read_csv("C:/Users/Student/Desktop/Honours/Second-Semester/SpecialTopics/Project/sneakers_Reviews_Dataset.csv",sep=";")

data=data.loc[:,["review_text","rating"]]

# Clean the data

#Check if the data has any missing values
print(data.isnull())

#Drop missing values
print("data shape before dropping missing values",data.shape)
data=data.dropna()
print("data shape after dropping missing values",data.shape)

#Check the distrinution of the data classes/ratings
print(data.loc[:,"rating"].value_counts())
# Remove the special characters
for i in range(len(data)):
    data.loc[i,"review_text"]=re.sub(r'[^\w\s]', '', data.loc[i,"review_text"])

#Change uppercase to lowercase
for i in range(len(data)):
    data.loc[i,"review_text"] = data.loc[i,"review_text"].lower()

#Remove stopwords
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    return ' '.join(filtered_tokens)
data['review_text'] = data['review_text'].apply(remove_stopwords)

#Remove unicode characters
for i in range(len(data)):
    data.loc[i,"review_text"] = re.sub(r'[^\x00-\x7F]+', '', data.loc[i,"review_text"])

#Remove numbers
for i in range(len(data)):
    data.loc[i,"review_text"] = re.sub(r'\d+', '', data.loc[i,"review_text"])

#Take target and features
x=data.loc[:,"review_text"]
y=data.loc[:,"rating"]

#Split data to be train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#Encode the text to be in numbers
count_vectorizer = CountVectorizer()
x_train = count_vectorizer.fit_transform(x_train)
x_train = x_train.toarray()
x_test = count_vectorizer.fit_transform(x_test)
x_test = x_test.toarray()
#print(x_train.shape,x_test.shape)
print(y_train)

#Train the model
classifier = LogisticRegression(penalty="l2",C=1)

#Cross-validate the model
cv_scores = cross_val_score(classifier, x_train, y_train, cv=5)
#Plot the validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Fold')
plt.ylabel('Validation Score')
plt.title('Cross-Validation Scores')
plt.grid(True)
plt.savefig("PlotofValidationScores.png")
plt.show()
plt.close()
#Save the validation scores
with open('cross-validation-scores.txt', 'w') as f:
    f.write(f"Cross-Validation-Scores: {cv_scores}")

#Fit the final model
classifier.fit(x_train, y_train)

# Make predictions
y_pred = classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(conf_matrix)
prediction=classifier.predict(x_test[0].reshape(1, -1))
def final_pred(prediction):
    if prediction==np.array([1]) or prediction==np.array([2]):
        prediction="Negative"
    elif prediction==np.array([3]):
        prediction="Neutral"
    else:
        prediction="Positive"
    return prediction
prediction=final_pred(prediction)
print(prediction)

#Save the results of the model
with open('results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}")

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png")
plt.show()
plt.close()

#Save the model and vectoriser for later use with new data
#with open('C:/Users/Student/Desktop/Honours/Second-Semester/SpecialTopics/Project/Django/MLModel/Model.pkl', 'wb') as model_file:
    #pickle.dump((classifier, count_vectorizer), model_file)
