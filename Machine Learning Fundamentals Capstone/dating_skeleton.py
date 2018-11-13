#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as pl
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
import re
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches



# Create your df here:
df = pd.read_csv("/Users/longvo/Downloads/ML Final Project/profiles.csv")

# df.job.head()
# df = df.dropna()
df.fillna({'age': 'NA',
           'body_type': 'NA',
           'diet': 'NA',
           'drugs': 'NA',
           'education': 'NA',
           'essay1': 'NA',
           'essay2': 'NA',
           'essay3': 'NA',
           'essay4': 'NA',
           'essay5': 'NA',
           'essay6': 'NA',
           'essay7': 'NA',
           'essay8': 'NA',
           'essay9': 'NA',
           'ethnicity': 'NA',
           'orientation': 'NA',
           'height': 'NA',
           'job': 'NA',
           'offspring': 'NA',
           'sign': 'NA',
           'pets': 'NA',
           'religion': 'NA',
           'smokes': 'NA',
           'speaks': 'NA',
           'drinks': 'NA',
           'essay0': 'NA'},
          inplace=True)

print(df.shape)

drink_mapping = {"NA": 0, "not at all": 1, "rarely": 2, "socially": 3, "often": 4, "very often": 5, "desperately": 6}
df["drinks_code"] = df.drinks.map(drink_mapping)
drinks_code = df["drinks_code"].replace(np.nan, '')

smoke_mapping = {"NA": 0, "no": 1, "sometimes": 2, "when drinking": 3, "yes": 4, "trying to quit": 5}
df["smokes_code"] = df.smokes.map(smoke_mapping)
smokes_code = df["smokes_code"].replace(np.nan, '')

drug_mapping = {"NA": 0, "never": 1, "sometimes": 2, "often": 3}
df["drugs_code"] = df.drugs.map(drug_mapping)
drugs_code = df["drugs_code"].replace(np.nan, '')

sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
sex_code = df["sex_code"].replace(np.nan, '')

orientation_mapping = {"straight": 0, "gay": 1, "bisexual": 2}
df["orientation_code"] = df.orientation.map(orientation_mapping)
orientation_code = df["orientation_code"].replace(np.nan, '')

df["age_bucket"] = pd.cut(df["age"], [0, 25, 30, 35, 120], include_lowest=True, labels=['0-25', '26-30', '31-35', '36-120'])
age_bucket_mapping = {'0-25': 0, '26-30': 1, '31-35': 2, '36-120': 3}
df["age_bucket_code"] = df.age_bucket.map(age_bucket_mapping)
age_bucket_code = df["age_bucket_code"].replace(np.nan, '')




print(df.groupby('orientation').size())
df = df[["sex", "sex_code", 'smokes_code', 'drinks_code', 'drugs_code', 'age_bucket_code', 'orientation_code']]

# sns.countplot(df['sex'], label="Count")
# plt.show()


# df.drop('sex_code', axis=1).plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(9, 9), title='Box Plot for each input variable')
# plt.savefig('sex_box')
# plt.show()

# df.drop('sex_code', axis=1).hist(bins=30, figsize=(9, 9))
# pl.suptitle("Histogram for each numeric input variable")
# plt.savefig('sex_hist')
# plt.show()

feature_names = ['smokes_code', 'drinks_code', 'drugs_code', 'age_bucket_code', 'orientation_code']
X = df[feature_names]
y = df['sex']

# cmap = cm.get_cmap('gnuplot')
# scatter = pd.scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('sex_scatter_matrix')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test)))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# LINEAR REGRESSION METHOD - Question: Predict age bucket by smoking, drug, alchol habits and gender

x = df[['smokes_code', 'drinks_code', 'drugs_code', 'sex_code']]
y = df["age_bucket_code"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=42)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=42)

mlr = LinearRegression()

model = mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

plt.show()
# Input code here:
print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))


plt.scatter(y_test, y_predict)
plt.plot(range(5), range(5))
plt.show()
