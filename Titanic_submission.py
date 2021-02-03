import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
path = <your path here>

train = pd.read_csv(path+r'\train.csv')
test = pd.read_csv(path+r'\test.csv')
age_bins = [0, 16, 32, 48, 64, 80] #min to max
fare_bins = [-0.001, 7.91, 14.454, 31, 520] # min to max

# Checking what affects what and to what extent xD
# sns.catplot(x='Sex', y='Survived', data=train, kind='bar', height=5)
# sns.scatterplot(x=train.Name, y=train.Age, hue=train.Survived).set(xticklabels=[])
# sns.FacetGrid(train, col='Survived').map(sns.histplot, 'Fare', bins=25)
# sns.FacetGrid(train, col='Survived').map(sns.histplot, 'Fare', bins=25)
# sns.scatterplot(x=train.Fare, y=train.Age, hue=train.Survived)
# sns.catplot(data=train, x='Pclass', y='Survived', kind='bar', hue='Sex')
# sns.countplot(data=train, x='Survived', hue='Pclass')
# sns.catplot(data=train, x='SibSp', y='Survived', height=5, kind='bar')
# sns.countplot(data=train, x='Survived', hue='Embarked')
# plt.show('hold')
train.Sex = train.Sex.replace('male', 0).replace('female', 1)
test.Sex = test.Sex.replace('male', 0).replace('female', 1)
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)
train.Embarked = train.Embarked.replace('S', 0).replace('C', 1).replace('Q', 2)
test.Embarked = test.Embarked.replace('S', 0).replace('C', 1).replace('Q', 2)
train.Fare = train.Fare.fillna(train.Fare.mean())
test.Fare = test.Fare.fillna(test.Fare.mean())

for _ in (train, test):
    _['Title'] = _.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train.Title = train.Title.replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Rest').replace('Mme', 'Mrs').replace('Ms', 'Miss').replace('Sir', 'Mr')
test.Title = test.Title.replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Rest').replace('Ms', 'Miss')

title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rest': 5}
for _ in (train, test):
	_['Title'] = _['Title'].map(title_map)

for _ in (train, test):
	_.loc[(_.Age.isna()) & (_.Title == 1), 'Age'] = round(_[_.Title == 1].Age.mean(), 2)
	_.loc[(_.Age.isna()) & (_.Title == 3), 'Age'] = round(_[_.Title == 3].Age.mean(), 2)
	_.loc[(_.Age.isna()) & (_.Title == 2), 'Age'] = round(_[_.Title == 2].Age.mean(), 2)
	_.loc[(_.Age.isna()) & (_.Title == 4), 'Age'] = round(_[_.Title == 4].Age.mean(), 2)
	_.loc[(_.Age.isna()) & (_.Title == 5), 'Age'] = round(_[_.Title == 5].Age.mean(), 2)

for _ in (train, test):
	for i in range(5):
		_.loc[(_.Age > age_bins[i]) & (_.Age <= age_bins[i+1]), 'Age'] = i

# Survival Rate by Title
# print(train[['Title', 'Survived']].groupby('Title').mean().sort_values(by='Survived', ascending=False))

# Survival Rate by Age group
# print(train[['Age', 'Survived']].groupby('Age').mean().sort_values(by='Survived', ascending=False))

for _ in (train, test):
	for i in range(4):
		_.loc[(_.Fare > fare_bins[i]) & (_.Fare <= fare_bins[i+1]), 'Fare'] = i

# Survival Rate by Fare group
# print(train[['Fare', 'Survived']].groupby('Fare').mean().sort_values(by='Survived', ascending=False))

# Survival Rate by Cabin assigned
# print(train[['CabinAssigned', 'Survived']].groupby('CabinAssigned').mean().sort_values(by='Survived', ascending=False))

# Dropping irrelevant data
columns_to_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket']
for _ in (train, test):
	_.drop(columns=columns_to_drop, axis=1, inplace=True)

X_train = train.drop(columns=['Survived'], axis=1)
Y_train = train['Survived']
X_test  = test.copy()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

test = pd.read_csv(path+r'\test.csv')
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Y_pred})

file_name = 'submission.csv'
submission.to_csv(path+r'\\'+file_name, index=False)
