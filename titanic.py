import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

print(df.head())

print(df.info())
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print(df.isnull().sum())

plt.figure()
df['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure()
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar')
plt.title("Gender vs Survival")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

plt.figure()
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar')
plt.title("Class vs Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(df['Age'], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df['Fare'], bins=10)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

print("Females have higher survival rate")
print("1st class passengers survived more")
print("Age and Fare show distribution patterns")