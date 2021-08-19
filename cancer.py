import numpy as np
from sklearn import preprocessing , cross_validation , neighbors
import pandas as pd
from sklearn.metrics import accuracy_score


#read file, replace missing data as outliers, drop unnecesary items(id here)
df=pd.read_csv('cancer_data.txt')
df.replace('?' , -99999 , inplace=True)
df.drop('id' , 1 , inplace=True)

# define features(x) and labels(y)
x=np.array(df.drop(['class'], 1))
y=np.array(df['class'])

#split 20% of data as test data
x_train , x_test , y_train , y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#fit an predict data
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
pre=clf.predict(x_test)

#both are same
confidence=clf.score(x_test,y_test)
acc=accuracy_score(pre,y_test)

print(pre)
print("Malign")
print(confidence)
print(acc)

#predict few particular cases
example_measure=np.array([[4,2,1,1,1,6,8,3,5],[4,2,1,2,1,6,7,3,5]])
prediction=clf.predict(example_measure)
print(prediction)
