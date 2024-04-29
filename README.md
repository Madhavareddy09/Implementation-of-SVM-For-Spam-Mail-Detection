# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.
## PROGRAM :
```python
'''
Program to implement the SVM For Spam Mail Detection..
Developed by : K MADHAVA REDDY
RegisterNumber : 212223240064
'''
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## OUTPUT :
### Encoding
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/65935976-e3c8-40ce-a32d-afc3f87d77d6)

### Head()
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/e275e762-2027-41fd-85f9-eab7f494fc90)

### Info()
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/1bae5c66-c39f-4c4c-8921-5a1e43af5eec)

### isnull().sum()
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/d5c41095-dfce-465b-84d5-5c27b54a69b5)

### Prediction of y
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/95704d37-5457-4602-8aee-277821111fbf)

### Accuracy
![image](https://github.com/Madhavareddy09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742470/fc1e1365-a570-4e2f-823c-31d1c38be700)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
