import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel


# TRAINING DATA
db = pd.read_csv("./stress.csv")
x = db.iloc[:, :4] # Features
y = db.iloc[:, 4] # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# print(clf.predict([[36,99,6,75]]))


# FAST API
class StressForm(BaseModel):
    temperature: float
    spo2: float
    sleepTime: float
    heartRate: float


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello AI"}


@app.post("/")
async def root(item: StressForm):
    itemDict = item.dict()
    temperature = itemDict["temperature"]
    spo2 = itemDict["spo2"]
    sleepTime = itemDict["sleepTime"]
    heartRate = itemDict["heartRate"]
    # stressLevel = classifier.predict(sc.transform(
    # [[itemDict["temperature"], itemDict["spo2"], itemDict["sleepTime"], itemDict["heartRate"]]]))
    # print(type(itemDict["temperature"]), itemDict["spo2"],
    #       itemDict["sleepTime"], itemDict["heartRate"])
    stressLevel = (clf.predict(
        [[temperature, spo2, sleepTime, heartRate]]))
    return {"stressLevel": str(stressLevel[0])}
