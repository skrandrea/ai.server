import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from pydantic import BaseModel


# TRAINING DATA
df = pd.read_csv("./stress.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# print(classifier.predict(sc.transform([[34.08, 91.344, 4.016, 63.36]])))


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
    stressLevel = (classifier.predict(sc.transform(
        [[temperature, spo2, sleepTime, heartRate]])))
    return {"stressLevel": str(stressLevel[0])}
