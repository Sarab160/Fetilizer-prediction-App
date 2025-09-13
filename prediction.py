import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
df=pd.read_csv("Fertilizer Prediction.csv")

# print(df.info())
# # print(df["Crop Type"].unique)
# # print(np.unique(df["Crop Type"]))
# # print(df.shape)
# # sns.pairplot(data=df)
# # plt.show()
# print(df.columns)
x=df[['Temparature', 'Humidity ', 'Moisture',
       'Nitrogen', 'Potassium', 'Phosphorous']]
le=LabelEncoder()
y=le.fit_transform(df["Fertilizer Name"])

oe=OrdinalEncoder()
fe=df[["Soil Type","Crop Type"]]
encode_array=oe.fit_transform(fe)
encodedata=pd.DataFrame(encode_array,columns=fe.columns)

final_x=pd.concat([x,encodedata],axis=1)

x_train,x_test,y_train,y_test=train_test_split(final_x,y,test_size=0.2,random_state=42)

lr=LogisticRegression()

lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))
print(lr.score(x_train,y_train))
print(precision_score(y_test,lr.predict(x_test),average="macro"))
print(recall_score(y_test,lr.predict(x_test),average="macro"))
print(f1_score(y_test,lr.predict(x_test),average="macro"))

cf=confusion_matrix(y_test,lr.predict(x_test))
sns.heatmap(data=cf,annot=True)
plt.show()

