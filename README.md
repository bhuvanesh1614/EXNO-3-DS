## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="657" height="449" alt="image" src="https://github.com/user-attachments/assets/a0780886-6758-4bea-afed-d4d284a66c9c" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="267" height="191" alt="image" src="https://github.com/user-attachments/assets/f3a11144-737f-4359-98f8-434810f127a5" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="490" height="361" alt="image" src="https://github.com/user-attachments/assets/b89b0227-a69a-4e02-a147-74542d2d9452" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="1860" height="569" alt="image" src="https://github.com/user-attachments/assets/0c412836-26bb-417b-9d7a-8a70c55b5bf9" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
<img width="1865" height="186" alt="image" src="https://github.com/user-attachments/assets/078d359b-2243-43e2-ac02-78c1b72ba6b0" />

```
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="526" height="351" alt="image" src="https://github.com/user-attachments/assets/fedd7a72-d534-4c5f-a2a5-94851b3c3200" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="762" height="343" alt="image" src="https://github.com/user-attachments/assets/ddb0b401-0f96-40bf-a631-9b26a03d35f1" />

```
pip install --upgrade category_encoders
```
<img width="1184" height="350" alt="image" src="https://github.com/user-attachments/assets/fabacaaa-f251-477a-99c8-6739434fee67" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="682" height="413" alt="image" src="https://github.com/user-attachments/assets/38bff869-3f30-4921-8752-0ceb9454cbbf" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
<img width="760" height="419" alt="image" src="https://github.com/user-attachments/assets/4a70ec94-004f-4d59-b0fe-0f30a5988304" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="754" height="422" alt="image" src="https://github.com/user-attachments/assets/2b9a64b7-1f3f-404c-ad82-112ae26553db" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="873" height="424" alt="image" src="https://github.com/user-attachments/assets/5c0007db-298e-4a0e-8ae5-0a2e3d40bd45" />

```
df.skew()
```
<img width="527" height="181" alt="image" src="https://github.com/user-attachments/assets/e8602c19-b235-487b-83a5-df2eff69080f" />

```
np.log(df["Highly Positive Skew"])
```
<img width="524" height="237" alt="image" src="https://github.com/user-attachments/assets/59f8564d-8f83-4e91-bbc4-00b694ec6b7f" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="564" height="231" alt="image" src="https://github.com/user-attachments/assets/003c8498-a321-4244-8b92-a8d4b25a6da1" />
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="566" height="234" alt="image" src="https://github.com/user-attachments/assets/4f99b0cb-fb75-4ce4-8e92-9c75f8d0990c" />
```
np.square(df["Highly Positive Skew"])
```
<img width="530" height="230" alt="image" src="https://github.com/user-attachments/assets/114c17ef-5fc3-47c4-b2de-2092e5dde68a" />
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="897" height="448" alt="image" src="https://github.com/user-attachments/assets/9714223a-093c-4fa5-aff7-8e9c0ca0ba9c" />
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
<img width="891" height="467" alt="image" src="https://github.com/user-attachments/assets/322d43fc-c3f4-4776-bb1f-6c2efeb53faa" />
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="652" height="450" alt="image" src="https://github.com/user-attachments/assets/314223ea-3ee9-4f80-8f00-ef1f49522a3b" />
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
<img width="688" height="464" alt="image" src="https://github.com/user-attachments/assets/b997c000-7f92-4250-98a1-258fe23b3d2b" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="891" height="459" alt="image" src="https://github.com/user-attachments/assets/66a4e3f1-dacd-4cb5-ab1c-7ea4aa561c91" />
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="669" height="448" alt="image" src="https://github.com/user-attachments/assets/cc56e3df-fc27-4ae2-aedd-19ba28b8e172" />
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="621" height="449" alt="image" src="https://github.com/user-attachments/assets/205f35d6-e641-4692-b2c3-39f00f91de07" />
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="656" height="457" alt="image" src="https://github.com/user-attachments/assets/f8933aac-2e8d-4ec0-8735-cdf127b58af5" />


       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
    Thus the given data,feature encoding,transformation process and save the data to a file was performed successfully.
       

       
