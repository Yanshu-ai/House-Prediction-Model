# Step 1: Import library
import pandas as pd

# Step 2: Read dataset
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv')
df.head()
df.info()
df.describe()
df.columns

# Step 3: Define your problem (y and X)
y = df['MEDV']
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
# Step 4: Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2529)

# Step 5: Select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6: Train model
model.fit(X_train,y_train)
model.intercept_
model.coef_

# Step 7: Make prediction
y_pred = model.predict(X_test)
y_pred

# Step 8: Evaluate
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)