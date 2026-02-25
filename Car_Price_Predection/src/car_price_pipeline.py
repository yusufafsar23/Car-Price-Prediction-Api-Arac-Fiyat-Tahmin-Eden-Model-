import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("C:/Users/90507/Desktop/Car_Price_Predection/data/car_price_prediction.csv")
print(data.head()) 
print(data.isnull().sum())


data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_clean = data.dropna(how="all")
print(df_clean.isnull().sum())
df_clean = df_clean.drop(["Car ID", "Model", "Engine Size"], axis=1)

X=df_clean.drop("Price",axis=1)
y=df_clean["Price"]

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42)

num_columns=X.select_dtypes(include=["int64","float64"]).columns
cat_columns=X.select_dtypes(include=["object"]).columns

numeratic_transform=Pipeline(steps=[
    ('scaler',StandardScaler())
])
categorical_transform=Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor=ColumnTransformer(
    transformers=[
        ("num",numeratic_transform,num_columns),
        ("cat",categorical_transform,cat_columns)
    ]
)

models={
    "Linear Regression":LinearRegression(),
    "Decision Tree":DecisionTreeRegressor(random_state=42),
    "Random Forest":RandomForestRegressor(random_state=42,n_estimators=100),
    "Gradient Boosting":GradientBoostingRegressor(random_state=42,n_estimators=100)

}
results={}
for name,model in models.items():
    pipeline=Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('regressor',model)
    ])
    pipeline.fit(X_train,y_train)
    tahmin=pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, tahmin))
    r2 = r2_score(y_test, tahmin)
    results[name] = {"RMSE": rmse, "R2": r2}

results_df = pd.DataFrame(results).T
print(results_df)

model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',GradientBoostingRegressor(random_state=42))
])
param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_split': [2, 5, 10],
    'model__subsample': [0.7, 0.85, 1.0]
}

grid=GridSearchCV(model,param_grid,cv=5,scoring='neg_root_mean_squared_error',n_jobs=-1,verbose=2)
grid.fit(X_train,y_train)
print("En iyi parametreler:", grid.best_params_)
print("En iyi CV RMSE:", grid.best_score_)

tahmin = grid.predict(X_test) 

rmse = np.sqrt(mean_squared_error(y_test, tahmin))
r2 = r2_score(y_test, tahmin)

print("RMSE:" ,rmse)
print("R²:" ,r2)

import joblib
joblib.dump(grid.best_estimator_, "car_prediction_model.pkl")