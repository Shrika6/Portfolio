import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# Load and preprocess the dataset
data = pd.read_csv('PS_20174392719_1491204439457_log.csv')

data = pd.get_dummies(data, columns=['type'])
data = data.drop(['nameOrig', 'nameDest'], axis=1)

scaler = StandardScaler()
data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = scaler.fit_transform(
    data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
)

X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = data['isFraud']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train the model
dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'auc'
}
model = xgb.train(params, dtrain, num_boost_round=200)

# Save the model
joblib.dump(model, 'xgboost_model.pkl')
