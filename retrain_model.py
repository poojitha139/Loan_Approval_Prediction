import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle as pk

# Load and preprocess data
data = pd.read_csv('loan_approval_dataset.csv')
data.drop(columns=['loan_id'], inplace=True)
data.columns = data.columns.str.strip()
data['Assets'] = (data.residential_assets_value + data.commercial_assets_value +
                  data.luxury_assets_value + data.bank_asset_value)
data.drop(columns=['residential_assets_value', 'commercial_assets_value',
                   'luxury_assets_value', 'bank_asset_value'], inplace=True)

def clean_data(st):
    return st.strip()

data.education = data.education.apply(clean_data)
data['education'] = data['education'].replace(['Graduate', 'Not Graduate'], [1, 0])
data.self_employed = data.self_employed.apply(clean_data)
data.self_employed = data.self_employed.replace(['No', 'Yes'], [0, 1])
data.loan_status = data.loan_status.apply(clean_data)
data.loan_status = data.loan_status.replace(['Approved', 'Rejected'], [1, 0])

# Split data
input_data = data.drop(columns=['loan_status'])
output_data = data['loan_status']
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

# Scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Save model and scaler
pk.dump(model, open('model.pkl', 'wb'))
pk.dump(scaler, open('scaler.pkl', 'wb'))

print("Model retrained and saved successfully.")
