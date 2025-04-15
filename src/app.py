from utils import db_connect
import pandas as pd
from sklearn.model_selection import train_test_split
engine = db_connect()

# your code here
with open("data/raw/.gitkeep","r") as file:
    data_url=file.read().strip()

data = pd.read_csv(data_url)

print(data.info())
cols = ["neighbourhood_group","room_type","price","number_of_reviews","availability_365"]
datax = data[cols].copy()

atyp_col = ["price","number_of_reviews"]
new_data = datax
for i in atyp_col:
    q1=new_data[i].quantile(0.25)
    q3=new_data[i].quantile(0.75)
    iqr = q3-q1
    low_lim = q1 - 1.5*iqr
    hi_lim = q3 + 1.5*iqr

    rem = new_data[(new_data[i]>=hi_lim) | (new_data[i]< low_lim)]
    new_data = new_data.drop(index=rem.index)
x_train, x_test = train_test_split(new_data, test_size = 0.2, random_state = 1)
new_data.to_csv("data/processed/filtered_data.csv", index=False)
new_data.to_csv("data/processed/x_train.csv", index=False)
new_data.to_csv("data/processed/x_test.csv", index=False)