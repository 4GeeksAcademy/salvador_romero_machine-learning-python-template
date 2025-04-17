from utils import db_connect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler

engine = db_connect()

# your code here
with open("data/raw/.gitkeep","r") as file:
    data_url=file.read().strip()

data = pd.read_csv(data_url)


atyp_col = ["price","minimum_nights","number_of_reviews","calculated_host_listings_count"]
new_data = data.drop("last_review",axis=1)
new_data = new_data.drop("host_name",axis=1)
new_data = new_data.drop("name",axis=1)

for i in atyp_col:
    q1=new_data[i].quantile(0.25)
    q3=new_data[i].quantile(0.75)
    iqr = q3-q1
    low_lim = q1 - 1.5*iqr
    hi_lim = q3 + 1.5*iqr

    rem = new_data[(new_data[i]>=hi_lim) | (new_data[i]< low_lim)]
    new_data = new_data.drop(index=rem.index)

# Facotrizar variables categóricas
new_data["room_type"] = pd.factorize(new_data["room_type"])[0]
new_data["neighbourhood_group"] = pd.factorize(new_data["neighbourhood_group"])[0]
new_data["neighbourhood"] = pd.factorize(new_data["neighbourhood"])[0]
new_data = new_data.dropna()


num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(new_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = new_data.index, columns = num_variables)
df_scal["price"] = new_data["price"]

# Dividir datos
x = df_scal.drop("price",axis=1)
y = df_scal["price"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

# Normalizar
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train_norm = scaler.transform(x_train)
# x_test_norm = scaler.transform(x_test)

sel_model = SelectKBest(chi2,k=4)
sel_model.fit(x_train,y_train)
ix = sel_model.get_support()
x_train_sel = pd.DataFrame(sel_model.transform(x_train),columns= x_train.columns.values[ix])
x_test_sel = pd.DataFrame(sel_model.transform(x_test),columns= x_test.columns.values[ix])
df_scal.to_csv("data/processed/filtered_data.csv", index=False)
x_train_sel.to_csv("data/processed/x_train.csv", index=False)
x_test_sel.to_csv("data/processed/x_test.csv", index=False)