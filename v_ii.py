# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("C:/Users/print/Downloads/online_retail.csv", dtype={'CustomerID': str,'InvoiceID': str}, encoding="ISO-8859-1")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format="%m/%d/%Y %H:%M")

print(df.info())
df.head()



df.isnull().sum()

df = df.dropna()
print(df.shape)



print(df[df['Quantity']<=0].shape[0])
df = df[df['Quantity']>0]

print(df[df['UnitPrice']<=0].shape[0])
df = df[df['UnitPrice']>0]

df['ContainDigit'] = df['StockCode'].apply(lambda x: any(c.isdigit() for c in x))
print(df[df['ContainDigit'] == False].shape[0])
df[df['ContainDigit'] == False].head()

df = df[df['ContainDigit'] == True]



df['date'] = df['InvoiceDate'].dt.date
print(df['date'].min())
print(df['date'].max())

date_quantity_series = df.groupby('date')['Quantity'].sum()
date_quantity_series.plot()

date_transaction_series = df.groupby('date')['InvoiceNo'].nunique()
date_transaction_series.plot()

date_unique_item_series = df.groupby('date')['StockCode'].nunique()
date_unique_item_series.plot()



print(len(df['CustomerID'].unique()))

customer_unique_transaction_series = df.groupby('CustomerID')['InvoiceNo'].nunique()
customer_unique_transaction_series.describe()

plt.boxplot(customer_unique_transaction_series.values)
plt.show()

customer_unique_item_series = df.groupby('CustomerID')['StockCode'].nunique()
customer_unique_item_series.describe()

plt.boxplot(customer_unique_item_series.values)
plt.show()



print(len(df['StockCode'].unique()))



df.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False)[:10]



print(df.groupby('StockCode')['Quantity'].sum().describe())
plt.plot(df.groupby('StockCode')['Quantity'].sum().values)
plt.show()

plt.plot(df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).values)
plt.show()



df['amount'] = df['Quantity'] * df['UnitPrice']
df.groupby('InvoiceNo')['amount'].sum().describe()

plt.plot(df.groupby('InvoiceNo')['amount'].sum().values)
plt.show()

plt.plot(df.groupby('InvoiceNo')['amount'].sum().sort_values(ascending=False).values)
plt.show()



import datetime

df_year_round = df[df['date'] < datetime.date(2011, 11, 1)]
df_year_end = df[df['date'] >= datetime.date(2011, 11, 1)]
print(df_year_round.shape)
print(df_year_end.shape)



customer_item_round_set = df_year_round.groupby('CustomerID')['StockCode'].apply(set)
print(customer_item_round_set)

customer_item_dict = {}

for customer_id, stocks in customer_item_round_set.items():
    customer_item_dict[customer_id] = {}
    for stock_code in stocks:
        customer_item_dict[customer_id][stock_code] = 'old'

print(str(customer_item_dict)[:100] + "...")

customer_item_end_set = df_year_end.groupby('CustomerID')['StockCode'].apply(set)
print(customer_item_end_set)

for customer_id, stocks in customer_item_end_set.items():
    if customer_id in customer_item_dict:
        for stock_code in stocks:
            if stock_code in customer_item_dict[customer_id]:
                customer_item_dict[customer_id][stock_code] = 'both'
            else:
                customer_item_dict[customer_id][stock_code] = 'new'

    else:
        customer_item_dict[customer_id] = {}
        for stock_code in stocks:
            customer_item_dict[customer_id][stock_code] = 'new'

print(str(customer_item_dict)[:100] + "...")

columns = ['CustomerID', 'old', 'new', 'both']
df_order_info = pd.DataFrame(columns=columns)

for customer_id in customer_item_dict:
    old = 0
    new = 0
    both = 0

    for stock_code in customer_item_dict[customer_id]:
        status = customer_item_dict[customer_id][stock_code]
        if status == 'old':
            old += 1
        elif status == 'new':
            new += 1
        else:
            both += 1

    row = [customer_id, old, new, both]
    series = pd.Series(row, index=columns)
    df_series = series.to_frame().T
    df_order_info = pd.concat([df_order_info, df_series], ignore_index=True)

df_order_info.head()

print(df_order_info.shape[0])

print(df_order_info[(df_order_info['old'] > 0) & (df_order_info['new'] > 0)].shape[0])

print(df_order_info[df_order_info['both'] > 0].shape[0])

df_order_info['new'].value_counts()

print(df_order_info['new'].value_counts()[1:].describe())



print(len(df_year_round['CustomerID'].unique()))
print(len(df_year_round['StockCode'].unique()))



uir_df = df_year_round.groupby(['CustomerID', 'StockCode'])['InvoiceNo'].nunique().reset_index()
uir_df.head()

uir_df['InvoiceNo'].hist(bins=20, grid=False)

uir_df['InvoiceNo'].apply(lambda x: np.log10(x)+1).hist(bins=20, grid=False)

uir_df['Rating'] = uir_df['InvoiceNo'].apply(lambda x: np.log10(x)+1)
uir_df['Rating'] = ((uir_df['Rating'] - uir_df['Rating'].min()) / (uir_df['Rating'].max() - uir_df['Rating'].min()) * 4) + 1
uir_df['Rating'].hist(bins=20, grid=False)



uir_df = uir_df[['CustomerID', 'StockCode', 'Rating']]
uir_df.head()

import time
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(uir_df[['CustomerID', 'StockCode', 'Rating']], reader)
train_data, test_data = train_test_split(data, test_size=0.2)

train_start = time.time()
model = SVD(n_factors=8, lr_all=0.005, reg_all=0.02, n_epochs=200)
model.fit(train_data)
train_end = time.time()
print("training time of model: %.2f seconds" % (train_end - train_start))

predictions = model.test(test_data)

print("RMSE of test dataset in SVD model:")
accuracy.rmse(predictions)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(uir_df[['CustomerID', 'StockCode', 'Rating']], reader)
train_data = data.build_full_trainset()

train_start = time.time()
model = SVD(n_factors=8, lr_all=0.005, reg_all=0.02, n_epochs=200)
model.fit(train_data)
train_end = time.time()
print("training time of model: %.2f seconds" % (train_end - train_start))



test_data = train_data.build_anti_testset()
target_user_predictions = model.test(test_data)

new_order_prediction_dict = {}
for customer_id, stock_code, _, predicted_rating, _ in target_user_predictions:
    if customer_id in new_order_prediction_dict:
        if stock_code in new_order_prediction_dict[customer_id]:
            pass
        else:
            new_order_prediction_dict[customer_id][stock_code] = predicted_rating
    else:
        new_order_prediction_dict[customer_id] = {}
        new_order_prediction_dict[customer_id][stock_code] = predicted_rating

print(str(new_order_prediction_dict)[:300] + "...")

test_data = train_data.build_testset()
target_user_predictions = model.test(test_data)

reorder_prediction_dict = {}
for customer_id, stock_code, _, predicted_rating, _ in target_user_predictions:
    if customer_id in reorder_prediction_dict:
        if stock_code in reorder_prediction_dict[customer_id]:
            pass
        else:
            reorder_prediction_dict[customer_id][stock_code] = predicted_rating
    else:
        reorder_prediction_dict[customer_id] = {}
        reorder_prediction_dict[customer_id][stock_code] = predicted_rating

print(str(reorder_prediction_dict)[:300] + "...")

total_prediction_dict = {}

for customer_id in new_order_prediction_dict:
    if customer_id not in total_prediction_dict:
        total_prediction_dict[customer_id] = {}
    for stock_code, predicted_rating in new_order_prediction_dict[customer_id].items():
        if stock_code not in total_prediction_dict[customer_id]:
            total_prediction_dict[customer_id][stock_code] = predicted_rating

for customer_id in reorder_prediction_dict:
    if customer_id not in total_prediction_dict:
        total_prediction_dict[customer_id] = {}
    for stock_code, predicted_rating in reorder_prediction_dict[customer_id].items():
        if stock_code not in total_prediction_dict[customer_id]:
            total_prediction_dict[customer_id][stock_code] = predicted_rating

print(str(total_prediction_dict)[:300] + "...")

simulation_test_df = df_year_end.groupby('CustomerID')['StockCode'].apply(set).reset_index()
simulation_test_df.columns = ['CustomerID', 'RealOrdered']
simulation_test_df.head()

def add_predicted_stock_set(customer_id, prediction_dict):
    if customer_id in prediction_dict:
        predicted_stock_dict = prediction_dict[customer_id]
        sorted_stocks = sorted(predicted_stock_dict, key=lambda x : predicted_stock_dict[x], reverse=True)
        return sorted_stocks
    else:
        return None

simulation_test_df['PredictedOrder(New)'] = simulation_test_df['CustomerID'].apply(lambda x: add_predicted_stock_set(x, new_order_prediction_dict))
simulation_test_df['PredictedOrder(Reorder)'] = simulation_test_df['CustomerID'].apply(lambda x: add_predicted_stock_set(x, reorder_prediction_dict))
simulation_test_df['PredictedOrder(Total)'] = simulation_test_df['CustomerID'].apply(lambda x: add_predicted_stock_set(x, total_prediction_dict))
simulation_test_df.head()



def calculate_recall(real_order, predicted_order, k):
    if predicted_order is None:
        return None

    predicted = predicted_order[:k]
    true_positive = 0
    for stock_code in predicted:
        if stock_code in real_order:
            true_positive += 1

    recall = true_positive / len(predicted)
    return recall

simulation_test_df['top_k_recall(Reorder)'] = simulation_test_df. apply(lambda x: calculate_recall(x['RealOrdered'], x['PredictedOrder(Reorder)'], 5), axis=1)
simulation_test_df['top_k_recall(New)'] = simulation_test_df. apply(lambda x: calculate_recall(x['RealOrdered'], x['PredictedOrder(New)'], 5), axis=1)
simulation_test_df['top_k_recall(Total)'] = simulation_test_df. apply(lambda x: calculate_recall(x['RealOrdered'], x['PredictedOrder(Total)'], 5), axis=1)

print(simulation_test_df['top_k_recall(Reorder)'].mean())
print(simulation_test_df['top_k_recall(New)'].mean())
print(simulation_test_df['top_k_recall(Total)'].mean())

simulation_test_df['top_k_recall(Reorder)'].value_counts()

simulation_test_df['top_k_recall(New)'].value_counts()

simulation_test_df['top_k_recall(Total)'].value_counts()

not_recommended_df = simulation_test_df[simulation_test_df['PredictedOrder(Reorder)'].isnull()]
print(not_recommended_df.shape)
not_recommended_df.head()



k = 5
result_df = simulation_test_df[simulation_test_df['PredictedOrder(Reorder)'].notnull()]
result_df['PredictedOrder(Reorder)'] = result_df['PredictedOrder(Reorder)'].apply(lambda x: x[:k])
result_df = result_df[['CustomerID', 'RealOrdered', 'PredictedOrder(Reorder)', 'top_k_recall(Reorder)']]
result_df.columns = [['구매자ID', '실제주문', '5개추천결과', 'Top5추천_주문재현도']]
result_df.sample(5).head()



