import pandas as pd

file_path = "C:/Users/print/Downloads/chipotle.tsv"
chipo = pd.read_csv(file_path, sep = '\t')

print(chipo.shape)
print("------------------------------------")
print(chipo.info())

chipo.head(10)

print(chipo.columns)
print("------------------------------------")
print(chipo.index)



chipo['order_id'] = chipo['order_id'].astype(str)

print(chipo.describe())

print(len(chipo['order_id'].unique()))
print(len(chipo['item_name'].unique()))



chipo['item_name'].value_counts().index.tolist()[0]



order_count = chipo.groupby('item_name')['order_id'].count()
order_count[:10]

item_quantity = chipo.groupby('item_name')['quantity'].sum()
item_quantity[:10]



# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align='center')
plt.ylabel('ordered_item_count')
plt.title('Distribution of all orderd item')

plt.show()



print(chipo.info())
print('-------------')
chipo['item_price'].head()

chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))
chipo.describe()

chipo['item_price'].head()



chipo.groupby('order_id')['item_price'].sum().mean()

chipo.groupby('order_id')['item_price'].sum().describe()[:10]



chipo_orderid_group = chipo.groupby('order_id').sum()
results = chipo_orderid_group[chipo_orderid_group.item_price >= 10]

print(results[:10])
print(results.index.values)



chipo_one_item = chipo[chipo.quantity == 1]
price_per_item = chipo_one_item.groupby('item_name').min()
price_per_item.sort_values(by = "item_price", ascending = False)[:10]

item_name_list = price_per_item.index.tolist()
x_pos = np.arange(len(item_name_list))
item_price = price_per_item['item_price'].tolist()

plt.bar(x_pos, item_price, align='center')
plt.ylabel('item price($)')
plt.title('Distribution of item price')

plt.show()

plt.hist(item_price)
plt.ylabel('counts')
plt.title('Histogram of item price')

plt.show()

chipo.groupby('order_id').sum().sort_values(by='item_price', ascending=False)[:5]

chipo_salad = chipo[chipo['item_name'] == "Veggie Salad Bowl"]
chipo_salad = chipo_salad.drop_duplicates(['item_name', 'order_id'])

print(len(chipo_salad))
chipo_salad.head(5)



chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken_result = chipo_chicken[chipo_chicken['quantity'] >= 2]
print(chipo_chicken_result.shape[0])

chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken_ordersum = chipo_chicken.groupby('order_id').sum()['quantity']
chipo_chicken_result = chipo_chicken_ordersum[chipo_chicken_ordersum >= 2]

print(len(chipo_chicken_result))
chipo_chicken_result.head(5)



