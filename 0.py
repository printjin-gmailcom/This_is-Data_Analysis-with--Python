import pandas as pd

names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
custom = [1, 5, 25, 13, 23232]

BabyDataSet = list(zip(names,births))
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

df.head()

print(df.dtypes)
print("-----------")

print(df.index)
print("-----------")

print(df.columns)

df['Names']

df[0:3]

df[df['Births'] > 100]



import numpy as np

arr1 = np.arange(15).reshape(3, 5)
print(arr1)

arr1.shape

arr1.dtype

arr2 = np.array([6, 7, 8])
print(arr2)

arr3 = np.zeros((3,4))
print(arr3)

arr4 = np.array([
    [1,2,3],
    [4,5,6]
], dtype = np.float64)

arr5 = np.array([
    [7,8,9],
    [10,11,12]
], dtype = np.float64)

print("arr4 + arr5 = ")
print(arr4 + arr5,"\n")
print("arr4 - arr5 = ")
print(arr4 - arr5,"\n")
print("arr4 * arr5 = ")
print(arr4 * arr5,"\n")
print("arr4 / arr5 = ")
print(arr4 / arr5,"\n")



# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

y = df['Births']
x = df['Names']

plt.bar(x, y)
plt.xlabel('Names')
plt.ylabel('Births')
plt.title('Bar plot')
plt.show()

np.random.seed(19920613)

x = np.arange(0.0, 100.0, 5.0)
y = (x * 1.5) + np.random.rand(20) * 50

plt.scatter(x, y, c="b", alpha=0.5, label="scatter point")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left')
plt.title('Scatter plot')
plt.show()

