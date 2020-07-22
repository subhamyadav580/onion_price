import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Onion_2020.csv')

column_values = df[["district"]].values
unique_values =  np.unique(column_values)
print('List of all the districts : ')
list_of_districts = list(unique_values)


for no,val in zip(range(1,len(list_of_districts)+1),list_of_districts):
    print('{}.{}'.format(no,val))

def dist(unique_values):
    district = str(input('Enter the district : '))
    if district not in unique_values:
        print('You have enterd wrong district!!! ')
        print('Enter the disrict again : ')
        dist(unique_values)
    else:
        return district


district = dist(unique_values)
# district = str(input('Enter the district : '))

rslt_df = df.loc[df['district'] == district]
days = [num for num in range(0,len(rslt_df))]


rslt_df['days'] = days
x_val = rslt_df[['days']]
y_val = rslt_df[['max_price']]

val = int(input('How many days you want to predict : '))
val_list = [x for x in range(val,(2*val))]
val_list = np.array(val_list)

val_list = val_list.reshape(-1, 1)

mlr = LinearRegression()

mlr.fit(x_val, y_val)
y_predict = mlr.predict(val_list)

val_list = range(1, len(val_list)+1)

for days, price in zip(val_list, y_predict):
    print('The price of {} days is {}'.format(days,price[0]))

plt.subplot(1,2,1)
plt.plot(x_val, y_val)
plt.title('Original Price Plot')
plt.xlabel('Days')
plt.ylabel('Price')


plt.subplot(1,2,2)
plt.plot(val_list, y_predict)
plt.title('Predicted Price Plot')
plt.xlabel('Days')
plt.ylabel('Price')


plt.subplots_adjust(wspace=0.5)


plt.show()













        