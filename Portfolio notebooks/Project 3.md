# Airbnb Open Data (Data Visualization)

##### This project focuses on exploring and visualizing the Airbnb Open Data (over 100,000 rows) using Python, specifically leveraging Seaborn and Matplotlib libraries. Through a series of visualizations, the project aims to uncover patterns, relationships, and trends within the dataset, highlighting key insights about listings, prices, room types, neighborhoods, and availability.

##### Tasks are:

1. Price Distribution **(Histogram/KDE)**
2. Room Type Distribution **(Pie Chart)**
3. Price vs. Reviews **(Scatter Plot)**
4. Average Price by Room Type **(Bar Chart)**
5. Top Neighborhoods **(Horizontal Bar Chart)**
6. Time Series Trends **(Line Plot)**
7. Correlation Heatmap **(Heatmap)**
8. Room Type by Neighborhood **(Stacked Bar Chart)**
9. Geographic Distribution **(Scatter Plot)**
10. Price Distribution by Neighborhood **(Box Plot)**

##### In section A., this involves purely cleaning data. You may skip this part and proceed to task 1.

##### A. Library Importing, Dataset Importing, and Data Cleaning section.


```python
# Importing Libraries and Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
df = pd.read_csv('Airbnb_Open_Data.csv')
```

    C:\Users\Josh09\AppData\Local\Temp\ipykernel_18620\3671619752.py:7: DtypeWarning: Columns (25) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv('Airbnb_Open_Data.csv')
    


```python
# Check dataset and its summary

pd.set_option('display.max_columns', None)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NAME</th>
      <th>host id</th>
      <th>host_identity_verified</th>
      <th>host name</th>
      <th>neighbourhood group</th>
      <th>neighbourhood</th>
      <th>lat</th>
      <th>long</th>
      <th>country</th>
      <th>country code</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>room type</th>
      <th>Construction year</th>
      <th>price</th>
      <th>service fee</th>
      <th>minimum nights</th>
      <th>number of reviews</th>
      <th>last review</th>
      <th>reviews per month</th>
      <th>review rate number</th>
      <th>calculated host listings count</th>
      <th>availability 365</th>
      <th>house_rules</th>
      <th>license</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001254</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>80014485718</td>
      <td>unconfirmed</td>
      <td>Madaline</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>strict</td>
      <td>Private room</td>
      <td>2020.0</td>
      <td>$966</td>
      <td>$193</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>10/19/2021</td>
      <td>0.21</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>286.0</td>
      <td>Clean up and treat the home the way you'd like...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002102</td>
      <td>Skylit Midtown Castle</td>
      <td>52335172823</td>
      <td>verified</td>
      <td>Jenna</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2007.0</td>
      <td>$142</td>
      <td>$28</td>
      <td>30.0</td>
      <td>45.0</td>
      <td>5/21/2022</td>
      <td>0.38</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>228.0</td>
      <td>Pet friendly but please confirm with me if the...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002403</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>78829239556</td>
      <td>NaN</td>
      <td>Elise</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>flexible</td>
      <td>Private room</td>
      <td>2005.0</td>
      <td>$620</td>
      <td>$124</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>352.0</td>
      <td>I encourage you to use my kitchen, cooking and...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002755</td>
      <td>NaN</td>
      <td>85098326012</td>
      <td>unconfirmed</td>
      <td>Garry</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2005.0</td>
      <td>$368</td>
      <td>$74</td>
      <td>30.0</td>
      <td>270.0</td>
      <td>7/5/2019</td>
      <td>4.64</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>322.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1003689</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>92037596077</td>
      <td>verified</td>
      <td>Lyndon</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2009.0</td>
      <td>$204</td>
      <td>$41</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>11/19/2018</td>
      <td>0.10</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>289.0</td>
      <td>Please no smoking in the house, porch or on th...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data Type check

df.dtypes
```




    id                                  int64
    NAME                               object
    host id                             int64
    host_identity_verified             object
    host name                          object
    neighbourhood group                object
    neighbourhood                      object
    lat                               float64
    long                              float64
    country                            object
    country code                       object
    instant_bookable                   object
    cancellation_policy                object
    room type                          object
    Construction year                 float64
    price                              object
    service fee                        object
    minimum nights                    float64
    number of reviews                 float64
    last review                        object
    reviews per month                 float64
    review rate number                float64
    calculated host listings count    float64
    availability 365                  float64
    house_rules                        object
    license                            object
    dtype: object



Column 'price' and 'service fee' should be float, column 'last review' should be a datetime data type.

- Convert 'price' and 'service fee column to numeric type. These columns are in object type due to '$' sign.

- We will rename these columns to 'price (USD)' and 'service fee (USD)', remove $ in the data, and then convert them to numeric type



```python
# Rename columns 'price' and 'service fee' to 'price (USD)' and 'service fee (USD)'.

df= df.rename(columns={'price': 'price (USD)', 'service fee': 'service fee (USD)'})

# Ensure all values are strings

df['price (USD)'] = df['price (USD)'].astype(str)
df['service fee (USD)'] = df['service fee (USD)'].astype(str)

# Remove '$' and ',' from 'price' and 'service fee' columns

df['price (USD)'] = df['price (USD)'].str.replace('[$,]', '', regex=True)
df['service fee (USD)'] = df['service fee (USD)'].str.replace('[$,]', '', regex=True)

df['price (USD)']

# Convert 'price (USD)' and 'service fee (USD)' columns to numeric values

df['price (USD)'] = pd.to_numeric(df['price (USD)'], errors='coerce')
df['service fee (USD)'] = pd.to_numeric(df['service fee (USD)'], errors='coerce')
```


```python
# Convert 'last review' column to datetime format

df['last review'] = pd.to_datetime(df['last review'], errors='coerce')
```


```python
df.shape
```




    (102599, 26)




```python
# Check for missing values

df.isna().sum()
```




    id                                     0
    NAME                                 250
    host id                                0
    host_identity_verified               289
    host name                            406
    neighbourhood group                   29
    neighbourhood                         16
    lat                                    8
    long                                   8
    country                              532
    country code                         131
    instant_bookable                     105
    cancellation_policy                   76
    room type                              0
    Construction year                    214
    price (USD)                          247
    service fee (USD)                    273
    minimum nights                       409
    number of reviews                    183
    last review                        15893
    reviews per month                  15879
    review rate number                   326
    calculated host listings count       319
    availability 365                     448
    house_rules                        52131
    license                           102597
    dtype: int64



After checking columns by missing values, considerations are as followed:

- Rows with missing values for Name, Host Name, Country, Latitude, Longitude, and Price will be removed because these columns are **mandatory** for an Airbnb listing to be published online. This decision is informed by my expertise in the field and my experience as a long-time Airbnb host.

- Other missing values might be attributed to unimplemented features rather than data errors. We will leave these as they are and address them based on the specific task at hand.

- Column license are almost missing of all data.


```python
# Remove missing data in columns NAME, host Name, country, lat, long, host_identity_verified and price USD.

df.dropna(subset=['NAME', 'host name', 'country', 'lat', 'long', 'host_identity_verified', 'price (USD)'], inplace=True)
```


```python
# Column license are almost all missing values. Let's evaluate it further.

df['license'].unique()
```




    array([nan, '41662/AL'], dtype=object)



After checking column 'license', we can see that there are no useable data in this column. Let's remove this column.


```python
# Remove 'license' column from the dataset

df.drop('license', axis=1, inplace=True)

df.columns

```




    Index(['id', 'NAME', 'host id', 'host_identity_verified', 'host name',
           'neighbourhood group', 'neighbourhood', 'lat', 'long', 'country',
           'country code', 'instant_bookable', 'cancellation_policy', 'room type',
           'Construction year', 'price (USD)', 'service fee (USD)',
           'minimum nights', 'number of reviews', 'last review',
           'reviews per month', 'review rate number',
           'calculated host listings count', 'availability 365', 'house_rules'],
          dtype='object')




```python
df.isna().sum()
```




    id                                    0
    NAME                                  0
    host id                               0
    host_identity_verified                0
    host name                             0
    neighbourhood group                  17
    neighbourhood                        14
    lat                                   0
    long                                  0
    country                               0
    country code                         36
    instant_bookable                     22
    cancellation_policy                   0
    room type                             0
    Construction year                   179
    price (USD)                           0
    service fee (USD)                   238
    minimum nights                      394
    number of reviews                   182
    last review                       15669
    reviews per month                 15656
    review rate number                  302
    calculated host listings count      310
    availability 365                    410
    house_rules                       51281
    dtype: int64




```python
# Check for duplicates

df.duplicated().sum()
```




    539




```python
# remove duplicates

df.drop_duplicates(inplace=True)
```


```python
df.duplicated().sum()
```




    0



### 1. Analyze the Distribution of Price (Histogram/KDE)

##### We will create a histogram to visualize the distribution of prices of Airbnb listings using Seaborn.


```python
hist = sns.histplot(df['price (USD)'], bins=50, color='red', kde=True)
plt.title('Distribution of Airbnb Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
counts = hist.patches 
bin_height = [patch.get_height() for patch in counts]
min_height = min(bin_height) - 50
max_height = max(bin_height) + 50
plt.ylim(min_height, max_height) # Set the lowest and highest bar (bins) to lowest and highest part of the histogram.
plt.show()


```


    
![png](Project%203_files/Project%203_20_0.png)
    


The histogram for Price (USD) appears to have a multimodal distribution. This indicates that Prices (USD) are clustered and have subgroups.

Task 1, done.

### 2. Room Type Distribution (Pie Chart)

##### We will use a pie chart to show the percentage distribution of room types in the dataset for visualization.


```python
# Create dataframe for this task

pie = df['room type'].value_counts().reset_index()
pie.columns = ['room type', 'count']
pie
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>room type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Entire home/apt</td>
      <td>52572</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private room</td>
      <td>45522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shared room</td>
      <td>2164</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hotel room</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create piechart

plt.figure(figsize=(15,9))
plt.pie(pie['count'], labels=pie['room type'], autopct='%1.1f%%', startangle=180, pctdistance=.8)
plt.title('Room Type Percentage')

plt.show()
```


    
![png](Project%203_files/Project%203_24_0.png)
    


Task 2, done.

### 3. Price vs. Reviews **(Scatter Plot)**

##### Exploring the relationship between column Price and Number of Reviews.


```python
# Due to high number of rows (100,370), I opted to acquire random sample of 1000 in order to avoid over-saturation of plots.

df_sample = df.sample(n=2000) 

# Create a scatter plot. Add column 'room type' for hue.

sns.scatterplot(df_sample, y='number of reviews', x='price (USD)', hue='room type')
plt.title('Price vs. Number of Reviews')
plt.show()
```


    
![png](Project%203_files/Project%203_27_0.png)
    


The scatter plot indicates that the relationship of column 'price' and 'number if reviews' are independent from each other. It shows that changes in 'price' does not significantly impact the column 'number of reviews'.

Task 3, done.

### 4. Average Price by Room Type **(Bar Chart)**

##### Let's create a Bar Chart where it displays the price average by room type.


```python
# Create dataframe where we group all the prices of each room type.

task4 = df.groupby('room type')['price (USD)']\
        .agg(Average_Price_USD =('mean'))

task4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Average_Price_USD</th>
    </tr>
    <tr>
      <th>room type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Entire home/apt</th>
      <td>625.458856</td>
    </tr>
    <tr>
      <th>Hotel room</th>
      <td>661.232143</td>
    </tr>
    <tr>
      <th>Private room</th>
      <td>624.843966</td>
    </tr>
    <tr>
      <th>Shared room</th>
      <td>635.486137</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Make a bar chart.

ax = sns.barplot(task4, x='room type', y='Average_Price_USD', palette='Set2')

# Bar values

for container in ax.containers:
    if len(container) > 0:  # Ensure non-empty containers
        ax.bar_label(container)

plt.title('Average Price Per Room Type')
plt.ylim(500,700)
plt.show()
```

    C:\Users\Josh09\AppData\Local\Temp\ipykernel_18620\712989193.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.barplot(task4, x='room type', y='Average_Price_USD', palette='Set2')
    


    
![png](Project%203_files/Project%203_31_1.png)
    


Hotel room seems to have the highest average price out of the four room types, but also have the lowest booked record. We can hypothesize that guests perception revolves around the interplay of price and space. Further study is needed. 

Task 4, done.

### 5. Top Neighborhoods **(Horizontal Bar Chart)**

##### Let's create a horizontal bar chart to show the top 10 neighbourhoods with the most Airbnb listings.


```python
plt.figure(figsize=(12,7))
ax = sns.barplot(df['neighbourhood'].value_counts().sort_values(ascending=False).head(10), orient='h', palette='Set1')

for container in ax.containers:
    if len(container) > 0: 
        ax.bar_label(container)

plt.title('Top 10 Neighbourhood with the Most Airbnb Listing')
plt.xlabel('Airbnb Listing Count')
plt.ylabel('Neighbourhood')
```

    C:\Users\Josh09\AppData\Local\Temp\ipykernel_18620\2638418034.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.barplot(df['neighbourhood'].value_counts().sort_values(ascending=False).head(10), orient='h', palette='Set1')
    




    Text(0, 0.5, 'Neighbourhood')




    
![png](Project%203_files/Project%203_34_2.png)
    


Task 5, done.

### 6. Time Series Trends **(Line Plot)**

##### Create a line plot showing the trend of listing availability over time using Seaborn's `lineplot`.

##### *Disclaimer: Column 'last review' is the only available datetime data in this dataset, instead of a more stable data such as check-in date, check-out date, or booking date. In reality, this can be a weak variable for the reason that not all that has booked leaves a review. This task is only for the purpose of highlighting visualization capability on Python.*


```python
# check columns last review and availability 365

df[['last review','availability 365']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last review</th>
      <th>availability 365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-10-19</td>
      <td>286.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-21</td>
      <td>228.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-11-19</td>
      <td>289.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-06-22</td>
      <td>374.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-10-05</td>
      <td>219.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102053</th>
      <td>2019-03-27</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>102054</th>
      <td>2017-08-31</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>102055</th>
      <td>2019-06-26</td>
      <td>235.0</td>
    </tr>
    <tr>
      <th>102056</th>
      <td>NaT</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>102057</th>
      <td>2019-06-15</td>
      <td>238.0</td>
    </tr>
  </tbody>
</table>
<p>100370 rows × 2 columns</p>
</div>




```python
df[['last review','availability 365']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last review</th>
      <th>availability 365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>84761</td>
      <td>99960.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2019-06-09 01:25:44.592442368</td>
      <td>141.151601</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2012-07-11 00:00:00</td>
      <td>-10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2018-10-26 00:00:00</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2019-06-13 00:00:00</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019-07-05 00:00:00</td>
      <td>269.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2040-06-16 00:00:00</td>
      <td>3677.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>135.435093</td>
    </tr>
  </tbody>
</table>
</div>



Several issues has been identified. The maximum value of 'last review' has a year 2040; minimum and maximum values of 'availability 365' are beyond day 1 and 365 of a year. This is illogical in the context of this task and should be corrected.


```python
# Remove all missing data, and data in column 'availability 365'.

task6 = df[['last review','availability 365']].dropna()
task6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last review</th>
      <th>availability 365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-10-19</td>
      <td>286.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-21</td>
      <td>228.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-11-19</td>
      <td>289.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-06-22</td>
      <td>374.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-10-05</td>
      <td>219.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102052</th>
      <td>2019-07-01</td>
      <td>323.0</td>
    </tr>
    <tr>
      <th>102053</th>
      <td>2019-03-27</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>102054</th>
      <td>2017-08-31</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>102055</th>
      <td>2019-06-26</td>
      <td>235.0</td>
    </tr>
    <tr>
      <th>102057</th>
      <td>2019-06-15</td>
      <td>238.0</td>
    </tr>
  </tbody>
</table>
<p>84615 rows × 2 columns</p>
</div>




```python
# Keep only between 1 to 365 values of 'availability 365' column. 

task6 = task6[(task6['availability 365'] >= 1) & (task6['availability 365'] <= 365) & (task6['last review'] != '2040-06-16')]
task6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last review</th>
      <th>availability 365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-10-19</td>
      <td>286.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-21</td>
      <td>228.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-11-19</td>
      <td>289.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-10-05</td>
      <td>219.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019-06-24</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102039</th>
      <td>2019-05-23</td>
      <td>147.0</td>
    </tr>
    <tr>
      <th>102040</th>
      <td>2019-06-29</td>
      <td>361.0</td>
    </tr>
    <tr>
      <th>102052</th>
      <td>2019-07-01</td>
      <td>323.0</td>
    </tr>
    <tr>
      <th>102055</th>
      <td>2019-06-26</td>
      <td>235.0</td>
    </tr>
    <tr>
      <th>102057</th>
      <td>2019-06-15</td>
      <td>238.0</td>
    </tr>
  </tbody>
</table>
<p>64121 rows × 2 columns</p>
</div>




```python
# Group by 'last review' and 'availability 365'.

task6 = task6.groupby('last review')['availability 365'].sum().reset_index()
```


```python
# Line plot for availability over time

plt.figure(figsize=(14,6))
sns.lineplot(task6, x='last review', y='availability 365', color='orange')
plt.title('Availability of Listings Over Time')
plt.xlabel('Date of Last Review')
plt.ylabel('Availability (Days per Year)')
plt.xlim(pd.Timestamp('2015-01-01'),task6['last review'].max())
plt.show()
```


    
![png](Project%203_files/Project%203_43_0.png)
    


Airbnb bookings seasonality is apparent around year end, while significant noise can be seen in the middle of 2019 and 2022. One obvious hypothesis may be due to the world wide pandemic events, such as the start and end of travel restrictions. Deeper insights in regards to it is subject for further analysis.

Task 6, done.

### 7. Correlation Heatmap **(Heatmap)**

##### We will create a heatmap to visualize the correlation between numeric variables in the dataset using Seaborn's `heatmap`.


```python
# Make a dataframe that are all numeric columns in this data set.

df_corr = df.select_dtypes(include='number')

# Remove columns host id and id. Though numerical, values in these fields are identifiers, not for measurement.

df_corr = df_corr.drop(columns=['id','host id'])

# Set up the new dataframe for heatmap.

df_corr = df_corr.dropna().corr()


df_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>long</th>
      <th>Construction year</th>
      <th>price (USD)</th>
      <th>service fee (USD)</th>
      <th>minimum nights</th>
      <th>number of reviews</th>
      <th>reviews per month</th>
      <th>review rate number</th>
      <th>calculated host listings count</th>
      <th>availability 365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lat</th>
      <td>1.000000</td>
      <td>0.073869</td>
      <td>0.007325</td>
      <td>-0.005777</td>
      <td>-0.005742</td>
      <td>0.017600</td>
      <td>-0.021724</td>
      <td>-0.020218</td>
      <td>-0.004315</td>
      <td>0.026308</td>
      <td>-0.013321</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.073869</td>
      <td>1.000000</td>
      <td>0.000317</td>
      <td>0.002333</td>
      <td>0.002321</td>
      <td>-0.033351</td>
      <td>0.065962</td>
      <td>0.119517</td>
      <td>0.015078</td>
      <td>-0.092338</td>
      <td>0.073670</td>
    </tr>
    <tr>
      <th>Construction year</th>
      <td>0.007325</td>
      <td>0.000317</td>
      <td>1.000000</td>
      <td>-0.003777</td>
      <td>-0.003767</td>
      <td>0.002375</td>
      <td>0.003627</td>
      <td>0.005432</td>
      <td>0.005610</td>
      <td>-0.002808</td>
      <td>-0.007513</td>
    </tr>
    <tr>
      <th>price (USD)</th>
      <td>-0.005777</td>
      <td>0.002333</td>
      <td>-0.003777</td>
      <td>1.000000</td>
      <td>0.999991</td>
      <td>-0.004300</td>
      <td>0.004417</td>
      <td>0.004266</td>
      <td>-0.007416</td>
      <td>-0.000328</td>
      <td>-0.000709</td>
    </tr>
    <tr>
      <th>service fee (USD)</th>
      <td>-0.005742</td>
      <td>0.002321</td>
      <td>-0.003767</td>
      <td>0.999991</td>
      <td>1.000000</td>
      <td>-0.004274</td>
      <td>0.004386</td>
      <td>0.004239</td>
      <td>-0.007430</td>
      <td>-0.000324</td>
      <td>-0.000694</td>
    </tr>
    <tr>
      <th>minimum nights</th>
      <td>0.017600</td>
      <td>-0.033351</td>
      <td>0.002375</td>
      <td>-0.004300</td>
      <td>-0.004274</td>
      <td>1.000000</td>
      <td>-0.047807</td>
      <td>-0.094703</td>
      <td>0.000467</td>
      <td>0.067328</td>
      <td>0.043162</td>
    </tr>
    <tr>
      <th>number of reviews</th>
      <td>-0.021724</td>
      <td>0.065962</td>
      <td>0.003627</td>
      <td>0.004417</td>
      <td>0.004386</td>
      <td>-0.047807</td>
      <td>1.000000</td>
      <td>0.594850</td>
      <td>-0.019378</td>
      <td>-0.079422</td>
      <td>0.108253</td>
    </tr>
    <tr>
      <th>reviews per month</th>
      <td>-0.020218</td>
      <td>0.119517</td>
      <td>0.005432</td>
      <td>0.004266</td>
      <td>0.004239</td>
      <td>-0.094703</td>
      <td>0.594850</td>
      <td>1.000000</td>
      <td>0.038269</td>
      <td>-0.024297</td>
      <td>0.079101</td>
    </tr>
    <tr>
      <th>review rate number</th>
      <td>-0.004315</td>
      <td>0.015078</td>
      <td>0.005610</td>
      <td>-0.007416</td>
      <td>-0.007430</td>
      <td>0.000467</td>
      <td>-0.019378</td>
      <td>0.038269</td>
      <td>1.000000</td>
      <td>0.023726</td>
      <td>-0.011840</td>
    </tr>
    <tr>
      <th>calculated host listings count</th>
      <td>0.026308</td>
      <td>-0.092338</td>
      <td>-0.002808</td>
      <td>-0.000328</td>
      <td>-0.000324</td>
      <td>0.067328</td>
      <td>-0.079422</td>
      <td>-0.024297</td>
      <td>0.023726</td>
      <td>1.000000</td>
      <td>0.135333</td>
    </tr>
    <tr>
      <th>availability 365</th>
      <td>-0.013321</td>
      <td>0.073670</td>
      <td>-0.007513</td>
      <td>-0.000709</td>
      <td>-0.000694</td>
      <td>0.043162</td>
      <td>0.108253</td>
      <td>0.079101</td>
      <td>-0.011840</td>
      <td>0.135333</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make a heatmap out from df_corr.

plt.figure(figsize=(14,6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', vmin=-1)
plt.title('Correlation Heatmap', fontsize=17)
plt.xticks(fontsize=14,rotation=60)
plt.yticks(fontsize=14)
plt.show()
```


    
![png](Project%203_files/Project%203_47_0.png)
    


Columns that has strong correlation are 'service fee (USD)' and 'price (USD)'; 'number of reviews' and 'reviews per month' has moderate correlation.

Task 7, done.

### 8. Room Type by Neighborhood **(Countplot)**

##### Let's create a stacked bar chart showing the number of listings per room type for the top 5 neighbourhoods using Seaborn's `countplot` and hue for stacking.



```python
# Obtain top 5 Neighbourhood and create new dataframe.

top5 = df['neighbourhood'].value_counts().head(5).index
df_top5 = df[df['neighbourhood'].isin(top5)].reset_index()
df_top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>id</th>
      <th>NAME</th>
      <th>host id</th>
      <th>host_identity_verified</th>
      <th>host name</th>
      <th>neighbourhood group</th>
      <th>neighbourhood</th>
      <th>lat</th>
      <th>long</th>
      <th>country</th>
      <th>country code</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>room type</th>
      <th>Construction year</th>
      <th>price (USD)</th>
      <th>service fee (USD)</th>
      <th>minimum nights</th>
      <th>number of reviews</th>
      <th>last review</th>
      <th>reviews per month</th>
      <th>review rate number</th>
      <th>calculated host listings count</th>
      <th>availability 365</th>
      <th>house_rules</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>1005202</td>
      <td>BlissArtsSpace!</td>
      <td>90821839709</td>
      <td>unconfirmed</td>
      <td>Emma</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>40.68688</td>
      <td>-73.95596</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>moderate</td>
      <td>Private room</td>
      <td>2009.0</td>
      <td>1060.0</td>
      <td>212.0</td>
      <td>45.0</td>
      <td>49.0</td>
      <td>2017-10-05</td>
      <td>0.40</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>219.0</td>
      <td>House Guidelines for our BnB We are delighted ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>1005754</td>
      <td>Large Furnished Room Near B'way</td>
      <td>79384379533</td>
      <td>verified</td>
      <td>Evelyn</td>
      <td>Manhattan</td>
      <td>Hell's Kitchen</td>
      <td>40.76489</td>
      <td>-73.98493</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>strict</td>
      <td>Private room</td>
      <td>2005.0</td>
      <td>1018.0</td>
      <td>204.0</td>
      <td>2.0</td>
      <td>430.0</td>
      <td>2019-06-24</td>
      <td>3.47</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>180.0</td>
      <td>- Please clean up after yourself when using th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>1010173</td>
      <td>Only 2 stops to Manhattan studio</td>
      <td>62566345680</td>
      <td>unconfirmed</td>
      <td>Heather</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.70837</td>
      <td>-73.95352</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2009.0</td>
      <td>778.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>148.0</td>
      <td>2019-06-29</td>
      <td>1.20</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>197.0</td>
      <td>Absolutely no smoking in the building, handlin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1012934</td>
      <td>Sweet and Spacious Brooklyn Loft</td>
      <td>86554611512</td>
      <td>verified</td>
      <td>Alissa</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.71842</td>
      <td>-73.95718</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>flexible</td>
      <td>Entire home/apt</td>
      <td>2016.0</td>
      <td>477.0</td>
      <td>95.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2021-12-28</td>
      <td>0.07</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>193.0</td>
      <td>- No smoking or open flames on the property - ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>1016800</td>
      <td>Midtown Pied-a-terre</td>
      <td>19382804591</td>
      <td>unconfirmed</td>
      <td>Andrew</td>
      <td>Manhattan</td>
      <td>Hell's Kitchen</td>
      <td>40.76715</td>
      <td>-73.98533</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2016.0</td>
      <td>209.0</td>
      <td>42.0</td>
      <td>10.0</td>
      <td>58.0</td>
      <td>2017-08-13</td>
      <td>0.49</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>103.0</td>
      <td>Please no pets or smoking in the house, though...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29486</th>
      <td>102038</td>
      <td>57356923</td>
      <td>HUGE BEDROOM LORIMER L TRAIN!!!</td>
      <td>29320426760</td>
      <td>unconfirmed</td>
      <td>Jose</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.71355</td>
      <td>-73.95003</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>flexible</td>
      <td>Private room</td>
      <td>2016.0</td>
      <td>570.0</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>17.0</td>
      <td>2019-04-30</td>
      <td>0.61</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>229.0</td>
      <td>No smoking</td>
    </tr>
    <tr>
      <th>29487</th>
      <td>102054</td>
      <td>57365760</td>
      <td>Private Bedroom with Amazing Rooftop View</td>
      <td>45936254757</td>
      <td>verified</td>
      <td>Trey</td>
      <td>Brooklyn</td>
      <td>Bushwick</td>
      <td>40.69872</td>
      <td>-73.92718</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>flexible</td>
      <td>Private room</td>
      <td>NaN</td>
      <td>909.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>2017-08-31</td>
      <td>0.72</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>#NAME?</td>
    </tr>
    <tr>
      <th>29488</th>
      <td>102055</td>
      <td>57366313</td>
      <td>Pretty Brooklyn One-Bedroom for 2 to 4 people</td>
      <td>23801060917</td>
      <td>verified</td>
      <td>Michael</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>40.67810</td>
      <td>-73.90822</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>NaN</td>
      <td>387.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>50.0</td>
      <td>2019-06-26</td>
      <td>3.12</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>235.0</td>
      <td>* Check out: 10am * We made an effort to keep ...</td>
    </tr>
    <tr>
      <th>29489</th>
      <td>102056</td>
      <td>57366865</td>
      <td>Room &amp; private bathroom in historic Harlem</td>
      <td>15593031571</td>
      <td>unconfirmed</td>
      <td>Shireen</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.81248</td>
      <td>-73.94317</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>strict</td>
      <td>Private room</td>
      <td>NaN</td>
      <td>848.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Each of us is working and/or going to school a...</td>
    </tr>
    <tr>
      <th>29490</th>
      <td>102057</td>
      <td>57367417</td>
      <td>Rosalee Stewart</td>
      <td>93578954226</td>
      <td>verified</td>
      <td>Stanley</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.81315</td>
      <td>-73.94747</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>flexible</td>
      <td>Entire home/apt</td>
      <td>2011.0</td>
      <td>1128.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>2019-06-15</td>
      <td>0.85</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>238.0</td>
      <td>Please remember that this is a residential bui...</td>
    </tr>
  </tbody>
</table>
<p>29491 rows × 26 columns</p>
</div>




```python
# Create countplot.

plt.figure(figsize=(10,7))
ax = sns.countplot(df_top5, x='neighbourhood', hue='room type', palette='Set1')
plt.ylabel('Number of Airbnb Listings')
plt.xlabel('Neighbourhood')
plt.title('Room Type Distribution by Neighbourhood')

# Apply bar labels to all bars.

for container in ax.containers:
    ax.bar_label(container)

# Adjust bar distance.

plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

plt.show()
```


    
![png](Project%203_files/Project%203_51_0.png)
    


Task 8, done.

### 9. Geographic Distribution **(Scatter Plot)**

##### Let's create a scatter plot using longitude and latitude to visualize the geographic distribution of Airbnb listings using Seaborn's `scatterplot`.


```python
plt.figure(figsize=(10,6))
sns.scatterplot(df, x='long', y='lat', hue='room type', palette='Set1', s=30)\
    .set_facecolor('darkgrey') # Set background color for a better contrast.
plt.title('Geographic Distribution of Airbnb Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
```


    
![png](Project%203_files/Project%203_54_0.png)
    


Task 9, done.

### 10. Price Distribution by Neighborhood **(Box Plot)**

##### Create a box plot to compare the distribution of prices across the top 5 neighbourhoods using Seaborn's `boxplot`.


```python
# Gather top 5 neighbourhood

task10 = df['neighbourhood'].value_counts().head(5).index

df_top5 = df[df['neighbourhood'].isin(task10)].reset_index()
df_top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>id</th>
      <th>NAME</th>
      <th>host id</th>
      <th>host_identity_verified</th>
      <th>host name</th>
      <th>neighbourhood group</th>
      <th>neighbourhood</th>
      <th>lat</th>
      <th>long</th>
      <th>country</th>
      <th>country code</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>room type</th>
      <th>Construction year</th>
      <th>price (USD)</th>
      <th>service fee (USD)</th>
      <th>minimum nights</th>
      <th>number of reviews</th>
      <th>last review</th>
      <th>reviews per month</th>
      <th>review rate number</th>
      <th>calculated host listings count</th>
      <th>availability 365</th>
      <th>house_rules</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>1005202</td>
      <td>BlissArtsSpace!</td>
      <td>90821839709</td>
      <td>unconfirmed</td>
      <td>Emma</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>40.68688</td>
      <td>-73.95596</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>moderate</td>
      <td>Private room</td>
      <td>2009.0</td>
      <td>1060.0</td>
      <td>212.0</td>
      <td>45.0</td>
      <td>49.0</td>
      <td>2017-10-05</td>
      <td>0.40</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>219.0</td>
      <td>House Guidelines for our BnB We are delighted ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>1005754</td>
      <td>Large Furnished Room Near B'way</td>
      <td>79384379533</td>
      <td>verified</td>
      <td>Evelyn</td>
      <td>Manhattan</td>
      <td>Hell's Kitchen</td>
      <td>40.76489</td>
      <td>-73.98493</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>strict</td>
      <td>Private room</td>
      <td>2005.0</td>
      <td>1018.0</td>
      <td>204.0</td>
      <td>2.0</td>
      <td>430.0</td>
      <td>2019-06-24</td>
      <td>3.47</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>180.0</td>
      <td>- Please clean up after yourself when using th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>1010173</td>
      <td>Only 2 stops to Manhattan studio</td>
      <td>62566345680</td>
      <td>unconfirmed</td>
      <td>Heather</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.70837</td>
      <td>-73.95352</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2009.0</td>
      <td>778.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>148.0</td>
      <td>2019-06-29</td>
      <td>1.20</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>197.0</td>
      <td>Absolutely no smoking in the building, handlin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1012934</td>
      <td>Sweet and Spacious Brooklyn Loft</td>
      <td>86554611512</td>
      <td>verified</td>
      <td>Alissa</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.71842</td>
      <td>-73.95718</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>flexible</td>
      <td>Entire home/apt</td>
      <td>2016.0</td>
      <td>477.0</td>
      <td>95.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2021-12-28</td>
      <td>0.07</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>193.0</td>
      <td>- No smoking or open flames on the property - ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>1016800</td>
      <td>Midtown Pied-a-terre</td>
      <td>19382804591</td>
      <td>unconfirmed</td>
      <td>Andrew</td>
      <td>Manhattan</td>
      <td>Hell's Kitchen</td>
      <td>40.76715</td>
      <td>-73.98533</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>2016.0</td>
      <td>209.0</td>
      <td>42.0</td>
      <td>10.0</td>
      <td>58.0</td>
      <td>2017-08-13</td>
      <td>0.49</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>103.0</td>
      <td>Please no pets or smoking in the house, though...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29486</th>
      <td>102038</td>
      <td>57356923</td>
      <td>HUGE BEDROOM LORIMER L TRAIN!!!</td>
      <td>29320426760</td>
      <td>unconfirmed</td>
      <td>Jose</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>40.71355</td>
      <td>-73.95003</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>flexible</td>
      <td>Private room</td>
      <td>2016.0</td>
      <td>570.0</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>17.0</td>
      <td>2019-04-30</td>
      <td>0.61</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>229.0</td>
      <td>No smoking</td>
    </tr>
    <tr>
      <th>29487</th>
      <td>102054</td>
      <td>57365760</td>
      <td>Private Bedroom with Amazing Rooftop View</td>
      <td>45936254757</td>
      <td>verified</td>
      <td>Trey</td>
      <td>Brooklyn</td>
      <td>Bushwick</td>
      <td>40.69872</td>
      <td>-73.92718</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>flexible</td>
      <td>Private room</td>
      <td>NaN</td>
      <td>909.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>2017-08-31</td>
      <td>0.72</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>#NAME?</td>
    </tr>
    <tr>
      <th>29488</th>
      <td>102055</td>
      <td>57366313</td>
      <td>Pretty Brooklyn One-Bedroom for 2 to 4 people</td>
      <td>23801060917</td>
      <td>verified</td>
      <td>Michael</td>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>40.67810</td>
      <td>-73.90822</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>moderate</td>
      <td>Entire home/apt</td>
      <td>NaN</td>
      <td>387.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>50.0</td>
      <td>2019-06-26</td>
      <td>3.12</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>235.0</td>
      <td>* Check out: 10am * We made an effort to keep ...</td>
    </tr>
    <tr>
      <th>29489</th>
      <td>102056</td>
      <td>57366865</td>
      <td>Room &amp; private bathroom in historic Harlem</td>
      <td>15593031571</td>
      <td>unconfirmed</td>
      <td>Shireen</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.81248</td>
      <td>-73.94317</td>
      <td>United States</td>
      <td>US</td>
      <td>True</td>
      <td>strict</td>
      <td>Private room</td>
      <td>NaN</td>
      <td>848.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Each of us is working and/or going to school a...</td>
    </tr>
    <tr>
      <th>29490</th>
      <td>102057</td>
      <td>57367417</td>
      <td>Rosalee Stewart</td>
      <td>93578954226</td>
      <td>verified</td>
      <td>Stanley</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.81315</td>
      <td>-73.94747</td>
      <td>United States</td>
      <td>US</td>
      <td>False</td>
      <td>flexible</td>
      <td>Entire home/apt</td>
      <td>2011.0</td>
      <td>1128.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>2019-06-15</td>
      <td>0.85</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>238.0</td>
      <td>Please remember that this is a residential bui...</td>
    </tr>
  </tbody>
</table>
<p>29491 rows × 26 columns</p>
</div>




```python
# Make a boxplot.

plt.figure(figsize=(10,7))
sns.boxplot(df_top5, hue='neighbourhood', y='price (USD)', palette=('coolwarm'))
plt.ylabel('Price (USD)')
plt.title('Price Distribution of Top 5 Neighbourhood')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
plt.show()
```


    
![png](Project%203_files/Project%203_58_0.png)
    


As we can see, the minimum and maximum prices of all top 5 neighbourhood are all equal. I conclude that this data has already been filtered, for levels of variation of minimum and maximum price are absent. Data within first and third quantile, and their mean also has very small differences.

Task 10, done.
