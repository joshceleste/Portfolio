# Weather History (Data Cleaning in Python)

##### In this project, I cleaned and transformed a dataset with 96,453 rows and 12 columns of raw weather data using Python to ensure accurate analysis and deeper insights. The following data cleaning tasks were performed:

1. Data Parsing and Formatting
2. Missing Values Handling
3. Row Duplicates Detection Handling 
4. Consistency Check and Data Validations
5. Data Subsetting to CSV Transformation
6. Outlier Detection and Handling

### 1. Data Parsing and Formatting

##### First, we review the columns and its data types to ensure that all data are consistent in format, identifying and correcting any discrepancies to standardize the data for accurate analysis.


```python
# Import the necessary libraries and data from the CSV file.

import pandas as pd
df = pd.read_csv('weatherHistory.csv')

```


```python
# Then let's take a look at the columns and data types of the dataset.

df.dtypes
```




    Formatted Date               object
    Summary                      object
    Precip Type                  object
    Temperature (C)             float64
    Apparent Temperature (C)    float64
    Humidity                    float64
    Wind Speed (km/h)           float64
    Wind Bearing (degrees)      float64
    Visibility (km)             float64
    Loud Cover                  float64
    Pressure (millibars)        float64
    Daily Summary                object
    dtype: object



 As we can see above, the Formatted Date column is an object.


```python
# Let's convert it to a datetime format.

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='raise')
```

    C:\Users\Josh09\AppData\Local\Temp\ipykernel_3792\1812845144.py:3: FutureWarning: In a future version of pandas, parsing datetimes with mixed time zones will raise an error unless `utc=True`. Please specify `utc=True` to opt in to the new behaviour and silence this warning. To create a `Series` with mixed offsets and `object` dtype, please use `apply` and `datetime.datetime.strptime`
      df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='raise')
    

FutureWarning occured because column 'Formatted Date' consists of utc date format.


```python
# Let's adjust our code.

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='raise', utc=True)
```


```python

# Let's check if the conversion was successful.

df.dtypes
```




    Formatted Date              datetime64[ns, UTC]
    Summary                                  object
    Precip Type                              object
    Temperature (C)                         float64
    Apparent Temperature (C)                float64
    Humidity                                float64
    Wind Speed (km/h)                       float64
    Wind Bearing (degrees)                  float64
    Visibility (km)                         float64
    Loud Cover                              float64
    Pressure (millibars)                    float64
    Daily Summary                            object
    dtype: object



Convertion of the column 'Formatted Date' from object to date time data type is successful.

Task 1, done.

### 2. Missing Values Handling

##### We determine which columns have missing values and decide whether to fill them with statistical measures, transform, or drop the affected rows.


```python
# Let's check if there are any missing values in the dataset based on the columns.

df.isna().sum()
```




    Formatted Date                0
    Summary                       0
    Precip Type                 517
    Temperature (C)               0
    Apparent Temperature (C)      0
    Humidity                      0
    Wind Speed (km/h)             0
    Wind Bearing (degrees)        0
    Visibility (km)               0
    Loud Cover                    0
    Pressure (millibars)          0
    Daily Summary                 0
    dtype: int64




```python
# As we can see above, the `Precip Type` column alone has missing values. Let's check its character first as a column.

df['Precip Type'].unique()
```




    array(['rain', 'snow', nan], dtype=object)



##### Here we can see that 'Precip Type' column contains object data, which only has 3 unique values; Rain, Snow, and NaN. 

##### After evaluating the context of the 'Nan' rows of the 'Precip Type' column, The 'NaN' values in the 'Precip Type' column indicates the absence precipitation, as supported by the 'Summary' and 'Daily Summary' columns; not due to data error. Therefore, we replace 'NaN' with 'No Precip' instead of dropping the rows.



```python
# Let's replace the missing values with 'no precip'.

df['Precip Type'] = df['Precip Type'].fillna('no precip')
```


```python
#Let's check if the replacement was successful and if there are no missing values.

df['Precip Type'].unique()
```




    array(['rain', 'snow', 'no precip'], dtype=object)




```python
df.isna().sum()
```




    Formatted Date              0
    Summary                     0
    Precip Type                 0
    Temperature (C)             0
    Apparent Temperature (C)    0
    Humidity                    0
    Wind Speed (km/h)           0
    Wind Bearing (degrees)      0
    Visibility (km)             0
    Loud Cover                  0
    Pressure (millibars)        0
    Daily Summary               0
    dtype: int64



Task 2, done.

### 3. Row Duplicates Detection Handling

##### Identifying and addressing duplicate rows to ensure each entry is unique, either by removing exact duplicates or consolidating redundant information.


```python
# Let's check if there are any duplicated rows by checking the index.

df[df.duplicated()].index
```




    Index([36072, 36073, 36074, 36075, 36076, 36077, 36078, 36079, 36080, 36081,
           36082, 36083, 36084, 36085, 36086, 36087, 36088, 36089, 36090, 36091,
           36092, 36093, 36094, 36095],
          dtype='int64')




```python
# Let's check the duplicated rows along with its identical rows and sort them for a much better understanding.

df[df.duplicated(keep=False)].sort_values(by='Formatted Date').head(5)
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
      <th>Formatted Date</th>
      <th>Summary</th>
      <th>Precip Type</th>
      <th>Temperature (C)</th>
      <th>Apparent Temperature (C)</th>
      <th>Humidity</th>
      <th>Wind Speed (km/h)</th>
      <th>Wind Bearing (degrees)</th>
      <th>Visibility (km)</th>
      <th>Loud Cover</th>
      <th>Pressure (millibars)</th>
      <th>Daily Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8040</th>
      <td>2010-08-01 22:00:00+00:00</td>
      <td>Clear</td>
      <td>rain</td>
      <td>18.800000</td>
      <td>18.800000</td>
      <td>0.93</td>
      <td>6.279</td>
      <td>270.0</td>
      <td>14.9086</td>
      <td>0.0</td>
      <td>1016.99</td>
      <td>Partly cloudy starting in the afternoon contin...</td>
    </tr>
    <tr>
      <th>36072</th>
      <td>2010-08-01 22:00:00+00:00</td>
      <td>Clear</td>
      <td>rain</td>
      <td>18.800000</td>
      <td>18.800000</td>
      <td>0.93</td>
      <td>6.279</td>
      <td>270.0</td>
      <td>14.9086</td>
      <td>0.0</td>
      <td>1016.99</td>
      <td>Partly cloudy starting in the afternoon contin...</td>
    </tr>
    <tr>
      <th>36073</th>
      <td>2010-08-01 23:00:00+00:00</td>
      <td>Clear</td>
      <td>rain</td>
      <td>18.222222</td>
      <td>18.222222</td>
      <td>0.97</td>
      <td>6.279</td>
      <td>291.0</td>
      <td>14.9086</td>
      <td>0.0</td>
      <td>1017.09</td>
      <td>Partly cloudy starting in the afternoon contin...</td>
    </tr>
    <tr>
      <th>8041</th>
      <td>2010-08-01 23:00:00+00:00</td>
      <td>Clear</td>
      <td>rain</td>
      <td>18.222222</td>
      <td>18.222222</td>
      <td>0.97</td>
      <td>6.279</td>
      <td>291.0</td>
      <td>14.9086</td>
      <td>0.0</td>
      <td>1017.09</td>
      <td>Partly cloudy starting in the afternoon contin...</td>
    </tr>
    <tr>
      <th>8042</th>
      <td>2010-08-02 00:00:00+00:00</td>
      <td>Clear</td>
      <td>rain</td>
      <td>18.072222</td>
      <td>18.072222</td>
      <td>0.98</td>
      <td>11.270</td>
      <td>290.0</td>
      <td>6.8425</td>
      <td>0.0</td>
      <td>1013.23</td>
      <td>Partly cloudy starting in the afternoon contin...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's drop the duplicated rows, reset the index count, and sort the remaining rows.

df = df.drop_duplicates().sort_values(by='Formatted Date').reset_index(drop=True)
```


```python
# Final duplicate check.

df[df.duplicated()].index
```




    Index([], dtype='int64')



Task 3, done.

### 4. Consistency Check and Data Validations

##### We will perform sanity checks on the dataset by identifying:
-Negative values in the 'Wind Speed (km/h)' column, as these should not exist since it measures speed, not direction.

-Values in the 'Temperature (C)' column that fall outside Earth's recorded temperature range of -89.2°C to 56.7°C.


```python
# Let's check first if there are any negative values in the columns 'Wind Speed (km/h)', then we will determine count of it.

df[df['Wind Speed (km/h)'] < 0].count()
```




    Formatted Date              0
    Summary                     0
    Precip Type                 0
    Temperature (C)             0
    Apparent Temperature (C)    0
    Humidity                    0
    Wind Speed (km/h)           0
    Wind Bearing (degrees)      0
    Visibility (km)             0
    Loud Cover                  0
    Pressure (millibars)        0
    Daily Summary               0
    dtype: int64




```python
# Column 'Wind Speed (km/h)' has no negative values, hence it is acceptable. 

# Now let's check the column 'Temperature (C)' and its acceptable range.

df[(df['Temperature (C)'] < -89.2) | (df['Temperature (C)'] > 56.7)].count()
```




    Formatted Date              0
    Summary                     0
    Precip Type                 0
    Temperature (C)             0
    Apparent Temperature (C)    0
    Humidity                    0
    Wind Speed (km/h)           0
    Wind Bearing (degrees)      0
    Visibility (km)             0
    Loud Cover                  0
    Pressure (millibars)        0
    Daily Summary               0
    dtype: int64



Since there are no values that is beyond the coldest and hottest earth surface temperature record, data in column 'Temperature (C)' is acceptable.

Task 4, done.

### 5. Data Subsetting to CSV Transformation

##### In an instance that we may need just a few specific columns to work on, we can create a subset of data from a large set of data, which are only necessary for a certain project. 


```python
# First we decide what columns to involve for a new set of data.

df.columns
```




    Index(['Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)',
           'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
           'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover',
           'Pressure (millibars)', 'Daily Summary'],
          dtype='object')




```python
# For the subdata, let's include columns 'Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)', 'Humidity', 'Visibility (km)', 'Loud Cover', and 'Daily Summary'.

#Then we save it as a stand alone data set, without it referencing to 'df'.

new_df = df[['Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)',
       #'Apparent Temperature (C)', 
       'Humidity', #'Wind Speed (km/h)',
       #'Wind Bearing (degrees)', 
       'Visibility (km)', #'Loud Cover',
       #'Pressure (millibars)', 
        'Daily Summary']].copy()
```


```python
# Now let's check if a new data has been created.

new_df.columns
```




    Index(['Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)',
           'Humidity', 'Visibility (km)', 'Daily Summary'],
          dtype='object')




```python
# Since we successfully created a new subset, let's create a new csv file out of it. 

new_df.to_csv('new df.csv')
```

Task 5, done.

### 6. Outlier Detection and Handling

##### We will detect outliers in column 'Wind Speed (km/h)' using statistical methods and then address them by correcting or removing anomalies to ensure the accuracy and reliability of the data.


```python
# In this task, scipy library will be needed.

import scipy.stats as stats
```


```python
# The most common statistical method when identifying outliers is the Interquartile Range (IQR) or Z-score. 

# We first need to know what the distribution of the column Wind Speed (km/h) is before we can choose the appropriate method.

stats.describe(df['Wind Speed (km/h)'])
```




    DescribeResult(nobs=96429, minmax=(0.0, 63.8526), mean=10.812460236028583, variance=47.79433597724738, skewness=1.1134639633369425, kurtosis=1.7692650251192665)



As seen above, the Skewness and Kurtosis values resulted to a non-normal distribution, so we will use the IQR method.



```python
# Let's find the IQR.

Q1 = df['Wind Speed (km/h)'].quantile(0.25)
Q3 = df['Wind Speed (km/h)'].quantile(0.75)
IQR = Q3 - Q1
IQR
```




    8.307599999999999




```python
# Let's find the upper and lower bounds.

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound, lower_bound
```




    (26.597199999999997, -6.633199999999997)




```python
# As we can observe here, the lower_bound is -6.63. This is deemed illogical since there are no negative wind speeds.

# We will replace the most logically correct value for the lower_bound with 0.

lower_bound = 0
upper_bound, lower_bound
```




    (26.597199999999997, 0)




```python
# Now that we have determined the bounds, Let's determine and check the outliers that are found beyond those.

Outliers = df[(df['Wind Speed (km/h)'] < lower_bound) | (df['Wind Speed (km/h)'] > upper_bound)]

Outliers['Wind Speed (km/h)']

```




    15       27.5954
    33       26.8226
    437      28.5292
    438      26.9353
    440      28.5292
              ...   
    93606    30.9764
    93803    27.2412
    93899    29.4469
    94305    27.5954
    94315    29.1249
    Name: Wind Speed (km/h), Length: 3028, dtype: float64




```python
# Let's drop the outliers.

df_after_drop = df.drop(Outliers.index)
df_after_drop
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
      <th>Formatted Date</th>
      <th>Summary</th>
      <th>Precip Type</th>
      <th>Temperature (C)</th>
      <th>Apparent Temperature (C)</th>
      <th>Humidity</th>
      <th>Wind Speed (km/h)</th>
      <th>Wind Bearing (degrees)</th>
      <th>Visibility (km)</th>
      <th>Loud Cover</th>
      <th>Pressure (millibars)</th>
      <th>Daily Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-12-31 23:00:00+00:00</td>
      <td>Partly Cloudy</td>
      <td>rain</td>
      <td>0.577778</td>
      <td>-4.050000</td>
      <td>0.89</td>
      <td>17.1143</td>
      <td>140.0</td>
      <td>9.9820</td>
      <td>0.0</td>
      <td>1016.66</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006-01-01 00:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>1.161111</td>
      <td>-3.238889</td>
      <td>0.85</td>
      <td>16.6152</td>
      <td>139.0</td>
      <td>9.9015</td>
      <td>0.0</td>
      <td>1016.15</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2006-01-01 01:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>1.666667</td>
      <td>-3.155556</td>
      <td>0.82</td>
      <td>20.2538</td>
      <td>140.0</td>
      <td>9.9015</td>
      <td>0.0</td>
      <td>1015.87</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006-01-01 02:00:00+00:00</td>
      <td>Overcast</td>
      <td>rain</td>
      <td>1.711111</td>
      <td>-2.194444</td>
      <td>0.82</td>
      <td>14.4900</td>
      <td>140.0</td>
      <td>9.9015</td>
      <td>0.0</td>
      <td>1015.56</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2006-01-01 03:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>1.183333</td>
      <td>-2.744444</td>
      <td>0.86</td>
      <td>13.9426</td>
      <td>134.0</td>
      <td>9.9015</td>
      <td>0.0</td>
      <td>1014.98</td>
      <td>Mostly cloudy throughout the day.</td>
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
    </tr>
    <tr>
      <th>96424</th>
      <td>2016-12-31 18:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>0.488889</td>
      <td>-2.644444</td>
      <td>0.86</td>
      <td>9.7566</td>
      <td>167.0</td>
      <td>8.0178</td>
      <td>0.0</td>
      <td>1020.03</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>96425</th>
      <td>2016-12-31 19:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>rain</td>
      <td>0.072222</td>
      <td>-3.050000</td>
      <td>0.88</td>
      <td>9.4185</td>
      <td>169.0</td>
      <td>7.2450</td>
      <td>0.0</td>
      <td>1020.27</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>96426</th>
      <td>2016-12-31 20:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>snow</td>
      <td>-0.233333</td>
      <td>-3.377778</td>
      <td>0.89</td>
      <td>9.2736</td>
      <td>175.0</td>
      <td>9.5795</td>
      <td>0.0</td>
      <td>1020.50</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>96427</th>
      <td>2016-12-31 21:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>snow</td>
      <td>-0.472222</td>
      <td>-3.644444</td>
      <td>0.91</td>
      <td>9.2414</td>
      <td>182.0</td>
      <td>8.4042</td>
      <td>0.0</td>
      <td>1020.65</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
    <tr>
      <th>96428</th>
      <td>2016-12-31 22:00:00+00:00</td>
      <td>Mostly Cloudy</td>
      <td>snow</td>
      <td>-0.677778</td>
      <td>-3.888889</td>
      <td>0.92</td>
      <td>9.2253</td>
      <td>189.0</td>
      <td>8.8711</td>
      <td>0.0</td>
      <td>1020.72</td>
      <td>Mostly cloudy throughout the day.</td>
    </tr>
  </tbody>
</table>
<p>93401 rows × 12 columns</p>
</div>



The dataframe df_after_drop is now free from outliers and is within the first and third quantile range.

Task 6, done.


