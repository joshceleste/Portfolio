# Transport Vehicles Sales (EDA in Python)

##### This project focuses on exploratory data analysis (EDA) of sales records to extract actionable insights and understand key sales trends. Using Python, I analyzed a comprehensive dataset that includes various attributes of sales transactions, customer details, and product information. 

##### The following tasks were performed:

1. Data Summary and Initial Inspection
2. Handling Missing Data
3. Data Type Conversion
4. Univariate 
5. Bivariate Analysis
6. Time Series Analysis
7. Categorical Data Analysis
8. Pivot Tables and Grouping
9. Correlation and Heatmap
10. Customer Segmentation (K-means Clustering)

### 1. Data Summary and Initial Inspection

##### First we review the dataset to understand its structure by displaying basic information, summary statistics, checking for missing values and duplicates, and overall row counts. This initial inspection will help identify data quality issues and provide an overview of the dataset.


```python
# Let's load necessary libraries and dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
df = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')
```


```python
# Display basic information about the dataset

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2823 entries, 0 to 2822
    Data columns (total 25 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   ORDERNUMBER       2823 non-null   int64  
     1   QUANTITYORDERED   2823 non-null   int64  
     2   PRICEEACH         2823 non-null   float64
     3   ORDERLINENUMBER   2823 non-null   int64  
     4   SALES             2823 non-null   float64
     5   ORDERDATE         2823 non-null   object 
     6   STATUS            2823 non-null   object 
     7   QTR_ID            2823 non-null   int64  
     8   MONTH_ID          2823 non-null   int64  
     9   YEAR_ID           2823 non-null   int64  
     10  PRODUCTLINE       2823 non-null   object 
     11  MSRP              2823 non-null   int64  
     12  PRODUCTCODE       2823 non-null   object 
     13  CUSTOMERNAME      2823 non-null   object 
     14  PHONE             2823 non-null   object 
     15  ADDRESSLINE1      2823 non-null   object 
     16  ADDRESSLINE2      302 non-null    object 
     17  CITY              2823 non-null   object 
     18  STATE             1337 non-null   object 
     19  POSTALCODE        2747 non-null   object 
     20  COUNTRY           2823 non-null   object 
     21  TERRITORY         1749 non-null   object 
     22  CONTACTLASTNAME   2823 non-null   object 
     23  CONTACTFIRSTNAME  2823 non-null   object 
     24  DEALSIZE          2823 non-null   object 
    dtypes: float64(2), int64(7), object(16)
    memory usage: 551.5+ KB
    


```python
# Display summary statistics for numerical columns.abs

df.describe()
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
      <th>ORDERNUMBER</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>ORDERLINENUMBER</th>
      <th>SALES</th>
      <th>QTR_ID</th>
      <th>MONTH_ID</th>
      <th>YEAR_ID</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.000000</td>
      <td>2823.00000</td>
      <td>2823.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10258.725115</td>
      <td>35.092809</td>
      <td>83.658544</td>
      <td>6.466171</td>
      <td>3553.889072</td>
      <td>2.717676</td>
      <td>7.092455</td>
      <td>2003.81509</td>
      <td>100.715551</td>
    </tr>
    <tr>
      <th>std</th>
      <td>92.085478</td>
      <td>9.741443</td>
      <td>20.174277</td>
      <td>4.225841</td>
      <td>1841.865106</td>
      <td>1.203878</td>
      <td>3.656633</td>
      <td>0.69967</td>
      <td>40.187912</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10100.000000</td>
      <td>6.000000</td>
      <td>26.880000</td>
      <td>1.000000</td>
      <td>482.130000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2003.00000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10180.000000</td>
      <td>27.000000</td>
      <td>68.860000</td>
      <td>3.000000</td>
      <td>2203.430000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2003.00000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10262.000000</td>
      <td>35.000000</td>
      <td>95.700000</td>
      <td>6.000000</td>
      <td>3184.800000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>2004.00000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10333.500000</td>
      <td>43.000000</td>
      <td>100.000000</td>
      <td>9.000000</td>
      <td>4508.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>2004.00000</td>
      <td>124.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10425.000000</td>
      <td>97.000000</td>
      <td>100.000000</td>
      <td>18.000000</td>
      <td>14082.800000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>2005.00000</td>
      <td>214.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for missing values

df.isna().sum()
```




    ORDERNUMBER            0
    QUANTITYORDERED        0
    PRICEEACH              0
    ORDERLINENUMBER        0
    SALES                  0
    ORDERDATE              0
    STATUS                 0
    QTR_ID                 0
    MONTH_ID               0
    YEAR_ID                0
    PRODUCTLINE            0
    MSRP                   0
    PRODUCTCODE            0
    CUSTOMERNAME           0
    PHONE                  0
    ADDRESSLINE1           0
    ADDRESSLINE2        2521
    CITY                   0
    STATE               1486
    POSTALCODE            76
    COUNTRY                0
    TERRITORY           1074
    CONTACTLASTNAME        0
    CONTACTFIRSTNAME       0
    DEALSIZE               0
    dtype: int64




```python
# Check for duplicates.

df[df.duplicated(keep=False)].sort_values(by='ORDERDATE')
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
      <th>ORDERNUMBER</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>ORDERLINENUMBER</th>
      <th>SALES</th>
      <th>ORDERDATE</th>
      <th>STATUS</th>
      <th>QTR_ID</th>
      <th>MONTH_ID</th>
      <th>YEAR_ID</th>
      <th>PRODUCTLINE</th>
      <th>MSRP</th>
      <th>PRODUCTCODE</th>
      <th>CUSTOMERNAME</th>
      <th>PHONE</th>
      <th>ADDRESSLINE1</th>
      <th>ADDRESSLINE2</th>
      <th>CITY</th>
      <th>STATE</th>
      <th>POSTALCODE</th>
      <th>COUNTRY</th>
      <th>TERRITORY</th>
      <th>CONTACTLASTNAME</th>
      <th>CONTACTFIRSTNAME</th>
      <th>DEALSIZE</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# let's determine numbers of rows.

df.shape
```




    (2823, 25)



Upon analyzing the dataset, it was noted that the buyers are not retail consumers, but rather resellers.

Task 1, done.

### 2. Handling Missing Data

##### After identifying missing values, it’s essential to handle them appropriately to ensure data quality. Depending on the nature of the data and the extent of missingness, different approaches can be used.

##### Referring to Task 1, let's observe the columns with missing values and its count: 
-Columns with missing data are ADDRESSLINE2, STATE, POSTALCODE, and TERRITORY.

-Around 80% rows are affected from missing values, column 'ADDRESSLINE2' being the highest.

-All columns with missing values are NOMINAL and Categorical in nature. 

##### Therefore, removing these rows will not provide any advantage for our analysis. Instead, we will explore alternative methods to address the missing data.


```python
# Let's check columns with missing values. First we deal with column ADDRESSLINE2.

df[df.isnull().any(axis=1)]
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
      <th>ORDERNUMBER</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>ORDERLINENUMBER</th>
      <th>SALES</th>
      <th>ORDERDATE</th>
      <th>STATUS</th>
      <th>QTR_ID</th>
      <th>MONTH_ID</th>
      <th>YEAR_ID</th>
      <th>PRODUCTLINE</th>
      <th>MSRP</th>
      <th>PRODUCTCODE</th>
      <th>CUSTOMERNAME</th>
      <th>PHONE</th>
      <th>ADDRESSLINE1</th>
      <th>ADDRESSLINE2</th>
      <th>CITY</th>
      <th>STATE</th>
      <th>POSTALCODE</th>
      <th>COUNTRY</th>
      <th>TERRITORY</th>
      <th>CONTACTLASTNAME</th>
      <th>CONTACTFIRSTNAME</th>
      <th>DEALSIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10107</td>
      <td>30</td>
      <td>95.70</td>
      <td>2</td>
      <td>2871.00</td>
      <td>2/24/2003 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>2</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Land of Toys Inc.</td>
      <td>2125557818</td>
      <td>897 Long Airport Avenue</td>
      <td>NaN</td>
      <td>NYC</td>
      <td>NY</td>
      <td>10022</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Yu</td>
      <td>Kwai</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10121</td>
      <td>34</td>
      <td>81.35</td>
      <td>5</td>
      <td>2765.90</td>
      <td>5/7/2003 0:00</td>
      <td>Shipped</td>
      <td>2</td>
      <td>5</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Reims Collectables</td>
      <td>26.47.1555</td>
      <td>59 rue de l'Abbaye</td>
      <td>NaN</td>
      <td>Reims</td>
      <td>NaN</td>
      <td>51100</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Henriot</td>
      <td>Paul</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10134</td>
      <td>41</td>
      <td>94.74</td>
      <td>2</td>
      <td>3884.34</td>
      <td>7/1/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>7</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Lyon Souveniers</td>
      <td>+33 1 46 62 7555</td>
      <td>27 rue du Colonel Pierre Avia</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>NaN</td>
      <td>75508</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Da Cunha</td>
      <td>Daniel</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10145</td>
      <td>45</td>
      <td>83.26</td>
      <td>6</td>
      <td>3746.70</td>
      <td>8/25/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>8</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Toys4GrownUps.com</td>
      <td>6265557265</td>
      <td>78934 Hillside Dr.</td>
      <td>NaN</td>
      <td>Pasadena</td>
      <td>CA</td>
      <td>90003</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Young</td>
      <td>Julie</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10159</td>
      <td>49</td>
      <td>100.00</td>
      <td>14</td>
      <td>5205.27</td>
      <td>10/10/2003 0:00</td>
      <td>Shipped</td>
      <td>4</td>
      <td>10</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Corporate Gift Ideas Co.</td>
      <td>6505551386</td>
      <td>7734 Strong St.</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>CA</td>
      <td>NaN</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Brown</td>
      <td>Julie</td>
      <td>Medium</td>
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
    </tr>
    <tr>
      <th>2818</th>
      <td>10350</td>
      <td>20</td>
      <td>100.00</td>
      <td>15</td>
      <td>2244.40</td>
      <td>12/2/2004 0:00</td>
      <td>Shipped</td>
      <td>4</td>
      <td>12</td>
      <td>2004</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Euro Shopping Channel</td>
      <td>(91) 555 94 44</td>
      <td>C/ Moralzarzal, 86</td>
      <td>NaN</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>28034</td>
      <td>Spain</td>
      <td>EMEA</td>
      <td>Freyre</td>
      <td>Diego</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2819</th>
      <td>10373</td>
      <td>29</td>
      <td>100.00</td>
      <td>1</td>
      <td>3978.51</td>
      <td>1/31/2005 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>1</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Oulu Toy Supplies, Inc.</td>
      <td>981-443655</td>
      <td>Torikatu 38</td>
      <td>NaN</td>
      <td>Oulu</td>
      <td>NaN</td>
      <td>90110</td>
      <td>Finland</td>
      <td>EMEA</td>
      <td>Koskitalo</td>
      <td>Pirkko</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2820</th>
      <td>10386</td>
      <td>43</td>
      <td>100.00</td>
      <td>4</td>
      <td>5417.57</td>
      <td>3/1/2005 0:00</td>
      <td>Resolved</td>
      <td>1</td>
      <td>3</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Euro Shopping Channel</td>
      <td>(91) 555 94 44</td>
      <td>C/ Moralzarzal, 86</td>
      <td>NaN</td>
      <td>Madrid</td>
      <td>NaN</td>
      <td>28034</td>
      <td>Spain</td>
      <td>EMEA</td>
      <td>Freyre</td>
      <td>Diego</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2821</th>
      <td>10397</td>
      <td>34</td>
      <td>62.24</td>
      <td>1</td>
      <td>2116.16</td>
      <td>3/28/2005 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>3</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Alpha Cognac</td>
      <td>61.77.6555</td>
      <td>1 rue Alsace-Lorraine</td>
      <td>NaN</td>
      <td>Toulouse</td>
      <td>NaN</td>
      <td>31000</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Roulet</td>
      <td>Annette</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2822</th>
      <td>10414</td>
      <td>47</td>
      <td>65.52</td>
      <td>9</td>
      <td>3079.44</td>
      <td>5/6/2005 0:00</td>
      <td>On Hold</td>
      <td>2</td>
      <td>5</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Gifts4AllAges.com</td>
      <td>6175559555</td>
      <td>8616 Spinnaker Dr.</td>
      <td>NaN</td>
      <td>Boston</td>
      <td>MA</td>
      <td>51003</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Yoshido</td>
      <td>Juri</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
<p>2676 rows × 25 columns</p>
</div>



As observed, the column **ADDRESSLINE2** appears to be associated with ADDRESSLINE1, typically containing a secondary address when applicable, making it optional. Based on a domain-knowledge imputation approach, we will remove the ADDRESSLINE2 column.


```python
# Remove ADDRESSLINE2 column.

df.drop('ADDRESSLINE2', axis=1, inplace=True)

# then check to confirm.

df.columns
```




    Index(['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER',
           'SALES', 'ORDERDATE', 'STATUS', 'QTR_ID', 'MONTH_ID', 'YEAR_ID',
           'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE',
           'ADDRESSLINE1', 'CITY', 'STATE', 'POSTALCODE', 'COUNTRY', 'TERRITORY',
           'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'DEALSIZE'],
          dtype='object')




```python
# Let's check the column STATE in a unique order sorted by COUNTRY . 

df[['COUNTRY', 'STATE']].drop_duplicates().sort_values(by='COUNTRY')
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
      <th>COUNTRY</th>
      <th>STATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Australia</td>
      <td>NSW</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Australia</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Australia</td>
      <td>Queensland</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Austria</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Belgium</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Canada</td>
      <td>BC</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Canada</td>
      <td>Quebec</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Denmark</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Finland</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Germany</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Ireland</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Italy</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Japan</td>
      <td>Tokyo</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Japan</td>
      <td>Osaka</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Norway</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Philippines</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Singapore</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Spain</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sweden</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>171</th>
      <td>Switzerland</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>UK</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>123</th>
      <td>UK</td>
      <td>Isle of Wight</td>
    </tr>
    <tr>
      <th>0</th>
      <td>USA</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>18</th>
      <td>USA</td>
      <td>PA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>USA</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>13</th>
      <td>USA</td>
      <td>CT</td>
    </tr>
    <tr>
      <th>12</th>
      <td>USA</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USA</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>45</th>
      <td>USA</td>
      <td>NH</td>
    </tr>
    <tr>
      <th>462</th>
      <td>USA</td>
      <td>NV</td>
    </tr>
  </tbody>
</table>
</div>



As observed, the column labeled **STATE** does not strictly adhere to the formal definition of a 'State'; instead, it includes regions or major divisions from other countries besides the USA.

I have decided to take the following actions:

-Rename column 'STATE' to 'STATE/REGION'.

-Replace 'NaN' rows in column STATE with 'Not Specified'.


```python
# Rename column STATE to STATE/REGION

df.rename(columns={'STATE':'STATE/REGION'}, inplace=True)
```


```python
# Replace NaN rows in column STATE/REGION with Not Specified.

df['STATE/REGION'] = df['STATE/REGION'].fillna('Not Specified')
```


```python
# Let's check again to confirm.

df[['COUNTRY', 'STATE/REGION']].drop_duplicates().sort_values(by='COUNTRY')
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
      <th>COUNTRY</th>
      <th>STATE/REGION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Australia</td>
      <td>NSW</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Australia</td>
      <td>Victoria</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Australia</td>
      <td>Queensland</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Austria</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Belgium</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Canada</td>
      <td>BC</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Canada</td>
      <td>Quebec</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Denmark</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Finland</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Germany</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Ireland</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Italy</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Japan</td>
      <td>Tokyo</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Japan</td>
      <td>Osaka</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Norway</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Philippines</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Singapore</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Spain</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sweden</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>171</th>
      <td>Switzerland</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>24</th>
      <td>UK</td>
      <td>Not Specified</td>
    </tr>
    <tr>
      <th>123</th>
      <td>UK</td>
      <td>Isle of Wight</td>
    </tr>
    <tr>
      <th>0</th>
      <td>USA</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>18</th>
      <td>USA</td>
      <td>PA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>USA</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>13</th>
      <td>USA</td>
      <td>CT</td>
    </tr>
    <tr>
      <th>12</th>
      <td>USA</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USA</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>45</th>
      <td>USA</td>
      <td>NH</td>
    </tr>
    <tr>
      <th>462</th>
      <td>USA</td>
      <td>NV</td>
    </tr>
  </tbody>
</table>
</div>



Now let's deal with missing values in **POSTALCODE**.


```python
# Show NaN rows in column POSTALCODE and its COUNTRY.

df[df['POSTALCODE'].isna()]['COUNTRY'].unique()
```




    array(['USA'], dtype=object)




```python
# The only country with missing POSTALCODE is USA. We can fill it with value in the column CITY instead.

df['POSTALCODE'] = df['POSTALCODE'].fillna(df['CITY'])
df
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
      <th>ORDERNUMBER</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>ORDERLINENUMBER</th>
      <th>SALES</th>
      <th>ORDERDATE</th>
      <th>STATUS</th>
      <th>QTR_ID</th>
      <th>MONTH_ID</th>
      <th>YEAR_ID</th>
      <th>PRODUCTLINE</th>
      <th>MSRP</th>
      <th>PRODUCTCODE</th>
      <th>CUSTOMERNAME</th>
      <th>PHONE</th>
      <th>ADDRESSLINE1</th>
      <th>CITY</th>
      <th>STATE/REGION</th>
      <th>POSTALCODE</th>
      <th>COUNTRY</th>
      <th>TERRITORY</th>
      <th>CONTACTLASTNAME</th>
      <th>CONTACTFIRSTNAME</th>
      <th>DEALSIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10107</td>
      <td>30</td>
      <td>95.70</td>
      <td>2</td>
      <td>2871.00</td>
      <td>2/24/2003 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>2</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Land of Toys Inc.</td>
      <td>2125557818</td>
      <td>897 Long Airport Avenue</td>
      <td>NYC</td>
      <td>NY</td>
      <td>10022</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Yu</td>
      <td>Kwai</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10121</td>
      <td>34</td>
      <td>81.35</td>
      <td>5</td>
      <td>2765.90</td>
      <td>5/7/2003 0:00</td>
      <td>Shipped</td>
      <td>2</td>
      <td>5</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Reims Collectables</td>
      <td>26.47.1555</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Not Specified</td>
      <td>51100</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Henriot</td>
      <td>Paul</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10134</td>
      <td>41</td>
      <td>94.74</td>
      <td>2</td>
      <td>3884.34</td>
      <td>7/1/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>7</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Lyon Souveniers</td>
      <td>+33 1 46 62 7555</td>
      <td>27 rue du Colonel Pierre Avia</td>
      <td>Paris</td>
      <td>Not Specified</td>
      <td>75508</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Da Cunha</td>
      <td>Daniel</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10145</td>
      <td>45</td>
      <td>83.26</td>
      <td>6</td>
      <td>3746.70</td>
      <td>8/25/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>8</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Toys4GrownUps.com</td>
      <td>6265557265</td>
      <td>78934 Hillside Dr.</td>
      <td>Pasadena</td>
      <td>CA</td>
      <td>90003</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Young</td>
      <td>Julie</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10159</td>
      <td>49</td>
      <td>100.00</td>
      <td>14</td>
      <td>5205.27</td>
      <td>10/10/2003 0:00</td>
      <td>Shipped</td>
      <td>4</td>
      <td>10</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Corporate Gift Ideas Co.</td>
      <td>6505551386</td>
      <td>7734 Strong St.</td>
      <td>San Francisco</td>
      <td>CA</td>
      <td>San Francisco</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Brown</td>
      <td>Julie</td>
      <td>Medium</td>
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
    </tr>
    <tr>
      <th>2818</th>
      <td>10350</td>
      <td>20</td>
      <td>100.00</td>
      <td>15</td>
      <td>2244.40</td>
      <td>12/2/2004 0:00</td>
      <td>Shipped</td>
      <td>4</td>
      <td>12</td>
      <td>2004</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Euro Shopping Channel</td>
      <td>(91) 555 94 44</td>
      <td>C/ Moralzarzal, 86</td>
      <td>Madrid</td>
      <td>Not Specified</td>
      <td>28034</td>
      <td>Spain</td>
      <td>EMEA</td>
      <td>Freyre</td>
      <td>Diego</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2819</th>
      <td>10373</td>
      <td>29</td>
      <td>100.00</td>
      <td>1</td>
      <td>3978.51</td>
      <td>1/31/2005 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>1</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Oulu Toy Supplies, Inc.</td>
      <td>981-443655</td>
      <td>Torikatu 38</td>
      <td>Oulu</td>
      <td>Not Specified</td>
      <td>90110</td>
      <td>Finland</td>
      <td>EMEA</td>
      <td>Koskitalo</td>
      <td>Pirkko</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2820</th>
      <td>10386</td>
      <td>43</td>
      <td>100.00</td>
      <td>4</td>
      <td>5417.57</td>
      <td>3/1/2005 0:00</td>
      <td>Resolved</td>
      <td>1</td>
      <td>3</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Euro Shopping Channel</td>
      <td>(91) 555 94 44</td>
      <td>C/ Moralzarzal, 86</td>
      <td>Madrid</td>
      <td>Not Specified</td>
      <td>28034</td>
      <td>Spain</td>
      <td>EMEA</td>
      <td>Freyre</td>
      <td>Diego</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2821</th>
      <td>10397</td>
      <td>34</td>
      <td>62.24</td>
      <td>1</td>
      <td>2116.16</td>
      <td>3/28/2005 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>3</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Alpha Cognac</td>
      <td>61.77.6555</td>
      <td>1 rue Alsace-Lorraine</td>
      <td>Toulouse</td>
      <td>Not Specified</td>
      <td>31000</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Roulet</td>
      <td>Annette</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2822</th>
      <td>10414</td>
      <td>47</td>
      <td>65.52</td>
      <td>9</td>
      <td>3079.44</td>
      <td>5/6/2005 0:00</td>
      <td>On Hold</td>
      <td>2</td>
      <td>5</td>
      <td>2005</td>
      <td>Ships</td>
      <td>54</td>
      <td>S72_3212</td>
      <td>Gifts4AllAges.com</td>
      <td>6175559555</td>
      <td>8616 Spinnaker Dr.</td>
      <td>Boston</td>
      <td>MA</td>
      <td>51003</td>
      <td>USA</td>
      <td>NaN</td>
      <td>Yoshido</td>
      <td>Juri</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
<p>2823 rows × 24 columns</p>
</div>



Lastly, let's check for **TERRITORY** and its missing values by COUNTRY.


```python
df[df['TERRITORY'].isna()]['COUNTRY'].unique()
```




    array(['USA', 'Canada'], dtype=object)




```python
# The only country with missing TERRITORY is USA and Canada; which both are within the territory of North America.

# Let's fill the missing TERRITORY with North America.

df['TERRITORY'] = df['TERRITORY'].fillna('North America')

df[['COUNTRY', 'TERRITORY']].query('TERRITORY == "North America"')
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
      <th>COUNTRY</th>
      <th>TERRITORY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>4</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>8</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2809</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2810</th>
      <td>Canada</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2812</th>
      <td>Canada</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2817</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
    <tr>
      <th>2822</th>
      <td>USA</td>
      <td>North America</td>
    </tr>
  </tbody>
</table>
<p>1074 rows × 2 columns</p>
</div>




```python
# Final check for missing values

df.isnull().sum()
```




    ORDERNUMBER         0
    QUANTITYORDERED     0
    PRICEEACH           0
    ORDERLINENUMBER     0
    SALES               0
    ORDERDATE           0
    STATUS              0
    QTR_ID              0
    MONTH_ID            0
    YEAR_ID             0
    PRODUCTLINE         0
    MSRP                0
    PRODUCTCODE         0
    CUSTOMERNAME        0
    PHONE               0
    ADDRESSLINE1        0
    CITY                0
    STATE/REGION        0
    POSTALCODE          0
    COUNTRY             0
    TERRITORY           0
    CONTACTLASTNAME     0
    CONTACTFIRSTNAME    0
    DEALSIZE            0
    dtype: int64



Task 2, done.

### 3. Data Type Conversion

##### After handling missing data, it’s important to ensure that each column has the correct data type for accurate analysis and efficient processing. Converting data types helps avoid errors and ensures the appropriate operations can be performed on the data.


```python
# let's check the columns, its data type, and value characteristics that each contain. 

df.dtypes
```




    ORDERNUMBER           int64
    QUANTITYORDERED       int64
    PRICEEACH           float64
    ORDERLINENUMBER       int64
    SALES               float64
    ORDERDATE            object
    STATUS               object
    QTR_ID                int64
    MONTH_ID              int64
    YEAR_ID               int64
    PRODUCTLINE          object
    MSRP                  int64
    PRODUCTCODE          object
    CUSTOMERNAME         object
    PHONE                object
    ADDRESSLINE1         object
    CITY                 object
    STATE/REGION         object
    POSTALCODE           object
    COUNTRY              object
    TERRITORY            object
    CONTACTLASTNAME      object
    CONTACTFIRSTNAME     object
    DEALSIZE             object
    dtype: object




```python
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
      <th>ORDERNUMBER</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>ORDERLINENUMBER</th>
      <th>SALES</th>
      <th>ORDERDATE</th>
      <th>STATUS</th>
      <th>QTR_ID</th>
      <th>MONTH_ID</th>
      <th>YEAR_ID</th>
      <th>PRODUCTLINE</th>
      <th>MSRP</th>
      <th>PRODUCTCODE</th>
      <th>CUSTOMERNAME</th>
      <th>PHONE</th>
      <th>ADDRESSLINE1</th>
      <th>CITY</th>
      <th>STATE/REGION</th>
      <th>POSTALCODE</th>
      <th>COUNTRY</th>
      <th>TERRITORY</th>
      <th>CONTACTLASTNAME</th>
      <th>CONTACTFIRSTNAME</th>
      <th>DEALSIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10107</td>
      <td>30</td>
      <td>95.70</td>
      <td>2</td>
      <td>2871.00</td>
      <td>2/24/2003 0:00</td>
      <td>Shipped</td>
      <td>1</td>
      <td>2</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Land of Toys Inc.</td>
      <td>2125557818</td>
      <td>897 Long Airport Avenue</td>
      <td>NYC</td>
      <td>NY</td>
      <td>10022</td>
      <td>USA</td>
      <td>North America</td>
      <td>Yu</td>
      <td>Kwai</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10121</td>
      <td>34</td>
      <td>81.35</td>
      <td>5</td>
      <td>2765.90</td>
      <td>5/7/2003 0:00</td>
      <td>Shipped</td>
      <td>2</td>
      <td>5</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Reims Collectables</td>
      <td>26.47.1555</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Not Specified</td>
      <td>51100</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Henriot</td>
      <td>Paul</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10134</td>
      <td>41</td>
      <td>94.74</td>
      <td>2</td>
      <td>3884.34</td>
      <td>7/1/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>7</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Lyon Souveniers</td>
      <td>+33 1 46 62 7555</td>
      <td>27 rue du Colonel Pierre Avia</td>
      <td>Paris</td>
      <td>Not Specified</td>
      <td>75508</td>
      <td>France</td>
      <td>EMEA</td>
      <td>Da Cunha</td>
      <td>Daniel</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10145</td>
      <td>45</td>
      <td>83.26</td>
      <td>6</td>
      <td>3746.70</td>
      <td>8/25/2003 0:00</td>
      <td>Shipped</td>
      <td>3</td>
      <td>8</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Toys4GrownUps.com</td>
      <td>6265557265</td>
      <td>78934 Hillside Dr.</td>
      <td>Pasadena</td>
      <td>CA</td>
      <td>90003</td>
      <td>USA</td>
      <td>North America</td>
      <td>Young</td>
      <td>Julie</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10159</td>
      <td>49</td>
      <td>100.00</td>
      <td>14</td>
      <td>5205.27</td>
      <td>10/10/2003 0:00</td>
      <td>Shipped</td>
      <td>4</td>
      <td>10</td>
      <td>2003</td>
      <td>Motorcycles</td>
      <td>95</td>
      <td>S10_1678</td>
      <td>Corporate Gift Ideas Co.</td>
      <td>6505551386</td>
      <td>7734 Strong St.</td>
      <td>San Francisco</td>
      <td>CA</td>
      <td>San Francisco</td>
      <td>USA</td>
      <td>North America</td>
      <td>Brown</td>
      <td>Julie</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>



We identified several columns that require conversion:

-The ORDERDATE column should be converted to the datetime data type.

-The columns CITY, STATE/REGION, COUNTRY, TERRITORY, STATUS, and DEALSIZE contain categorical values and should be converted to the category data type.


```python
# let's convert the ORDERDATE column to datetime format.

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
```


```python
# let's convert columns CITY, STATE/REGION, COUNTRY, TERRITORY, and DEALSIZE to category data type.

df[['CITY', 'STATE/REGION', 'COUNTRY', 'TERRITORY', 'DEALSIZE', 'STATUS']] = df[['CITY', 'STATE/REGION', 'COUNTRY', 'TERRITORY', 'DEALSIZE', 'STATUS']].astype('category')
```


```python
# Final data type checkabs

df.dtypes
```




    ORDERNUMBER                  int64
    QUANTITYORDERED              int64
    PRICEEACH                  float64
    ORDERLINENUMBER              int64
    SALES                      float64
    ORDERDATE           datetime64[ns]
    STATUS                    category
    QTR_ID                       int64
    MONTH_ID                     int64
    YEAR_ID                      int64
    PRODUCTLINE                 object
    MSRP                         int64
    PRODUCTCODE                 object
    CUSTOMERNAME                object
    PHONE                       object
    ADDRESSLINE1                object
    CITY                      category
    STATE/REGION              category
    POSTALCODE                  object
    COUNTRY                   category
    TERRITORY                 category
    CONTACTLASTNAME             object
    CONTACTFIRSTNAME            object
    DEALSIZE                  category
    dtype: object



Task 3, done.

### 4. Univariate Analysis

##### Univariate analysis focuses on examining the distribution and properties of a single variable (column) at a time. For this example, we will create a Histogram for the column SALES and Boxplot for the column QUANTITYORDERED.



```python
# Histogram for SALES

kde_ax = df['SALES'].plot(kind='kde', figsize=(12, 5), title='Distribution of Sales')

kde_ax.set_xlabel('Sales')
plt.show()
```


    
![png](project%202_files/project%202_35_0.png)
    


As observed above, SALES distribution is NOT normally distributed.


```python
# Boxplot for QUANTITYORDERED

boxplt_ax = df['QUANTITYORDERED'].plot(kind='box', figsize=(5, 5), title='Boxplot of Quantity Ordered')

boxplt_ax.set_xlabel('Quantity Ordered')
plt.show()
```


    
![png](project%202_files/project%202_37_0.png)
    



```python
# let's identify rows with oultiers.
q1 = df['QUANTITYORDERED'].quantile(0.25)
q3 = df['QUANTITYORDERED'].quantile(0.75)
iqr = q3 - q1

# identify the lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# identify outliers in the QUANTITYORDERED column and their index.
outliers = df[(df['QUANTITYORDERED'] < lower_bound) | (df['QUANTITYORDERED'] > upper_bound)]
outliers.index

```




    Index([418, 598, 1666, 1714, 1995, 1996, 2586, 2689], dtype='int64')



Task 4, done.

### 5. Bivariate Analysis

##### Bivariate analysis involves examining the relationship between two variables to understand their interactions, correlations, and potential impacts on each other. Let's use SALES and QUANTITYORDERED for this part.


```python
# Let's make a scatter plot of SALES vs. QUANTITYORDERED

sns.scatterplot(df, x='QUANTITYORDERED', y='SALES')
```




    <Axes: xlabel='QUANTITYORDERED', ylabel='SALES'>




    
![png](project%202_files/project%202_41_1.png)
    



```python
#lets make a correlation matrix column included.

corr_matrix = df[['SALES', 'QUANTITYORDERED', 'PRICEEACH']].corr()
corr_matrix

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
      <th>SALES</th>
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SALES</th>
      <td>1.000000</td>
      <td>0.551426</td>
      <td>0.657841</td>
    </tr>
    <tr>
      <th>QUANTITYORDERED</th>
      <td>0.551426</td>
      <td>1.000000</td>
      <td>0.005564</td>
    </tr>
    <tr>
      <th>PRICEEACH</th>
      <td>0.657841</td>
      <td>0.005564</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Task 5, done.

### 6. Time Series Analysis   

##### This is just a simple Time Series analysis to explore sales trends over time and identify patterns or seasonality in the data.




```python
# Extract year and month from the ORDERDATE

df['Year'] = df['ORDERDATE'].dt.year
df['Month'] = df['ORDERDATE'].dt.month
```


```python
# Group sales by month and year

sales_trends = df.groupby(['Year', 'Month'])['SALES'].sum().reset_index()

```


```python
# Line plot of sales over time

plt.figure(figsize=(10, 6))
sns.lineplot(x=sales_trends.index, y='SALES', data=sales_trends)
plt.title('Sales Trends Over Time')
plt.xticks(ticks=sales_trends.index, labels=sales_trends['Year'].astype(str) + '-' + sales_trends['Month'].astype(str), rotation=45)
plt.ticklabel_format(style='plain', axis='y')
plt.show()
```


    
![png](project%202_files/project%202_48_0.png)
    


It's apparent that from 2003 to 2005, the highest sales peaks at around October and November; probably due to preparation for the December holiday rush.

Task 6, done.

### 7. Categorical Data Analysis

##### In this section, we will explore and analyze the categorical variables present in the dataset to uncover trends, relationships, and patterns. Categorical data refers to variables that can take on a limited, fixed number of possible values representing distinct groups or categories.


```python
# Scatterplot for STATUS
plt.figure(figsize=(10, 5))
sns.scatterplot(df, x='ORDERDATE', y='SALES', hue='STATUS')
plt.xticks(rotation=45)
plt.show()
```


    
![png](project%202_files/project%202_51_0.png)
    


A large cluster of cancelled orders occured between May to July of 2004. Reasons for it are yet to be known.


```python
# Boxplot for SALES by PRODUCTLINE

plt.figure(figsize=(10, 6))
sns.boxplot(x='PRODUCTLINE', y='SALES', data=df)
plt.title('Sales by Product Line')
plt.xticks(rotation=45)
plt.show()
```


    
![png](project%202_files/project%202_53_0.png)
    


Task 7, done.

### 8. Pivot Tables and Grouping

##### Pivot tables and grouping are powerful tools for summarizing and analyzing data, particularly when dealing with large datasets. They allow us to aggregate data, perform calculations, and visualize relationships between different variables.


```python
# Pivot table for total sales by country

pivot_sales_country = df.pivot_table(values='SALES', index='COUNTRY', aggfunc='sum', observed=False)\
                    .sort_values(by='SALES', ascending=False)
print(pivot_sales_country)
```

                      SALES
    COUNTRY                
    USA          3627982.83
    Spain        1215686.92
    France       1110916.52
    Australia     630623.10
    UK            478880.46
    Italy         374674.31
    Finland       329581.91
    Norway        307463.70
    Singapore     288488.41
    Denmark       245637.15
    Canada        224078.56
    Germany       220472.09
    Sweden        210014.21
    Austria       202062.53
    Japan         188167.81
    Switzerland   117713.56
    Belgium       108412.62
    Philippines    94015.73
    Ireland        57756.43
    


```python
# Group sales by CUSTOMERNAME and aggregate total sales

df.groupby('CUSTOMERNAME')['SALES'].agg(TOTAL_SALES='sum')


#Sort by 'Total Sales'
df.groupby('CUSTOMERNAME')['SALES'].agg(TOTAL_SALES='sum')\
        .sort_values(by='TOTAL_SALES', ascending=False)
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
      <th>TOTAL_SALES</th>
    </tr>
    <tr>
      <th>CUSTOMERNAME</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Euro Shopping Channel</th>
      <td>912294.11</td>
    </tr>
    <tr>
      <th>Mini Gifts Distributors Ltd.</th>
      <td>654858.06</td>
    </tr>
    <tr>
      <th>Australian Collectors, Co.</th>
      <td>200995.41</td>
    </tr>
    <tr>
      <th>Muscle Machine Inc</th>
      <td>197736.94</td>
    </tr>
    <tr>
      <th>La Rochelle Gifts</th>
      <td>180124.90</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Royale Belge</th>
      <td>33440.10</td>
    </tr>
    <tr>
      <th>Microscale Inc.</th>
      <td>33144.93</td>
    </tr>
    <tr>
      <th>Auto-Moto Classics Inc.</th>
      <td>26479.26</td>
    </tr>
    <tr>
      <th>Atelier graphique</th>
      <td>24179.96</td>
    </tr>
    <tr>
      <th>Boards &amp; Toys Co.</th>
      <td>9129.35</td>
    </tr>
  </tbody>
</table>
<p>92 rows × 1 columns</p>
</div>



Task 8, done.

### 9. Correlation and Heatmap

##### Just like task 5, but with more numerical variables.


```python
# Calculate correlation between numerical variables

corr_matrix = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MONTH_ID']].corr()
corr_matrix
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
      <th>QUANTITYORDERED</th>
      <th>PRICEEACH</th>
      <th>SALES</th>
      <th>MONTH_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>QUANTITYORDERED</th>
      <td>1.000000</td>
      <td>0.005564</td>
      <td>0.551426</td>
      <td>-0.039048</td>
    </tr>
    <tr>
      <th>PRICEEACH</th>
      <td>0.005564</td>
      <td>1.000000</td>
      <td>0.657841</td>
      <td>0.005152</td>
    </tr>
    <tr>
      <th>SALES</th>
      <td>0.551426</td>
      <td>0.657841</td>
      <td>1.000000</td>
      <td>-0.009605</td>
    </tr>
    <tr>
      <th>MONTH_ID</th>
      <td>-0.039048</td>
      <td>0.005152</td>
      <td>-0.009605</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()

#clear
```


    
![png](project%202_files/project%202_61_0.png)
    


Task 9, done.

### 10. Customer Segmentation (K-means Clustering)

##### Customer Segmentation using K-means clustering is a common technique in data analysis and machine learning to group customers into distinct segments based on their characteristics.


```python
# First let's import scikit-learn library.

from sklearn.cluster import KMeans as KM
from sklearn.preprocessing import StandardScaler as SS
```

Then we collect the variables that are needed for this task.


```python
# We aggregate the data by CUSTOMERNAME to compute the total SALES (named TOTAL_SALES) and the count of ORDERNUMBER (named ORDER_FREQUENCY). 
# 
# The resulting DataFrame is then reset to have a standard index.

customer_data = df.groupby('CUSTOMERNAME').agg({'SALES': 'sum', 'ORDERNUMBER': 'count'}).reset_index()\
                .rename(columns={'SALES': 'Total_Sales', 'ORDERNUMBER': 'Order_Frequency'})
customer_data
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
      <th>CUSTOMERNAME</th>
      <th>Total_Sales</th>
      <th>Order_Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AV Stores, Co.</td>
      <td>157807.81</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alpha Cognac</td>
      <td>70488.44</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amica Models &amp; Co.</td>
      <td>94117.26</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna's Decorations, Ltd</td>
      <td>153996.13</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Atelier graphique</td>
      <td>24179.96</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Vida Sport, Ltd</td>
      <td>117713.56</td>
      <td>31</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Vitachrome Inc.</td>
      <td>88041.26</td>
      <td>25</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Volvo Model Replicas, Co</td>
      <td>75754.88</td>
      <td>19</td>
    </tr>
    <tr>
      <th>90</th>
      <td>West Coast Collectables Co.</td>
      <td>46084.64</td>
      <td>13</td>
    </tr>
    <tr>
      <th>91</th>
      <td>giftsbymail.co.uk</td>
      <td>78240.84</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>92 rows × 3 columns</p>
</div>



After selecting the specified variables and assuming the data has been cleaned, the next steps for k-means involve standardizing the data, determining the optimal number of clusters, and finally performing the clustering process.


```python
# Standardize the data by scaling both variables involved.

customer_data_scaled = SS().fit_transform(customer_data[['Total_Sales', 'Order_Frequency']])
customer_data_scaled
```




    array([[ 4.44431867e-01,  6.60315577e-01],
           [-3.51497241e-01, -3.47292784e-01],
           [-1.36117015e-01, -1.52271810e-01],
           [ 4.09687834e-01,  4.97798100e-01],
           [-7.73605958e-01, -7.69838225e-01],
           [-4.05249096e-01, -2.49782297e-01],
           [ 8.38093296e-01,  7.90329559e-01],
           [-4.51939990e-01, -5.09810261e-01],
           [-4.03035390e-01, -4.12299775e-01],
           [-1.44745416e-01, -1.19768315e-01],
           [-7.52647495e-01, -7.37334730e-01],
           [ 6.88091368e-02,  4.27491626e-02],
           [-6.75035096e-01, -5.42313757e-01],
           [-2.17658135e-01, -2.82285792e-01],
           [-9.10794518e-01, -8.99852207e-01],
           [-5.41515202e-01, -5.74817252e-01],
           [-6.64373106e-01, -6.39824243e-01],
           [-3.08195906e-01, -2.82285792e-01],
           [-3.78673786e-01, -3.14789288e-01],
           [-2.84895032e-01, -3.47292784e-01],
           [-4.67551416e-01, -4.77306766e-01],
           [-1.96532500e-01, -1.84775306e-01],
           [-2.50414427e-01, -2.17278801e-01],
           [ 3.72191482e-01,  3.35280622e-01],
           [ 1.05416397e-01,  4.27491626e-02],
           [-1.37042477e-01, -1.52271810e-01],
           [-3.64586869e-01, -3.47292784e-01],
           [ 3.28065955e-01,  1.72763145e-01],
           [ 1.19297494e-01,  1.02456671e-02],
           [-3.48112422e-01, -4.12299775e-01],
           [-6.65690974e-01, -6.07320748e-01],
           [ 5.82816882e-01,  4.00287613e-01],
           [-2.79274084e-01, -2.49782297e-01],
           [ 7.32168726e+00,  7.42104264e+00],
           [-9.23053230e-02, -1.52271810e-01],
           [-6.52236670e-02, -1.84775306e-01],
           [-4.71762706e-01, -3.79796279e-01],
           [-2.35539414e-01, -1.52271810e-01],
           [ 5.87782794e-02,  1.72763145e-01],
           [-7.70664338e-02, -1.19768315e-01],
           [ 2.36079310e-02, -5.47613240e-02],
           [-4.95195932e-01, -5.09810261e-01],
           [ 3.05822529e-01,  2.70273631e-01],
           [-1.07983836e-01, -2.49782297e-01],
           [ 6.47855478e-01,  7.25322568e-01],
           [ 5.01507561e-01,  5.95308586e-01],
           [-2.77829515e-01, -3.47292784e-01],
           [-3.10955791e-01, -1.84775306e-01],
           [-5.44168369e-02, -1.19768315e-01],
           [-5.56041013e-01, -5.42313757e-01],
           [-6.91888917e-01, -6.72327739e-01],
           [-5.17616647e-01, -5.09810261e-01],
           [-2.60801140e-01, -3.79796279e-01],
           [-2.14154271e-01, -1.52271810e-01],
           [-9.04073220e-04,  1.40259649e-01],
           [ 4.97511919e+00,  4.85326650e+00],
           [-3.15148395e-01, -3.14789288e-01],
           [-2.31234512e-01, -2.49782297e-01],
           [ 8.08391855e-01,  5.62805091e-01],
           [-2.71869210e-01, -2.17278801e-01],
           [ 2.06321284e-01,  1.07756154e-01],
           [-4.72641953e-01, -5.09810261e-01],
           [-3.77779590e-01, -3.47292784e-01],
           [-4.26582941e-02,  4.27491626e-02],
           [-3.10624182e-01, -1.84775306e-01],
           [-3.17622156e-01, -2.82285792e-01],
           [ 2.36926675e-01,  3.35280622e-01],
           [ 2.63477100e-01,  5.62805091e-01],
           [-3.13702095e-01, -1.52271810e-01],
           [-6.89198398e-01, -7.37334730e-01],
           [ 3.71426995e-01,  3.02777127e-01],
           [ 3.08310236e-01,  3.35280622e-01],
           [ 2.29783952e-01,  2.37770136e-01],
           [-5.36260683e-01, -5.09810261e-01],
           [-2.39721445e-01, -5.47613240e-02],
           [ 3.87582230e-01,  4.97798100e-01],
           [-1.84543617e-01, -1.52271810e-01],
           [ 4.47630079e-02, -2.22578284e-02],
           [-2.69610111e-01, -4.44803270e-01],
           [ 1.06945828e-01,  1.07756154e-01],
           [-2.35372516e-01, -3.14789288e-01],
           [ 4.64507619e-01,  3.02777127e-01],
           [ 1.04937487e-01,  4.27491626e-02],
           [-7.97004385e-02, -1.52271810e-01],
           [ 2.00539342e-02, -2.22578284e-02],
           [-4.09120138e-02, -2.22578284e-02],
           [ 8.16531116e-02, -5.47613240e-02],
           [ 7.89667859e-02,  1.02456671e-02],
           [-1.91500664e-01, -1.84775306e-01],
           [-3.03492854e-01, -3.79796279e-01],
           [-5.73941526e-01, -5.74817252e-01],
           [-2.80832957e-01, -1.52271810e-01]])




```python
# Now let's generate elbow plot in order to find the optimal number of clusters.

wcss = []
for i in range(1, 11):
    kmeans = KM(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(customer_data_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```


    
![png](project%202_files/project%202_69_0.png)
    


The graph revealed that 3 clusters form the final elbow point, after which the curve becomes smooth. Therefore, we select 3 clusters as our optimal choice for grouping the data.


```python
# KMeans clustering

kmeans = KM(n_clusters=3)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total_Sales', y='Order_Frequency', hue='Cluster', data=customer_data, palette='Set1')
plt.title('Customer Segmentation')
plt.show()

```


    
![png](project%202_files/project%202_71_0.png)
    


We can conclude that the vast majority of buyers, represented by two clusters, had total sales below 200,000 and placed around 60 orders. In contrast, the remaining 2.17% of buyers accounted for the majority of total sales and order frequency.

Task 10, done.
