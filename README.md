# Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning

## overview

“ A company can develop rapidly when it knows the behavior of it’s customer personality, so that it can provide better services and benefits to customers who have the potential to become loyal customers. By processing historical marketing campaign data to improve performance and target the right customers, so they can transcat on the company’s platform, from this data insight our focus is to create a cluster prediction model to make it easir for companies to make decisions. “

## Load Data
```shell
df = pd.read_csv("marketing_campaign_data.csv")
df.head(12)
```

## Feature Engineering
### Conversion rate
```shell
df['conversion_rate'] = df['Response'] / df['NumWebVisitsMonth']
df
```

### Age 
```shell
def kelompok_usia(x):
    if x['Year_Birth'] <= 1954:
        kelompok = 'Lansia'
    elif x['Year_Birth'] >= 1955 and x['Year_Birth'] <= 1993: 
        kelompok = 'Dewasa'
    else: 
        kelompok  = 'Remaja'
    return kelompok  

df['grup_umur'] = df.apply(lambda x: kelompok_usia(x), axis=1)
```

### Social status 
```shell
def kesejahteraan_masyakat(x):
    if x['Income'] >= 5.174150e+07:
        kelompok = 'Kaya'
    else: 
        kelompok  = 'Biasa aja'
    return kelompok  

df['grup_income'] = df.apply(lambda x: kesejahteraan_masyakat(x), axis=1) 
```

### Number of children, Total transactions and Total expenses
```shell
df['Total_Purchases'] = df['NumDealsPurchases'] + df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']+df['NumWebVisitsMonth']
df['jumlah_anak'] = df['Kidhome'] + df['Teenhome']
df['total_pembelian'] = df['MntCoke']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']
df['Total_Transaksi'] = df['Income'] - df['total_pembelian'] 
df['total_acc_cmp'] = df['AcceptedCmp2'] + df['AcceptedCmp1'] + df['AcceptedCmp5'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] 
```

## Exploratory Data Analysis

### Univariate Analysis

#### Numerical Boxplot
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/EDA%20uni%20num.png)

From the boxplot above, it can be seen that there is an outlier that is not too far from the other data. This outlier is located between the upper and lower bounds of the boxplot. This indicates that the outlier is still within the reasonable range of values for the data.

#### Numerical Distplot
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/EDA%20uni%20num2.png)<br>

1. The following variables are normally distributed: 'total_transaksi', NumWebVisitsMonth, NumStorePurchases, NumWebPurchases, NumDealsPurchases, Recency, Year_Birth
2. The following variables are positively skewed: MntCoke, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, conversion_rate
3. The following variables are bimodal or have more than 1 mode: total_acc_cmp, jumlah_anak, Kidhome, Teenhome
   
### Multivariate Analysis
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/Multivariate%20Analysis.png
)
<br>

## Conversion Rate 
### Conversion Rate Based On Age
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/Conversion%20Ratio%20Based%20on%20Age.png
)<br>
There is a significant relationship between customer age and conversion rate, where adults tend to have a greater impact on conversion rate than teenagers and the elderly. This is because adults are in their active age and have a higher income than teenagers and are more active than the elderly.

### Conversion Rate Based On Jumlah Anak
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/Conversion%20Ratio%20Based%20on%20Anak.png
) <br>
The graph above shows the relationship between conversion rate and the number of children. It can be seen that people with no children tend to have a higher conversion rate than people with one or more children.

## Data Preprocessing  <br>
According to the results, Income has 24 null values, conversion_rate has 11, and Total_Transaksi has 24.

### Handling Missing Value
We handle missing values using the following query,
```shell
df['Income'].fillna(df['Income'].mean(), inplace=True) 
df['conversion_rate'] = df['conversion_rate'].fillna(0) 
df['Total_Transaksi'].fillna(df['Total_Transaksi'].mean(), inplace=True) 
```
### Handling Duplicated Data
there are no duplicates in our data
```shell
df.duplicated().sum()
```
### Drop Data
we will remove unnecessary data. <br>
```shell
df.drop(columns = ['Unnamed: 0','ID', 'Kidhome', 'Teenhome','Z_CostContact', 'Z_Revenue','Dt_Customer'], inplace=True)
 ```

## Feature Encoding
```shell
df['Education'] = df['Education'].map({'S3' : 4, 'S2' : 3, 'S1':2, 'D3':1, 'SMA':0})
df['grup_income'] = df['grup_income'].map({'Kaya':1, 'Biasa aja':0})
df['grup_umur'] = df['grup_umur'].map({'Dewasa' : 1, 'Lansia': 0, 'Remaja':2})
df['Marital_Status'] = df['Marital_Status'].map({'Single' : 0, 'Couple' : 1})
```
## Standardization of Features
```shell
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scd = StandardScaler()
y_fit = scd.fit_transform(df.astype(float))
y_fit
```
### K-Means Clustering - PCA
```shell
cluster = df[['Recency', 'Total_Purchases', 'total_pembelian']].copy()
cluster.columns = ['Recency','Frequency','Monetary']
features = ['Recency','Frequency','Monetary']
cluster.describe(include='all')
```

<br>
We want to see a graph of the RFM

#### Subplot

```shell
cols = cluster.columns
plt.figure(figsize= (15, 20))
for i in range(len(cols)):
    plt.subplot(6, 2, i+1)
    sns.kdeplot(x = cluster[cols[i]])
    plt.tight_layout()
```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/RFMpng.png)


#### Boxplot

```shell
cols = cluster.columns
plt.figure(figsize= (10,15))
for i in range(len(cols)):
    plt.subplot(4, 4, i+1)
    sns.boxplot(y = cluster[cols[i]], orient='v')
    plt.tight_layout()
```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/download.png)

Looks like we have a few outliers. Time to handle them.

## Handling Outliers

```shell
for col in cols:
    high_cut = cluster[col].quantile(q=0.99)
    low_cut= cluster[col].quantile(q=0.01)
    cluster.loc[cluster[col]>high_cut,col]=high_cut
    cluster.loc[cluster[col]<low_cut,col]=low_cut
```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/after%20handling%20outlier.png
)
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/monetary%20need%20handling%20outlier%20more.png) <br>
It turns out that there are still some outliers in the monetary data. Let's handle with transformation.

```shell
tf_log = cluster.copy()
tf_log['Monetary'] = np.log(cluster['Monetary'])

plt.figure(figsize= (5, 5))
sns.kdeplot(x = tf_log['Monetary'])
plt.tight_layout()
```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/monetary%20handling%20outlier%20with%20transformasipng.png
)




```shell
```


