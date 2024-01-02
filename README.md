# Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning

## Overview

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

## Implementing clustering using k-means clustering


```shell

inertia = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300, n_init=10, random_state = 42)
    kmeans.fit(y_fit)
    inertia.append(kmeans.inertia_)

sns.lineplot(x=range(1,11), y = inertia, color = 'purple')
sns.scatterplot(x=range(1,11), y = inertia, s = 50, color = 'blue')
circle = Ellipse((4, 45000), width=0.3, height=2000, color='red', fill=False, linewidth=2)
plt.gca().add_patch(circle)
# plt.gca().autoscale_view()
plt.show()

```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/kmeans.png) <br>
The slope appears to be decreasing from 4 to 5. Therefore, n_cluster = 4 will be chosen to perform the k-means clustering model.


## Calculating the silhouette score to see how the model performance is obtained.
```shell

n_cluster = [4,5,6,8,9,10]
fig, ax = plt.subplots(2, 3, figsize=(15,8))
for i in n_cluster:
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 4)
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(df_std)
```
![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/siloute.png) <br>
The silhouette score that is good is the one on the lower right with an average value of 0.6, so the performance of the model obtained from the silhouette score is also better. In addition, if you pay attention. In general, a silhouette value that approaches 1 indicates that the data clustering within that cluster is very good.



## Principal Component Analysis

```shell
# Membandingkan hasil scatter plot PCA dengan scatter plot sebelumnya
sns.pairplot(data=df_pca, hue='Labels', diag_kind='kde', palette=(random.shuffle(colors)))
plt.tight_layout(rect = (2,2,2,2))
```

![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/Customer%20Segmentation%20Based%20on%20RFM%20Model.png) <br>

## Customer Personality Analysis for Marketing Retargeting
```shell

c = ['#957DAD','#E0BBe4','#B7D3DF','#CDE8E6']

def dist_feats(features):
    plt.figure(figsize=[len(features)*5,3])
    i = 1
    for feats in features:
        ax = plt.subplot(1,len(features),i)
        ax.vlines(cluster[feats].median(), ymin=-0.5, ymax=3, color='black', linewidth=1,  linestyle='--')
        dfg = cluster.groupby('Labels')
        x = dfg[feats].median().index
        y = dfg[feats].median().values
        ax.barh(x,y, color=c)
        plt.title(feats)
        i = i+1

dist_feats(features)

```

![image](https://github.com/ariniamsr/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/Pic/RFM.png) <br>

#### Insights for each feature:<br>

R, Recency: The higher the value of frequency, the more often the customer makes a purchase.<br>
F, Total_Purchases: The higher the value of frequency, the more often the customer makes a purchase.<br>
M, total Purchases: The higher the value of monetary, the more money the customer spends on purchases.<br>

#### From the visualization above, we can draw the following conclusions:<br>

Label 0 = has a high R pattern as well as F and M below the median.<br>
Label 1 = has a high F and M pattern as well as R below the median.<br>
Label 2 = has a low F, M, and R pattern.<br>
Label 3 = has a high F, M, and R pattern.<br>


Cluster 0: Most Loyal Customers:<br>
Customers in this cluster last interacted with the business 74 days ago, with low shopping frequency and the highest spending.<br>

Cluster 1: New Customers:<br>
Customers in this cluster have just interacted with the business within the last 22 days, with high shopping frequency and significant spending.<br>

Cluster 2: Impactful Customers:<br>
Customers in this cluster have just interacted with the business within the last 24 days, with low shopping frequency and a fair amount of spending.<br>

Cluster 3: Passive Customers:<br>
Customers in this cluster last interacted with the business 73 days ago, with high shopping frequency and significant spending.<br>


#### Selecting clusters for marketing retargeting:<br>

Cluster 3 & Cluster 1: These clusters are good targets for retargeting because of their high shopping frequency and spending. Marketing strategies can focus on offering exclusive deals or purchase bonuses to increase customer loyalty in these groups.<br>

## Calculating the potential impact of marketing retargeting results from existing clusters
```shell

Cluster_0 = cluster[cluster['Labels'] == 0]['Monetary'].sum()
Cluster_1 = cluster[cluster['Labels'] == 1]['Monetary'].sum()
Cluster_2 = cluster[cluster['Labels'] == 2]['Monetary'].sum()
Cluster_3 = cluster[cluster['Labels'] == 3]['Monetary'].sum()
total_spent  = Cluster_0 + Cluster_1 + Cluster_2 + Cluster_3
potential_impact_cluster_3 = (Cluster_3 / total_spent) * 100
potential_impact_cluster_1 = (Cluster_1 / total_spent) * 100


print('Total Spent of Cluster 0: Rp', Cluster_0)
print('Total Spent of Cluster 1: Rp', Cluster_1)
print('Total Spent of Cluster 2: Rp', Cluster_2)
print('Total Spent of Cluster 3: Rp', Cluster_3)
print('Total Spent: Rp', total_spent)
print('Potential Impact of Cluster 3: {:.2f}%'.format(potential_impact_cluster_3))
print('Potential Impact of Cluster 1: {:.2f}%'.format(potential_impact_cluster_1))
```

output:<br>
Total Spent of Cluster 0: Rp 88453000<br>
Total Spent of Cluster 1: Rp 557668000<br>
Total Spent of Cluster 2: Rp 80137000<br>
Total Spent of Cluster 3: Rp 626855000<br>
Total Spent: Rp 1353113000<br>
Potential Impact of Cluster 3: 46.33%<br>
Potential Impact of Cluster 1: 41.21%<br>


If we calculate the potential impact by focusing on retargeting marketing on Cluster 3 and Cluster 1, the total spending we will receive is Rp 62,685,500,000 for Cluster 3 and Rp 55,766,800,000 for Cluster 1, with a potential impact of 46.33% and 41.21%.
