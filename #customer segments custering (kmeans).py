#customer segments custering (kmeans) 
import matplotlib
matplotlib.use('TkAgg') # Fix multi-plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np 
import networkx as nx
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 1. Load your dataset
# ==============================
# Change "data.csv" to your file path
df = pd.read_csv("/Users/mia/Downloads/SEM 2 - 2025/Personalized Project/marketing_campaign.csv", sep="\t")

#Data Cleaning and Feature Engineering
#Removing missing data
df=df.dropna(subset=['Income'])
df=df[df['Income']<600000]
#Age of customer today
df['Age'] = 2025 - df['Year_Birth']
#Total spendings on items
df['Spending']=df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']
#Segmenting marital status in 2 groups
df['Marital_Status']=df['Marital_Status'].replace({'Divorced':'Alone','Single':'Alone','Married':'In couple','Together':'In couple','Absurd':'Alone','Widow':'Alone','YOLO':'Alone'})
#Feature indicates total children in the household
df['Children'] = df['Kidhome'] + df['Teenhome']
#Feature indicates if the customer has children or not
df['Has_child'] = np.where(df.Children > 0, 'Has child', 'No child')
#Replacing number of children with categorical values
df['Children'].replace({3: "3 children",2:'2 children',1:'1 child',0:"No child"},inplace=True)
#Senority variable creating
last_date = pd.to_datetime('2014-10-04')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
df['Seniority'] = (last_date - df['Dt_Customer']).dt.days / 30
df['Seniority'] = df['Seniority'].astype(int)
#segmenting education levels in 3 groups
df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
#Rename columns for clarity
df=df.rename(columns={'MntWines': "Wines",'MntFruits':'Fruits','MntMeatProducts':'Meat','MntFishProducts':'Fish','MntSweetProducts':'Sweets','MntGoldProds':'Gold'})
#Selecting columns
df=df[['Age','Education','Marital_Status','Income','Spending', 'Seniority','Has_child','Children','Wines','Fruits','Meat','Fish','Sweets','Gold']]

# 1. Xác định cột số và cột phân loại
num_features = ['Age','Income','Spending','Seniority','Wines','Fruits','Meat','Fish','Sweets','Gold']
cat_features = ['Education','Marital_Status','Has_child','Children']

# 2. Tạo transformer: scale cho số, onehot cho phân loại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])

# 3. Chuẩn hoá + onehot cho dữ liệu
X_transformed = preprocessor.fit_transform(df)

# 4. Elbow method để tìm k tối ưu
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(X_transformed)
visualizer.show()

# Giả sử X_transformed là dữ liệu đã scale + one-hot encode
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_transformed)

# Xem số lượng mỗi cụm
print(df['Cluster'].value_counts())

# Xem trung bình các biến số theo cụm
num_features = ['Age','Income','Spending','Seniority','Wines','Fruits','Meat','Fish','Sweets','Gold']
cluster_profile = df.groupby('Cluster')[num_features].mean()
print(cluster_profile)\

cat_features = ['Education','Marital_Status','Has_child','Children']
cat_summary = df.groupby('Cluster')[cat_features].agg(lambda x: x.value_counts(normalize=True).to_dict())
print(cat_summary)

plt.figure(figsize=(8,6))
plt.scatter(df['Income'], df['Spending'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Customer Segments (k=5)')
plt.show()

