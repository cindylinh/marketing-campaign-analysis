# eda_template.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 1. Load your dataset
# ==============================
# Change "data.csv" to your file path
df = pd.read_csv("/Users/mia/Downloads/SEM 2 - 2025/Personalized Project/marketing_campaign.csv", sep="\t")

# ==============================
# 2. Quick overview
# ==============================
print("\n🔹 Shape of dataset:", df.shape)
print("\n🔹 First 5 rows:\n", df.head())
print("\n🔹 Info:")
print(df.info())
print("\n🔹 Summary statistics (numeric):\n", df.describe())
print("\n🔹 Summary statistics (categorical):\n", df.describe(include='object'))
print("\n🔹 Missing values:\n", df.isnull().sum())

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


# ==============================
# 3. Numeric analysis
# ==============================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_cols) > 0:
    print("\n🔹 Numeric columns:", list(numeric_cols))

    # Histograms
    df[numeric_cols].hist(figsize=(12, 8))
    plt.suptitle("Histograms of Numeric Columns")
    plt.show()

    # Boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[numeric_cols], orient="h")
    plt.title("Boxplots of Numeric Columns")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# ==============================
# 4. Categorical analysis
# ==============================
cat_cols = df.select_dtypes(include=["object"]).columns

if len(cat_cols) > 0:
    print("\n🔹 Categorical columns:", list(cat_cols))

    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.xticks(rotation=45)
        plt.title(f"Countplot of {col}")
        plt.show()

# ==============================
# 5. Pairplot (relationship overview)
# ==============================
#if len(numeric_cols) > 1:
    #sns.pairplot(df[numeric_cols].dropna())
    #plt.suptitle("Pairplot of Numeric Features", y=1.02)
    #plt.show()

# Tính ma trận tương quan
corr = df.corr(numeric_only=True)

# Lấy ma trận upper triangle để loại bỏ trùng lặp
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Chuyển sang dạng cột để dễ sắp xếp
corr_pairs = upper.unstack().dropna()

# Lấy top 5 correlations mạnh nhất (theo giá trị tuyệt đối)
top5 = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).head(5)

print("Top 5 strongest correlations:")
print(top5)

top5_sorted = top5.sort_values()  # sắp xếp để bar chart nhìn đẹp

# Vẽ bar chart
plt.figure(figsize=(8,5))
top5_sorted.plot(kind='barh', color='skyblue')
plt.xlabel("Correlation")
plt.title("Top 5 Strongest Correlations")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()