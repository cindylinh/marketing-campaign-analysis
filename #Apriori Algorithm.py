#Apriori Algorithm

import matplotlib
matplotlib.use('TkAgg') # Fix multi-plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import networkx as nx
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')


# 1. Load dataset
# ==============================
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

#Create Age segment
cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
cut_bins = [0, 30, 45, 65, 120]
df['Age_group'] = pd.cut(df['Age'], bins=cut_bins, labels=cut_labels_Age)
#Create Income segment
cut_labels_Income = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
df['Income_group'] = pd.qcut(df['Income'], q=4, labels=cut_labels_Income)
#Create Seniority segment
cut_labels_Seniority = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
df['Seniority_group'] = pd.qcut(df['Seniority'], q=4, labels=cut_labels_Seniority)
df=df.drop(columns=['Age','Income','Seniority'])

cut_labels = ['Low consumer', 'Frequent consumer', 'Biggest consumer']
df['Wines_segment'] = pd.qcut(df['Wines'][df['Wines']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df['Fruits_segment'] = pd.qcut(df['Fruits'][df['Fruits']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df['Meat_segment'] = pd.qcut(df['Meat'][df['Meat']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df['Fish_segment'] = pd.qcut(df['Fish'][df['Fish']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df['Sweets_segment'] = pd.qcut(df['Sweets'][df['Sweets']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df['Gold_segment'] = pd.qcut(df['Gold'][df['Gold']>0],q=[0, .25, .75, 1], labels=cut_labels).astype("object")
df.replace(np.nan, "Non consumer",inplace=True)
df.drop(columns=['Spending','Wines','Fruits','Meat','Fish','Sweets','Gold'],inplace=True)
df = df.astype(object)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 999)
pd.options.display.float_format = "{:.3f}".format
association=df.copy() 
data = pd.get_dummies(association)
min_support = 0.08
max_len = 10
frequent_items = apriori(data, use_colnames=True, min_support=min_support, max_len=max_len + 1)
rules = association_rules(frequent_items, metric='lift', min_threshold=1)

product='Wines'
segment='Biggest consumer'
target = f"Wines_segment_{segment}"  # no curly braces
results = rules[rules['consequents'].astype(str).str.contains(target)]
results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnal_care.head())

import matplotlib.pyplot as plt

# Filter to just rules you’re interested in
rules_plot = results_personnal_care.copy()

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    rules_plot['support'],
    rules_plot['confidence'],
    c=rules_plot['lift'],
    cmap='viridis',
    s=100, edgecolors='k'
)

plt.colorbar(scatter, label='Lift')
plt.title('Association Rules: Support vs Confidence (color=Lift)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()

# Take top N rules by lift
top_rules = results_personnal_care.sort_values('lift', ascending=False).head(10)

G = nx.DiGraph()

for _, row in top_rules.iterrows():
    for a in row['antecedents']:
        for c in row['consequents']:
            G.add_edge(a, c, weight=row['lift'])

plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
plt.title('Top Association Rules Network')
plt.show()

def get_rules(product, segment='Biggest consumer', rules=rules):
    target = f"{product}_segment_{segment}"
    filtered = rules[rules['consequents'].astype(str).str.contains(target, na=False)]
    return filtered.sort_values(by='confidence', ascending=False)

products = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]

for prod in products:
    rules_for_prod = get_rules(prod, 'Biggest consumer')
    print(f"\n--- {prod} Rules ---")
    print(rules_for_prod.head(5))