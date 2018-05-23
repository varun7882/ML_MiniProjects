import pandas as pd
dataset=pd.read_csv("W1data.csv")

X=dataset.drop(['Cultivar 1','Cultivar 2','Cultivar 3'],axis=1).values
y=dataset.loc[:,['Cultivar 1','Cultivar 2','Cultivar 3']].values
