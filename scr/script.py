# import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
data_tennis = pd.read_csv("../data/tennis_stats.csv")
df = pd.DataFrame(data_tennis)


# afficher les infos generales

print("Les infos generales de tennis_stats \n",df.info())
print("\n\n\cn")
# afficher les collones du dataframe
print("Les collonnes du DataFrame: \n: ",df.columns)
print("\n\n\n")
#Afficher les types de variable du DataFrame
print("Les types variables du DataFrame:\n",df.dtypes) # est ce que on peut dire que ce sont les types des variables des series??
#print pour explorer les etiquettes/valeurs uniques/variables categorielles
print(df['Player'].value_counts().head(30))






# perform exploratory analysis here:
print("# plotting of feature: ", df['Aces'].name,"on",df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['Aces'], df['Wins'],color = 'orange',label = 'Donnees reelles', alpha = 0.4)
plt.xlabel("Variable Ind: (Aces)")
plt.ylabel("Variable Dep: (Wins)")
plt.title("Relation entre Aces et Wins")
plt.grid(True)
plt.legend()
plt.show()
print('\n\n')


























## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here: