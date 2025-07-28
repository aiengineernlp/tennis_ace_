# import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from textblob.en.inflect import plural_rules

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



print("""Service Game Columns (Offensive)""")

"""CAS DE LA VARIABLE DEPENDANTE: Wins"""

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
#
#
print("ploting of features: ", df['DoubleFaults'].name, "on", df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['DoubleFaults'],df['Wins'],color = 'orange',label= "donnees reelles", alpha=0.4)
plt.xlabel("Variable Ind: (DoubleFaults)")
plt.ylabel("Variable Dep: (Wins)")
plt.title("relation entre DoubleFaults, 'et',Wins")
plt.grid(True)
plt.legend()
plt.show()
#
#
print("ploting of feature: ",df['FirstServe'].name, "on",df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['FirstServe'],df['Wins'],color= "orange",label ="valeur reelles",alpha= 0.4)
plt.xlabel("Variable Ind: (FirstServe)")
plt.ylabel("Variable DePendante:(Wins)")
plt.title("Relation entre FirstServe et Wins")
plt.legend()
plt.grid(True)
plt.show()
# #
# #
print(("ploting of features : ",df['FirstServePointsWon'].name, "on",df['Wins'].name))
plt.figure(figsize=(8,5))
plt.scatter(df['FirstServePointsWon'],df['Wins'],color='orange',label='valeurs reelles',alpha=0.4)
plt.title("Relation entre FirstServePointsWon et Wins")
plt.xlabel("Variable inde: (FirstServePointsWon)")
plt.ylabel("Variable depend: (Wins)")
plt.legend()
plt.grid(True)
plt.show()
print("\n\n")
# #
# #
# #
print('ploting of feature :', df['SecondServePointsWon'].name, "on", df['Wins'].name)
plt.figure(figsize = (8,5))
plt.scatter(df['SecondServePointsWon'],df['Wins'],label = 'Valeur Reells', alpha =0.4,color = 'orange')
plt.xlabel('Variable ind (SecondServePointsWon)')
plt.ylabel("Variable dependante (Wins)")
plt.title('Relation entre SecondServePointsWon et Wins')
plt.legend()
plt.grid(True)
plt.show()
#
print('ploting of feature: ', df['BreakPointsFaced'].name , df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['BreakPointsFaced'],df['Wins'],color='orange',label='valeur reelles', alpha= 0.4)
plt.title("Relation entre BreakPointsFaced et  Wins")
plt.xlabel("Variable independante : (BreakPointsFaced)")
plt.ylabel("varible dependante: (Wins)")
plt.legend()
plt.grid(True)
plt.show()


print('ploting of feature: ', df['BreakPointsSaved'].name, "on", df['Wins'])
plt.figure(figsize=(8,5))
plt.scatter(df['BreakPointsSaved'], df["Wins"], color = 'orange', alpha=0.4,label='Valeurs reelles')
plt.xlabel('Variable independate: BreakPointsSaved')
plt.ylabel('Vriable dependante: Wins ')
plt.grid(True)
plt.legend()
plt.show()


print('ploting of feature:',df['ServiceGamesPlayed'].name , 'on',df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ServiceGamesPlayed'],df['Wins'],color='orange',label='Valeurs reelles',alpha = 0.4)
plt.xlabel('valeur independante : ServiceGamesPlayed ')
plt.ylabel('valeur dependante:  Wins')
plt.title("Relation entre ServiceGamesPlayed et Wins")
plt.grid(True)
plt.legend()
plt.show()

print('ploting of feature:',df['ServiceGamesWon'].name , 'on',df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ServiceGamesWon'],df['Wins'],color='orange',label='Valeurs reelles',alpha = 0.4)
plt.xlabel('valeur independante : ServiceGamesWon ')
plt.ylabel('valeur dependante:  Wins')
plt.title("Relation entre ServiceGamesWon et Wins")
plt.grid(True)
plt.legend()
plt.show()


print("ploting of fetures :",df['ServiceGamesWon'].name , "on", df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ServiceGamesWon'], df['Wins'],color = 'orange',alpha = 0.4, label = "valeurs reelles")
plt.xlabel("Variable indep:ServiceGamesWon")
plt.ylabel("Variable dependante: Wins ")
plt.title('Relation entre ServiceGamesWon et Wins')
plt.legend()
plt.grid(True)
plt.show()

#
print("ploting of feature:", df['TotalServicePointsWon'].name, 'on',df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['TotalServicePointsWon'],df['Wins'], color= 'orange',label = 'Valeurs reelles', alpha = 0.4)
plt.xlabel("Variable  ind: TotalServicePointsWon")
plt.ylabel("Variable dep: Wins")
plt.title("Relation entre TotalServicePointsWon er Wins")
plt.legend()
plt.grid(True)
plt.show()

print('ploting of feature ',df['SecondServeReturnPointsWon'].name ,'on',df['Wins'])
plt.scatter(df['SecondServeReturnPointsWon'],df['Wins'],color = 'orange',label = 'Valeur reelle',alpha = 0.4)
plt.xlabel("Variable indep: ")
plt.ylabel("Variable dependante:")
plt.title('Relation entre SecondServeReturnPointsWon et Wins')
plt.grid(True)
plt.legend()
plt.show()


print("ploting of feature:  ",df['BreakPointsOpportunities'].name, 'on' ,df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['BreakPointsOpportunities'],df['Wins'],color = 'orange',label = 'Valeurs reelles',alpha=0.4)
plt.xlabel('Variable independante: BreakPointsOpportunities ')
plt.ylabel('Variable dependante: Wins')
plt.title('Relation entre BreakPointsOpportunities et Wins')
plt.legend()
plt.grid(True)
plt.show()

print('ploting of feature ',df['BreakPointsConverted'].name, 'on', df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['BreakPointsConverted'],df['Wins'],color = 'orange',label = 'Valeur reellees')
plt.xlabel("Variable independante: BreakPointsConverted")
plt.ylabel("Variable dependante: `WIns'")
plt.legend()
plt.grid(True)
plt.show()


print("ploting of feature,",df['ReturnGamesPlayed'].name, 'on', df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ReturnGamesPlayed'],df['Wins'],color = 'orange',label='Valeur reelle',alpha = 0.4)
plt.xlabel('Variable indep: ReturnGamesPlayed')
plt.ylabel('Variable dependante: Wins')
plt.legend()
plt.grid(True)
plt.show()

print("Ploting of feature,",df['ReturnGamesWon'].name, 'on',df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ReturnGamesWon'],df['Wins'],color = 'orange',alpha=0.4,label = 'Valeurs reelles')
plt.title("relation entre ReturnGamesWon et Wins")
plt.xlabel("Variable independante: ReturnGamesWon ")
plt.ylabel("variable dependante: Wins ")
plt.legend()
plt.grid(True)
plt.show()


print("Ploting of features,", df['ReturnPointsWon'].name, 'on', df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['ReturnPointsWon'],df['Wins'],color = 'orange',label = 'Variable reellee',alpha = 0.5)
plt.title('Relation entre ReturnPointsWon et Wins')
plt.xlabel('variable independante: ReturnPointsWon ')
plt.ylabel('variable dependante:  Wins')
plt.legend()
plt.grid(True)
plt.show()



print('ploting of ',df['TotalPointsWon'].name, 'on',df['Wins'].name)
plt.figure(figsize=(8,5))
plt.scatter(df['TotalPointsWon'],df['Wins'],color = 'orange', alpha = 0.4, label= "Valeure reelles")
plt.title('Relatiion entre  TotalPointsWon et  Wins')
plt.xlabel("Variable independante  TotalPointsWon ")
plt.ylabel("Variable dependante: Wins ")
plt.legend()
plt.grid(True)
plt.show()



















print("""Return Game Columns (Defensive)""")

# print('ploting of the feature:', df['FirstServeReturnPointsWon'].name, 'on', df['Wins'].name)
# plt.figure(figsize=(8,5))
# plt.scatter(df['FirstServeReturnPointsWon'],df['Wins'],color = 'orange',alpha = 0.4)
# plt.xlabel('variable independante :FirstServeReturnPointsWon ')
# plt.ylabel('variable dependante : Wins')
# plt.title('Relation entre  FirstServeReturnPointsWon et Wins')
# plt.legend()
# plt.grid(True)
# plt.show()
#
















"""CAS DE LA VARIABLE DEPENDANTE  Losses """

# print("Ploting of features ",df['Aces'].name , "on", df['Losses'].name)
# plt.figure(figsize=(8,5))
# plt.scatter(df['Aces'],df['Losses'],color = 'orange', label = 'valeurs reellees', alpha= 0.4)
# plt.xlabel('Variable inde: (Aces)')
# plt.ylabel('Variable dependante: (Losses)')
# plt.title('Relation entre Aces and Losses')
# plt.legend()
# plt.grid(True)
# plt.show()

























## perform single feature linear regressions here:






















## perform two feature linear regressions here:





















## perform multiple feature linear regressions here:




