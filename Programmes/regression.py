import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv('modified_file.csv')

# Convertir les pourcentages de cacao en nombres
df['cocoa_percentage'] = df['ingredients'].str.extract('(\d+)').astype(int)

# Préparer les données pour le modèle de régression
X = df['cocoa_percentage'].values.reshape(-1,1)
y = df['Satisfaction'].values.reshape(-1,1)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Créer et entraîner le modèle de régression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# Prédire les valeurs de satisfaction pour l'ensemble de test
y_pred = regressor.predict(X_test)

# Afficher les coefficients de la régression
print(regressor.coef_)

# Comparer les valeurs prédites avec les valeurs réelles
df_compare = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df_compare)

# Visualiser les résultats
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
