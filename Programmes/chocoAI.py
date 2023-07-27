import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re

def extract_number(s):
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    match = re.search(r'(\d+(\.\d+)?)', s)
    if match:
        return float(match.group())
    else:
        return None








# Charger les données
df = pd.read_csv('products.csv')

# Prétraitement des données
# Convertir les données de quantité en chiffres (exemple : 100 g à 100)
df['quantity'] = df['quantity'].apply(extract_number)

# Normaliser les données
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Séparer les données en caractéristiques (X) et cible (y)
X = df[['quantity', 'nutrition_score', 'carbon_footprint', 'co2_packaging']]
y = df['store_count']

# Séparer les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir le modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compiler le modèle
model.compile(loss='mean_squared_error', optimizer='adam')

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Visualiser les résultats
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
