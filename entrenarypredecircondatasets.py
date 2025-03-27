import pickle
import pandas as pd
import matplotlib.pyplot as plt


# Cargar datos de prueba de helathinsrance
train_data = pd.read_csv('C:/Users/Usuario/Documents/PROYECTOSACC/Dataset/healtinsurance/insurance.csv')

# Mostrar información detallada de las-m  columnas
print("Información del DataFrame:")
print (train_data.head())
print (train_data.info())
print (train_data.describe())

# Mostrar si hay valores nulos en cada columna
train_data['age'].fillna(train_data['age'].median(), inplace=True)
print("\nValores nulos por columna:")
print(train_data.isnull().sum())

#rellenar variables categoricas

train_data ['region'] = train_data ['region'].astype('category').cat.codes

#graficar histogramas de todas las columnas# Graficar histogramas para cada columna numérica
for column in train_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 6))  # Ajustar el tamaño del gráfico
    train_data[column].hist(bins=30, color='skyblue', edgecolor='black')  # Personalizar el histograma
    plt.title(f'Histograma de {column}', fontsize=14)  # Título para cada gráfico
    plt.xlabel(f'{column} (Valor)', fontsize=12)  # Etiqueta del eje X
    plt.ylabel('Frecuencia', fontsize=12)  # Etiqueta del eje Y
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Agregar una cuadrícula ligera
    plt.show()
