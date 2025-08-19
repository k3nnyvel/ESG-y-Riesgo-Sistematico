# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:53:53 2025

@author: Kevin Gutiérrez 
"""

#%%
# 1. Cargamos el archivo con el que trabajaremos

# Asegúrese que el archivo con el que trabajaremos se encuentre en su carpeta de trabajo
# antes de correr la celda

import pandas as pd # Cargamos la librería pandas

ESG = pd.read_excel(io='Database ESG BOOK.xlsx',
                    sheet_name='Datos ESG')  
#%%
# 2. Eliminamos las empresas de la base de datos que no tenían datos disponibles en la página de ESG BOOK o que hayan quebrado

ESG_limpio = ESG.dropna()
ESG_limpio = ESG_limpio[~ESG_limpio.astype(str).apply(lambda fila: fila.str.contains("quiebra")).any(axis=1)]
ESG_limpio = ESG_limpio.sort_values(by="name").reset_index(drop=True)

#%%
# 3. Transformamos Inputs Scored de fracción a decimal 
def convertir_inputs(valor):
    try:
        numerador, denominador = map(int, valor.split("/"))
        return numerador / denominador
    except:
        return None  # por si algo falla

ESG_limpio["Inputs Scored"] = ESG_limpio["Inputs Scored"].apply(convertir_inputs)
# %%
# 4. Convertimos todas las columnas relevantes a tipo numérico (float)
columnas_a_convertir = [
    "Performance Score",
    "Risk Score",
    "Sector Percentile PS",
    "Sector Percentile RS",
    "Materiality Disclosure",
    "Exposure Flags",
    "Frameworks",
    "Completed Public Disclosures",
    "Data Points"
]

for col in columnas_a_convertir:
    ESG_limpio[col] = pd.to_numeric(ESG_limpio[col], errors='coerce')


# %%
# 5. Invertimos las escalas de Risk Score y Sector Percentile RS
# Esto lo hacemos porque los datos entregados por ESG book de los Atributos Risk Score y Sector Percentile RS siguen
# una relación inversa. Es decir, Mientras más alto el valor, menos exposición al riesgo ESG se tiene.
ESG_limpio["Risk Score Invertido"] = 100 - ESG_limpio["Risk Score"]
ESG_limpio["Sector Percentile RS Invertido"] = 1 - ESG_limpio["Sector Percentile RS"]
# De esta forma ahora siguen una relación directa, mientras más alto el valor, más riesgo  

# %%
# 6. Eliminar las columnas originales que seguían una relación inversa
ESG_limpio = ESG_limpio.drop(columns=["Risk Score", "Sector Percentile RS"])

# 7. Insertar las columnas invertidas en la posición 10 y 11
ESG_limpio.insert(10, "Risk Score", ESG_limpio.pop("Risk Score Invertido"))
ESG_limpio.insert(11, "Sector Percentile RS", ESG_limpio.pop("Sector Percentile RS Invertido"))


# %%
# NORMALIZACIÓN DE ATRIBUTOS ESG
# 1. Importamos las librerías necesarias
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 2. Separamos los nombres de las empresas (columna 0)
nombres_empresas = ESG_limpio.iloc[:, 0]  # la columna 0 es el nombre de las empresas

# 3. Seleccionamos los atributos ESG para clustering
atributos_clustering = [
    "Performance Score",
    "Sector Percentile PS",
    "Risk Score",
    "Sector Percentile RS",
    "Materiality Disclosure",
    "Exposure Flags",
    "Frameworks",
    "Completed Public Disclosures",
    "Inputs Scored",
    "Data Points"
]

X = ESG_limpio[atributos_clustering].copy()

# 4. Normalizamos con z-score
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 5. Volvemos a convertirlo a DataFrame y agregamos nombres de empresas
X_normalizado = pd.DataFrame(X_normalizado, columns=atributos_clustering)
X_normalizado.insert(0, "Empresa", nombres_empresas.values)

# %%
# NUMERO DE CLUSTERS
# Ahora que tenemos todo listo, con nombres, atributos normalizados, necesitamos fijar el número de clusters
# Para poder obtener el número de clusters idóneo, realizaremos el "método del codo".
# 1. Importamos las librerias 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Eliminamos la columna 'Empresa' para quedarnos solo con los atributos numéricos
X_numeric = X_normalizado.drop(columns=["Empresa"])

# 3. Calulamos la inercia para distintos valores de k
inertias = []
k_range = range(1, 16)

for k in k_range:
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X_numeric)
    inertias.append(modelo.inertia_)
    
# 4. Graficamos la curva para encontrar el "codo"
plt.figure(figsize=(25, 5))
sns.lineplot(x=list(k_range), y=inertias, marker='o')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia Total')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# PRUEBA 1 PARA OBTENER EL NUMERO DE CLUSTERS A USAR 
# Basados en el método del codo, parece ser que el número de clusters óptimo son k = 10
# Esto se establece en base al gráfico generado, donde k = 10, se logra observar un punto de inflexión y donde la
# variación de la inercia comienza a ser cada vez menor.  
# Entonces comenzamos a generar los clusters
# 1. Creamos el modelo con 10 clusters
k = 10 # definimos el número de clusters
modelo_kmeans = KMeans(n_clusters=k, random_state=42)

# Entrenamos el modelo usando los datos normalizados (sin la columna 'Empresa')
clusters = modelo_kmeans.fit_predict(X_numeric)

# 2. Agregamos el número de cluster como nueva columna en el DataFrame original
X_normalizado["Cluster"] = clusters 
# Mostramos las primeras 10 empresas con su cluster asignado
X_normalizado[["Empresa", "Cluster"]].head(10)

# 3. Creamos una tabla para observar la canitdad de empresas por cluster 
# Contamos cuántas empresas hay en cada cluster
resumen_clusters = X_normalizado["Cluster"].value_counts().sort_index()

# Lo convertimos en una tabla con nombre más claro
tabla_resumen = pd.DataFrame({
    "Cluster": resumen_clusters.index,
    "Cantidad de Empresas": resumen_clusters.values
}) # Observar la tabla resumen para verificar que no hayan outliers o ruido 
print(tabla_resumen)
# %%
# PRUEBA 2 PARA OBTENER EL NUMERO DE CLUSTERS A USAR
# Como podemos observar en la tabla resumen, hay 1 cluster que solo contiene 1 empresa, esta empresa la podemos
# considerar como "outlier" o "ruido", por lo que volveremos a generar los cluster pero en este caso con k = 9, 
# ya que en base al gráfico, podemos observar que el primer punto de inflexión generado es en k = 9
k = 9 #fijamos el número de clusters
modelo_kmeans = KMeans(n_clusters=k, random_state=42)

# 1. Entrenamos el modelo usando los datos normalizados (sin la columna 'Empresa')
clusters = modelo_kmeans.fit_predict(X_numeric)

# 2. Agregamos el número de cluster como nueva columna en el DataFrame original
X_normalizado["Cluster"] = clusters

# 3. Creamos una tabla para observar la canitdad de empresas por cluster 
# Contamos cuántas empresas hay en cada cluster
resumen_clusters = X_normalizado["Cluster"].value_counts().sort_index()

# Actualizamos la tabla para visualizar si ahora los clusters están bien formados
tabla_resumen = pd.DataFrame({
    "Cluster": resumen_clusters.index,
    "Cantidad de Empresas": resumen_clusters.values
})
#Si apreciamos la tabla resumen, ahora si tenemos los cluster mejor definidos, sin outliers o ruido
print(tabla_resumen)
# %%
# EVALUACIÓN ESTADÍSTICA DE LA CALIDAD DE CLUSTERING
# Luego de aplicar el método del codo, se realizaron pruebas con k = 10 y k = 9.
# En k = 10, se observó la existencia de clusters con solo una empresa, los cuales
# pueden considerarse outliers o ruido, lo que afecta la calidad del análisis.
#
# Al reducir a k = 9, los clusters se redistribuyen y cada grupo contiene
# al menos más de una empresa, lo cual sugiere una agrupación más útil y sin ruido evidente.
#
# Sin embargo, la ausencia de outliers visuales no garantiza que los clusters estén
# bien definidos estructuralmente. Por eso, se procede a calcular el coeficiente
# de silueta promedio para distintos valores de k (desde 2 hasta 9), con 1000 repeticiones
# aleatorias para cada k. Esto permite:
#
# - Evaluar la robustez del agrupamiento (promedio del coeficiente de silueta),
# - Medir su estabilidad (desviación estándar entre ejecuciones),
# - Y comparar objetivamente qué valor de k ofrece el mejor balance entre
#   separación entre clusters y utilidad interpretativa.
#
# Este análisis complementa al método del codo, permitiendo justificar de forma
# estadística la elección del número óptimo de clusters.

from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Definir rango de k a evaluar
rango_k = range(2, 10)
n_repeats = 1000

# Almacenar resultados
promedios_silueta = []
desviaciones_silueta = []

for k in rango_k:
    siluetas = []
    for seed in range(n_repeats):
        # Reordenamos datos y entrenamos KMeans con k clusters y seed distinta
        X_shuffled = shuffle(X_numeric, random_state=seed).reset_index(drop=True)
        modelo = KMeans(n_clusters=k, random_state=seed)
        etiquetas = modelo.fit_predict(X_shuffled)
        score = silhouette_score(X_shuffled, etiquetas)
        siluetas.append(score)

    # Guardamos estadísticas
    promedios_silueta.append(np.mean(siluetas))
    desviaciones_silueta.append(np.std(siluetas))
    print(f"k={k}: Promedio silueta = {np.mean(siluetas):.4f} | Std = {np.std(siluetas):.4f}")

# Graficamos los promedios
plt.figure(figsize=(10, 5))
plt.errorbar(rango_k, promedios_silueta, yerr=desviaciones_silueta, fmt='-o', capsize=5, color='darkorange')
plt.title("Promedio y desviación del coeficiente de silueta por k")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette promedio")
plt.grid(True)
plt.show()

#%%
from tabulate import tabulate
import matplotlib.pyplot as plt

# Tabla Resumen de los Coeficientes Silueta promedio para valores de k de 2 a 9 
tabla_silueta = []
for k, sil, std in zip(range(2, 10), promedios_silueta, desviaciones_silueta):
    tabla_silueta.append([f"k={k}", f"{sil:.4f}", f"{std:.4f}"])

# Convertir en texto tabulado 
print(tabulate(tabla_silueta, headers=["K", "Promedio silueta", "Desviación estándar"], tablefmt="grid"))

# Crear imagen con matplotlib
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')
table = ax.table(
    cellText=tabla_silueta,
    colLabels=["K", "Promedio silueta", "Desviación estándar"],
    loc='center',
    cellLoc='center',
    colLoc='center'
)
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Guardar como imagen
plt.savefig("tabla_silueta_promedios.png", bbox_inches='tight', dpi=300)
plt.show()


# %%
# DEFINICIÓN DEL NÚMERO DE CLUSTERS: ELECCIÓN DE K

# Para determinar el número óptimo de clusters, se aplicaron dos enfoques complementarios:
#
# a) Método del Codo:
#    Se graficó la inercia (suma de distancias cuadradas dentro de los clusters) para valores de k entre 1 y 16.
#    El gráfico permitió identificar puntos de inflexión, destacando especialmente los valores de k = 9 y k = 10,
#    donde la ganancia marginal en reducción de inercia comienza a estabilizarse.
#
# b) Evaluación del Coeficiente de Silueta Promedio:
#    A continuación, se calculó el coeficiente de silueta promedio para valores de k entre 2 y 10.
#    Este indicador mide qué tan bien separados están los clusters, considerando tanto su cohesión interna
#    como su separación respecto a otros grupos.
#    Para cada valor de k, se realizaron 1000 ejecuciones aleatorias (shuffle + inicialización distinta) con KMeans,
#    evaluando así la robustez y estabilidad del agrupamiento.
#
# Resultados:
# - k = 2 obtuvo el valor de silueta más alto (~0.33), pero genera una segmentación demasiado general (binaria).
# - A partir de k = 4, se observa una caída razonable en el coeficiente, pero con estabilidad (std < 0.008).
# - En k > 6, la calidad del clustering disminuye significativamente y los grupos se vuelven poco definidos.
#
# Finalmente, se decidió utilizar k = 4 como número óptimo de clusters por las siguientes razones:
# - Ofrece un buen equilibrio entre interpretabilidad y separación estructural.
# - Permite identificar distintos perfiles ESG (líderes, rezagados, balanceados, etc.).
# - Conserva una estructura robusta (coef. silueta promedio > 0.20 con baja desviación).
# - Evita clusters vacíos o con una sola empresa, como se observó en valores más altos de k.
#
# Esta decisión proporciona una segmentación ESG rica en matices, útil tanto para análisis estratégicos como para vincular
# los resultados a industrias, disclosure, riesgo y desempeño en sostenibilidad.
    # %%
# CLUSTERING FINAL CON k = 4 (seleccionado tras evaluación estadística)

# Definimos el número final de clusters
k = 4
modelo_kmeans = KMeans(n_clusters=k, random_state=42)

# Entrenamos el modelo con los datos normalizados
clusters = modelo_kmeans.fit_predict(X_numeric)

# Asignamos el cluster a cada empresa
X_normalizado["Cluster"] = clusters

# Creamos resumen de cantidad de empresas por cluster
resumen_clusters = X_normalizado["Cluster"].value_counts().sort_index()
tabla_resumen = pd.DataFrame({
    "Cluster": resumen_clusters.index,
    "Cantidad de Empresas": resumen_clusters.values
})
print(tabla_resumen)
# %% Comparación de métodos de clustering: KMeans vs GMM vs Clustering Jerárquico

from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns

# Configuración
k = 4
# Método 2: Gaussian Mixture Model
modelo_gmm = GaussianMixture(n_components=k, random_state=42)
labels_gmm = modelo_gmm.fit_predict(X_numeric)
silueta_gmm = silhouette_score(X_numeric, labels_gmm)

# Método 3: Clustering Jerárquico (agglomerative)
modelo_jerarquico = AgglomerativeClustering(n_clusters=k)
labels_jerarquico = modelo_jerarquico.fit_predict(X_numeric)
silueta_jerarquico = silhouette_score(X_numeric, labels_jerarquico)

# Mostrar resultados comparativos
print("Comparación del Coeficiente de Silueta para k = 4:")
print(f"K-Means (k=4): {promedios_silueta[2]:.4f}")
print(f"Gaussian Mixture: {silueta_gmm:.4f}")
print(f"Jerárquico: {silueta_jerarquico:.4f}")

# Visualización básica en 2D con PCA (no sé que tan factible sea incluirlo debido a que el modelo trabaja con 10 dimensiones y aquí solo se presentan 2)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_numeric)

plt.figure(figsize=(16, 4))

# KMeans
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=clusters, palette="tab10", s=30)
plt.title("K-Means Clustering")

# GMM
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels_gmm, palette="tab10", s=30)
plt.title("Gaussian Mixture Model")

# Jerárquico
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels_jerarquico, palette="tab10", s=30)
plt.title("Clustering Jerárquico")

plt.tight_layout()
plt.show()

#%% Tabla para comparar modelos
# Valores obtenidos 
# Crear tabla resumen
tabla_modelos = [
    ["K-Means", f"{promedios_silueta[2]:.4f}"],
    ["Gaussian Mixture", f"{silueta_gmm:.4f}"],
    ["Clustering Jerárquico", f"{silueta_jerarquico:.4f}"]
]

# Mostrar como tabla en texto 
print(tabulate(tabla_modelos, headers=["Modelo", "Coef. Silueta"], tablefmt="grid"))

# Crear imagen de tabla con matplotlib
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis('off')

table = ax.table(
    cellText=tabla_modelos,
    colLabels=["Modelo", "Coef. Silueta"],
    cellLoc='center',
    loc='center'
)

table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Guardar como imagen
plt.savefig("tabla_silueta_modelos.png", bbox_inches='tight', dpi=300)
plt.show()


# %%
# DESNORMALIZAMOS LOS CENTROIDES PARA INTERPRETACIÓN

centroides_z = modelo_kmeans.cluster_centers_
centroides_originales = scaler.inverse_transform(centroides_z)
centroides_desnormalizados = pd.DataFrame(centroides_originales, columns=atributos_clustering)
centroides_desnormalizados["Cluster"] = centroides_desnormalizados.index

# %%
# UNIMOS LOS CLUSTERS AL DATAFRAME ORIGINAL

# Añadimos el ticker al DataFrame X_normalizado para facilitar el merge
X_normalizado["ticker"] = ESG_limpio["ticker"].values

# Hacemos merge en vez de asignar por posición, así evitamos duplicados
ESG_clusters = ESG_limpio.merge(
    X_normalizado[["ticker", "Cluster"]],
    on="ticker",
    how="left"
)

# Ordenamos por cluster (opcional)
ESG_clusters = ESG_clusters.sort_values(by="Cluster").reset_index(drop=True)


# %%
# Corregir el nombre de las industria mal escritas
ESG_clusters["industry"] = ESG_clusters["industry"].replace(
    {"Aerospace & Defense": "Aerospace and Defense"})
ESG_clusters["industry"] = ESG_clusters["industry"].replace(
    {"Hotels, Restaurants & Leisure": "Hotels Restaurants and Leisure"})
ESG_clusters["industry"] = ESG_clusters["industry"].replace(
    {"Metals & Mining": "Metals and Mining"})
# %%
# Calcular la tabla original (absolutos)
tabla_industrias = ESG_clusters.groupby(["Cluster", "industry"]).size().unstack(fill_value=0)

# Calcular porcentajes por fila (cluster)
tabla_porcentajes = tabla_industrias.div(tabla_industrias.sum(axis=1), axis=0) * 100

# Crear tabla combinada con formato "n (x%)"
tabla_combinada = tabla_industrias.astype(str) + " (" + tabla_porcentajes.round(1).astype(str) + "%)"

# Transponer para mejor visualización
tabla_industrias_por_cluster = tabla_combinada.transpose()

# Mostrar resultado
print(tabla_industrias_por_cluster)
# Convertir la tabla a formato compatible
datos_tabla = tabla_industrias_por_cluster.values.tolist()
etiquetas_filas = tabla_industrias_por_cluster.index.tolist()

# Mostrar resultado
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')
tabla = ax.table(
    cellText=datos_tabla,
    rowLabels=etiquetas_filas,
    colLabels=["0", "1", "2", "3"],
    loc='center',
    cellLoc='center',
    colLoc='center'
)

tabla.scale(1, 2)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

# Guardar como imagen
plt.savefig("tabla_industrias_por_cluster.png", bbox_inches='tight', dpi=300)
plt.show()

# %%
"""
# Este bloque obtiene el beta (5Y Monthly) directamente desde Yahoo Finance,
# utilizando los tickers de cada empresa. Fue ejecutado en **mayo de 2025**,
# coincidiendo con la fecha de corte de ESG Book (08-05-2025), por lo tanto
# los valores de beta corresponden al mismo horizonte temporal que los datos ESG.
# Crear una función para obtener el beta de un ticker
import yfinance as yf
def obtener_beta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("beta", None)
    except:
        return None

# Aplicar la función a la columna de tickers
ESG_clusters["Beta"] = ESG_clusters["ticker"].apply(obtener_beta)
# Este archivo contiene los betas descargados en la misma fecha que los datos ESG,
# y se debe utilizar en análisis futuros para mantener la integridad del horizonte temporal.

ESG_clusters[["name", "Cluster", "ticker", "Beta"]].to_excel("betas_empresas_2025-05-08.xlsx", index=False)
"""

# %%
# CARGAR LOS BETAS GUARDADOS DESDE ARCHIVO EXCEL (MAYO 2025)
# Esto asegura que el análisis utilice los mismos betas que se descargaron a la fecha del ESG Book.

betas_guardados = pd.read_excel("betas_empresas_2025-05-08.xlsx")

# %%
# CALCULAR EL PROMEDIO DE BETA POR CLUSTER USANDO LOS DATOS CARGADOS

promedio_beta = betas_guardados.groupby("Cluster")["Beta"].mean()

# Crear nueva columna en el DataFrame de centroides con el promedio por cluster
centroides_desnormalizados["Beta Promedio"] = centroides_desnormalizados["Cluster"].map(promedio_beta)
#%% CALCULO DEL BETA SEGÚN MODELOS FF3 y CAPM

#Paso 1: Librerías y configuración
import pandas as pd
import yfinance as yf
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
# Fechas del análisis
start_date = "2020-04-01" #se considera un mes antes del periodo contemplado porque al cálcular los retornos, el primero se elimina.
end_date = "2025-05-01"

#%% Paso 3: Extraer tickers únicos
tickers = ESG_clusters["ticker"].dropna().unique().tolist()
historical_data_FF3 = {}
fallidos_FF3 = []
exitosos_FF3 = []

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=False, progress=False)

        if not data.empty and 'Adj Close' in data.columns:
            historical_data_FF3[ticker] = data
            exitosos_FF3.append(ticker)
        else:
            print(f" Sin columnas esperadas para {ticker}")
            fallidos_FF3.append(ticker)
    except Exception as e:
        print(f" Error con {ticker}: {e}")
        fallidos_FF3.append(ticker)
print(f"\n Descarga completada. {len(exitosos_FF3)} exitosos | {len(fallidos_FF3)} fallidos")
#%% Paso 4: Calculamos los retornos mensuales para cada empresa
retornos_por_empresa = {}
for ticker, df in historical_data_FF3.items():
    try:
        precios = df['Adj Close']
        retornos = precios.pct_change().dropna()
        retornos_por_empresa[ticker] = retornos  
    except Exception as e:
        print(f"❗ Error al calcular retorno para {ticker}: {e}")
#%% Paso 5: Cargamos el archivo con los datos FF3
FF3 = pd.read_csv("FF3.csv", sep = ";")
#%% Paso 6: Ajustamos el dataframe del modelo FF3
FF3.rename(columns={FF3.columns[0]: 'Date'}, inplace=True)
FF3.rename(columns={FF3.columns[1]: 'MKT'}, inplace=True)
FF3.rename(columns={FF3.columns[4]: 'RiskFree'}, inplace=True)
# Convertir columna a datetime (último día del mes)
FF3['Date'] = pd.to_datetime(FF3['Date'], format="%Y%m") + pd.offsets.MonthEnd(0)
#Establecer 'Date' como índice (opcional, si tus retornos también lo usan como índice)
FF3.set_index('Date', inplace=True)
# Verificar resultado
print(FF3.head())
#%% Paso 7: Hacemos el merge de los factores para cada uno de los tickers y sus respectivos retornos según cada período
# Creamos nuevo diccionario con los retornos + factores FF3 por ticker
retornos_con_factores = {}

for ticker, df_retornos in retornos_por_empresa.items():
    try:
        # Aseguramos que el índice es tipo datetime mensual
        df_retornos.index = pd.to_datetime(df_retornos.index)
        df_retornos.index = df_retornos.index.to_period('M').to_timestamp('M')  # Fin de mes

        # Hacemos merge con los factores
        combinado = df_retornos.merge(FF3, left_index=True, right_index=True, how='inner')

        # Guardamos el resultado
        retornos_con_factores[ticker] = combinado

    except Exception as e:
        print(f"Error al hacer merge con {ticker}: {e}")

# %% Paso 8: Calculamos primero el modelo de CAPM para cada uno de las empresas
import statsmodels.api as sm
betas_capm = {}
rsquared_capm = {}
for ticker, df in retornos_con_factores.items():
    try:
        retorno_col = ticker.upper()
        columnas_necesarias = ['MKT', 'RiskFree', retorno_col]

        # Verificar existencia de columnas
        if not all(col in df.columns for col in columnas_necesarias):
            print(f"⚠️ Columnas faltantes para {ticker}: {[col for col in columnas_necesarias if col not in df.columns]}")
            continue

        # Filtrar filas completas
        df_clean = df[columnas_necesarias].dropna()
        if len(df_clean) < 12:
            print(f"⚠️ Muy pocos datos para {ticker} ({len(df_clean)} filas), se omite")
            continue

        # Convertir porcentajes a proporciones
        df_clean[['MKT']] = df_clean[['MKT']] / 100
        df_clean['RiskFree'] = df_clean['RiskFree'] / 100

        # Variables del modelo CAPM
        x_CAPM = sm.add_constant(df_clean[['MKT']])
        y_CAPM = df_clean[retorno_col] - df_clean['RiskFree']

        # Regresión
        model = sm.OLS(y_CAPM, x_CAPM).fit()
        betas_capm[ticker] = model.params  # incluye constante y beta
        rsquared_capm[ticker] = model.rsquared

    except Exception as e:
        print(f"❌ Error en regresión para {ticker}: {e}")
#%% Paso 9: Guardamos solo los betas obtenidos del modelo CAPM para cada una de las empresas
#Creamos un nuevo diccionario con solo el beta de mercado (coeficiente 'MKT')
betas_MKT_CAPM = {
    ticker: coeficientes["MKT"]
    for ticker, coeficientes in betas_capm.items()
    if "MKT" in coeficientes
}

# Lo convertimos en DataFrame
betas_limpios_capm = pd.DataFrame.from_dict(betas_MKT_CAPM, orient="index", columns=["Beta_CAPM_MKT"])

# Convertimos el índice a columna
betas_limpios_capm = betas_limpios_capm.reset_index().rename(columns={"index": "Ticker"})

# Opcional: mostramos los primeros valores
betas_limpios_capm.head()
#%% Paso 10: Calculamos el modelo de FF3 para cada una de las empresas 
betas_ff3 = {}
rsquared_ff3 = {}
for ticker, df in retornos_con_factores.items():
    try:
        # Aseguramos que el nombre del retorno esté en mayúscula
        retorno_col = ticker.upper()
        columnas_necesarias = ['MKT', 'SMB', 'HML', 'RiskFree', retorno_col]

        # Verificar que existan las columnas necesarias
        if not all(col in df.columns for col in columnas_necesarias):
            print(f"⚠️ Columnas faltantes para {ticker}: {[col for col in columnas_necesarias if col not in df.columns]}")
            continue

        # Filtrar filas completas
        df_clean2 = df[columnas_necesarias].dropna()
        if len(df_clean2) < 12:
            print(f"⚠️ Muy pocos datos para {ticker} ({len(df_clean)} filas), se omite")
            continue
        # Convertir factores Fama-French de porcentaje a proporción
        df_clean2[['MKT', 'SMB', 'HML']] = df_clean2[['MKT', 'SMB', 'HML']] / 100


        # Variables del modelo
        x = sm.add_constant(df_clean2[['MKT', 'SMB', 'HML']])
        y = df_clean2[ticker.upper()] - df_clean2["RiskFree"] / 100

        # Regresión
        model = sm.OLS(y, x).fit()
        betas_ff3[ticker] = model.params
        rsquared_ff3[ticker] = model.rsquared

    except Exception as e:
        print(f"❌ Error en regresión para {ticker}: {e}")
#%% Paso 11: Guardamos solo los betas obtenidos del modelo FF3 para cada una de las empresas
#Creamos un nuevo diccionario con solo el beta de mercado (coeficiente 'MKT')
betas_mkt = {
    ticker: coeficientes["MKT"]
    for ticker, coeficientes in betas_ff3.items()
    if "MKT" in coeficientes
}

# Lo convertimos en DataFrame
betas_limpios_FF3 = pd.DataFrame.from_dict(betas_mkt, orient="index", columns=["Beta_FF3_MKT"])
betas_limpios_FF3 = betas_limpios_FF3.reset_index().rename(columns={"index": "Ticker"})
# Opcional: mostramos los primeros valores
betas_limpios_FF3.head()
#%% Paso 12: Generamos un Dataframe con los betas de Yfinance, CAPM y FF3 para compararlos posteriormente
# COPIAMOS LOS DATAFRAMES BASE
df_ff3 = betas_limpios_FF3.copy()
df_capm = betas_limpios_capm.copy()
df_yf = betas_guardados.copy()

# ASEGURAMOS EL MISMO FORMATO EN LOS TICKERS (minúscula)
df_ff3["Ticker"] = df_ff3["Ticker"].astype(str).str.lower()
df_capm["Ticker"] = df_capm["Ticker"].astype(str).str.lower()
df_yf["ticker"] = df_yf["ticker"].astype(str).str.lower()

# RENOMBRAMOS COLUMNAS PARA HACER MERGE
df_ff3 = df_ff3.rename(columns={"Ticker": "ticker", "Beta_FF3_MKT": "Beta_FF3"})
df_capm = df_capm.rename(columns={"Ticker": "ticker", "Beta_CAPM_MKT": "Beta_CAPM"})
df_yf = df_yf.rename(columns={"Beta": "Beta_YF"})

# CONVERTIMOS R2 A DATAFRAMES
df_r2_ff3 = pd.DataFrame.from_dict(rsquared_ff3, orient='index', columns=["R2_FF3"])
df_r2_ff3.index.name = "ticker"
df_r2_ff3 = df_r2_ff3.reset_index()

df_r2_capm = pd.DataFrame.from_dict(rsquared_capm, orient='index', columns=["R2_CAPM"])
df_r2_capm.index.name = "ticker"
df_r2_capm = df_r2_capm.reset_index()

# MERGE DE TODOS LOS DATOS
comparacion_betas = df_yf.copy()
comparacion_betas = comparacion_betas.merge(df_capm, on="ticker", how="left")
comparacion_betas = comparacion_betas.merge(df_ff3, on="ticker", how="left")
comparacion_betas = comparacion_betas.merge(df_r2_capm, on="ticker", how="left")
comparacion_betas = comparacion_betas.merge(df_r2_ff3, on="ticker", how="left")
# CREAMOS UN DF PARA COMPARAR LOS PROMEDIOS DE BETA Y R2 SEGÚN CLUSTER Y MODELO

# Calcular promedios por cluster
betas_originales = centroides_desnormalizados[["Cluster", "Beta Promedio"]]
promedio_betaCAPM = comparacion_betas.groupby("Cluster")["Beta_CAPM"].mean()
promedio_betaFF3 = comparacion_betas.groupby("Cluster")["Beta_FF3"].mean()
promedio_R2_CAPM = comparacion_betas.groupby("Cluster")["R2_CAPM"].mean()
promedio_R2_FF3 = comparacion_betas.groupby("Cluster")["R2_FF3"].mean()

# Unirlos en un solo DataFrame
df_cluster = pd.concat(
    [betas_originales, promedio_betaCAPM, promedio_betaFF3, promedio_R2_CAPM, promedio_R2_FF3],
    axis=1
)
df_cluster.columns = ["Cluster", "Beta Original", "Beta Promedio CAPM", "Beta Promedio FF3", "R2 Promedio CAPM", "R2 Promedio FF3"]

df_cluster.head()

#%% Guardar df_cluster como archivo Excel
ruta_guardado = "FF3_y_CAPM_cluster.xlsx"
df_cluster.to_excel(ruta_guardado, index=True)
print(f"✅ Archivo guardado como: {ruta_guardado}")
'''
#%% Cargamos el archivo con la comparación entre modelo CAPM y FF3
df_cluster = pd.read_excel("FF3_y_CAPM_cluster.xlsx")

#%%
# Crear etiquetas interpretativas manualmente en base a los centroides observados

# Asumimos que ya tenemos centroides_desnormalizados con la siguiente estructura:
# Columnas clave: 'Sector Percentile PS', 'Sector Percentile RS', 'Frameworks', 'Completed Public Disclosures'

# Clasificamos con etiquetas interpretativas
def clasificar_cluster(row):
    beta = row["Beta Promedio"]
    ps = row["Performance Score"]
    disclosures = row["Completed Public Disclosures"]
    frameworks = row["Frameworks"]
    
    if beta < 1.0:
        if ps >= 60 and disclosures >= 30 and frameworks >= 7:
            return "Líderes ESG con bajo riesgo financiero"
        else:
            return "Desempeño ESG aceptable con bajo riesgo financiero"
    elif 1.0 <= beta <= 1.2:
        return "Desempeño ESG intermedio con riesgo financiero moderado"
    else:
        return "Rezago ESG con alto riesgo financiero"


# Crear nueva columna en el DataFrame
centroides_desnormalizados["Perfil Interpretativo"] = centroides_desnormalizados.apply(clasificar_cluster, axis=1)

#%%
from tabulate import tabulate
import matplotlib.pyplot as plt

# Selección de columnas
tabla_clusters = centroides_desnormalizados[["Cluster", "Beta Promedio", "Perfil Interpretativo"]].copy()
tabla_clusters["Beta Promedio"] = tabla_clusters["Beta Promedio"].round(3)
tabla_clusters = tabla_clusters.sort_values("Cluster")

# Convertir a lista
contenido_tabla = tabla_clusters.values.tolist()

# Crear figura con más ancho para texto largo
fig, ax = plt.subplots(figsize=(12, 4))  # puedes ajustar el tamaño si es necesario
ax.axis('off')

tabla = ax.table(cellText=contenido_tabla,
                 colLabels=["Cluster", "Beta Promedio", "Perfil Interpretativo"],
                 loc='center',
                 cellLoc='left',   # alinear el texto a la izquierda
                 colLoc='center')

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.0, 1.5)

# Ajuste de anchos manual (opcional pero útil)
for i in range(len(contenido_tabla) + 1):  # +1 incluye encabezado
    tabla.auto_set_column_width(i)

# Guardar como imagen
plt.savefig("tabla_clusters_beta_perfil_ajustada.png", bbox_inches='tight', dpi=300)
plt.show()

#%%
from tabulate import tabulate

# Renombrar columnas para mayor claridad visual
tabla_centroides = centroides_desnormalizados.rename(columns={
    'Performance Score': 'Perf.',
    'Sector Percentile PS': 'Pct. PS',
    'Risk Score': 'Risk',
    'Sector Percentile RS': 'Pct. RS',
    'Materiality Disclosure': 'Mat. Disc.',
    'Exposure Flags': 'Exp.',
    'Frameworks': 'Frm.',
    'Completed Public Disclosures': 'Pub. Disc.',
    'Inputs Scored': 'Inputs',
    'Data Points': 'Pts.',
    'Perfil Interpretativo': 'Perfil',
    'Beta Promedio': 'Beta μ'
})

# Redondear para que se vea limpio
tabla_centroides = tabla_centroides.round(3)

# Desacoplar consola para ver tabla ajustada 
print(tabulate(tabla_centroides, headers='keys', tablefmt='grid', showindex=True))


# %%
# Añadimos una columna con los betas de cada empresa al DataFrame ESG_clusters
ESG_clusters = ESG_clusters.merge(
    betas_guardados[["name", "Beta"]],
    on="name",  # O usa 'ticker' si esa columna es la que coincide
    how="left")
# %%
atributos_esg = [
    "Performance Score", "Sector Percentile PS", "Risk Score", "Sector Percentile RS",
    "Materiality Disclosure", "Exposure Flags", "Frameworks",
    "Completed Public Disclosures", "Inputs Scored", "Data Points"
]
    
# Calcular promedios ESG y beta por industria
resumen_promedios = ESG_clusters.groupby("industry").agg(
    Beta_Promedio=("Beta", "mean"),
    **{col: (col, "mean") for col in atributos_esg})
# Unir tablas por índice (industry)
tabla_completa = tabla_industrias_por_cluster.join(resumen_promedios, how="left")

# Ver resultado
print(tabla_completa)

# (Opcional) Guardar en Excel
tabla_completa.to_excel("Distribucion_Industria_Cluster_ESG_Beta.xlsx")

# %%# Crear boxplot por los betas de cada cluster  

# Filtrar datos válidos
df_betas = ESG_clusters[['Beta', 'Cluster']].dropna()

# Mapear nombres personalizados para la leyenda (hue)
cluster_labels = {
    0: '0 - Desempeño ESG aceptable',
    1: '1 - Desempeño ESG intermedio',
    2: '2 - Rezagados ESG',
    3: '3 - Líderes ESG'
}
df_betas['Cluster_leyenda'] = df_betas['Cluster'].map(cluster_labels)

# Configurar estilo
sns.set(style="whitegrid")

# Crear boxplot (Cluster en eje X como número, leyenda descriptiva en hue)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_betas, x="Cluster", y="Beta", hue="Cluster_leyenda", palette="pastel", orient="v", dodge=False)
plt.title("Distribución del Beta Patrimonial por Clúster", fontsize=14)
plt.xlabel("Clúster")
plt.ylabel("Beta Patrimonial")

# Línea de referencia para beta de mercado = 1
plt.ylim(-1, 6)
plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Beta de mercado (β = 1)')

# Mostrar leyenda con nombres y línea de mercado
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#%% para hacer el Heatmap de atributos ESG por cluster
centroides_heatmap = pd.DataFrame(centroides_z)
#%% Heatmap de atributos ESG por cluster
centroides_heatmap.columns = atributos_esg
centroides_heatmap.index.name = "Clúster"

plt.figure(figsize=(12, 6))
sns.heatmap(centroides_heatmap, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
# Títulos
plt.title("Mapa de Calor de atributos ESG normalizados por Clúster", fontsize=14)
plt.xlabel("Atributos ESG")
plt.ylabel("Clúster")
plt.tight_layout()
plt.show()

#%%
ESG_limpio.to_excel("ESG LIMPIO.xlsx")
#%% CÁLCULO DEL RATIO DE TREYNOR    
'''
# Como ya obtuvimos los retornos de cada empresa para los modelos de CAPM y FF3, reutilizamos los mismos retornos para calcular
# los retornos promedio mensuales y así obtener el ratio de Treynor.
# Cargamos los retornos mensuales por empresa en un nuevo diccionario y calculamos el promedio para cada uno
df_retornos_mensuales = pd.DataFrame({
    'ticker': list(retornos_por_empresa.keys()),
    'Retorno Mensual Promedio': [df.mean().values[0] for df in retornos_por_empresa.values()]
})

df_retornos_mensuales.set_index("ticker", inplace=True)
df_retornos_mensuales.head()
df_retornos_mensuales.to_excel("Retornos Mensuales Promedio.xlsx") #Guardamos los retornos mensuales promedio en un archivo xlsx para reproducibilidad
'''
#%% Cargar retornos mensuales promedio desde archivo guardado
# Cargamos los retornos mensuales guardados para evitar cambios en el análisis
df_retornos_mensuales = pd.read_excel("Retornos Mensuales Promedio.xlsx")

# Limpiar nombres de columnas: quitar espacios y pasar a minúsculas
df_retornos_mensuales.columns = df_retornos_mensuales.columns.str.strip().str.lower()

# Renombrar a formato correcto para el merge
df_retornos_mensuales.rename(columns={
    "ticker": "ticker",  # por si tiene otro nombre
    "retorno mensual promedio": "Retorno Mensual Promedio"
}, inplace=True)
#%% Paso 6: Incluimos el retorno promedio calculado para cada una de las empresas del dataframe
ESG_clusters = ESG_clusters.merge(df_retornos_mensuales, how='left', on='ticker')
ESG_clusters.to_excel("ESG CLUSTERS.xlsx") #actualizar el archivo excel con el Treynor Ratio y retornos promedios x empresa
#%% Paso 7: Calculamos el Ratio de Treynor para ver cuanto retorno tiene cada empresa sobre cada unidad de su riesgo sistemático
# Además se calcula el promedio por cada cluster y se cruza con las interpretaciones de los clusters.
# Supuesto: tasa libre de riesgo anual (como antes)
r_f_anual = 0.03 
r_f_mensual = r_f_anual / 12  # Ajuste para retorno mensual

# Calcular el Ratio de Treynor para cada empresa
ESG_clusters["Treynor Ratio"] = (ESG_clusters["Retorno Mensual Promedio"] - r_f_mensual) / ESG_clusters["Beta"]

# Calcular el promedio del Treynor Ratio por cluster
treynor_promedios = ESG_clusters.groupby("Cluster")["Treynor Ratio"].mean().reset_index()
treynor_promedios.columns = ["Cluster", "Treynor Ratio Promedio"]

# Unir al DataFrame de centroides
centroides_desnormalizados = centroides_desnormalizados.merge(treynor_promedios, how="left", on="Cluster")
#%% Para verificar la dispersión de los Ratios de Treynor de cada cluster
# Creamos una copia del dataframe para trabajar
df_treynor = ESG_clusters[['ticker', 'Cluster', 'Treynor Ratio']].dropna().copy()
df_treynor['Cluster'] = df_treynor['Cluster'].astype(int)
# Estadísticas descriptivas por cluster
stats_clusters = df_treynor.groupby('Cluster')['Treynor Ratio'].agg(
    Media='mean',
    Mediana='median',
    Desviacion='std',
    Minimo='min',
    Maximo='max',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
    Conteo='count'
).reset_index()
print(stats_clusters)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %% Preparar datos con leyenda personalizada
# Mapear nombres explicativos para la leyenda
cluster_labels = {
    0: '0 - Desempeño ESG aceptable',
    1: '1 - Desempeño ESG intermedio',
    2: '2 - Rezagados ESG',
    3: '3 - Líderes ESG'
}
ESG_clusters['Cluster_leyenda'] = ESG_clusters['Cluster'].map(cluster_labels)

# Gráfico 1: Dispersión de Treynor Ratio por empresa en cada clúster (con leyenda)
plt.figure(figsize=(10, 6))
sns.stripplot(data=ESG_clusters, x='Cluster', y='Treynor Ratio',
              hue='Cluster_leyenda', jitter=True, alpha=0.7, palette='viridis', dodge=False)
plt.title('Dispersión del Treynor Ratio por Clúster')
plt.xlabel('Clúster')
plt.ylabel('Treynor Ratio')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Gráfico 2: Boxplot del Treynor Ratio por clúster (con leyenda)
plt.figure(figsize=(10, 6))
sns.boxplot(data=ESG_clusters, x='Cluster', y='Treynor Ratio',
            hue='Cluster_leyenda', palette="Set2", dodge=False)
plt.title('Distribución del Treynor Ratio por Clúster (Boxplot)')
plt.xlabel('Clúster')
plt.ylabel('Treynor Ratio')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


#%% PARA IDENTIFICAR OUTLIERS DENTRO DE CADA CLUSTER
# Aplicar eliminación de outliers por IQR a cada cluster individualmente
# y calcular el nuevo promedio del Treynor Ratio para comparar con el original

clusters = ESG_clusters['Cluster'].unique()
print("Clusters identificados:", clusters)

# Crear lista para almacenar promedios antes y después
resultados = []

#%% 
for cluster_id in clusters:
    print(f"\n--- Análisis del Cluster {cluster_id} ---")

    # Filtrar empresas del cluster actual
    cluster_df = ESG_clusters[ESG_clusters['Cluster'] == cluster_id].copy()

    # Calcular IQR y límites
    Q1 = cluster_df['Treynor Ratio'].quantile(0.25)
    Q3 = cluster_df['Treynor Ratio'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"Límite inferior: {limite_inferior:.4f}, Límite superior: {limite_superior:.4f}")

    # Detectar outliers
    outliers = cluster_df[
        (cluster_df['Treynor Ratio'] < limite_inferior) |
        (cluster_df['Treynor Ratio'] > limite_superior)
    ]

    # Mostrar empresas outliers 
    print(f"Empresas outliers en cluster {cluster_id}:")
    print(outliers[['ticker', 'Treynor Ratio']].to_string(index=False))

    # Eliminar outliers
    cluster_filtrado = cluster_df[
        (cluster_df['Treynor Ratio'] >= limite_inferior) &
        (cluster_df['Treynor Ratio'] <= limite_superior)
    ]

    # Calcular promedios
    promedio_original = cluster_df['Treynor Ratio'].mean()
    promedio_filtrado = cluster_filtrado['Treynor Ratio'].mean()

    print(f"Promedio original: {promedio_original:.6f}")
    print(f"Promedio sin outliers: {promedio_filtrado:.6f}")

    # Guardar resultados
    resultados.append({
        "Cluster": cluster_id,
        "Promedio Original": promedio_original,
        "Promedio Filtrado": promedio_filtrado,
        "Outliers Detectados": len(outliers)
    })
#%% 
# Mostrar resumen final de resultados por cluster
df_resultados = pd.DataFrame(resultados)

print("\nResumen de promedio del Treynor Ratio antes y después de eliminar outliers:")
print(df_resultados.to_string(index=False))
#%%
'''
# Asegurarse de que los nombres de columna coincidan
df_resultados.rename(columns={"Promedio Filtrado": "Treynor Ratio Promedio"}, inplace=True)

# Actualizar valores en centroides_desnormalizados según el cluster
for i, row in df_resultados.iterrows():
    cluster_id = row["Cluster"]
    nuevo_promedio = row["Treynor Ratio Promedio"]

    centroides_desnormalizados.loc[
        centroides_desnormalizados["Cluster"] == cluster_id,
        "Treynor Ratio Promedio"
    ] = nuevo_promedio

# Verificación rápida
print("\nCentroides actualizados con Treynor Ratio Promedio sin outliers:")
print(centroides_desnormalizados[["Cluster", "Treynor Ratio Promedio"]].to_string(index=False))
'''
#%% CÁLCULO Y CONSOLIDACIÓN DE RATIOS DE TREYNOR Y BETAS POR CLUSTER

# 1. Retornos promedio por cluster 
df_retornos_mensuales = df_retornos_mensuales.merge(
    ESG_clusters[["ticker", "Cluster"]],
    on="ticker",
    how="left"
)
Retornos_por_cluster = df_retornos_mensuales.groupby("Cluster")["Retorno Mensual Promedio"].mean()

# 2. Asegurar que betas estén en Series indexadas por Cluster
betas_yf = betas_guardados.groupby("Cluster")["Beta"].mean()
betas_capm = df_cluster.set_index("Cluster")["Beta Promedio CAPM"]
betas_ff3 = df_cluster.set_index("Cluster")["Beta Promedio FF3"]

# 3. Alinear índices
for s in [Retornos_por_cluster, betas_yf, betas_capm, betas_ff3]:
    s.index = s.index.astype(int)

# 4. Calcular Treynor Ratio para cada modelo
Treynor_yf   = (Retornos_por_cluster - r_f_mensual) / betas_yf
Treynor_capm = (Retornos_por_cluster - r_f_mensual) / betas_capm
Treynor_ff3  = (Retornos_por_cluster - r_f_mensual) / betas_ff3

# 5. Construir DataFrame final consolidado
df_final_treynor = pd.DataFrame({
    "Retorno Promedio Mensual": Retornos_por_cluster,
    "Beta YF": betas_yf,
    "Beta CAPM": betas_capm,
    "Beta FF3": betas_ff3,
    "Treynor YF": Treynor_yf,
    "Treynor CAPM": Treynor_capm,
    "Treynor FF3": Treynor_ff3
}).reset_index()


#%% Exportar df_cluster como imagen
df_cluster_fmt = df_cluster.copy()
df_cluster_fmt["Cluster"] = df_cluster_fmt["Cluster"].astype(int)

for col in df_cluster_fmt.columns:
    if col != "Cluster":
        df_cluster_fmt[col] = df_cluster_fmt[col].round(7)

datos = df_cluster_fmt.values.tolist()
columnas = df_cluster_fmt.columns.tolist()

fig, ax = plt.subplots(figsize=(9, 2.5))
ax.axis('off')
tabla = ax.table(
    cellText=datos,
    colLabels=columnas,
    loc='center',
    cellLoc='center'
)

tabla.auto_set_column_width(col=list(range(len(columnas))))  # Ajuste al contenido
tabla.scale(1, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

plt.savefig("tabla_df_cluster.png", bbox_inches='tight', dpi=300)
plt.show()

#%% Exportar df_final_treynor como imagen
df_treynor_fmt = df_final_treynor.copy()
df_treynor_fmt["Cluster"] = df_treynor_fmt["Cluster"].astype(int)

for col in df_treynor_fmt.columns:
    if col != "Cluster":
        df_treynor_fmt[col] = df_treynor_fmt[col].round(7)

datos = df_treynor_fmt.values.tolist()
columnas = df_treynor_fmt.columns.tolist()

fig, ax = plt.subplots(figsize=(12, 2.8))  # Puedes ajustar el alto si se ve muy apretado
ax.axis('off')
tabla = ax.table(
    cellText=datos,
    colLabels=columnas,
    loc='center',
    cellLoc='center'
)

tabla.auto_set_column_width(col=list(range(len(columnas))))
tabla.scale(1, 1.5)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

plt.savefig("tabla_df_final_treynor.png", bbox_inches='tight', dpi=300)
plt.show()

#%%
# ==== Beta Promedio por Clúster (colores + leyenda) ====
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dataframe de centroides
df_cent = centroides_desnormalizados[['Cluster', 'Beta Promedio']].copy()
df_cent = df_cent.sort_values('Cluster').reset_index(drop=True)

# Etiquetas (eje X y leyenda)
cluster_labels_short = {
    0: 'ESG aceptable',
    1: 'ESG intermedio',
    2: 'Rezagados ESG',
    3: 'Líderes ESG'
}

# Paleta de colores fija por clúster (consistente con otras figuras)
palette = sns.color_palette("Set2", 4)
cluster_palette = {0: palette[0], 1: palette[1], 2: palette[2], 3: palette[3]}

sns.set(style="whitegrid")
plt.figure(figsize=(9, 5))

x_pos = np.arange(len(df_cent))

# Puntos por clúster, con color y etiqueta en la leyenda
for i, row in df_cent.iterrows():
    c = int(row['Cluster'])
    plt.scatter(
        x_pos[i], row['Beta Promedio'],
        s=140, color=cluster_palette[c],
        label=f"{c} - {cluster_labels_short[c]}"
    )
    # Valor numérico sobre el punto
    plt.text(x_pos[i], row['Beta Promedio'] + 0.03, f"{row['Beta Promedio']:.3f}",
             ha='center', va='bottom', fontsize=10)

# Línea de referencia β=1
plt.axhline(y=1, linestyle='--', linewidth=1.2, color="gray")

# Ejes y ticks
plt.title('Beta promedio (centroide) por Clúster', fontsize=14)
plt.ylabel('Beta promedio (5Y)')
plt.xlabel('Clúster')
plt.xticks(x_pos, [cluster_labels_short[c] for c in df_cent['Cluster']], rotation=0)

# Leyenda (evitar duplicados por seguridad)
handles, labels = plt.gca().get_legend_handles_labels()
uniq = dict(zip(labels, handles))
plt.legend(uniq.values(), uniq.keys(), title='Clúster', frameon=True, loc='best')

# Márgenes Y
ymin = max(0, df_cent['Beta Promedio'].min() - 0.1)
ymax = df_cent['Beta Promedio'].max() + 0.25
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.savefig('fig_beta_promedio_por_cluster_colores.png', dpi=300, bbox_inches='tight')
plt.show()
