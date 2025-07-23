# Importando bibliotecas necessárias para análise, visualização e machine learning
import pandas as pd                      # Manipulação de dados em tabelas
import seaborn as sns                   # Visualizações estatísticas (gráficos)
import matplotlib.pyplot as plt         # Visualização de gráficos
from sklearn.model_selection import train_test_split  # Separar dados em treino e teste
from sklearn.preprocessing import StandardScaler      # Padronizar (normalizar) os dados
from sklearn.linear_model import LinearRegression     # Modelo de Regressão Linear
from sklearn.metrics import mean_squared_error, r2_score  # Avaliação do modelo
from sklearn.datasets import fetch_california_housing      # Dataset com dados reais da Califórnia

# Carrega o dataset da Califórnia (preços de casas e variáveis demográficas)
california = fetch_california_housing()

# Cria um DataFrame com os dados e adiciona a coluna PRICE (preço das casas)
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Exibe estatísticas descritivas dos dados (média, desvio padrão, min, max, etc.)
print(data.describe())

# Cria um mapa de calor com as correlações entre as variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlação")
plt.show()

# Separa as variáveis independentes (X) e a variável alvo (y = preço da casa)
X = data.drop(columns=['PRICE'])  # Todas menos o preço
y = data['PRICE']                 # Apenas o preço

# Divide os dados em conjuntos de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normaliza os dados para que todas as variáveis fiquem na mesma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Ajusta e transforma os dados de treino
X_test = scaler.transform(X_test)         # Transforma os dados de teste com base no treino

# Cria e treina um modelo de Regressão Linear com os dados normalizados
model = LinearRegression()
model.fit(X_train, y_train)

# Usa o modelo treinado para fazer previsões com os dados de teste
y_pred = model.predict(X_test)

# Avalia o modelo com duas métricas:
mse = mean_squared_error(y_test, y_pred)  # Erro médio quadrático
r2 = r2_score(y_test, y_pred)             # Coeficiente R² (qualidade da previsão)

# Imprime as métricas de avaliação
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared (R2): {r2:.2f}")

# Cria um gráfico comparando os valores reais com os valores previstos
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7, color='b', label="Valores Reais")  # Pontos azul = previsão vs real
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label="Valores Perfeitos")  # Linha vermelha ideal
plt.xlabel("Valores Reais")       # Rótulo do eixo X
plt.ylabel("Valores Previstos")   # Rótulo do eixo Y
plt.title("Valores Reais vs. Previstos")  # Título do gráfico
plt.legend()                      # Adiciona legenda
plt.show()
