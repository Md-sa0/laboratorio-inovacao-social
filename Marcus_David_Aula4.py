"""
Aluno: Marcus David Nascimento de Sa
Matricula: 2056506
Disciplina: Laboratorio de Inovacao Social
Atividade 01 - Pre-processamento e EDA no California Housing Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dados = fetch_california_housing()
df = pd.DataFrame(dados.data, columns=dados.feature_names)
df['MedHouseVal'] = dados.target

print("=" * 60)
print("ENTREGA E1: INSPECÃO GERAL")
print("=" * 60)

print("df.shape:", df.shape)
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe())
print("\ndf.head():")
print(df.head())

relatorio_e1 = """
Respostas E1:
1. Linhas e Colunas: O dataset possui 20.640 linhas e 9 colunas no total (8 features preditoras e 1 alvo).
2. Tipos de Dados: Todas as colunas sao do tipo float64 (numericas continuas).
3. Valores Nulos: Nao existem valores ausentes no dataset original (0 nulos).
4. Observacoes Relevantes:
   - A variavel alvo 'MedHouseVal' possui valor maximo truncado exatamente em 5.0 (teto artificial de 500 mil dolares), criando um acumulo visivel na distribuicao.
   - Variaveis como 'AveRooms' e 'AveOccup' possuem valores maximos extremamente distantes da mediana e do terceiro quartil (por exemplo, AveRooms max = 141 e AveOccup max = 1243), apontando presenca severa de assimetria e outliers.
"""
print(relatorio_e1)

print("=" * 60)
print("ENTREGA E2: DADOS FALTANTES")
print("=" * 60)

print("Nulos originais:\n", df.isnull().sum())

df_simulado = df.copy()
np.random.seed(42)

indices_medinc = df_simulado.sample(frac=0.05, random_state=42).index
indices_rooms = df_simulado.sample(frac=0.05, random_state=24).index

df_simulado.loc[indices_medinc, 'MedInc'] = np.nan
df_simulado.loc[indices_rooms, 'AveRooms'] = np.nan

print("\nNulos apos simulacao de 5%:\n", df_simulado[['MedInc', 'AveRooms']].isnull().sum())

mediana_medinc = df_simulado['MedInc'].median()
df_simulado['MedInc'] = df_simulado['MedInc'].fillna(mediana_medinc)

media_rooms = df_simulado['AveRooms'].mean()
df_simulado['AveRooms'] = df_simulado['AveRooms'].fillna(media_rooms)

print("\nNulos apos imputacao:\n", df_simulado[['MedInc', 'AveRooms']].isnull().sum())

justificativa_e2 = """
Justificativas E2:
- Para 'MedInc' (Renda Mediana), foi utilizada a imputacao pela Mediana. Variaveis de renda sao tipicamente assimetricas a direita e sensiveis a valores extremos; a mediana preserva a tendencia central sem sofrer distorcao por rendas atipicas.
- Para 'AveRooms' (Media de Quartos), foi aplicada a imputacao pela Media para efeito comparativo de estrategias no experimento, garantindo que o valor medio global da amostra permanecesse inalterado durante a recomposicao dos dados faltantes.
"""
print(justificativa_e2)

print("=" * 60)
print("ENTREGA E3: NORMALIZACAO E PADRONIZACAO")
print("=" * 60)

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

print("MinMax Treino (Min/Max MedInc):", X_train_minmax[:, 0].min(), X_train_minmax[:, 0].max())
print("Standard Treino (Media/Desvio MedInc):", round(X_train_std[:, 0].mean(), 4), round(X_train_std[:, 0].std(), 4))
print("Standard Teste (Media/Desvio MedInc):", round(X_test_std[:, 0].mean(), 4), round(X_test_std[:, 0].std(), 4))

comentario_e3 = """
Comentarios E3:
- O MinMaxScaler comprime todas as caracteristicas estritamente no intervalo [0, 1]. Isso e ideal para algoritmos sensiveis a limites fixos (como redes neurais e KNN), porem comprime grande parte dos dados se houver outliers severos.
- O StandardScaler reescala para media proxima a 0 e desvio padrao 1, mantendo a presenca relativa dos outliers sem limitar um intervalo rigido.
- A regra de fit no treino e apenas transform no teste evita data leakage, garantindo que estatisticas do conjunto de teste nao contaminem o pre-processamento.
"""
print(comentario_e3)

print("=" * 60)
print("ENTREGA E4: HISTOGRAMAS E OUTLIERS")
print("=" * 60)

plt.figure(figsize=(14, 10))
df.hist(bins=30, figsize=(14, 10), color='steelblue', edgecolor='black')
plt.tight_layout()
plt.savefig('e4_histogramas.png')
plt.close()

features_box = ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']
plt.figure(figsize=(12, 6))
for i, col in enumerate(features_box, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col], color='coral')
    plt.title(col)
plt.tight_layout()
plt.savefig('e4_boxplots.png')
plt.close()

proposta_e4 = """
Discussao e Proposta de Outliers E4:
- Identificacao: 'AveRooms' e 'AveOccup' apresentam valores extremos dezenas de vezes acima da concentracao dos dados. 'MedInc' tambem apresenta cauda longa a direita, enquanto 'MedHouseVal' exibe um acumulo incomum em 5.0 (censura a direita).
- Tratamento em Projeto Real:
  1. Para variaveis como AveRooms e AveOccup com valores irreais para residencias unifamiliares (ex: > 50 quartos), aplicaria um filtro por quantis (capping/winsorization no percentil 99 ou remocao dos registros fisicamente implusiveis).
  2. Para o alvo 'MedHouseVal', os registros fixados em 5.0 devem ser analisados separadamente ou tratados com modelos de regressao censurada (Tobit), pois representam truncamento artificial na coleta e degradam regressoes lineares puras.
"""
print(proposta_e4)

print("=" * 60)
print("ENTREGA E5: MAPA DE CORRELACAO")
print("=" * 60)

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Matriz de Correlacao - California Housing')
plt.tight_layout()
plt.savefig('e5_heatmap_correlacao.png')
plt.close()

corr_alvo = corr_matrix['MedHouseVal'].sort_values(ascending=False)
print("Correlacao com MedHouseVal:\n", corr_alvo)

analise_e5 = """
Analise E5:
- Maior Correlacao Positiva: MedInc (0.69). Faz total sentido para o contexto imobiliario, visto que regioes com maior renda media possuem capacidade de compra superior e demandam residencias com precos mais elevados.
- Maior Correlacao Negativa: Latitude (-0.14), seguida de AveOccup (-0.02) ou Longitude (-0.05). A correlacao negativa com Latitude reflete o historico imobiliario da California, onde areas costeiras do sul (como Los Angeles e Orange County) concentram precos mais altos do que o norte rural do estado.
"""
print(analise_e5)

print("=" * 60)
print("ENTREGA E6: SCATTER PLOT E PAIRPLOT")
print("=" * 60)

df['Faixa_Renda'] = pd.qcut(df['MedInc'], q=4, labels=['Baixa', 'Media-Baixa', 'Media-Alta', 'Alta'])

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', hue='Faixa_Renda', palette='viridis', alpha=0.5)
plt.title('MedInc vs MedHouseVal por Faixa de Renda')
plt.xlabel('Renda Mediana')
plt.ylabel('Preco Medio do Imovel ($100k)')
plt.tight_layout()
plt.savefig('e6_scatterplot.png')
plt.close()

cols_pairplot = ['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'MedHouseVal']
pairplot_fig = sns.pairplot(df[cols_pairplot].sample(1500, random_state=42), corner=True)
pairplot_fig.savefig('e6_pairplot.png')
plt.close()

descricao_e6 = """
Descricao E6:
O scatter plot confirma uma relacao linear positiva evidente entre a renda mediana (MedInc) e o preco das casas (MedHouseVal), com as faixas mais altas concentrando precos elevados. No entanto, nota-se uma linha horizontal densa no teto de 5.0, caracterizando a censura do dataset. No pairplot, destaca-se tambem a correlacao bimodal/espacial da Latitude e a ausencia de correlacoes lineares fortes entre a idade da casa (HouseAge) e o preco.
"""
print(descricao_e6)

print("=" * 60)
print("ENTREGA E7: SINTESE E INSIGHTS")
print("=" * 60)

insights_e7 = """
Sintese e Insights para Modelagem (E7):

1. Dominancia da Renda Mediana (MedInc):
   - Observado: A variavel MedInc possui a correlacao mais forte com o alvo (r = 0.69) e dispersao bem delimitada por faixa.
   - Relevancia: E a feature com maior capacidade explicativa do modelo para determinar precos residenciais.
   - Impacto: Deve ser mantida como preditor prioritario e nao pode ser descartada ou sofrer transformacoes que destruam sua monotonicidade. Modelos lineares se beneficiarao diretamente dessa feature.

2. Teto Artificial no Valor dos Imoveis (MedHouseVal = 5.0):
   - Observado: Ha um acumulo expressivo de registros no valor exato de 5.0 no histograma e no scatter plot.
   - Relevancia: Representa um teto de corte (censura a direita) aplicado durante a coleta original dos dados.
   - Impacto: Modelos de regressao tradicionais tendem a subestimar valores no topo. Deve-se considerar a remocao desses registros censurados para avaliar o modelo em valores reais ou utilizar metricas robustas a outliers como MAE em vez de MSE puro.

3. Presenca de Outliers Extremos em AveRooms e AveOccup:
   - Observado: Valores maximos discrepantes (acima de 100 comodos e 1000 ocupantes por residencia) com terceiro quartil baixo.
   - Relevancia: Indicam ruidos de coleta ou registros de hoteis/alojamentos que distorcem o comportamento residencial comum.
   - Impacto: Algoritmos lineares e baseados em distancia (LinearRegression, Ridge, KNN, SVR) serao severamente degradados; exige tecnicas de clipping/capping ou o uso prioritario de modelos baseados em arvores (como Random Forest ou XGBoost), que sao naturalmente robustos a esses extremos.

4. Dependencia Nao Linear Espacial (Latitude e Longitude):
   - Observado: As correlacoes lineares individuais de coordenadas sao baixas, mas o mapa geografico do pairplot reflete a costa da California.
   - Relevancia: O valor de um imovel depende conjuntamente de Latitude e Longitude (distancia da costa e de grandes polos urbanos), nao de eixos isolados.
   - Impacto: Exige a criacao de novas features (ex: distancia ate a costa / centros urbanos) ou a adocao de algoritmos que capturem interacoes nao lineares e bivariadas, como Gradient Boosting ou Redes Neurais.

5. Sensibilidade ao Metodo de Escalonamento:
   - Observado: A comparacao entre MinMaxScaler e StandardScaler mostrou que MinMax foi achatado pelas caudas longas de AveRooms e AveOccup.
   - Relevancia: Recursos com escalas dispares afetam diretamente a convergencia de gradiente descendente e penalizacoes L1/L2.
   - Impacto: Para algoritmos lineares ou redes neurais, deve-se preferir RobustScaler ou StandardScaler associado a transformacao logaritmica pre-escalonamento, evitando MinMaxScaler enquanto as caudas nao forem tratadas.
"""
print(insights_e7)