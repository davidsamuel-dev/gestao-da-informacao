# -*- coding: utf-8 -*-
"""Gera o notebook do trabalho. Arquivo auxiliar, apagado depois da geração."""
import json
from pathlib import Path

def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }

def code(source):
    text = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

cells.append(md("""
# Mineração de Dados — Heart Disease

Pergunta do trabalho: **entre pacientes avaliados para doença arterial coronariana, quais perfis clínicos se parecem e é possível classificar a presença da doença a partir dos exames disponíveis?**

Este notebook segue o fluxo visto em aula:

1. Preparar os dados
2. Explorar os dados
3. Agrupar registros parecidos (K-Means)
4. Classificar um registro em uma classe conhecida (árvore de decisão)

A classe usada na classificação é o diagnóstico que já vem na base. O agrupamento não inventa esse rótulo: ele só procura pacientes com exames parecidos.
"""))

cells.append(md("""
## Base escolhida

| Item | Conteúdo |
|---|---|
| Nome | Heart Disease |
| Link | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Instituições | Cleveland Clinic Foundation; Hungarian Institute of Cardiology (Budapeste); University Hospital, Zurique; V.A. Medical Center, Long Beach |
| Autores | Andras Janosi, William Steinbrunn, Matthias Pfisterer e Robert Detrano (1989) |
| Licença | CC BY 4.0 |
| Citação | Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X |

A página do UCI destaca 303 pacientes de Cleveland. Neste trabalho usamos os **quatro hospitais**, no mesmo formato de 14 colunas, para enxergar a qualidade dos dados de cada origem. São 920 fichas no total.

A classe original se chama `num`: 0 é ausência de estreitamento relevante da artéria; 1 a 4 são graus de presença. Como nos experimentos clássicos desta base, vamos transformar isso em duas classes: **ausência (0)** e **presença (1)**.
"""))

cells.append(md("""
## Instalação e importação das bibliotecas

No Google Colab estas bibliotecas já estão instaladas. Usamos:

- **pandas:** manipulação dos dados
- **numpy:** operações numéricas
- **matplotlib e seaborn:** gráficos
- **scikit-learn:** agrupamento e classificação
"""))

cells.append(code("""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
"""))

cells.append(md("""
## Carregando o dataset

Os arquivos processados do UCI não trazem cabeçalho. O valor ausente está marcado com `?`. Cada arquivo é um hospital. Juntamos os quatro e guardamos a origem na coluna `hospital`.
"""))

cells.append(code("""
colunas = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

arquivos = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Hungria": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "Suica": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "Long Beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

nomes_hospitais = {
    "Cleveland": "Cleveland",
    "Hungria": "Hungria",
    "Suica": "Suíça",
    "Long Beach": "Long Beach",
}

partes = []
for hospital, url in arquivos.items():
    parte = pd.read_csv(url, header=None, names=colunas, na_values="?")
    parte["hospital"] = hospital
    partes.append(parte)

df = pd.concat(partes, ignore_index=True)
df["hospital_nome"] = df["hospital"].map(nomes_hospitais)

print("Fichas carregadas:", len(df))
print("Hospitais:")
display(df["hospital_nome"].value_counts().rename("fichas"))
"""))

cells.append(md("""
## Conhecendo os dados

Antes de aplicar qualquer algoritmo, precisamos entender o que cada coluna significa e onde os dados faltam.

| Coluna | Significado |
|---|---|
| age | Idade, em anos |
| sex | Sexo (0 = feminino, 1 = masculino) |
| cp | Tipo de dor no peito (1 = angina típica, 2 = atípica, 3 = dor não anginosa, 4 = assintomático) |
| trestbps | Pressão arterial em repouso, em mm Hg |
| chol | Colesterol sérico, em mg/dl |
| fbs | Glicemia de jejum acima de 120 mg/dl (1 = sim, 0 = não) |
| restecg | Eletrocardiograma em repouso (0 = normal, 1 = anomalia de ST-T, 2 = hipertrofia ventricular) |
| thalach | Frequência cardíaca máxima atingida |
| exang | Angina provocada por exercício (1 = sim, 0 = não) |
| oldpeak | Depressão do segmento ST induzida pelo exercício |
| slope | Inclinação do segmento ST |
| ca | Número de vasos principais visíveis na fluoroscopia (0 a 3) |
| thal | Resultado do exame de tálio |
| num | Diagnóstico angiográfico (0 = ausência, 1 a 4 = presença em graus) |
| hospital | Origem da ficha |
"""))

cells.append(code("""
print("Tipos e quantidade de valores preenchidos:")
df.info()

print("\\nPrimeiras linhas:")
display(df.head())

print("\\nEstatísticas das variáveis numéricas:")
display(df.describe().round(2))
"""))

cells.append(code("""
print("Valores ausentes (marcados como ? no arquivo original):")
ausentes = df[colunas].isna().sum().sort_values(ascending=False)
display(ausentes)

print("\\nDiagnóstico original (num), por hospital:")
display(pd.crosstab(df["hospital_nome"], df["num"], margins=True))
"""))

cells.append(md("""
### O que devemos observar?

- `ca`, `thal` e `slope` faltam em grande parte das fichas fora de Cleveland.
- Na Suíça, o colesterol veio todo como **0**. Zero não é um colesterol medido: é ausência de dado escrita de outro jeito.
- Em Long Beach também aparecem colesteróis iguais a zero.
- A Hungria só usa `num` 0 e 1. Os outros hospitais usam 0 a 4. Por isso a classe do trabalho será binária: ausência ou presença.
- A proporção de doença muda muito de um hospital para outro. Isso precisa aparecer na interpretação.
"""))

cells.append(code("""
chol_zero = df["chol"].eq(0).groupby(df["hospital_nome"]).mean().mul(100).round(1)
print("Percentual de colesterol igual a zero, por hospital:")
display(chol_zero.rename("% colesterol = 0"))

qualidade = df.copy()
qualidade.loc[qualidade["chol"].eq(0), "chol"] = np.nan
percentual_ausente = (
    qualidade.groupby("hospital_nome")[colunas]
    .apply(lambda g: g.isna().mean().mul(100))
    .round(1)
)

plt.figure(figsize=(12, 5))
sns.heatmap(percentual_ausente, annot=True, fmt=".0f", cmap="YlOrRd", vmin=0, vmax=100)
plt.title("Percentual de valores ausentes por hospital")
plt.xlabel("Variável")
plt.ylabel("Hospital")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## Preparação dos dados

Regras usadas neste trabalho:

1. Tratar colesterol igual a zero como valor ausente.
2. Retirar `slope`, `ca` e `thal`. Essas colunas passam de 30% de ausência e, fora de Cleveland, quase não existem. Mantê-las reduziria o estudo a um único hospital.
3. Manter idade, sexo, tipo de dor, pressão, colesterol, glicemia, eletrocardiograma, frequência cardíaca máxima, angina de exercício e depressão de ST.
4. Remover a ficha inteira quando um desses exames mantidos estiver vazio. Não vamos inventar o valor que faltou.
5. Criar a coluna `doenca`: 0 se `num` é 0, e 1 se `num` é 1, 2, 3 ou 4.

A Suíça sai da análise final porque nenhuma ficha de lá tem colesterol real. Isso fica registrado de propósito: é uma decisão de qualidade de dados, não um filtro escondido.
"""))

cells.append(code("""
dados = df.copy()
dados.loc[dados["chol"].eq(0), "chol"] = np.nan

colunas_descartadas = ["slope", "ca", "thal"]
colunas_modelo = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
]

antes = dados.groupby("hospital_nome").size().rename("fichas_originais")
completas = dados.dropna(subset=colunas_modelo)
depois = completas.groupby("hospital_nome").size().rename("fichas_usadas")

comparacao = pd.concat([antes, depois], axis=1).fillna(0).astype(int)
comparacao["removidas"] = comparacao["fichas_originais"] - comparacao["fichas_usadas"]
print("Efeito da limpeza por hospital:")
display(comparacao)
print("Colunas retiradas da modelagem:", ", ".join(colunas_descartadas))
print("Fichas usadas na mineração:", len(completas), "de", len(df))

dados = completas.copy()
dados["doenca"] = (dados["num"] > 0).astype(int)

print("\\nClasse binária (0 = ausência, 1 = presença):")
display(dados["doenca"].value_counts().rename("fichas"))
print("Proporção de presença: {:.1%}".format(dados["doenca"].mean()))
"""))

cells.append(md("""
## Visualização inicial

A primeira pergunta descritiva é: **a presença da doença está distribuída do mesmo jeito nos hospitais e nos tipos de dor?**

Estes gráficos ainda não são o agrupamento nem a classificação. Eles mostram o terreno em que os algoritmos vão trabalhar.
"""))

cells.append(code("""
rotulos_cp = {
    1: "Angina típica",
    2: "Angina atípica",
    3: "Dor não anginosa",
    4: "Assintomático",
}

prevalencia = (
    dados.groupby("hospital_nome")["doenca"]
    .agg(fichas="size", proporcao_presenca="mean")
    .sort_values("proporcao_presenca")
)
prevalencia["proporcao_presenca"] = (prevalencia["proporcao_presenca"] * 100).round(1)
print("Presença de doença nas fichas que permaneceram:")
display(prevalencia)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=prevalencia.reset_index(),
    x="proporcao_presenca",
    y="hospital_nome",
    color="#b4534b",
)
plt.xlabel("Presença de doença (%)")
plt.ylabel("Hospital")
plt.title("Presença de doença por hospital, após a limpeza")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
por_dor = dados.copy()
por_dor["tipo_dor"] = por_dor["cp"].astype(int).map(rotulos_cp)
resumo_dor = (
    por_dor.groupby("tipo_dor")["doenca"]
    .agg(fichas="size", proporcao_presenca="mean")
    .reindex(rotulos_cp.values())
)
resumo_dor["proporcao_presenca"] = (resumo_dor["proporcao_presenca"] * 100).round(1)
print("Presença de doença por tipo de dor no peito:")
display(resumo_dor)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=resumo_dor.reset_index(),
    x="proporcao_presenca",
    y="tipo_dor",
    color="#3d5a80",
)
plt.xlabel("Presença de doença (%)")
plt.ylabel("Tipo de dor")
plt.title("Presença de doença por tipo de dor no peito")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=dados, x="doenca", y="thalach", ax=axes[0], color="#7d9d8a")
axes[0].set_title("Frequência cardíaca máxima")
axes[0].set_xlabel("Doença (0 = ausência, 1 = presença)")
axes[0].set_ylabel("Batimentos")

sns.boxplot(data=dados, x="doenca", y="oldpeak", ax=axes[1], color="#d4a373")
axes[1].set_title("Depressão de ST no exercício")
axes[1].set_xlabel("Doença (0 = ausência, 1 = presença)")
axes[1].set_ylabel("Oldpeak")

plt.tight_layout()
plt.show()
"""))

cells.append(md("""
# Agrupamento com K-Means

O K-Means junta pacientes com exames parecidos. Ele **não recebe** a coluna `doenca`. Se os grupos coincidirem em parte com a presença da doença, isso é uma leitura feita depois, não uma resposta que o algoritmo já conhecia.

`cp` e `restecg` são códigos de categoria. O número 4 não significa “duas vezes o tipo 2”. Por isso essas colunas viram indicadores (uma coluna para cada categoria) antes do agrupamento.

As escalas também são diferentes: idade fica perto de 50, colesterol perto de 250 e angina é 0 ou 1. A padronização coloca tudo na mesma escala para nenhuma coluna dominar a distância.
"""))

cells.append(code("""
def montar_atributos(base):
    tabela = base[colunas_modelo].copy()
    for coluna in ["sex", "cp", "fbs", "restecg", "exang"]:
        tabela[coluna] = tabela[coluna].astype(int)
    tabela = pd.get_dummies(tabela, columns=["cp", "restecg"])
    return tabela.rename(columns={
        "age": "idade",
        "sex": "sexo_masculino",
        "trestbps": "pressao_repouso",
        "chol": "colesterol",
        "fbs": "glicemia_alta",
        "thalach": "freq_cardiaca_max",
        "exang": "angina_exercicio",
        "oldpeak": "depressao_st",
        "cp_1": "dor_angina_tipica",
        "cp_2": "dor_angina_atipica",
        "cp_3": "dor_nao_anginosa",
        "cp_4": "assintomatico",
        "restecg_0": "ecg_normal",
        "restecg_1": "ecg_anomalia_st",
        "restecg_2": "ecg_hipertrofia",
    })

X = montar_atributos(dados)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Atributos usados no agrupamento e na classificação:")
print(", ".join(X.columns))
print("Pacientes:", X.shape[0])
"""))

cells.append(md("""
## Escolha do número de grupos

O Silhouette Score resume se os pacientes ficam mais perto do próprio grupo do que dos outros. Perto de 1, os grupos estão bem separados. Perto de 0, eles se misturam.

Nesta base o escore fica baixo para qualquer quantidade de grupos. Isso já é um resultado: os perfis clínicos se sobrepõem. Vamos testar de 2 a 6 grupos e ficar com **2**, porque é a divisão que ainda dá para explicar na apresentação. Grupos a mais só quebram o lado de menor risco em perfis muito parecidos entre si.
"""))

cells.append(code("""
silhuetas = []
for k in range(2, 7):
    rotulos = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X_scaled)
    escore = silhouette_score(X_scaled, rotulos)
    silhuetas.append({"grupos": k, "silhouette": round(escore, 3)})

tabela_silhueta = pd.DataFrame(silhuetas)
display(tabela_silhueta)

plt.figure(figsize=(7, 4))
sns.lineplot(data=tabela_silhueta, x="grupos", y="silhouette", marker="o")
plt.title("Silhouette Score por quantidade de grupos")
plt.xlabel("Quantidade de grupos")
plt.ylabel("Silhouette")
plt.xticks(tabela_silhueta["grupos"])
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
k_escolhido = 2
dados["cluster"] = KMeans(
    n_clusters=k_escolhido, n_init=20, random_state=42
).fit_predict(X_scaled)

perfil = (
    dados.groupby("cluster")[["age", "trestbps", "chol", "thalach", "oldpeak", "exang", "doenca"]]
    .mean()
    .round(2)
)
perfil["fichas"] = dados.groupby("cluster").size()
print("Perfil médio de cada grupo:")
display(perfil)

print("Grupos por hospital:")
display(pd.crosstab(dados["cluster"], dados["hospital_nome"]))
"""))

cells.append(code("""
contagem = dados["cluster"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
contagem.plot(kind="bar", color="#3d5a80")
plt.title("Quantidade de pacientes por grupo")
plt.xlabel("Grupo")
plt.ylabel("Pacientes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=dados,
    x="thalach",
    y="oldpeak",
    hue="cluster",
    style="doenca",
    palette="tab10",
    alpha=0.75,
)
plt.title("Frequência cardíaca máxima e depressão de ST, por grupo")
plt.xlabel("Frequência cardíaca máxima")
plt.ylabel("Depressão de ST (oldpeak)")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Como ler os grupos

O número do grupo é só um identificador. A leitura usa as médias.

No resultado desta execução, um grupo reúne pacientes mais velhos, com frequência cardíaca máxima mais baixa, mais angina de exercício, depressão de ST mais alta e cerca de 80% de presença da doença. O outro reúne pacientes mais novos, com frequência cardíaca máxima mais alta, pouca angina e cerca de 20% de presença.

Isso é associação entre o perfil e o diagnóstico. Não quer dizer que a idade ou o exame tenha causado a doença. O Silhouette baixo confirma que a fronteira entre os grupos não é nítida: muita gente fica no meio do caminho.
"""))

cells.append(md("""
# Classificação

A classificação é diferente do agrupamento. Aqui existe uma classe conhecida: ausência ou presença de doença arterial coronariana, vinda da angiografia (`num`).

A árvore de decisão aprende regras do tipo “se o paciente é assintomático e a depressão de ST passa de um certo valor, classifique como presença”. Ela é adequada a este trabalho porque a regra pode ser mostrada e explicada.

O hospital **não entra** como atributo. Se entrasse, a árvore poderia aprender “este hospital tem mais doentes” em vez de aprender o exame.

Separamos 75% das fichas para treinar e 25% para testar. O teste é estratificado: a proporção de presença é parecida nos dois lados. A profundidade máxima é 4, para a árvore caber na apresentação.
"""))

cells.append(code("""
y = dados["doenca"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_treino, y_treino)
y_previsto = modelo.predict(X_teste)

maioria = y.value_counts(normalize=True).max()
print("Acerto se chute sempre a classe mais comum: {:.1%}".format(maioria))
print("Fichas de treino:", len(X_treino), "| Fichas de teste:", len(X_teste))
"""))

cells.append(md("""
## Avaliando o classificador

A acurácia sozinha engana quando uma classe é muito maior que a outra. Aqui as duas classes estão próximas, mas ainda assim olhamos precisão, recall e a matriz de confusão.

- **Precisão da presença:** entre quem o modelo marcou como presença, quantos realmente tinham doença.
- **Recall da presença:** entre quem tinha doença, quantos o modelo encontrou.
- O erro mais grave neste tema é o falso negativo: o modelo diz ausência e a ficha é presença.
"""))

cells.append(code("""
print("Relatório de classificação:")
print(classification_report(
    y_teste,
    y_previsto,
    target_names=["Ausência", "Presença"],
    digits=3,
))

matriz = pd.DataFrame(
    confusion_matrix(y_teste, y_previsto),
    index=["Ausência real", "Presença real"],
    columns=["Prevista ausência", "Prevista presença"],
)
print("Matriz de confusão:")
display(matriz)
"""))

cells.append(code("""
plt.figure(figsize=(22, 10))
plot_tree(
    modelo,
    feature_names=list(X.columns),
    class_names=["Ausência", "Presença"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Árvore de decisão (profundidade máxima 4)")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## Exemplo de previsão para uma ficha nova

A ficha abaixo não está no treino. Ela descreve um paciente de 62 anos, sexo masculino, sem queixa de dor típica (código assintomático), pressão 150, colesterol 280, eletrocardiograma com hipertrofia, frequência máxima 115, angina no exercício e depressão de ST de 2,2.

O modelo devolve a classe aprendida. Isso ilustra a técnica. Não é um diagnóstico e não deve ser usado para decidir cuidado de uma pessoa.
"""))

cells.append(code("""
ficha_nova = pd.DataFrame([{
    "age": 62,
    "sex": 1,
    "cp": 4,
    "trestbps": 150,
    "chol": 280,
    "fbs": 0,
    "restecg": 2,
    "thalach": 115,
    "exang": 1,
    "oldpeak": 2.2,
}])

X_nova = montar_atributos(ficha_nova).reindex(columns=X.columns, fill_value=0)
classe = modelo.predict(X_nova)[0]
nomes_classe = {0: "ausência", 1: "presença"}
print("Classe prevista:", nomes_classe[int(classe)])
"""))

cells.append(md("""
## Interpretação e aplicação possível

Os exames que mais separam os perfis são a frequência cardíaca máxima, a angina de exercício, a depressão de ST e o tipo de dor. O grupo de maior risco clínico concentra a maior parte das fichas com presença da doença. A árvore acerta acima do chute da classe mais comum, então os exames carregam informação útil para a classe.

Uma aplicação possível, só como exercício de gestão da informação, é organizar filas de revisão de fichas em um arquivo histórico: fichas com o perfil de maior risco poderiam ser conferidas primeiro por um profissional. O resultado deste notebook não autoriza essa fila num hospital real.

## Limitações

- Os pacientes foram avaliados no fim dos anos 1980, em centros que já investigavam doença coronariana. Não representam a população geral.
- Associação não é causa. Ver colesterol ou dor junto com a classe não prova que aquele exame produziu a doença.
- A Suíça inteira ficou de fora porque o colesterol não foi registrado. Long Beach continua com proporção de doença bem mais alta que Cleveland e Hungria. Parte do padrão pode refletir o mix de hospitais.
- `ca`, `thal` e `slope` foram retirados por falta de preenchimento. Em Cleveland eles existem e são clinicamente importantes. O modelo não os vê.
- O Silhouette dos grupos é baixo. Os grupos são tendências, não caixas fechadas.
- A árvore foi limitada a quatro níveis e testada em cerca de um quarto das fichas. Outra divisão treino/teste muda os números.
- Este trabalho não é um dispositivo de diagnóstico.

## Referência

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). **Heart Disease** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Página da base: https://archive.ics.uci.edu/dataset/45/heart+disease
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

destino = Path(r"c:\Gestão da Informação\Mineracao_de_Dados_Heart_Disease.ipynb")
destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(destino, "cells", len(cells))
