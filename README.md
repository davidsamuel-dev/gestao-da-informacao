# Mineração de Dados — Heart Disease

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Mineração de Dados](https://img.shields.io/badge/Minera%C3%A7%C3%A3o%20de%20Dados-1f6feb)](https://pt.wikipedia.org/wiki/Minera%C3%A7%C3%A3o_de_dados)
[![Sistemas de Informação](https://img.shields.io/badge/Sistemas%20de%20Informa%C3%A7%C3%A3o-8250df)](https://www.ifto.edu.br/)
[![IFTO](https://img.shields.io/badge/IFTO-Para%C3%ADso%20do%20Tocantins-2ea44f)](https://www.ifto.edu.br/)

Trabalho de **Mineração de Dados** do curso de Sistemas de Informação do [IFTO — Campus Paraíso do Tocantins](https://www.ifto.edu.br/), 6º período, orientado pelo **Prof. Marcos Raimundo**.

A análise usa a base **Heart Disease**, do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease). O notebook junta fichas de quatro hospitais, limpa o que não está preenchido e responde:

> Entre pacientes avaliados para doença arterial coronariana, quais perfis clínicos se parecem e é possível classificar a presença da doença a partir dos exames disponíveis?

O fluxo completo está em [Mineracao_de_Dados_Heart_Disease.ipynb](Mineracao_de_Dados_Heart_Disease.ipynb).

## Equipe

**David Samuel · Valdinei Sousa · Gabriel Cunha · Lohan dos Reis**

| | |
|---|---|
| Instituição | Instituto Federal do Tocantins — Campus Paraíso do Tocantins |
| Curso | Sistemas de Informação, 6º período |
| Disciplina | Gestão da Informação |
| Professor | Marcos Raimundo |

## Objetivos

- Conhecer e descrever a base Heart Disease, com origem, licença e variáveis;
- Carregar as fichas dos quatro hospitais;
- Tratar colesterol zerado, colunas muito vazias e a classe de diagnóstico;
- Explorar a presença da doença por hospital e por tipo de dor;
- Agrupar pacientes com K-Means e avaliar a separação com Silhouette Score;
- Classificar ausência ou presença de doença com uma árvore de decisão;
- Registrar que associação entre exame e diagnóstico não é causa.

## Tecnologias

- Python 3;
- Jupyter Notebook ou Google Colab;
- pandas e NumPy;
- Matplotlib e Seaborn;
- scikit-learn.

## Dataset

| Item | Conteúdo |
|---|---|
| Nome | Heart Disease |
| Link | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Instituições | Cleveland Clinic Foundation; Hungarian Institute of Cardiology (Budapeste); University Hospital, Zurique; V.A. Medical Center, Long Beach |
| Autores | Andras Janosi, William Steinbrunn, Matthias Pfisterer e Robert Detrano (1989) |
| Licença | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Tarefa no UCI | Classificação |
| Registros | 920 fichas nos quatro hospitais; 303 na ficha principal do UCI (Cleveland) |

Os arquivos processados não têm cabeçalho. O valor ausente vem marcado como `?`. As 14 colunas usadas na literatura são idade, sexo, tipo de dor no peito, pressão em repouso, colesterol, glicemia, eletrocardiograma, frequência cardíaca máxima, angina de exercício, depressão de ST, inclinação do ST, número de vasos, tálio e o diagnóstico `num` (0 = ausência; 1 a 4 = graus de presença).

O notebook baixa esses arquivos direto do UCI. A pasta `dados/` guarda a mesma cópia para consulta local.

## Como executar

### Google Colab

1. Envie `Mineracao_de_Dados_Heart_Disease.ipynb` para o [Google Colab](https://colab.research.google.com/).
2. Confirme que o ambiente tem acesso à internet, porque a carga lê os arquivos no UCI.
3. No menu, use **Ambiente de execução → Executar tudo**.
4. Leia as células na ordem. Tabelas e gráficos aparecem abaixo de cada bloco.

### Ambiente local

Instale as dependências:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

Inicie o Jupyter na pasta do projeto:

```bash
jupyter notebook
```

Abra `Mineracao_de_Dados_Heart_Disease.ipynb` e execute as células de cima para baixo.

## Fluxo da análise

### 1. Preparação dos dados

O notebook junta Cleveland, Hungria, Suíça e Long Beach. Colesterol igual a zero vira valor ausente. As colunas `slope`, `ca` e `thal` saem da modelagem porque faltam em mais de 30% das fichas e quase só existem em Cleveland. Fichas com exame incompleto entre os atributos mantidos também saem. A classe final é binária: ausência ou presença.

Depois dessa limpeza permanecem **661 fichas**. A Suíça sai por completo: nas 123 fichas o colesterol estava zerado.

### 2. Exploração

Os gráficos mostram a proporção de doença em cada hospital que permaneceu, a proporção por tipo de dor no peito e a diferença de frequência cardíaca máxima e de depressão de ST entre ausência e presença.

### 3. Agrupamento

O K-Means forma **dois perfis** com os exames padronizados. A coluna de diagnóstico não entra no algoritmo. O Silhouette Score fica baixo, entre 0,16 e 0,23: os grupos são tendências, não caixas separadas.

- Um grupo reúne pacientes mais velhos, com frequência cardíaca máxima mais baixa, mais angina de exercício e cerca de 80% de presença.
- O outro reúne pacientes mais novos, com frequência máxima mais alta, pouca angina e cerca de 21% de presença.

### 4. Classificação

Uma árvore de decisão de profundidade 4 aprende a classe real da base, em um teste com 25% das fichas. O acerto foi de **77,7%**, acima dos **52,5%** de chutar sempre a classe mais comum. Entre quem tinha doença, o modelo encontrou **86%**. O hospital não é usado como atributo.

## Estrutura do projeto

O notebook está organizado nas 15 seções do roteiro da disciplina: título e integrantes, pergunta, fonte, bibliotecas, carga, exploração, limpeza, agrupamento, classificação, nota sobre padrões frequentes, avaliação, interpretação, limitações, conclusão e referências.

```text
.
├── Mineracao_de_Dados_Heart_Disease.ipynb   # Notebook com a análise
├── Relatorio_Mineracao_Heart_Disease.docx   # Relatório de 5 páginas
├── dados/                                    # Cópia local dos arquivos do UCI
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   └── processed.va.data
└── README.md                                 # Esta documentação
```

## Observações

- Os números dos grupos são só identificadores. A leitura usa idade, pressão, colesterol, frequência cardíaca, depressão de ST e angina.
- A proporção de doença não é a mesma nos hospitais. Long Beach continua bem mais alto do que Cleveland e Hungria.
- O resultado organiza fichas históricas para estudo. Não diagnostica e não substitui avaliação médica.
- Associação entre um exame e a classe não prova causa.

## Referência

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). **Heart Disease** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
