import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Título e Descrição
st.set_page_config(page_title="Simulador do Teorema Central do Limite", layout="centered")

st.title("Simulador TCL")
st.markdown("""
O intuito desta aplicação é demonstrar visualmente o **Teorema Central do Limite (TCL)**.

Não importa a distribuição original dos dados, se ela possui *variância finita* e *valor esperado definido*, 
então a **média** de uma amostra de tamanho $n$ suficientemente grande convergirá para o ***formato* de uma Distribuição Normal**.

---

##### Instruções
Use o painel à esquerda para configurar a simulação e os parâmetros da distribuição:

- **Número de Simulações $m$:** Torna o desenho do histograma mais "nítido", pois estamos coletando mais dados para desenhar as barras. Aumentar o m não transforma o gráfico em uma Normal.
> É como se estivéssemos aumentando a "resolução" de uma imagem, usando maix *pixels*.

- **Tamanho de cada amostra $n$:** É o aumento de $n$ que de fato altera o formato do gráfico, expalhando a aleatoriedade de modo que distribuição das médias amostrais assumam o formato de um sino, característico da Distribuição Normal

---
""")

# Painel de Controle
st.sidebar.header("Configurações da Simulação")

dist_nome = st.sidebar.selectbox( "**1. Escolha a Distribuição Original:**", ("Exponencial", "Uniforme", "Bernoulli"))
n = st.sidebar.slider("**2. Tamanho da Amostra ($n$):**", min_value=1, max_value=300, value=2, step=1)
m = st.sidebar.slider("**3. Quantidade de Simulações ($m$):**", min_value=250, max_value=5000, value=250, step=1)

# Configurando os parâmetros da distribuição
if dist_nome == "Exponencial":
    param_lambda = st.sidebar.slider("**Parâmetro $\\lambda$:**", min_value=2, max_value=100, value=2, step=1)
    mu_teorico, sigma_teorico = param_lambda, param_lambda
    dados_originais = np.random.exponential(scale=param_lambda, size=(m, n))
    
elif dist_nome == "Uniforme":
    
    a = st.sidebar.slider("**Parâmetro $a$ (início do intervalo):**", min_value=0, max_value=20, value=0, step=1)
    tam_intervalo = st.sidebar.slider("**Tamanho do intervalo $b - a$:**", min_value=10, max_value=40, value=10, step=1)
    b = a + tam_intervalo

    mu_teorico = (a + b) / 2
    sigma_teorico = np.sqrt((b - a)**2 / 12)
    dados_originais = np.random.uniform(low=a, high=b, size=(m, n))
    
else: # Bernoulli
    p = 0.5 if st.sidebar.checkbox("Usar p = 0.5 (Equilibrado)", value=True) else 0.05
    mu_teorico = p
    sigma_teorico = np.sqrt(p * (1 - p))
    dados_originais = np.random.binomial(n=1, p=p, size=(m, n))



# Criação dos Gráficos
medias_amostrais = dados_originais.mean(axis=1)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Formato Original")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.hist(dados_originais[:, 0], bins=20, color='gray', alpha=0.7, density=True)
    ax1.set_title("Amostra Original")
    st.pyplot(fig1)

with col2:
    st.subheader("Resultado das Médias")
    fig2, ax2 = plt.subplots(figsize=(5, 4))

    # Histograma das médias e curva normal
    ax2.hist(medias_amostrais, bins=30, color='skyblue', alpha=0.7, density=True, label="Médias reais")
    std_erro = sigma_teorico / np.sqrt(n)
    x = np.linspace(mu_teorico - 4*std_erro, mu_teorico + 4*std_erro, 100)
    ax2.plot(x, stats.norm.pdf(x, mu_teorico, std_erro), 'r-', lw=2, label="Normal Esperada")
    ax2.set_title(f"Distribuição das Médias (n={n})")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

# Comentários
st.markdown(f"#### Distribuição escolhida: {dist_nome}")

if dist_nome == "Exponencial":
    st.markdown(f""" ###### $X \\sim \\operatorname{{Exponencial}}({param_lambda})$
    Como a distribuição Exponencial é muito assimétrica, é necessário um tamanho amostral maior, $n \\approx 10$.
    Além disso, dependendo do valor do parâmetro $\\lambda$, pode ser que seja preciso também um tamanho muito grande $m$ de simulações.
    """)

elif dist_nome == "Uniforme":
    st.markdown(f""" ###### $X \\sim \\operatorname{{Uniforme}}({a},{b})$
    Na distribuição Uniforme, todos os intervalos de mesmo tamanho possuem exatamente a mesma probabilidade de acontecer, logo o histograma gerado tende a um formato retangular, com alguns picos gerados pelo acaso.


    Por ser uma distribuição simétrica na origem, a distribuição Uniforme Contínua converge muito rápido pelo TCL, mesmo para valores pequenos de $n$, como $5$ ou $10$. 
    """)

else: # Bernoulli
    st.markdown(f""" ###### $X \\sim \\operatorname{{Bernoulli}}({p})$
    A distribuição Bernoulli é estritamente discreta, então o gráfico original não possui curvas: ele mostra apenas duas barras isoladas (o valor 0 para fracasso e o valor 1 para sucesso).

    Tirar a média de uma Bernoulli significa calcular uma proporção ($\\hat{{p}}$):
    - Se a chance original for equilibrada $(p = 0.5)$, as médias assumem o formato de sino rapidamente.
    - Se a chance for muito desbalanceada $(p = 0.05)$, pequenas amostras geram gráficos extremamente assimétricos. Logo, são necessárias amostras de tamanho $n$ grande para que o gráfico se assemelhe ao de uma Distribuição Normal.
    """)
