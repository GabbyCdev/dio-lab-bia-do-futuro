import json
import pandas as pd
import requests
import streamlit as st

# ========= CONFIG =========
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO = "llama3"

st.set_page_config(page_title="FinAI 💰", layout="wide")
st.title("💰 FinAI - Assistente Financeira Inteligente")

# ========= CARREGAR DADOS =========
@st.cache_data
def carregar_dados():
    with open('./data/perfil_investidor.json', encoding='utf-8') as f:
        perfil = json.load(f)

    transacoes = pd.read_csv('./data/transacoes.csv')
    historico = pd.read_csv('./data/historico_atendimento.csv', sep=';')

    with open('./data/produtos_financeiros.json', encoding='utf-8') as f:
        produtos = json.load(f)

    return perfil, transacoes, historico, produtos


perfil, transacoes, historico, produtos = carregar_dados()

# ========= DADOS SEGUROS =========
nome = perfil.get('cliente', {}).get('nome', 'Cliente')
perfil_risco = perfil.get('perfil_financeiro', {}).get('perfil_risco', 'N/A')
renda = perfil.get('cliente', {}).get('renda_mensal', 0)
objetivo = perfil.get('situacao_atual', {}).get('objetivo_principal', 'N/A')

# ========= KPIs =========
receita = transacoes[transacoes['tipo'] == 'entrada']['valor'].sum()
gastos = transacoes[transacoes['tipo'] == 'saida']['valor'].sum()
saldo = receita - gastos

st.subheader(f"Olá, {nome} 👋")

col1, col2, col3 = st.columns(3)
col1.metric("💵 Receita", f"R$ {receita:.2f}")
col2.metric("💸 Gastos", f"R$ {gastos:.2f}")
col3.metric("📊 Saldo", f"R$ {saldo:.2f}")

st.divider()

# ========= GRÁFICO =========
st.subheader("📊 Gastos por Categoria")
gastos_categoria = transacoes.groupby('categoria')['valor'].sum()
st.bar_chart(gastos_categoria)

st.divider()

# ========= CONTEXTO =========
def montar_contexto():
    contexto = f"""
Cliente: {nome}
Perfil de risco: {perfil_risco}
Renda mensal: {renda}
Objetivo: {objetivo}
Saldo atual: {saldo}

Gastos por categoria:
{gastos_categoria.to_string()}

Últimas transações:
{transacoes.tail(5).to_string(index=False)}

Histórico recente:
{historico.tail(3).to_string(index=False)}

Produtos disponíveis:
{json.dumps(produtos, indent=2, ensure_ascii=False)}
"""
    return contexto

# ========= SYSTEM PROMPT =========
SYSTEM_PROMPT = """
Você é a FinAI, uma consultora financeira inteligente.

REGRAS:
- Use apenas os dados fornecidos
- Seja clara, objetiva e prática
- Não invente informações
- Dê sugestões financeiras úteis
- Seja personalizada com base nos dados do cliente
"""

# ========= FUNÇÃO IA =========
def perguntar(pergunta):
    contexto = montar_contexto()

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXTO:
{contexto}

Pergunta: {pergunta}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODELO,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json().get("response", "Sem resposta do modelo.")
        else:
            return f"Erro na API: {response.status_code}"

    except Exception as e:
        return f"Erro ao conectar com Ollama: {e}"

# ========= CHAT =========
st.subheader("🤖 Converse com a FinAI")

# memória do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input do usuário
if pergunta := st.chat_input("Pergunte sobre suas finanças..."):
    # salva pergunta
    st.session_state.messages.append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.write(pergunta)

    # resposta IA
    with st.chat_message("assistant"):
        with st.spinner("Pensando... 🤔"):
            resposta = perguntar(pergunta)
            st.write(resposta)

    # salva resposta
    st.session_state.messages.append({"role": "assistant", "content": resposta})