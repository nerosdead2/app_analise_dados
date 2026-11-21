import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. INGESTÃO, PERFORMANCE E LIMPEZA
# Leitura garantindo o encoding 'latin1' 
df = pd.read_csv(
    'transacoes.csv', encoding='latin1'
)

# Imputação de valores NaN em valor usando a mediana por estado_cliente
df['valor'] = df['valor'].fillna(
    df.groupby('estado_cliente')['valor'].transform('median')
)

# Adição da coluna estática
df['plataforma'] = 'Mobile'

# 2. Engenharia de dados & alinhamento temporal
# Conversão para datetime e aplicação do fuso horário 
df['data_transacao'] = pd.to_datetime(df['data_transacao']).dt.tz_localize(
    'America/Sao_Paulo'
)

# Extração de dia da semana e mês usando 
df['dia_semana'] = df['data_transacao'].dt.day_name()
df['mes'] = df['data_transacao'].dt.month

# Descarte de transações duplicadas mantendo apenas a primeira ocorrência
df = df.drop_duplicates(keep='first').copy()

# 3. Operações vetorizadas e filtros bitwise
# Filtro para mês 9, estados SP ou RJ, e valor > 5000 usando operadores
condicao_mes = df['mes'] == 9
condicao_estado = (df['estado_cliente'] == 'SP') | (
    df['estado_cliente'] == 'RJ'
)
condicao_valor = df['valor'] > 5000.0

df_filtrado_sp_rj = df[condicao_mes & condicao_estado & condicao_valor].copy()

# 4. Cruzamento de dados & agregação
# Dicionário de risco mapeado aos clientes existentes no dataset
risco_dict = {
    'C100': 'Baixo',
    'C101': 'Alto',
    'C102': 'Médio',
    'C103': 'Baixo',
    'C104': 'Crítico',
}

df['nivel_risco'] = df['id_cliente'].map(risco_dict)

# Tabela agrupando por mês e nível de risco com subtotais
tabela_dinamica = pd.pivot_table(
    df,
    values='valor',
    index='mes',
    columns='nivel_risco',
    aggfunc='sum',
    margins=True,
    margins_name='Total Geral',
)

print('--- Tabela Dinâmica: Volume por Mês e Nível de Risco ---')
print(tabela_dinamica)
print('\n')

# 5. Detecção de outliers 


def calcular_zscore_por_grupo(
    dataframe: pd.DataFrame, col_grupo: str, col_valor: str
) -> pd.Series:
  medias = dataframe.groupby(col_grupo)[col_valor].transform('mean')
  desvios = dataframe.groupby(col_grupo)[col_valor].transform('std')
  return (dataframe[col_valor] - medias) / desvios


df['z_score'] = calcular_zscore_por_grupo(df, 'estado_cliente', 'valor')

# Filtro de potenciais anomalias/fraudes (Z > 2.5)
df_anomalias = df[df['z_score'] > 2.5].copy()

print(f'Total de anomalias detectadas (Z > 2.5): {len(df_anomalias)}')
print(
    df_anomalias[
        ['id_cliente', 'data_transacao', 'estado_cliente', 'valor', 'z_score']
    ].head()
)
print('\n')

# 6. Visualização gráfica orientada a objetos
# Agregação diária do valor total das transações
df_diario = (
    df.set_index('data_transacao')['valor'].resample('D').sum().reset_index()
)

# Cálculo da média móvel de 7 dias
df_diario['media_movel_7d'] = df_diario['valor'].rolling(window=7).mean()

# Criação da figura e eixos via API Orientada a Objetos
fig, ax = plt.subplots(figsize=(12, 6))

# Plotagem das duas séries
ax.plot(
    df_diario['data_transacao'],
    df_diario['valor'],
    label='Valor Total Diário (R$)',
    color='#1f77b4',
    alpha=0.6,
    marker='o',
)
ax.plot(
    df_diario['data_transacao'],
    df_diario['media_movel_7d'],
    label='Média Móvel (7 dias)',
    color='#d62728',
    linewidth=2.5,
)

# Ajustes de eixos, títulos e legendas
ax.set_ylim(bottom=0)
ax.set_title(
    'Evolução Diária de Transações vs. Média Móvel de 7 Dias',
    fontsize=14,
    pad=15,
)
ax.set_xlabel('Data da Transação', fontsize=11)
ax.set_ylabel('Volume Financeiro (R$)', fontsize=11)
ax.legend(loc='upper left', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5)

# Formatação visual das datas no eixo X
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

