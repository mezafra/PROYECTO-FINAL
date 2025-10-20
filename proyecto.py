# -----------------------------------------------------
# 📊 PROYECTO STREAMLIT - ANÁLISIS ESTOCÁSTICO Y RSI (INTERACTIVO)
# -----------------------------------------------------

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------
# 🏁 CONFIGURACIÓN DE LA APP
# -----------------------------------------------------
st.set_page_config(page_title="Análisis Estocástico y RSI", layout="centered")
st.title("📈 Análisis de Acciones: RSI, Proceso Estocástico y Tendencia")
st.markdown("Analiza el comportamiento de una acción combinando **RSI**, **tendencia lineal** y un modelo estocástico (GBM).")

# -----------------------------------------------------
# 🎛️ PARÁMETROS INTERACTIVOS
# -----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    ticker = st.text_input("Símbolo de la acción", "F")  # Ejemplo: F, AAPL, GOOG
with col2:
    periodo = st.selectbox("Período histórico", ["6mo", "1y", "2y", "5y"])
with col3:
    horizonte = st.selectbox("Horizonte de proyección", ["1 día", "1 semana", "1 mes", "6 meses", "1 año", "5 años", "10 años"])
with col4:
    n_sim = st.slider("N° de simulaciones", 5, 100, 20)

# Conversión del horizonte a años (para GBM)
horizonte_map = {
    "1 día": 1/252,
    "1 semana": 5/252,
    "1 mes": 21/252,
    "6 meses": 0.5,
    "1 año": 1,
    "5 años": 5,
    "10 años": 10
}
T = horizonte_map[horizonte]

# -----------------------------------------------------
# 📥 DESCARGA DE DATOS
# -----------------------------------------------------
data = yf.Ticker(ticker)
hist = data.history(period=periodo, interval="1d")

if hist.empty:
    st.error("No se pudieron descargar los datos. Verifica el símbolo.")
    st.stop()

st.subheader(f"📊 Datos históricos del último período de {ticker}")
st.line_chart(hist['Close'], use_container_width=True)

# -----------------------------------------------------
# 📈 CÁLCULO DEL RSI (ARREGLADO: Wilder / EWM)
# -----------------------------------------------------
window = 14
delta = hist['Close'].diff()

# ganancias y pérdidas
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# Promedios tipo Wilder usando EWM (alpha = 1/window, adjust=False)
avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

# evitar división por cero
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Manejo de casos extremos:
# cuando avg_loss == 0 => RSI = 100 (máximo), cuando avg_gain == 0 y avg_loss>0 => RSI=0
rsi_filled = rsi.copy()
rsi_filled = rsi_filled.fillna(50)  # valores iniciales neutrales
rsi_filled.loc[avg_loss == 0] = 100
rsi_filled.loc[(avg_gain == 0) & (avg_loss > 0)] = 0

hist['RSI'] = rsi_filled
rsi_actual = hist['RSI'].iloc[-1]

# Interpretación del RSI
if rsi_actual > 70:
    decision_rsi = f"RSI={rsi_actual:.2f} → **Sobrecompra. Riesgo de corrección. No invertir.**"
    rsi_signal = -1
elif rsi_actual < 30:
    decision_rsi = f"RSI={rsi_actual:.2f} → **Sobreventa. Potencial de subida. Oportunidad de compra.**"
    rsi_signal = 1
else:
    decision_rsi = f"RSI={rsi_actual:.2f} → **Zona neutral. Esperar confirmaciones.**"
    rsi_signal = 0

# -----------------------------------------------------
# 📊 GRÁFICO DEL RSI MEJORADO (CORREGIDO)
# -----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,6), sharex=True)

# Precio histórico
ax1.plot(hist['Close'], color='steelblue', linewidth=2)
ax1.set_title(f'Precio histórico de {ticker}')
ax1.set_ylabel('Precio (USD)')

# RSI: línea + banda 30-70 + resaltados de sobrecompra/sobreventa
ax2.plot(hist.index, hist['RSI'], color='#FFA500', label=f'RSI ({window} días)', linewidth=1.5)

# Banda neutral sombreada (30 a 70)
ax2.fill_between(hist.index, 30, 70, color='purple', alpha=0.08)

# Resaltado de zonas extremas (opcional)
ax2.fill_between(hist.index, hist['RSI'], 70, where=(hist['RSI'] >= 70), color='red', alpha=0.25, interpolate=True)
ax2.fill_between(hist.index, hist['RSI'], 30, where=(hist['RSI'] <= 30), color='green', alpha=0.25, interpolate=True)

ax2.axhline(70, color='red', linestyle='--', linewidth=1)
ax2.axhline(30, color='green', linestyle='--', linewidth=1)
ax2.set_title('Índice RSI (14 días)')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# -----------------------------------------------------
# 📉 TENDENCIA (Regresión Lineal)
# -----------------------------------------------------
X = np.arange(len(hist)).reshape(-1,1)
y = hist['Close'].values.reshape(-1,1)
model = LinearRegression().fit(X, y)
tendencia = model.coef_[0][0]

if tendencia > 0:
    decision_trend = f"Tendencia reciente: **Alcista (+{tendencia:.4f} por día).**"
    trend_signal = 1
else:
    decision_trend = f"Tendencia reciente: **Bajista ({tendencia:.4f} por día).**"
    trend_signal = -1

# -----------------------------------------------------
# 🔮 PROCESO ESTOCÁSTICO (GBM)
# -----------------------------------------------------
returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
mu = returns.mean() * 252
sigma = returns.std() * np.sqrt(252)
S0 = hist['Close'].iloc[-1]

def simular_precio_accion(S0, mu, sigma, T=1, N=252, n_simulaciones=10):
    N = int(N * T)
    dt = T / N if N > 0 else 1/252
    precios = np.zeros((N+1, n_simulaciones))
    precios[0] = S0
    for i in range(1, N+1):
        z = np.random.standard_normal(n_simulaciones)
        precios[i] = precios[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return precios

precios_simulados = simular_precio_accion(S0, mu, sigma, T=T, n_simulaciones=n_sim)

fig2, ax3 = plt.subplots(figsize=(9,5))
ax3.plot(precios_simulados, lw=1)
ax3.set_title(f"Simulación estocástica del precio futuro de {ticker} ({horizonte})")
ax3.set_xlabel("Días")
ax3.set_ylabel("Precio (USD)")
st.pyplot(fig2)

precio_esperado = precios_simulados[-1].mean()
variacion = ((precio_esperado - S0) / S0) * 100

if variacion > 0:
    decision_stoc = f"Simulación: el precio podría **subir {variacion:.2f}%**. Escenario optimista."
    stoc_signal = 1
else:
    decision_stoc = f"Simulación: el precio podría **bajar {abs(variacion):.2f}%**. Escenario conservador."
    stoc_signal = -1

# -----------------------------------------------------
# 🧠 CONCLUSIÓN FINAL COMBINADA
# -----------------------------------------------------
total_signal = rsi_signal + trend_signal + stoc_signal

if total_signal >= 2:
    final = "✅ Señales alineadas al **alza** → Recomendación: **COMPRAR**"
elif total_signal <= -2:
    final = "🚫 Señales **bajistas** → Recomendación: **NO INVERTIR o VENDER**"
else:
    final = "⚖️ Señales **mixtas** → Recomendación: **ESPERAR o MANTENER posición**"

vol_hist = sigma * 100

# -----------------------------------------------------
# 📋 RESULTADOS
# -----------------------------------------------------
st.subheader("📋 Resultados del análisis")
st.write(decision_rsi)
st.write(decision_trend)
st.write(decision_stoc)
st.write(f"📉 Volatilidad histórica anual: **{vol_hist:.2f}%**")
st.markdown(f"### 🔎 Conclusión final: {final}")

# Precio esperado
st.metric(label=f"💰 Precio actual de {ticker}", value=f"${S0:.2f}")
st.metric(label=f"📈 Precio esperado en {horizonte}", value=f"${precio_esperado:.2f}", delta=f"{variacion:.2f}%")
