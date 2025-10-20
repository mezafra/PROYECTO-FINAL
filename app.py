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

# -----------------------------------------------------
# 🧾 CADENA DE OPCIONES, SELECCIÓN Y SIMULADOR DE ORDEN (AGREGADO)
# (NO SE MODIFICÓ NADA DE LO ANTERIOR; esto es adición)
# -----------------------------------------------------
st.markdown("---")
st.subheader("🧾 Cadena de opciones y recomendación")

# Intentar obtener expiraciones
try:
    expirations = data.options  # lista de fechas en formato 'YYYY-MM-DD'
except Exception as e:
    expirations = []
    st.warning("No se pudieron obtener las expiraciones de opciones desde Yahoo Finance.")

if not expirations:
    st.info("No hay cadena de opciones disponible para este ticker o Yahoo no devuelve expiraciones.")
else:
    # Selector de fecha de vencimiento
    exp_choice = st.selectbox("Seleccione fecha de vencimiento", expirations, index=0)

    # Descargar cadena para la expiración seleccionada
    try:
        chain = data.option_chain(exp_choice)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception as e:
        st.error("Error al descargar la cadena de opciones para la expiración seleccionada.")
        calls = pd.DataFrame()
        puts = pd.DataFrame()

    # Si hay datos, procesarlos
    if not calls.empty or not puts.empty:
        # Añadir columna tipo
        if not calls.empty:
            calls = calls.assign(type='call')
        if not puts.empty:
            puts = puts.assign(type='put')

        # Unir ambos
        options = pd.concat([calls, puts], ignore_index=True, sort=False)
        # limpiar columnas faltantes
        if 'lastPrice' not in options.columns:
            options['lastPrice'] = np.nan
        if 'strike' not in options.columns:
            st.error("La cadena de opciones no contiene la columna 'strike'.")
        else:
            # Calcular distancia al precio esperado
            options['dist_to_pred'] = (options['strike'] - precio_esperado).abs()

            # opción más cercana en general
            idx_min = options['dist_to_pred'].idxmin()
            closest_option = options.loc[idx_min]

            # opción recomendada según señal total: si total_signal >0 -> call, <0 -> put, =0 -> neutral
            if total_signal > 0:
                preferred_type = 'call'
            elif total_signal < 0:
                preferred_type = 'put'
            else:
                preferred_type = None

            # buscar la opción del tipo preferido más cercana (si existe)
            preferred_option = None
            if preferred_type is not None:
                subset = options[options['type'] == preferred_type]
                if not subset.empty:
                    preferred_option = subset.loc[subset['dist_to_pred'].idxmin()]

            # Mostrar la opción más cercana en pantalla
            st.markdown("**Opción cuyo strike está más cercano al precio predicho:**")
            st.dataframe(closest_option.to_frame().T)

            if preferred_option is not None:
                st.markdown(f"**Opción recomendada por la señal ({'alcista' if preferred_type=='call' else 'bajista'}):**")
                st.dataframe(preferred_option.to_frame().T)
            else:
                st.info("Señal mixta/neutral — no se sugiere explícitamente call ni put. Revisa RSI y tendencia.")

            # Permitir al usuario seleccionar cuál de las dos (closest / preferred) quiere "ejecutar"
            choice_map = {"Opción más cercana": closest_option}
            if preferred_option is not None:
                choice_map["Opción recomendada"] = preferred_option

            sel_key = st.selectbox("Selecciona qué opción simular la ejecución", list(choice_map.keys()))
            option_to_trade = choice_map[sel_key]

            # Input de cantidad / direction
            colq1, colq2 = st.columns([2,2])
            with colq1:
                qty = st.number_input("Cantidad de contratos (enteros)", min_value=1, value=1, step=1)
            with colq2:
                action = st.selectbox("Acción simulada", ["Comprar (Long)", "Vender (Short)"])

            # Mostrar resumen antes de ejecutar
            st.markdown("**Resumen de la orden simulada**")
            st.write(f"Ticker: **{ticker}**")
            st.write(f"Vencimiento: **{exp_choice}**")
            st.write(f"Tipo: **{option_to_trade['type']}**")
            st.write(f"Strike: **{option_to_trade['strike']}**")
            last_price = option_to_trade.get('lastPrice', np.nan)
            st.write(f"Prima (lastPrice): **{last_price}**")
            st.write(f"Cantidad de contratos: **{qty}**")
            st.write(f"Acción: **{action}**")

            # Botón de ejecución simulada
            if st.button("Ejecutar orden simulada"):
                # Mensaje de confirmación simple
                notional = qty * 100 * (last_price if not np.isnan(last_price) else 0)
                st.success(f"Orden simulada: {action} {qty} contratos de {ticker} ({option_to_trade['type']}) strike {option_to_trade['strike']} para vencimiento {exp_choice}. Prima por contrato: {last_price}. Notional aproximado: ${notional:,.2f}")

            # Graficar payoff simplificado (neto de prima) en vencimiento
            try:
                K = float(option_to_trade['strike'])
                premium = float(option_to_trade.get('lastPrice', 0.0))  # usar lastPrice como prima aproximada
                # rango de precios en vencimiento alrededor del strike
                S_min = max(0, S0 * 0.5)
                S_max = S0 * 1.5
                S_range = np.linspace(S_min, S_max, 200)
                if option_to_trade['type'] == 'call':
                    payoff = np.maximum(S_range - K, 0) - premium
                else:
                    payoff = np.maximum(K - S_range, 0) - premium

                # neto por contrato y por la cantidad
                payoff_net = payoff * 100 * qty

                fig3, ax4 = plt.subplots(figsize=(8,4))
                ax4.plot(S_range, payoff_net, linewidth=1.5)
                ax4.axhline(0, color='black', linestyle='--')
                ax4.set_title(f"Payoff en vencimiento ({option_to_trade['type'].upper()}) - neto de prima")
                ax4.set_xlabel("Precio subyacente en vencimiento (USD)")
                ax4.set_ylabel("Payoff neto (USD)")
                st.pyplot(fig3)

            except Exception as e:
                st.info("No se pudo graficar el payoff por falta de datos de strike/prima.")

            # -----------------------------------------------------
            # Conclusión final automática que integra RSI + precio predicho + opción
            # -----------------------------------------------------
            conclusion_lines = []
            conclusion_lines.append(f"Precio actual: ${S0:.2f}. Precio esperado ({horizonte}): ${precio_esperado:.2f} ({variacion:.2f}%).")
            conclusion_lines.append(decision_rsi)
            conclusion_lines.append(decision_trend)
            conclusion_lines.append(decision_stoc)

            if preferred_option is not None:
                recommended_text = f"Según la señal combinada, se sugiere considerar la **{preferred_option['type'].upper()}** con strike **{preferred_option['strike']}** y vencimiento **{exp_choice}** (prima aprox. {preferred_option.get('lastPrice', 'N/A')})."
            else:
                recommended_text = "Señal mixta/neutral: no hay recomendación clara de CALL o PUT; considerar esperar o hacer hedging."

            conclusion_lines.append(recommended_text)

            # Texto final resumido
            st.markdown("### 🧾 Conclusión automática integrada")
            st.write(" ".join(conclusion_lines))
    else:
        st.info("La cadena de opciones no contiene calls ni puts para la expiración seleccionada.")


