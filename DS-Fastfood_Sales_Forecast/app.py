import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import yaml

# Carga de Configuración
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuración de la App
st.set_page_config(page_title=config['project']['name'], page_icon="🍔", layout="wide")

# Título Principal
st.title(f"🍔 {config['project']['name']} - Dashboard")
st.markdown("""
Este panel interactivo permite navegar entre el **Análisis Exploratorio de Datos (EDA)** para entender el comportamiento histórico de las ventas, 
y las **Predicciones del Modelo de Regresión** para optimizar inventarios futuros.
""")

st.divider()

# Definir Pestañas (Tabs)
tab1, tab2 = st.tabs(["🔍 Análisis Exploratorio (EDA)", "📈 Predicciones ML"])

# ----------------- Funciones de Carga de Datos -----------------
@st.cache_data
def load_raw_data():
    raw_path = config['data']['raw']
    if not os.path.exists(raw_path):
        return None
    df = pd.read_csv(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['item_price', 'transaction_amount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
    return df

@st.cache_data
def load_predictions():
    predictions_path = config['data']['predictions']
    if not os.path.exists(predictions_path):
        return None
    df = pd.read_csv(predictions_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

@st.cache_data
def load_model_metrics():
    metrics_path = config['data']['metrics']
    if not os.path.exists(metrics_path):
        return None
    return pd.read_csv(metrics_path)

raw_data = load_raw_data()
df_results = load_predictions()
df_metrics = load_model_metrics()

# ================= TAB 1: Análisis Exploratorio =================
with tab1:
    st.header("Análisis Exploratorio y Descriptivo de Ventas")
    if raw_data is None:
        st.warning(f"⚠️ No se encontraron datos crudos en `{config['data']['raw']}`.")
    else:
        df = raw_data.copy()
        
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("💰 Ingresos Totales", f"${df['transaction_amount'].sum():,.2f}")
        with colB:
            top_product = df.groupby('item_name')['quantity'].sum().idxmax()
            st.metric("🍔 Más Vendido", top_product)
        with colC:
            avg_ticket = df['transaction_amount'].mean()
            st.metric("📊 Ticket Promedio", f"${avg_ticket:.2f}")
        
        st.divider()
        
        left_col, right_col = st.columns((2, 3))
        with left_col:
            st.markdown("#### Popularidad de Productos")
            product_sales = df.groupby('item_name')['quantity'].sum().sort_values(ascending=False).reset_index()
            fig_prod = px.bar(product_sales, x='quantity', y='item_name', orientation='h', 
                             color='quantity', color_continuous_scale='Sunset')
            fig_prod.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig_prod, use_container_width=True)
        
        with right_col:
            st.markdown("#### Evolución de Ingresos Mensuales")
            df['year_month'] = df['date'].dt.to_period('M').astype(str)
            monthly_revenue = df.groupby('year_month')['transaction_amount'].sum().reset_index()
            fig_month = px.line(monthly_revenue, x='year_month', y='transaction_amount', markers=True)
            fig_month.update_traces(line_color='#FF4B4B', line_width=4)
            fig_month.update_layout(xaxis_title="Mes", yaxis_title="Ingresos ($)")
            st.plotly_chart(fig_month, use_container_width=True)

# ================= TAB 2: Predicciones ML =================
with tab2:
    if df_results is None or df_metrics is None:
        st.warning("⚠️ No se encontraron predicciones. Ejecuta el pipeline para entrenar los modelos.")
    else:
        st.markdown("<h3 style='text-align: center;'>🏆 Leaderboard de Modelos (Enterprise Metrics)</h3>", unsafe_allow_html=True)
        
        # Enterprise Leaderboard
        st.dataframe(df_metrics.style.highlight_min(subset=['Test_RMSE', 'MAE', 'MAPE', 'SMAPE'], color='#d4edda')
                                        .highlight_max(subset=['R2', 'Waste_Reduction_%'], color='#d4edda')
                                        .format({
                                            "CV_RMSE": "{:.4f}", 
                                            "Test_RMSE": "{:.4f}", 
                                            "MAE": "{:.4f}", 
                                            "R2": "{:.4f}",
                                            "MAPE": "{:.2%}",
                                            "SMAPE": "{:.2%}",
                                            "Waste_Reduction_%": "{:.1f}%"
                                        }), 
                      use_container_width=True)
        
        st.divider()
        
        best_model_name = df_metrics.iloc[0]['Model']
        model_cols = [col for col in df_results.columns if col.startswith('Predicted_')]
        model_names = [col.replace('Predicted_', '') for col in model_cols]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_model = st.selectbox("Modelo a visualizar:", model_names, index=model_names.index(best_model_name))
            
            # Show specific metrics for selected model
            m = df_metrics[df_metrics['Model'] == selected_model].iloc[0]
            st.metric("Waste Reduction", f"{m['Waste_Reduction_%']:.1f}%")
            st.metric("Accuracy (1-MAPE)", f"{1-m['MAPE']:.1%}")

        with col2:
            selected_col = f"Predicted_{selected_model}"
            daily_results = df_results.groupby('date').sum()
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=daily_results.index, y=daily_results['Actual'], name='Real', line=dict(color='#2E86AB')))
            fig_pred.add_trace(go.Scatter(x=daily_results.index, y=daily_results[selected_col], name=f'Pred ({selected_model})', line=dict(color='#F24236', dash='dash')))
            fig_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_pred, use_container_width=True)
