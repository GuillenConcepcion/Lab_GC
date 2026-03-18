import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Carga de Configuración si existe
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = None

def main(input_path, output_dir="reports/figures/eda"):
    print(f"[EDA] Iniciando EDA Avanzado desde: {input_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"[Error] No se encontró el dataset en {input_path}")
        return
    
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Preprocesamiento inicial (Limpieza)
    for col in ['item_price', 'transaction_amount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)

    # ---------------------------------------------------------
    # 1. Identificación de Discrepancias (Valores Faltantes y Outliers)
    # ---------------------------------------------------------
    print("\n[1] Analizando Discrepancias (Missing Values & Outliers)...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title("Mapa de Calor de Valores Faltantes")
    plt.savefig(f"{output_dir}/discrepancies_missing_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['quantity'], color='skyblue')
    plt.title("Outliers en Cantidad (Quantity)")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['transaction_amount'], color='salmon')
    plt.title("Outliers en Monto de Transacción")
    plt.savefig(f"{output_dir}/discrepancies_outliers.png")
    plt.close()

    # ---------------------------------------------------------
    # 2 & 5. Relaciones, Patrones y Tendencias
    # ---------------------------------------------------------
    print("\n[2 & 5] Detectando Relaciones y Tendencias...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.savefig(f"{output_dir}/relationships_correlation.png")
    plt.close()

    # Tendencias temporales (Ventas diarias)
    daily_sales = df.groupby('date')['transaction_amount'].sum().reset_index()
    daily_sales['rolling_mean'] = daily_sales['transaction_amount'].rolling(window=7).mean()
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales['date'], daily_sales['transaction_amount'], label='Ventas Diarias', alpha=0.3)
    plt.plot(daily_sales['date'], daily_sales['rolling_mean'], label='Media Móvil 7 días', color='red')
    plt.title("Tendencia Temporal de Ventas")
    plt.legend()
    plt.savefig(f"{output_dir}/trends_temporal.png")
    plt.close()

    # ---------------------------------------------------------
    # 3 & 7. Calidad de Datos y Base para Modelado Estadístico
    # ---------------------------------------------------------
    print("\n[3 & 7] Verificando Calidad y Fundamentos Estadísticos...")
    # Verificación de consistencia en nombres de productos
    unique_items = sorted(df['item_name'].unique())
    print(f"Items únicos detectados ({len(unique_items)}): {unique_items[:5]}...")
    
    # Test de Normalidad (Shapiro-Wilk) - Punto 7
    # Sample to avoid p-value issues with very large datasets
    sample_size = min(len(df), 500)
    stat, p = stats.shapiro(df['transaction_amount'].sample(sample_size))
    print(f"Test de Normalidad (Shapiro) en sample de {sample_size}: p-value = {p:.4f}")
    if p < 0.05:
        print("Insight: El monto de transacción no sigue una distribución normal. Se sugiere transformación Log.")

    # ---------------------------------------------------------
    # 4. Eficiencia de Análisis (Scaling & Distributions)
    # ---------------------------------------------------------
    print("\n[4] Mejorando la Eficiencia de Análisis (Scaling)...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['transaction_amount'], kde=True, color='blue')
    plt.title("Distribución Original")
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df['transaction_amount']), kde=True, color='green')
    plt.title("Distribución (Log-Transform)")
    plt.savefig(f"{output_dir}/efficiency_normalization.png")
    plt.close()

    # ---------------------------------------------------------
    # 6. Facilitación de la Comunicación
    # ---------------------------------------------------------
    print("\n[6] Generando Resumen Ejecutivo para Comunicación...")
    # Gráfico de barras resumen por tipo de item
    plt.figure(figsize=(10, 6))
    df.groupby('item_type')['transaction_amount'].sum().sort_values().plot(kind='barh', color='teal')
    plt.title("Ventas Totales por Tipo de Producto")
    plt.xlabel("Ingresos Totales ($)")
    plt.savefig(f"{output_dir}/communication_summary.png")
    plt.close()

    # ---------------------------------------------------------
    # 8. Generalizabilidad del Modelo
    # ---------------------------------------------------------
    print("\n[8] Evaluando Generalizabilidad (ANOVA)...")
    # Análisis de varianza (Anova) por momento del día
    times = df['time_of_sale'].unique()
    groups = [df[df['time_of_sale'] == t]['transaction_amount'] for t in times]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"ANOVA (Impacto de 'Time of Sale'): F={f_stat:.2f}, p={p_anova:.4f}")
    if p_anova < 0.05:
        print("Insight: 'Time of Sale' es un predictor fuerte (p < 0.05). Incluir en el modelo final.")

    print(f"\n[Terminado] EDA Avanzado completado. Gráficos guardados en '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Avanzado de Ventas")
    parser.add_argument("--input", type=str, default="data/raw/balaji_fast_food_sales.csv")
    parser.add_argument("--output_dir", type=str, default="reports/figures/eda")
    args = parser.parse_args()
    main(args.input, args.output_dir)
