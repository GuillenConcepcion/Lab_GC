# 🏀 DS_ETL_NBA

Un proyecto de portafolio que demuestra un pipeline **ETL (Extract, Transform, Load)** robusto implementado en Python. Integra datos de ventas locales en formato CSV y datos deportivos externos consumidos a través de una API JSON, procesándolos y cargándolos en una base de datos PostgreSQL local gestionada con Docker.


##  Descripción del Proyecto
- E/S de archivos y API: trabajar con diferentes formatos de archivos, obtener datos de las API
- Gestión de bases de datos: Interfaz con bases de datos mediante SQLAlchemy para gestionar la persistencia de datos
- Manejo de errores: Implementar los mecanismos de manejo de errores necesarios para garantizar la integridad de los datos
- Programación: automatización del proceso ETL mediante trabajos cron

## 🚀 Arquitectura del Pipeline

1. **Extract**: 
   - Lee el dataset de ventas local (`sales_data.csv`).
   - Consume datos de jugadores de la API pública de la NBA ([balldontlie.io](https://balldontlie.io/)).
2. **Transform**: 
   - Limpia datos faltantes en ventas, estandariza columnas y formatea fechas usando `pandas`.
   - Aplana la estructura JSON anidada de la API de la NBA para convertirla en tablas relacionales.
3. **Load**: 
   - Guarda los datos transformados en CSVs de respaldo en `data/processed/`.
   - Carga la información validada en una base de datos PostgreSQL mediante `SQLAlchemy`.
4. **Automatización & Logging**: 
   - Registro detallado de eventos y manejo de excepciones mediante el módulo `logging`.
   - Estructura preparada para ejecución periódica usando la librería `schedule`.

## 📂 Estructura del Proyecto

```text
etl_nba_portafolio/
├── data/
│   ├── raw/             # Ubica aquí tu `sales_data.csv`
│   └── processed/       # Datos limpios y estructurados de salida
├── logs/                # Archivos de registro de errores/eventos (etl_pipeline.log)
├── src/
│   ├── config.py        # Configuración y variables de entorno variables
│   ├── etl_pipeline.py  # Script principal del pipeline ETL
│   └── requirements.txt # Dependencias del proyecto
├── .env.example         # Plantilla de variables de entorno
├── docker-compose.yml   # Definición del servicio local de PostgreSQL
└── README.md            # Documentación del proyecto
```

## 🛠️ Requisitos de Instalación

- Python 3.9+
- Docker y Docker Compose (para correr la base de datos localmente)

### 1. Preparar Entorno de Python

Clona el repositorio y crea un entorno virtual (recomendado):
```bash
python -m venv venv
# Activar en Windows
venv\Scripts\activate
# Activar en Mac/Linux
source venv/bin/activate
```

Instala las dependencias:
```bash
pip install -r src/requirements.txt
```

### 2. Configurar la Base de Datos con Docker

Inicia el contenedor de PostgreSQL en segundo plano:
```bash
docker-compose up -d
```
Esto levantará una instancia de PostgreSQL accesible en el puerto `5432` con usuario y contraseña por defecto definidos en el archivo `docker-compose.yml`.

### 3. Configurar Variables de Entorno y Datos

1. Renombra el archivo `.env.example` a `.env` (si usas contraseñas personalizadas, actualízalo para que coincida con el docker-compose).
2. **Descarga de Datos de Ventas:**
   - Descarga el dataset `Sample Sales Data` de [Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data).
   - Coloca el archivo descargado como `sales_data.csv` dentro de la carpeta `data/raw/`.

## 🏃‍♂️ Ejecución

Para iniciar el pipeline y ver los logs en tiempo real por consola:

```bash
python src/etl_pipeline.py
```

Al terminar, los datos limpios estarán disponibles en las tablas `sales_data` y `nba_players` de la base de datos `portfolio_db`, así como en la carpeta `data/processed/`.
