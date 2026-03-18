# Guía de Uso: Plantilla Cookiecutter DS (Custom)

Esta carpeta contiene los cimientos para convertir tu proyecto de Forecast en una plantilla reutilizable.

## 🏁 Cómo usar la plantilla
1. **Instala Cookiecutter**:
   ```bash
   pip install cookiecutter
   ```
2. **Genera un nuevo proyecto**:
   Ejecuta el siguiente comando apuntando a la carpeta de este proyecto (o a un repo de git):
   ```bash
   cookiecutter .
   ```
3. **Responde a las preguntas**:
   El archivo `cookiecutter.json` te pedirá:
   - `project_name`: "DS_Fastfood_Sales_Forecast".
   - `environment_manager`: conda
   - `license`: MIT

## 📂 Archivo `cookiecutter.json`
He creado un archivo `cookiecutter.json` en la raíz con las siguientes opciones configuradas:
- Variables dinámicas para nombres de repositorio.
- Opciones para incluir Docker y MLflow.
- Selección de entorno virtual.

## 📝 Próximos Pasos para la Automatización
Para que la plantilla sea 100% funcional, debes:
1. Crear una carpeta llamada `{{ cookiecutter.repo_name }}/` en tu repositorio de plantillas.
2. Mover todos los archivos (`src/`, `data/`, `app/`) dentro de esa carpeta.
3. Reemplazar las menciones "Fastfood" por `{{ cookiecutter.project_name }}` en los archivos `.py` y `.yaml`.
