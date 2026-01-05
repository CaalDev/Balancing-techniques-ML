# Proyecto de Clasificación de Espectros de Banano

Este proyecto contiene notebooks para el análisis y clasificación de datos espectrales de plantas de banano, utilizando diferentes técnicas de deep learning para manejar el desbalanceo de clases.

## 📋 Descripción General

El proyecto se enfoca en la clasificación de plantas de banano según su estado de salud y tratamiento aplicado, utilizando datos espectrales. Se implementan dos técnicas principales para abordar el problema de desbalanceo de clases:

- **LDAM Loss (Label-Distribution-Aware Margin Loss)**: Función de pérdida que ajusta los márgenes de decisión basándose en la distribución de las clases
- **MixUp**: Técnica de aumento de datos que genera muestras sintéticas mediante interpolación lineal

## 📂 Estructura de Notebooks

### Notebooks con LDAM Loss

#### 1. [LDAM_Loss_Sana_2 Clases.ipynb](LDAM_Loss_Sana_2%20Clases.ipynb)
**Objetivo:** Clasificación binaria de plantas sanas vs enfermas usando LDAM Loss

**Características:**
- Clasificación de 2 clases basada en la columna "Sana"
- Implementa LDAM Loss para manejar desbalanceo de clases
- Incluye visualizaciones con PCA, LDA y t-SNE
- Análisis exploratorio de distribución de datos
- Entrenamiento con redes neuronales profundas

#### 2. [LDAM_Loss_Tratamiento_2 Clases.ipynb](LDAM_Loss_Tratamiento_2%20Clases.ipynb)
**Objetivo:** Clasificación binaria basada en dos tipos de tratamiento usando LDAM Loss

**Características:**
- Clasificación de 2 clases basada en la columna "Tratamiento"
- Utiliza LDAM Loss para ajustar márgenes de decisión
- Visualización de distribución de clases
- Análisis de correlación entre variables
- Métricas de evaluación especializadas para datos desbalanceados

#### 3. [LDAM_Loss_Tratamiento_3 Clases.ipynb](LDAM_Loss_Tratamiento_3%20Clases.ipynb)
**Objetivo:** Clasificación multiclase de tres tipos de tratamiento usando LDAM Loss

**Características:**
- Clasificación de 3 clases basada en la columna "Tratamiento"
- Implementación de LDAM Loss para múltiples clases
- Mayor complejidad en el modelo debido a más clases
- Visualizaciones multidimensionales
- Evaluación con matrices de confusión y métricas por clase

### Notebooks con MixUp

#### 4. [MixUp_Sana_2_Clases.ipynb](MixUp_Sana_2_Clases.ipynb)
**Objetivo:** Clasificación binaria de plantas sanas vs enfermas usando MixUp

**Características:**
- Clasificación de 2 clases basada en la columna "Sana"
- Implementa técnica MixUp para aumento de datos sintéticos
- Generación de muestras interpoladas para balancear clases
- Visualizaciones con PCA, LDA y t-SNE
- Comparación de rendimiento con y sin MixUp

#### 5. [MixUp_Tratamiento_2 Clases.ipynb](MixUp_Tratamiento_2%20Clases.ipynb)
**Objetivo:** Clasificación binaria basada en dos tipos de tratamiento usando MixUp

**Características:**
- Clasificación de 2 clases basada en la columna "Tratamiento"
- Utiliza MixUp para crear muestras sintéticas mediante interpolación
- Mejora la generalización del modelo
- Análisis de distribución de clases post-MixUp
- Evaluación de robustez del modelo

#### 6. [MixUp_Tratamiento_3 Clases.ipynb](MixUp_Tratamiento_3%20Clases.ipynb)
**Objetivo:** Clasificación multiclase de tres tipos de tratamiento usando MixUp

**Características:**
- Clasificación de 3 clases basada en la columna "Tratamiento"
- Aplicación de MixUp en escenario multiclase
- Interpolación entre múltiples clases para balanceo
- Análisis detallado de rendimiento por clase
- Visualizaciones complejas de fronteras de decisión

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **PyTorch**: Framework principal para deep learning
- **Keras**: API de alto nivel para construcción de modelos
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Scikit-learn**: Preprocesamiento y métricas
- **Matplotlib & Seaborn**: Visualización de datos
- **Imbalanced-learn**: Manejo de datasets desbalanceados

## 📊 Flujo de Trabajo Común

Todos los notebooks siguen un flujo similar:

1. **Instalación de dependencias**
2. **Carga de datos** desde Google Drive
3. **Análisis exploratorio**:
   - Visualización de distribución de clases
   - Reducción de dimensionalidad (PCA, LDA, t-SNE)
   - Análisis de correlaciones
4. **Preprocesamiento**:
   - Limpieza de datos
   - Normalización/estandarización
   - División en train/test
5. **Entrenamiento del modelo**:
   - Implementación de LDAM Loss o MixUp
   - Optimización de hiperparámetros
6. **Evaluación**:
   - Matrices de confusión
   - Métricas de clasificación
   - Visualización de resultados

## 🚀 Uso

Los notebooks están diseñados para ejecutarse en **Google Colab**. Para usarlos:

1. Sube los notebooks a tu Google Drive
2. Asegúrate de tener los datos en la ruta correcta en tu Drive
3. Abre el notebook en Google Colab
4. Ejecuta las celdas secuencialmente

## 📈 Resultados Esperados

Cada notebook genera:
- Gráficos de distribución de clases
- Visualizaciones de reducción de dimensionalidad
- Matrices de confusión
- Métricas de clasificación (accuracy, precision, recall, F1-score)
- Curvas de entrenamiento (loss y accuracy)

## 🔍 Comparación de Técnicas

- **LDAM Loss**: Mejor para datasets con desbalanceo moderado, ajusta los márgenes de decisión sin aumentar datos
- **MixUp**: Efectivo para mejorar generalización, crea datos sintéticos, útil con desbalanceo severo

## 📝 Notas

- Los datasets deben estar en formato CSV con separador `;`
- Se recomienda usar GPU para acelerar el entrenamiento
- Los paths a los datos deben ajustarse según la ubicación en Google Drive

## 👥 Contribuciones

Este proyecto forma parte de un trabajo de investigación en clasificación de espectros de plantas de banano para detección temprana de enfermedades.

## 👥 Autores

Alejandro Martinez Valencia, Carlos Andres Aguirre Lopez



---
---

# Banana Spectrum Classification Project

This project contains notebooks for the analysis and classification of spectral data from banana plants, using different deep learning techniques to handle class imbalance.

## 📋 General Description

The project focuses on classifying banana plants according to their health status and applied treatment, using spectral data. Two main techniques are implemented to address the class imbalance problem:

- **LDAM Loss (Label-Distribution-Aware Margin Loss)**: Loss function that adjusts decision margins based on class distribution
- **MixUp**: Data augmentation technique that generates synthetic samples through linear interpolation

## 📂 Notebook Structure

### Notebooks with LDAM Loss

#### 1. [LDAM_Loss_Sana_2 Clases.ipynb](LDAM_Loss_Sana_2%20Clases.ipynb)
**Objective:** Binary classification of healthy vs diseased plants using LDAM Loss

**Features:**
- 2-class classification based on the "Sana" column
- Implements LDAM Loss to handle class imbalance
- Includes visualizations with PCA, LDA, and t-SNE
- Exploratory analysis of data distribution
- Training with deep neural networks

#### 2. [LDAM_Loss_Tratamiento_2 Clases.ipynb](LDAM_Loss_Tratamiento_2%20Clases.ipynb)
**Objective:** Binary classification based on two treatment types using LDAM Loss

**Features:**
- 2-class classification based on the "Tratamiento" column
- Uses LDAM Loss to adjust decision margins
- Visualization of class distribution
- Correlation analysis between variables
- Specialized evaluation metrics for imbalanced data

#### 3. [LDAM_Loss_Tratamiento_3 Clases.ipynb](LDAM_Loss_Tratamiento_3%20Clases.ipynb)
**Objective:** Multiclass classification of three treatment types using LDAM Loss

**Features:**
- 3-class classification based on the "Tratamiento" column
- LDAM Loss implementation for multiple classes
- Higher model complexity due to more classes
- Multidimensional visualizations
- Evaluation with confusion matrices and per-class metrics

### Notebooks with MixUp

#### 4. [MixUp_Sana_2_Clases.ipynb](MixUp_Sana_2_Clases.ipynb)
**Objective:** Binary classification of healthy vs diseased plants using MixUp

**Features:**
- 2-class classification based on the "Sana" column
- Implements MixUp technique for synthetic data augmentation
- Generation of interpolated samples to balance classes
- Visualizations with PCA, LDA, and t-SNE
- Performance comparison with and without MixUp

#### 5. [MixUp_Tratamiento_2 Clases.ipynb](MixUp_Tratamiento_2%20Clases.ipynb)
**Objective:** Binary classification based on two treatment types using MixUp

**Features:**
- 2-class classification based on the "Tratamiento" column
- Uses MixUp to create synthetic samples through interpolation
- Improves model generalization
- Analysis of post-MixUp class distribution
- Model robustness evaluation

#### 6. [MixUp_Tratamiento_3 Clases.ipynb](MixUp_Tratamiento_3%20Clases.ipynb)
**Objective:** Multiclass classification of three treatment types using MixUp

**Features:**
- 3-class classification based on the "Tratamiento" column
- Application of MixUp in multiclass scenario
- Interpolation between multiple classes for balancing
- Detailed per-class performance analysis
- Complex visualizations of decision boundaries

## 🛠️ Technologies Used

- **Python 3.x**
- **PyTorch**: Main framework for deep learning
- **Keras**: High-level API for model building
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Scikit-learn**: Preprocessing and metrics
- **Matplotlib & Seaborn**: Data visualization
- **Imbalanced-learn**: Handling imbalanced datasets

## 📊 Common Workflow

All notebooks follow a similar flow:

1. **Dependencies installation**
2. **Data loading** from Google Drive
3. **Exploratory analysis**:
   - Class distribution visualization
   - Dimensionality reduction (PCA, LDA, t-SNE)
   - Correlation analysis
4. **Preprocessing**:
   - Data cleaning
   - Normalization/standardization
   - Train/test split
5. **Model training**:
   - LDAM Loss or MixUp implementation
   - Hyperparameter optimization
6. **Evaluation**:
   - Confusion matrices
   - Classification metrics
   - Results visualization

## 🚀 Usage

The notebooks are designed to run on **Google Colab**. To use them:

1. Upload the notebooks to your Google Drive
2. Ensure you have the data in the correct path in your Drive
3. Open the notebook in Google Colab
4. Execute the cells sequentially

## 📈 Expected Results

Each notebook generates:
- Class distribution plots
- Dimensionality reduction visualizations
- Confusion matrices
- Classification metrics (accuracy, precision, recall, F1-score)
- Training curves (loss and accuracy)

## 🔍 Technique Comparison

- **LDAM Loss**: Better for datasets with moderate imbalance, adjusts decision margins without augmenting data
- **MixUp**: Effective for improving generalization, creates synthetic data, useful with severe imbalance

## 📝 Notes

- Datasets must be in CSV format with `;` separator
- GPU usage is recommended to accelerate training
- Data paths must be adjusted according to location in Google Drive

## 👥 Contributions

This project is part of a research work on banana plant spectrum classification for early disease detection.

## 👥 Authors

Alejandro Martinez Valencia, Carlos Andres Aguirre Lopez
