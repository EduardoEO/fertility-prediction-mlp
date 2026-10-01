# 🧬 Predicción de Fertilidad Seminal mediante Redes Neuronales Artificiales (MLP)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/imblearn-SMOTE-orange.svg)](https://imbalanced-learn.org/)
[![Dataset](https://img.shields.io/badge/UCI-Fertility_Dataset-lightgrey.svg)](https://archive.ics.uci.edu/dataset/244/fertility)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Proyecto Final** — Asignatura de *Redes Neuronales y Aprendizaje Profundo*  
> **Grado en Ingeniería Informática**  
> **Autores:** Eduardo Estefanía Ovejero & Guillem Alcubilla Pueyo  

---

## 📌 Documentación y Enlaces Rápidos

- 📄 **[Memoria Técnica Completa (PDF)](./docs/trabajo_final_fertilidad.pdf)** — Documento académico formal con fundamentos matemáticos, metodología, experimentación y conclusiones.
- 📊 **[Presentación del Proyecto (PPTX)](./docs/presentacion_fertilidad.pptx)** — Diapositivas de defensa del proyecto.
- 📓 **[Notebook Principal (Modelo Final con SMOTE)](./notebooks/04_fertility_SMOTE.ipynb)** — Flujo de trabajo completo y validación del mejor modelo.

---

## 📖 Descripción General

La fertilidad es un componente crucial de la salud reproductiva, pero los métodos diagnósticos habituales (seminogramas y análisis clínicos de laboratorio) suelen ser complejos, lentos, costosos e invasivos. Sin embargo, factores ambientales, hábitos de vida y antecedentes clínicos influyen de forma directa y no lineal en la calidad del esperma.

Este proyecto implementa y evalúa una arquitectura de **Red Neuronal Artificial tipo Perceptrón Multicapa (MLP)** como herramienta de cribado médico temprano no invasivo. El objetivo es determinar si un individuo presenta alteraciones en la calidad seminal basándose exclusivamente en sus hábitos sociodemográficos, estilos de vida y antecedentes médicos.

---

## 📂 Estructura del Repositorio

El repositorio sigue una organización modular estándar en Ciencia de Datos y Machine Learning:

```text
├── data/
│   ├── raw/
│   │   ├── fertility_Diagnosis.txt             # Dataset original en formato texto
│   │   └── fertility_DiagnosisCSV.csv          # Dataset original en formato CSV
│   └── processed/
│       └── fertility_Diagnosis_Preprocesado.csv # Dataset con target binarizado (0 = Normal, 1 = Alterado)
│
├── notebooks/
│   ├── 01_fertility_baseline.ipynb             # Exploración inicial, preprocesado y modelo base
│   ├── 02_fertility_FeatureEng.ipynb           # Experimentación con ingeniería de características (descartada)
│   ├── 03_fertility_Undersampling.ipynb        # Experimentación con balanceo por submuestreo (descartado)
│   └── 04_fertility_SMOTE.ipynb                # Modelo definitivo: SMOTE calibrado (k=3) + MLPClassifier
│
├── docs/
│   ├── trabajo_final_fertilidad.pdf            # Memoria técnica completa y fundamentación teórica
│   └── presentacion_fertilidad.pptx            # Presentación para la defensa del proyecto
│
├── .gitignore                                  # Reglas de exclusión para Git (venv, checkpoints, etc.)
├── requirements.txt                            # Dependencias necesarias para reproducir el entorno
└── README.md                                   # Documentación del proyecto
```

---

## 📊 Datos y Problemática Clínica

El conjunto de datos utilizado procede del repositorio oficial de la Universidad de California en Irvine:
🔗 **[UCI Machine Learning Repository: Fertility Dataset](https://archive.ics.uci.edu/dataset/244/fertility)** *(Gil et al., 2012)*.

### Desafíos Principales
1. **Muestra Reducida ($N = 100$):** Un tamaño muestral muy pequeño que aumenta el riesgo de sobreajuste y reduce el poder de las pruebas estadísticas univariables.
2. **Severo Desbalanceo de Clases:** 88% de casos con fertilidad normal (`N`) frente a un 12% con alteraciones (`O`). Sin técnicas adecuadas, los clasificadores tienden a predecir siempre la clase mayoritaria.

### Variables del Modelo (9 predictoras + 1 objetivo)

| Variable | Tipo | Rango / Codificación | Descripción |
| :--- | :--- | :--- | :--- |
| `season` | Categórica | `[-1, 1]` | Estación del análisis (-1: Invierno, -0.33: Primavera, 0.33: Verano, 1: Otoño) |
| `age` | Numérica | `[0, 1]` | Edad normalizada del paciente (original: 18 a 36 años) |
| `child_diseases` | Binaria | `{0, 1}` | Padecimiento de enfermedades infantiles graves (varicela, sarampión, etc.) |
| `accident` | Binaria | `{0, 1}` | Traumatismos o accidentes graves sufridos |
| `surgical_intervention`| Binaria | `{0, 1}` | Intervenciones quirúrgicas previas |
| `high_fevers` | Categórica | `{-1, 0, 1}` | Fiebres altas en el último año (-1: <3 meses, 0: >3 meses, 1: Ninguna) |
| `alcohol` | Numérica | `[0, 1]` | Frecuencia de consumo de alcohol |
| `smoking` | Categórica | `{-1, 0, 1}` | Hábito de tabaquismo (-1: Nunca, 0: Ocasional, 1: Diario) |
| `hrs_sitting` | Numérica | `[0, 1]` | Horas al día en posición sedente (original: 0 a 16 h) |
| **`diagnosis` (Target)** | **Binaria** | **`{0, 1}`** | **Diagnóstico: `0 = Normal`, `1 = Alterado`** |

---

## 🔬 Metodología y Decisiones de Diseño

```
  Dataset UCI (100 muestras)
              │
              ▼
   Preprocesado y Pipeline
   ├─ Codificación Target (0 = Normal, 1 = Alterado)
   ├─ Verificación de Nulos y Duplicados
   └─ Escalado Robusto (StandardScaler)
              │
              ▼
   Selección de Características
   └─ Pruebas ANOVA y Chi-cuadrado (χ²)
      ↳ Se decide conservar las 9 variables (evitar pérdida de contexto no lineal)
              │
              ▼
   Tratamiento del Desbalanceo
   ├─ Submuestreo ──► Descartado (colapso con ~24 muestras totales)
   └─ SMOTE Calibrado (k=3) ──► Seleccionado (evita solapamiento espacial)
              │
              ▼
   Red Neuronal MLPClassifier
   ├─ Entrada: 9 neuronas
   ├─ Capas Ocultas: (10, 5) con ReLU
   ├─ Optimizador: Adam (lr=0.005, alpha=1e-5, batch_size=16)
   └─ Validación: Stratified 3-Fold Cross-Validation
              │
              ▼
   Evaluación en Test (20% Hold-out)
   └─ Accuracy: 90% | Recall Minoritaria: 50%
```

1. **Selección de Características:** Las pruebas univariables (ANOVA para variables continuas y $\chi^2$ para categóricas) arrojaron $p > 0.05$ debido al tamaño muestral reducido. Se conservaron las 9 características originales, ya que el MLP está capacitado para extraer relaciones no lineales cruzadas que los tests individuales no detectan.
2. **Ingeniería de Características:** Se experimentó creando variables sintéticas (`trauma_fisico` y `riesgo_sedentario`). La matriz de correlación reveló severa multicolinealidad ($r = 0.91$ con `hrs_sitting`), provocando degradación en la matriz de confusión. En consecuencia, se descartaron formalmente.
3. **Calibración de SMOTE ($k=3$):** El valor habitual de $k=5$ producía interpolaciones artificiales demasiado distantes hacia la clase mayoritaria debido a la escasez de muestras alteradas (~10 por pliegue). Reducir a $k=3$ garantizó interpolaciones estrictamente locales sin introducir ruido.
4. **Arquitectura MLP:**
   - **Activación:** Se utilizó **ReLU** en lugar de Sigmoide para prevenir el desvanecimiento del gradiente (*vanishing gradient*).
   - **Regularización y Optimización:** Optimizador **Adam**, tasa de aprendizaje inicial de `0.005`, regularización L2 reducida ($\alpha = 10^{-5}$) y lotes pequeños (`batch_size=16`) durante 500 épocas sin detención prematura (*Early Stopping*) para aprovechar al máximo la información disponible.

---

## 📈 Resultados Principales

Evaluación sobre el conjunto de test independiente (20% de los datos, 20 pacientes no observados durante el entrenamiento):

| Métrica | Valor Obtenido | Relevancia Clínica |
| :--- | :---: | :--- |
| **Accuracy Global** | **90.00%** | Precisión global honesta y robusta |
| **Recall (Clase Minoritaria)** | **50.00%** | Detecta el 50% de pacientes alterados (1 de 2 en test) |
| **Verdaderos Negativos** | **17 / 18** | Especificidad del 94.4% en pacientes sanos |
| **Falsos Positivos** | **1** | Tasa mínima de falsas alarmas |
| **Falsos Negativos** | **1** | Mínimo error crítico médico |

> ⚠️ **La Paradoja de la Precisión:**  
> En conjuntos de datos altamente desbalanceados, predecir siempre la clase mayoritaria otorgaría un 88% de *Accuracy*, pero con una utilidad diagnóstica nula (Recall = 0%). Priorizar el **Recall** y el **F1-Score** sobre la clase minoritaria mediante SMOTE y ajuste de hiperparámetros rompió el sesgo trivial, dotando a la red de sensibilidad diagnóstica real.

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. Crear y activar un entorno virtual (recomendado)
```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar los Notebooks
```bash
jupyter lab
# o bien
jupyter notebook
```
Se recomienda abrir y ejecutar en orden numérico:
1. `notebooks/01_fertility_baseline.ipynb`
2. `notebooks/02_fertility_FeatureEng.ipynb`
3. `notebooks/03_fertility_Undersampling.ipynb`
4. `notebooks/04_fertility_SMOTE.ipynb` *(Mejor modelo final)*

---

## 📚 Referencias Bibliográficas

1. **Gil, D., Girela, J. L., De Juan, J., Gomez-Torres, M. J., & Johnsson, M.** (2012). *Predicting seminal quality with artificial intelligence methods*. Expert Systems with Applications, 39(16), 12564–12573.
2. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
4. **Montgomery, D. C.** (2019). *Design and Analysis of Experiments* (10th ed.). John Wiley & Sons.
5. **Siristatidis, C., Pouliakis, A., Chrelias, C., & Kassanos, D.** (2016). *Artificial intelligence in medicine: today and tomorrow in reproductive medicine*. Gynecological Endocrinology, 32(8), 597–604.

---

## 👥 Autores

- **Eduardo Estefanía Ovejero**
- **Guillem Alcubilla Pueyo**

*Trabajo académico realizado para la asignatura de Redes Neuronales y Aprendizaje Profundo (Grado en Ingeniería Informática).*
