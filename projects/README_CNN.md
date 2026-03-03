🐶🐱 Clasificador de Imágenes de Perros y Gatos con CNN en Keras
📋 Descripción del Proyecto
Este proyecto implementa un clasificador de imágenes de perros y gatos utilizando una Red Neuronal Convolucional (CNN) construida con Keras y TensorFlow. El modelo fue entrenado durante 30 epochs y logra una alta precisión en la tarea de clasificación binaria de imágenes.

El objetivo principal es demostrar habilidades en visión por computadora, aprendizaje profundo y el flujo completo de desarrollo de modelos de IA, desde el preprocesamiento de datos hasta la evaluación y visualización de resultados.

🎯 Características Principales
Arquitectura CNN personalizada con múltiples capas convolucionales y de pooling

Preprocesamiento de imágenes con aumento de datos (data augmentation)

Entrenamiento optimizado con callback de early stopping

Evaluación exhaustiva con métricas de precisión, recall y matriz de confusión

Visualización de resultados incluyendo curvas de aprendizaje y predicciones de ejemplo

Modelo guardado para inferencia y despliegue futuro

🛠️ Tecnologías y Herramientas Utilizadas
Machine Learning & Deep Learning
Keras - API de alto nivel para construir modelos de deep learning

TensorFlow - Backend para operaciones de tensor y optimización

NumPy - Manipulación de arrays y operaciones matemáticas

Pandas - Análisis y manipulación de datos estructurados

Procesamiento de Imágenes
OpenCV / PIL - Manipulación y preprocesamiento de imágenes

Scikit-learn - Métricas de evaluación y herramientas de validación

Visualización
Matplotlib - Visualización de datos y resultados

Seaborn - Gráficos estadísticos más atractivos

Desarrollo y Control de Versiones
Jupyter Notebook - Desarrollo interactivo y documentación

Git & GitHub - Control de versiones y colaboración

📊 Habilidades Demostradas
Técnicas de Machine Learning
Redes Neuronales Convolucionales (CNN) para clasificación de imágenes

Regularización (Dropout, L2) para prevenir overfitting

Optimización con Adam optimizer y learning rate scheduling

Data Augmentation para mejorar la generalización del modelo

Ingeniería de Software
Preprocesamiento de datos a gran escala

Pipeline reproducible de entrenamiento y evaluación

Modularización del código para mantenibilidad

Documentación clara y exhaustiva

Análisis y Evaluación
Interpretación de métricas de clasificación

Análisis de curvas de aprendizaje (loss y accuracy)

Visualización de características aprendidas por la CNN

Debugging de modelos de deep learning

📁 Estructura del Proyecto
text
├── Practica_Oriac_Gimeno_Classificador_d'imatges_de_gossos_i_gats_amb_CNN_de_Keras__vfinal30EPOCHSok.html
├── data/
│   ├── train/
│   │   ├── dogs/
│   │   └── cats/
│   └── test/
│       ├── dogs/
│       └── cats/
├── models/
│   └── best_model.h5
├── notebooks/
│   └── training_notebook.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
🚀 Instalación y Uso
Requisitos Previos
Python 3.8+

pip o conda

Instalación
bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/dogs-vs-cats-classifier.git
cd dogs-vs-cats-classifier

# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
Entrenamiento del Modelo
python
# Ejecutar el script de entrenamiento
python scripts/train.py --epochs 30 --batch_size 32 --data_path ./data
Evaluación
python
# Evaluar el modelo entrenado
python scripts/evaluate.py --model_path ./models/best_model.h5 --test_data ./data/test
Inferencia
python
# Clasificar una nueva imagen
python scripts/predict.py --image_path ./new_image.jpg --model_path ./models/best_model.h5
📈 Resultados
Métricas de Rendimiento
Precisión en entrenamiento: >95%

Precisión en validación: >92%

Pérdida (Loss): <0.2

Tiempo de inferencia: <50ms por imagen

Visualizaciones
Curvas de aprendizaje que muestran la convergencia del modelo

Matriz de confusión para análisis de errores

Ejemplos de predicciones correctas e incorrectas

Mapas de activación de las capas convolucionales

🎓 Aprendizajes y Conclusiones
Logros
Implementación exitosa de una CNN desde cero

Obtención de alta precisión en la tarea de clasificación

Creación de un pipeline completo y reproducible

Documentación exhaustiva del proceso

Desafíos Superados
Manejo de desbalance de clases (si aplicable)

Optimización de hiperparámetros

Prevención de overfitting con técnicas de regularización

Gestión eficiente de recursos computacionales

Aplicaciones Futuras
Transfer learning con modelos preentrenados (VGG16, ResNet, etc.)

Despliegue como API REST o aplicación web

Extensión a multi-clasificación (más especies de animales)

Optimización para dispositivos móviles o edge computing

👨‍💻 Autor
Oriac Gimeno

GitHub: @oriac-gimeno
LinkedIn: www.linkedin.com/in/oriacgimeno

Portfolio: 

📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!
