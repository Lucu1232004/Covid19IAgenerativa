# Covid19IAgenerativa - Practical Exam 1 | IA Generativa

Repo: https://github.com/Lucu1232004/Covid19IAgenerativa
Materia: IA Generativa - Prof. Jose Luis Paniagua Jaramillo - Universidad Autonoma de Occidente
Equipo: Samuel Patiño y Nicolás Peña (grupo COVID-19)
Dataset: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## Objetivo
Autoencoders convolucionales para reconstruccion (CIFAR-10) y denoising (COVID X-Ray). La parte 3 la trabaja el compañero por separado.

## Estructura que se sube
```
├── notebooks/
│   ├── 01cifar-autoencoder.ipynb   # Parte 1: CIFAR-10 + PCA y t-SNE
│   └── 02covid-denoising.ipynb     # Parte 2: Denoising COVID con ruido 0.5
├── README.md
└── requirements.txt
```

## Dataset COVID-19 (pranavraikokte)
- Rayos X de torax en gris, 3 clases: Covid / Normal / Viral Pneumonia
- Estructura: train y test, cada uno con 3 subcarpetas por clase
- Es pequeno (unos 300 imgs) asi que ojo con el overfitting
- Todo a 64x64 y entre 0 y 1. Lo cargamos como rgb para reusar la misma arquitectura.

### Como bajarlo en Kaggle (sin subir zip a mano)
```python
import kagglehub
path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
print("Path to dataset files:", path)
```

## Como correr
- Todo en Kaggle con GPU: Settings > Accelerator > GPU + Internet ON
- Abrir los notebooks de la carpeta notebooks y dar Run All
