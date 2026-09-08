# Covid19IAgenerativa - Practical Exam 1 | IA Generativa

Repo: https://github.com/Lucu1232004/Covid19IAgenerativa
Materia: IA Generativa - Prof. Jose Luis Paniagua Jaramillo - Universidad Autonoma de Occidente
Equipo: Samuel Patiño y Nicolás Peña (grupo 7, COVID-19)
Dataset asignado: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## Objetivo
Autoencoders convolucionales para reconstruccion (CIFAR-10), denoising (COVID X-Ray) y clasificacion con stacked CAE + SVM (COVID X-Ray). Todo con MSE y evaluacion honesta en test.

## Estructura que se entrega
```
├── notebooks/
│   ├── 00cnn-color-images.ipynb              # Parte 0: color, carga y prepro (base de todo)
│   ├── 01cifar-autoencoder.ipynb             # Parte 1: CIFAR-10 + PCA/t-SNE + kNN (train 0.00253/val 0.00117)
│   ├── 02covid-denoising.ipynb               # Parte 2: 8 estrategias x 2 ruidos, gana parches (25.16 dB/0.819 en 0.1)
│   ├── 03covid-denoising-preentrenado.ipynb  # Parte 2 ext: CORRIDO test 26.92 dB/0.8721 (pre 27.04 vs cero 26.88 val, 16200 parches)
│   └── 04stacked-clasificacion.ipynb         # Parte 3: CV 5-fold, base_profe + SVM -> test acc/F1 0.91
├── doc/                                      # papers consultados por parte
├── README.md
└── requirements.txt
```
Autoria: una sola firma en `00` celda 0. Los demas cuadernos no repiten nombres.

## Dataset COVID-19 (pranavraikokte)
- Rayos X de torax en gris, 3 clases: Covid (111 train) / Normal (70) / Viral Pneumonia (70); test 66
- Estructura con subcarpeta intermedia: `Covid19-dataset/train/{...} + test/{...}`
- Pequeno (251 train): riesgo de overfitting -> split 200/51 estratificado + test intacto + parches + aumento
- Todo a 64x64 RGB (gris repetido x3) en [0,1] float32 para reusar la misma arquitectura

### Como bajarlo en Kaggle (sin subir zip a mano)
```python
import kagglehub
path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
print("Path to dataset files:", path)
```
Externo solo para preentrenar (03, corrido): `tawsifurrahman/covid19-radiography-database` (usadas 12000 imgs 64x64, no supervisado, eval solo en asignado). Parches traslapados paso 8: 200x81 = 16200 muestras.

## Como correr
- Todo en Kaggle con GPU: Settings > Accelerator > GPU T4 + Internet ON
- Orden: 00 -> 01 -> 02 -> 03 -> 04 (todos corridos con GPU T4)
- Cada notebook guarda sus PDFs en `/kaggle/working/` (curvas, grillas, barras, matrices). Ver `presentacion.md` (fuera del repo) para correspondencia figura-celda.
