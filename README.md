# Reconocimiento del Alfabeto Completo de Lengua de Señas Mexicana (LSM)

Sistema de aprendizaje profundo para clasificar las **27 letras del alfabeto dactilológico de la Lengua de Señas Mexicana (LSM)** combinando modelos estáticos sobre **landmarks 3D + ingeniería geométrica** y una comparación directa contra **Transfer Learning en CNN RGB**, además de modelos secuenciales (**LSTM** y **Bi-GRU con Atención Temporal**) para las 6 letras dinámicas.

---

## 1. Datasets y División Estricta por Persona (0 Fuga de Datos)

Los experimentos se construyeron a partir de dos datasets públicos licenciados bajo **CC-BY-4.0**:

| Subconjunto | Letras | Fuente original (Zenodo) | Muestras | División estricta por participante |
|---|---|---|---|---|
| **Estático** | 21 (`A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y`) | *Mexican Sign Language Alphabet (static signs only)* (`10.5281/zenodo.10067509`) | 279,716 imágenes (niveles angulares `A`, `B`, `C`) | **Train:** 15 sujetos (`209,399`)<br>**Val:** `S5, S12, S17` (`42,698`)<br>**Test:** `S18, S20` (`27,619`) |
| **Dinámico** | 6 (`J, K, Ñ, Q, X, Z`) | *Mexican Sign Language Alphabet (dynamic signs only)* (`10.5281/zenodo.14689869`) | 1,242 videos (vista frontal y perfil 45°) | **Train:** 15 sujetos (`930`)<br>**Val:** `S5, S13, S17` (`192`)<br>**Test:** `S20, S21` (`120`) |

Para que el proyecto sea liviano, portable y reproducible en local o Google Colab sin descargar ~8 GB en archivos `.7z` (~25 GB descomprimidos), el repositorio incluye en `data/hf_lsm_dataset/` los landmarks 3D extraídos con **MediaPipe Hands (`0.10.21`)** en formato **Parquet ZSTD (~88 MB en total)**.

---

## 2. Resultados Comparativos en Conjunto de Prueba (Sujetos No Vistos)

Los resultados completos están consolidados en [outputs/tabla_comparativa_final.csv](outputs/tabla_comparativa_final.csv). Para una comparación rigurosa frente a la CNN RGB (evaluada sobre el 100% de las imágenes), en los modelos basados en landmarks se reporta tanto la **exactitud condicional** (sobre manos detectadas por MediaPipe: `98.95%` global y `97.27%` en Sub C) como la **exactitud efectiva** (contando las no-detecciones como error sobre las 27,619 imágenes de prueba):

| Tarea | Modelo | Parámetros | Val Macro-F1 | Acc. Detectada | Det. MP | Acc. Efectiva (100%) | Test Macro-F1 | Sub A / B / C (Efec.) | Tiempo |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **Estático (21)** | `MLP-Base` (60D crudas) | 51,989 | 0.9700 | 93.84% | 98.95% | 92.85% | 0.9385 | 99.17% / 95.10% / 84.33% | 50.6 s |
| **Estático (21)** | `MobileNetV3-Small` (CNN RGB, Transfer Learning) | 1,539,381 | 0.9682 | 94.53% | 100.00% | 94.53% | 0.9453 | 99.43% / 96.36% / 87.84% | 547.8 s |
| **Estático (21)** | **`HandGeoResNet`** (184D geométricas + aug 3D) | 317,829 | **0.9744** | **95.92%** | 98.95% | **94.92%** | **0.9586** | **99.67% / 97.30% / 87.86%** | 151.6 s |
| **Dinámico (6)** | `LSTM-Base` (63D secuencia cruda) | 99,590 | 0.9895 | 100.00% | 100.00% | 100.00% | 1.0000 | — | 20.9 s |
| **Dinámico (6)** | **`DynamicBiGRUAttn`** (214D esp-temp + Atención) | 637,971 | **1.0000** | **100.00%** | 100.00% | **100.00%** | **1.0000** | — | 123.4 s |

> **Observación clave:** En el subconjunto estático, `HandGeoResNet` supera a `MobileNetV3-Small` en exactitud efectiva global (`94.92%` vs `94.53%`) y en Macro-F1 (`0.9586` vs `0.9453`) utilizando **~4.8× menos parámetros** y **~3.6× menos tiempo de entrenamiento**, mientras que en señas dinámicas `DynamicBiGRUAttn` alcanza convergencia estable y `1.0000` de F1 en validación sin las oscilaciones de `LSTM-Base`.

---

## 3. Estructura del Repositorio

```text
├── LSM_Alfabeto_Completo.ipynb     # Notebook principal autocontenido y ejecutado (Partes A y B)
├── requirements.txt                # Dependencias del proyecto
├── data/
│   └── hf_lsm_dataset/             # Landmarks 3D en Parquet ZSTD (~88 MB, splits train/val/test)
├── lsm/
│   ├── features.py                 # Normalización 3D, marco canónico de palma, 184 features y aug 3D
│   ├── train_static.py             # Entrenamiento y comparación MLP-Base vs HandGeoResNet
│   ├── train_static_cnn.py         # Transfer Learning con MobileNetV3-Small sobre imágenes RGB
│   ├── train_dynamic.py            # Entrenamiento y comparación LSTM-Base vs DynamicBiGRUAttn
│   ├── summary_report.py           # Consolidación de tablas CSV y gráficas comparativas
│   ├── extract_landmarks.py        # Extractor MediaPipe Hands para imágenes estáticas
│   ├── extract_dynamic.py          # Extractor MediaPipe Hands para secuencias de video
│   ├── prepare_hf_package.py       # Empaquetado a Parquet ZSTD y subida opcional a HF Hub
│   ├── build_notebook.py           # Generador reproducible de LSM_Alfabeto_Completo.ipynb
│   └── demo_camera.py              # Demo en tiempo real con cámara web (27 letras)
└── outputs/
    ├── tabla_comparativa_final.csv
    ├── comparacion_global.png
    ├── comparacion_f1_landmarks_vs_cnn.png
    ├── static/                     # Checkpoints, métricas y gráficas de MLP-Base y HandGeoResNet
    ├── static_cnn/                 # Checkpoint, métricas y gráficas de MobileNetV3-Small
    └── dynamic/                    # Checkpoints, métricas y gráficas de LSTM-Base y DynamicBiGRUAttn
```

---

## 4. Cómo Reproducir el Proyecto

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

> **Nota sobre MediaPipe:** solo se necesita `mediapipe==0.10.21` y `opencv-python` si vas a re-extraer landmarks desde imágenes/videos crudos o ejecutar la demo de cámara en vivo. Las versiones `>= 0.10.30` / `1.x` eliminaron `mp.solutions`.

### Opción A: Notebook autocontenido (local o Google Colab)

1. Abre `LSM_Alfabeto_Completo.ipynb` (o en Google Colab clona el repo con `!git clone <URL_DEL_REPO> && cd <CARPETA>`).
2. Ejecuta todas las celdas; el notebook lee automáticamente los archivos Parquet locales en `data/hf_lsm_dataset/` y entrena/evalúa `MLP-Base`, `HandGeoResNet`, `LSTM-Base` y `DynamicBiGRUAttn`.

### Opción B: Scripts modulares en línea de comandos

```bash
# 1. Modelos estáticos sobre landmarks (MLP-Base y HandGeoResNet)
python lsm/train_static.py

# 2. Modelos dinámicos sobre secuencias (LSTM-Base y DynamicBiGRUAttn)
python lsm/train_dynamic.py

# 3. Consolidar tablas y gráficas finales en outputs/
python lsm/summary_report.py
```

### Opción C: Reproducir el experimento CNN RGB (`MobileNetV3-Small`)

Descarga `MSL-ABC.7z` desde [Zenodo 10067509](https://zenodo.org/records/10067509), extráelo en `data/raw/MSL-ABC` y ejecuta:

```bash
python lsm/train_static_cnn.py --root data/raw/MSL-ABC
```

### Opción D: Demo en tiempo real con cámara web

```bash
python lsm/demo_camera.py
```

Controles en ventana: `M` cambia el modo (`AUTO` / `ESTATICA` / `DINAMICA`), `L` alterna mano derecha/izquierda, `Espacio` agrega letra al texto, `Backspace` borra, `C` limpia y `Q`/`Esc` sale.
