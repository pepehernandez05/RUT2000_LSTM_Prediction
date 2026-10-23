# Predicción del Russell 2000 (RUT) con LSTM multivariable

Proyecto de series temporales financieras: se limpia y prepara el histórico diario del índice **US Small Cap 2000** y se entrena una red **LSTM multivariable** para predecir el precio de cierre.

## Contenido

| Archivo | Descripción |
|---|---|
| `Preprocesado_RUT2000.ipynb` | Limpieza (formatos, nulos, duplicados, consistencia High/Low), feature engineering, split cronológico 80/20, escalado con `MinMaxScaler` y creación de secuencias. |
| `Modelo_LSTM_multivariable.ipynb` | Definición, entrenamiento (con `EarlyStopping`) y evaluación del modelo LSTM. |

## Variables utilizadas

`Price`, `Open`, `High`, `Low`, `return`, `ma_7`, `ma_14`, `volatility_7`, `high_low_diff`, `open_close_diff`, `momentum_7`. Objetivo: `Price`.

## Modelo

- 1 capa LSTM (32 unidades) con regularización L2 (0.001)
- Dropout 0.3 + capa Dense(1)
- Adam (lr = 0.001), pérdida MSE, `EarlyStopping` (paciencia 15, restaura mejores pesos)
- Métricas: MSE, MAE y MAPE sobre el test, ya desescalado a puntos del índice

## Cómo ejecutarlo

```bash
git clone https://github.com/TU_USUARIO/NOMBRE_REPO.git
cd NOMBRE_REPO
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Descarga el histórico del índice (formato CSV de Investing.com) y guárdalo como  
   `datasets/unprocessed/US Small Cap 2000 Historical Data.csv`.
2. Ejecuta `Preprocesado_RUT2000.ipynb`. Genera los `.npy` y `dataset_limpio.csv` en `datasets/processed/`.
3. Ejecuta `Modelo_LSTM_multivariable.ipynb`.

> Los datos no se incluyen en el repositorio (ver `.gitignore`).

## Estructura esperada

```
.
├── Preprocesado_RUT2000.ipynb
├── Modelo_LSTM_multivariable.ipynb
├── requirements.txt
├── README.md
└── datasets/
    ├── unprocessed/   # CSV original (no versionado)
    └── processed/     # secuencias .npy y dataset_limpio.csv (no versionado)
```

## Autor

José Ignacio Hernández Pablos y Mencía Marzal Higuero
