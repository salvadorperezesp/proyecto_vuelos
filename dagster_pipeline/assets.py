import pandas as pd
import numpy as np
from dagster import asset, get_dagster_logger

# ── Asset 1: Cargar datos ─────────────────────────────────
@asset
def datos_crudos():
    """Carga el dataset de vuelos desde el fichero feather limpio."""
    logger = get_dagster_logger()
    logger.info("Cargando dataset de vuelos...")

    vuelos = pd.read_feather("vuelos_2024_limpio.feather")

    logger.info(f"Dataset cargado: {len(vuelos):,} filas, {vuelos.shape[1]} columnas")
    return vuelos


# ── Asset 2: Limpiar y preparar features ─────────────────
@asset
def datos_procesados(datos_crudos):
    """Selecciona las features relevantes y elimina nulos."""
    logger = get_dagster_logger()
    logger.info("Procesando datos...")

    numericas   = ["crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance"]
    categoricas = ["aerolinea", "origin", "dest", "month", "day_of_month", "day_of_week"]

    X = datos_crudos[numericas + categoricas].copy()
    y = datos_crudos["tipo_retraso"]

    # Eliminar filas con nulos
    antes = len(X)
    mask  = X.notna().all(axis=1) & y.notna()
    X     = X[mask]
    y     = y[mask]
    logger.info(f"Filas eliminadas por nulos: {antes - len(X):,}")

    # Unir X e y en un solo DataFrame para devolverlo
    datos = X.copy()
    datos["tipo_retraso"] = y.values

    logger.info(f"Distribución de clases:\n{datos['tipo_retraso'].value_counts().to_string()}")
    logger.info(f"Datos procesados: {len(datos):,} filas listas para entrenar")
    return datos


# ── Asset 3: Muestra de entrenamiento ────────────────────
@asset
def muestra_entrenamiento(datos_procesados):
    """Toma una muestra de 100.000 filas para entrenar."""
    logger = get_dagster_logger()

    N = min(100_000, len(datos_procesados))
    muestra = datos_procesados.sample(n=N, random_state=42)

    logger.info(f"Muestra seleccionada: {len(muestra):,} filas")
    return muestra


# ── Asset 4: Entrenar modelo ──────────────────────────────
@asset
def modelo_entrenado(muestra_entrenamiento):
    """Entrena la red neuronal MLP y devuelve las métricas."""
    logger = get_dagster_logger()
    logger.info("Iniciando entrenamiento de la red neuronal...")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import joblib

    numericas   = ["crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance"]
    categoricas = ["aerolinea", "origin", "dest", "month", "day_of_month", "day_of_week"]

    X = muestra_entrenamiento[numericas + categoricas]
    y = muestra_entrenamiento["tipo_retraso"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Preprocesamiento
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ct  = make_column_transformer((ohe, categoricas), remainder="passthrough")
    X_train_arr = ct.fit_transform(X_train).astype(np.float32)
    X_test_arr  = ct.transform(X_test).astype(np.float32)

    # Normalizar numéricas
    n_ohe  = X_train_arr.shape[1] - len(numericas)
    mean_  = X_train_arr[:, n_ohe:].mean(axis=0)
    std_   = X_train_arr[:, n_ohe:].std(axis=0) + 1e-8
    X_train_arr[:, n_ohe:] = (X_train_arr[:, n_ohe:] - mean_) / std_
    X_test_arr[:, n_ohe:]  = (X_test_arr[:,  n_ohe:] - mean_) / std_

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_arr).to(device)
    X_test_t  = torch.tensor(X_test_arr).to(device)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long).to(device)
    y_test_t  = torch.tensor(y_test_enc,  dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)

    # Arquitectura
    class MLP(nn.Module):
        def __init__(self, n_features, n_clases):
            super().__init__()
            self.red = nn.Sequential(
                nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_clases)
            )
        def forward(self, x):
            return self.red(x)

    modelo = MLP(X_train_arr.shape[1], len(le.classes_)).to(device)

    pesos   = compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
    pesos_t = torch.tensor(pesos, dtype=torch.float32).to(device)
    criterio    = nn.CrossEntropyLoss(weight=pesos_t)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)

    # Entrenamiento (20 épocas para el pipeline)
    mejor_acc = 0.0
    for epoca in range(1, 21):
        modelo.train()
        for X_batch, y_batch in train_loader:
            optimizador.zero_grad()
            loss = criterio(modelo(X_batch), y_batch)
            loss.backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            preds    = modelo(X_test_t).argmax(1)
            accuracy = (preds == y_test_t).float().mean().item()
            if accuracy > mejor_acc:
                mejor_acc = accuracy

        logger.info(f"Época {epoca:02d}/20 | Accuracy: {accuracy:.3f}")

    # Guardar modelo y preprocesador
    torch.save(modelo.state_dict(), "mlp_vuelos.pt")
    joblib.dump(ct, "preprocesador_vuelos.pkl")
    joblib.dump(le, "label_encoder_vuelos.pkl")

    metricas = {
        "accuracy_final": round(accuracy, 4),
        "mejor_accuracy": round(mejor_acc, 4),
        "n_features":     X_train_arr.shape[1],
        "clases":         list(le.classes_),
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }

    logger.info(f"Entrenamiento completado. Mejor accuracy: {mejor_acc:.3f}")
    logger.info(f"Modelo guardado en mlp_vuelos.pt")
    return metricas


# ── Asset 5: Evaluación y reporte ────────────────────────
@asset
def reporte_evaluacion(modelo_entrenado):
    """Genera un resumen de los resultados del modelo."""
    logger = get_dagster_logger()

    logger.info("=== REPORTE FINAL ===")
    logger.info(f"Clases predichas:  {modelo_entrenado['clases']}")
    logger.info(f"Features totales:  {modelo_entrenado['n_features']}")
    logger.info(f"Filas entrenamiento: {modelo_entrenado['n_train']:,}")
    logger.info(f"Filas test:          {modelo_entrenado['n_test']:,}")
    logger.info(f"Accuracy final:    {modelo_entrenado['accuracy_final']:.4f}")
    logger.info(f"Mejor accuracy:    {modelo_entrenado['mejor_accuracy']:.4f}")

    return modelo_entrenado
