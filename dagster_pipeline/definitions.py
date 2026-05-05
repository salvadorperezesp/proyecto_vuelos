from dagster import (
    Definitions,
    define_asset_job,
    ScheduleDefinition,
    AssetSelection,
)
from dagster_pipeline.assets import (
    datos_crudos,
    datos_procesados,
    muestra_entrenamiento,
    modelo_entrenado,
    reporte_evaluacion,
)

# ── Job: agrupa todos los assets en un pipeline ───────────
pipeline_vuelos = define_asset_job(
    name="pipeline_vuelos",
    selection=AssetSelection.all(),
    description="Pipeline completo: carga → limpieza → muestra → entrenamiento → evaluación",
)

# ── Schedule: ejecuta el pipeline cada semana ─────────────
schedule_semanal = ScheduleDefinition(
    job=pipeline_vuelos,
    cron_schedule="0 6 * * 1",   # cada lunes a las 6:00 AM
    name="schedule_semanal_vuelos",
)

# ── Definitions: punto de entrada de Dagster ──────────────
defs = Definitions(
    assets=[
        datos_crudos,
        datos_procesados,
        muestra_entrenamiento,
        modelo_entrenado,
        reporte_evaluacion,
    ],
    jobs=[pipeline_vuelos],
    schedules=[schedule_semanal],
)
