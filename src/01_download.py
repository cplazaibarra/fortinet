"""
FASE 1 — Descarga/Extracción de datos OHLCV 15m de FTNT (Fortinet) desde PostgreSQL
Carga los datos desde la base de datos nativa fortinet_db e exporta a parquet para el pipeline.
"""

import os
import psycopg2
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "fortinet_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

RAW_DIR = Path("data/raw/spot")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH_FTNT = RAW_DIR / "ftnt_15m_raw.parquet"
OUTPUT_PATH_BTC = RAW_DIR / "btcusdt_15m_raw.parquet"


def main():
    console.print(Panel(
        f"[bold cyan]Cargando velas de 15m de Fortinet (FTNT) desde PostgreSQL[/bold cyan]\n"
        f"Base de Datos: [green]{DB_NAME}[/green] ({DB_HOST}:{DB_PORT})",
        title="FASE 1 — Extracción de Datos FTNT",
        border_style="cyan"
    ))

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    df = pd.read_sql("SELECT * FROM ohlcv_15m ORDER BY timestamp ASC;", conn)
    conn.close()

    console.print(f"[green]Se leyeron {len(df):,} registros desde PostgreSQL.[/green]")

    # Guardar en parquet para compatibilidad con todo el pipeline de backtesting y ML
    df.to_parquet(OUTPUT_PATH_FTNT, index=False)
    df.to_parquet(OUTPUT_PATH_BTC, index=False)  # Alias para compatibilidad con scripts originales

    console.print(f"[bold green]✓ Archivos parquet guardados exitosamente en {RAW_DIR}[/bold green]")


if __name__ == "__main__":
    main()
