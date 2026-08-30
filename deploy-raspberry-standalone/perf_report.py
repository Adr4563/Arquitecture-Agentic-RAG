"""
Resumen de logs/tiempos.csv y logs/recursos.csv (generados por
perf_monitor.py) -- responde "qué componente tarda más" y "cómo vino el
CPU/memoria durante la sesión". Sin pandas: es un CSV chico, no hace falta
la dependencia extra en la Pi.

Uso:
    python perf_report.py                      # todo el historial
    python perf_report.py --desde 2026-08-30   # solo desde esa fecha (ISO)
"""

import argparse
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUTA_TIEMPOS = os.path.join(HERE, "logs", "tiempos.csv")
RUTA_RECURSOS = os.path.join(HERE, "logs", "recursos.csv")


def _leer_csv(ruta, desde=None):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if desde:
        filas = [f for f in filas if f["timestamp"] >= desde]
    return filas


def _percentil(valores_ordenados, pct):
    if len(valores_ordenados) == 1:
        return valores_ordenados[0]
    idx = max(0, int(len(valores_ordenados) * pct) - 1)
    return valores_ordenados[idx]


def _reporte_tiempos(desde):
    filas = _leer_csv(RUTA_TIEMPOS, desde)
    if not filas:
        print(f"(sin datos en {RUTA_TIEMPOS} -- ¿ya corrió una sesión con las funciones @medir?)")
        return

    por_componente = {}
    errores_por_componente = {}
    for f in filas:
        por_componente.setdefault(f["componente"], []).append(float(f["duracion_ms"]))
        if f["error"] == "1":
            errores_por_componente[f["componente"]] = errores_por_componente.get(f["componente"], 0) + 1

    print(f"\n=== Tiempo por componente ({len(filas)} llamadas totales) ===")
    print(f"{'componente':<24}{'llamadas':>9}{'total_s':>9}{'media_ms':>10}{'p90_ms':>9}{'max_ms':>9}{'errores':>9}")
    orden = sorted(por_componente.items(), key=lambda kv: sum(kv[1]), reverse=True)
    for componente, duraciones in orden:
        duraciones_ord = sorted(duraciones)
        p90 = _percentil(duraciones_ord, 0.9)
        errores = errores_por_componente.get(componente, 0)
        print(f"{componente:<24}{len(duraciones):>9}{sum(duraciones)/1000:>9.1f}"
              f"{statistics.mean(duraciones):>10.1f}{p90:>9.1f}{max(duraciones):>9.1f}{errores:>9}")


def _reporte_recursos(desde):
    filas = _leer_csv(RUTA_RECURSOS, desde)
    if not filas:
        print(f"\n(sin datos en {RUTA_RECURSOS} -- ¿psutil instalado? ver perf_monitor.iniciar_muestreo_recursos)")
        return

    cpu = [float(f["cpu_percent"]) for f in filas]
    mem = [float(f["memoria_rss_mb"]) for f in filas]
    print(f"\n=== Recursos del proceso ({len(filas)} muestras, cada {INTERVALO_MUESTREO_SEG_INFO}) ===")
    print(f"CPU%:    media {statistics.mean(cpu):.1f}   max {max(cpu):.1f}")
    print(f"Memoria: media {statistics.mean(mem):.1f} MB   max {max(mem):.1f} MB   "
          f"(inicio {mem[0]:.1f} MB -> fin {mem[-1]:.1f} MB)")


# Solo informativo en el encabezado del reporte -- el intervalo real que se
# usó pudo ser otro si PERF_MUESTREO_SEG estaba seteado distinto en esa sesión.
INTERVALO_MUESTREO_SEG_INFO = f"{os.environ.get('PERF_MUESTREO_SEG', '5')}s (o el que estaba seteado al correr)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", help="timestamp ISO mínimo a incluir (ej. 2026-08-30)")
    args = parser.parse_args()
    _reporte_tiempos(args.desde)
    _reporte_recursos(args.desde)
