"""Tablero en consola con el avance de las corridas que quedan andando (se cierra con Ctrl+C o al terminar todo).

No toca las corridas: solo mira los archivos que van dejando en data/processed y se actualiza cada minuto. Mientras
está abierto, le pide a Windows que no suspenda el equipo por inactividad; no cambia ninguna configuración, y al cerrar
se quita. Avisa si una corrida se demora de más, si no hay ninguna activa o si el equipo se suspendió.
Uso: .venv/Scripts/python.exe scripts/ver_avance.py
"""
import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ESCENARIO = "0111_2026-01_E3"
CONFIGURACIONES_MEMETICO = ["base_10_20_3", "menos_individuos_5_20_3", "menos_vecinos_10_10_3", "sin_memoria_10_20_1"]
SALTO_SEGUNDOS = 60     # un paso del historial más largo que esto solo se explica por una suspensión
ANCHO_BARRA = 30


def pending_stages(escenario: str = ESCENARIO) -> list:
    # Las corridas en el orden en que están programadas; "archivos" es lo que deja cada unidad al terminar
    tenure = [f"tabu_tenure5_{escenario}_p{punto}_s{semilla}_historial.csv"
              for semilla in range(5) for punto in (86, 100, 111, 120)]
    comparacion_s0 = f"genetico_gen0_{escenario}_comparacion_s0.csv"
    memetico = [f"memetico_calibracion_{escenario}_{nombre}_historial.csv" for nombre in CONFIGURACIONES_MEMETICO]
    return [{"nombre": "Tabú con tenure 5", "archivos": tenure, "minutos": 10},
            {"nombre": "Repetir semilla 0 del genético", "archivos": [comparacion_s0], "minutos": 12,
             "requiere": f"comparacion_interferida/{comparacion_s0}"},   # el archivo viejo se mueve antes de repetir
            {"nombre": "Calibrar el memético", "archivos": memetico, "minutos": 10}]


def stage_progress(etapa: dict, carpeta: Path) -> dict:
    # Cuántas unidades van, cuánto falta y si la que está corriendo ya se pasó del doble de lo normal
    total, minutos = len(etapa["archivos"]), etapa["minutos"]
    if "requiere" in etapa and not (carpeta / etapa["requiere"]).exists():
        hechos = []
    else:
        hechos = [carpeta / nombre for nombre in etapa["archivos"] if (carpeta / nombre).exists()]
    completa = len(hechos) == total

    transcurrido = None
    if hechos and not completa:
        transcurrido = (time.time() - max(ruta.stat().st_mtime for ruta in hechos)) / 60
    restantes = (total - len(hechos)) * minutos - min(transcurrido or 0, minutos)
    return {"hechos": len(hechos), "total": total, "completa": completa, "minutos_restantes": round(restantes, 1),
            "demorada": transcurrido is not None and transcurrido > 2 * minutos, "rutas": hechos}


def time_jump(ruta: Path, umbral: float = SALTO_SEGUNDOS) -> bool:
    # ¿Un paso del historial tardó más que el umbral? Es la huella de que el equipo se suspendió en esa corrida
    with open(ruta, encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if "segundos" not in (lector.fieldnames or []):
            return False
        segundos = [float(fila["segundos"]) for fila in lector if fila["segundos"]]
    return any(despues - antes > umbral for antes, despues in zip(segundos, segundos[1:]))


def format_minutes(minutos: float) -> str:
    horas, resto = divmod(round(minutos), 60)
    return f"{horas} h {resto:02d} min" if horas else f"{resto} min"


def active_runs() -> list:
    # Scripts de corridas que están activos ahora (None si no se pudo consultar)
    consulta = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                "ForEach-Object { $_.CommandLine }")
    try:
        salida = subprocess.run(["powershell", "-NoProfile", "-Command", consulta], capture_output=True, text=True,
                                timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    activos = set()
    for linea in salida.splitlines():
        for nombre in ("comparar_metaheuristicas.py", "calibrar_memetico.py"):
            if nombre in linea:
                partes = linea.split("--parte ")
                activos.add(nombre + (f" ({partes[1].split()[0]})" if len(partes) > 1 else ""))
    return sorted(activos)


def keep_awake(activar: bool) -> None:
    # Pide a Windows no suspender por inactividad mientras este programa siga abierto (no aplica a cerrar la tapa)
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if activar else 0))


def show(avances: list, activos: list | None, saltos: list, suspensiones: list, intervalo: int) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    ahora = datetime.now()
    print(f"AVANCE DE LAS CORRIDAS · {ahora:%H:%M:%S} · se actualiza cada {intervalo} s · Ctrl+C para cerrar\n")
    for etapa, avance in avances:
        llenos = round(ANCHO_BARRA * avance["hechos"] / avance["total"])
        estado = "LISTA" if avance["completa"] else f"faltan {format_minutes(avance['minutos_restantes'])}"
        print(f"  {etapa['nombre']:<32} [{'#' * llenos}{'-' * (ANCHO_BARRA - llenos)}] "
              f"{avance['hechos']:>2}/{avance['total']:<2} {estado}")
    faltan = sum(avance["minutos_restantes"] for _, avance in avances)
    print(f"\n  Termina aproximadamente: {ahora + timedelta(minutes=faltan):%d/%m %H:%M} "
          f"(faltan {format_minutes(faltan)})")
    print(f"  Corriendo ahora: {', '.join(activos) if activos else 'nada' if activos is not None else 'no se pudo consultar'}")
    print("  Equipo: este tablero evita la suspensión por inactividad. No cierres la tapa.")

    alertas = [f"{etapa['nombre']}: lleva más del doble de lo normal; puede estar detenida"
               for etapa, avance in avances if avance["demorada"]]
    if activos == [] and faltan > 0:
        alertas.append("No hay ninguna corrida activa. Es normal solo por unos minutos entre etapas")
    alertas += [f"El equipo se suspendió hacia las {momento}" for momento in suspensiones]
    alertas += [f"Salto de tiempo, corrida posiblemente dañada: {nombre}" for nombre in saltos]
    if alertas:
        print("\n  ALERTAS")
        for alerta in alertas:
            print(f"  ! {alerta}")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tablero de avance de las corridas")
    parser.add_argument("--intervalo", type=int, default=60, help="segundos entre actualizaciones (por defecto 60)")
    args = parser.parse_args()

    etapas = pending_stages()
    revisados, saltos, suspensiones = set(), [], []
    keep_awake(True)
    try:
        ultima_vuelta = time.time()
        while True:
            avances = [(etapa, stage_progress(etapa, PROCESSED)) for etapa in etapas]
            for _, avance in avances:
                for ruta in avance["rutas"]:
                    if ruta.name not in revisados:
                        revisados.add(ruta.name)
                        if time_jump(ruta):
                            saltos.append(ruta.name)
            if time.time() - ultima_vuelta > args.intervalo + 120:
                suspensiones.append(f"{datetime.fromtimestamp(ultima_vuelta):%H:%M}")
            show(avances, active_runs(), saltos, suspensiones, args.intervalo)
            if all(avance["completa"] for _, avance in avances):
                print("\n  Todas las corridas terminaron. Ya se puede cerrar y el equipo vuelve a su configuración normal.")
                break
            ultima_vuelta = time.time()
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\n  Tablero cerrado. Las corridas siguen andando; el equipo vuelve a su configuración normal de suspensión.")
    finally:
        keep_awake(False)
