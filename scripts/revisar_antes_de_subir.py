"""Control antes de subir a GitHub: revisa lo que se va a subir y bloquea la subida si encuentra datos de la empresa.

Lo llama git solo antes de cada `git push` (hook .git/hooks/pre-push). Revisa el contenido de los archivos que cambia
cada commit nuevo y su mensaje, y busca:
  - el nombre de la empresa
  - números de guía: 9 o más dígitos seguidos (salvo los de ejemplo, que empiezan por 0000)
  - cifras con forma de pesos de 100 millones o más (nueve dígitos o más con puntos de miles), el tamaño de los costos del mes
  - valores reales escritos con o sin separadores: las guías del archivo privado y las tarifas reales de camión
    que no son redondas (un valor redondo como 2.000.000 no se distingue de uno inventado)
Además no deja subir ramas original/... (historial anterior, con cifras) ni etiquetas.
Los valores prohibidos se leen de data/ (que no se publica): este archivo no contiene ninguno.

Uso:
  .venv/Scripts/python.exe scripts/revisar_antes_de_subir.py --instalar        instala el hook en .git/hooks/pre-push
  .venv/Scripts/python.exe scripts/revisar_antes_de_subir.py --revisar main    revisa todo el contenido de una rama
Variables:
  prohibidos: textos de dígitos que no pueden aparecer (guías reales y tarifas reales de camión)
  nombres: nombres de la empresa que no pueden aparecer (del archivo privado, llave nombre_empresa)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PRIVADO = RAIZ / "data" / "external" / "datos_empresa.json"
TARIFAS = RAIZ / "data" / "external" / "tarifas_ftl.csv"
CERO = "0" * 40

GUIA = re.compile(r"(?<![\d.,_])\d{9,}(?!\d)")
PESOS = re.compile(r"(?<![\d.,])(?:\d{3}(?:\.\d{3}){2,}|\d{1,3}(?:\.\d{3}){3,})(?![\d.]*\d)")
NUMERO = re.compile(r"\d[\d_.]*\d")


def forbidden_values() -> tuple[set[str], set[str]]:
    # Guías reales y tarifas reales de camión (como texto de dígitos) y nombres de la empresa, leídos de data/
    prohibidos, nombres = set(), set()
    if PRIVADO.exists():
        datos = json.loads(PRIVADO.read_text(encoding="utf-8"))
        prohibidos |= set(datos.get("guia_region", {})) | set(datos.get("guia_ciudad", {}))
        nombres |= {nombre.upper() for nombre in datos.get("nombre_empresa", [])}
    else:
        print(f"ADVERTENCIA: no existe {PRIVADO}; solo se revisan los patrones generales")
    if TARIFAS.exists():
        for linea in TARIFAS.read_text(encoding="utf-8").splitlines()[1:]:
            for campo in linea.split(","):
                valor = campo.strip().split(".")[0]
                if re.fullmatch(r"\d{5,}(\.0+)?", campo.strip()) and int(valor) % 100_000:   # solo las no redondas
                    prohibidos.add(valor)
    return prohibidos, nombres


def find_sensitive(texto: str, prohibidos: set[str], nombres: set[str]) -> list[str]:
    # Lista de hallazgos (vacía si no hay nada sensible)
    hallazgos = []
    mayusculas = texto.upper()
    hallazgos += [f"nombre de la empresa ({nombre.title()})" for nombre in nombres if nombre in mayusculas]
    hallazgos += [f"posible guía {g}" for g in GUIA.findall(texto) if not g.startswith("0000")]
    hallazgos += [f"cifra con forma de pesos {c}" for c in PESOS.findall(texto)]
    for numero in NUMERO.findall(texto):
        sin_separadores = numero.replace("_", "").replace(".", "")
        entero = numero.replace("_", "").split(".")[0] if re.search(r"\.\d{1,2}$", numero) else sin_separadores
        if sin_separadores in prohibidos or entero in prohibidos:
            hallazgos.append(f"valor real de la empresa ({numero})")
    return hallazgos


def check_refs(lineas: list[str]) -> list[str]:
    # Ramas que no se pueden subir: original/... (historial con cifras) y etiquetas
    errores = []
    for linea in lineas:
        partes = linea.split()
        if len(partes) == 4 and partes[1] != CERO:
            if partes[0].startswith("refs/heads/original/") or partes[0].startswith("refs/tags/"):
                errores.append(f"no se sube {partes[0]}: es del historial anterior, con datos de la empresa")
    return errores


def _git(*argumentos: str) -> str:
    return subprocess.run(["git", *argumentos], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def _review_commit(commit: str, prohibidos: set[str], nombres: set[str], todo: bool = False) -> list[str]:
    # Hallazgos en el mensaje y en los archivos que cambia el commit (todos sus archivos si todo=True)
    hallazgos = [f"{commit[:7]} mensaje: {h}" for h in find_sensitive(_git("log", "-1", "--format=%B", commit),
                                                                       prohibidos, nombres)]
    archivos = (_git("ls-tree", "-r", "--name-only", commit) if todo
                else _git("diff-tree", "--root", "--no-commit-id", "-r", "--name-only", "--diff-filter=AM", commit))
    for archivo in archivos.splitlines():
        try:
            contenido = _git("show", f"{commit}:{archivo}")
        except subprocess.CalledProcessError:
            continue
        if "\0" in contenido[:8000]:
            continue                                                  # binario
        hallazgos += [f"{commit[:7]} {archivo}: {h}" for h in find_sensitive(contenido, prohibidos, nombres)]
    return hallazgos


def review_push(lineas: list[str]) -> list[str]:
    # Lo que git entrega al hook: "<ref local> <sha local> <ref remota> <sha remota>" por cada rama que se sube
    errores = check_refs(lineas)
    prohibidos, nombres = forbidden_values()
    for linea in lineas:
        partes = linea.split()
        if len(partes) != 4 or partes[1] == CERO:
            continue                                                  # borrar una rama remota: nada que revisar
        # Rama nueva, o remota que no está en este equipo (p. ej. el README que crea GitHub, que se reemplaza
        # con --force): se revisan todos los commits que no están ya en el remoto
        conocida = partes[3] != CERO and subprocess.run(["git", "cat-file", "-e", f"{partes[3]}^{{commit}}"],
                                                         cwd=RAIZ, capture_output=True).returncode == 0
        rango = [f"{partes[3]}..{partes[1]}"] if conocida else [partes[1], "--not", "--remotes"]
        for commit in _git("rev-list", *rango).split():
            errores += _review_commit(commit, prohibidos, nombres)
    return errores


def install_hook() -> None:
    hook = RAIZ / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\n"
                    "# Control de datos de la empresa antes de subir (ver scripts/revisar_antes_de_subir.py)\n"
                    'RAIZ="$(git rev-parse --show-toplevel)"\n'
                    'export PYTHONIOENCODING=utf-8\n'
                    'exec "$RAIZ/.venv/Scripts/python.exe" "$RAIZ/scripts/revisar_antes_de_subir.py" "$@"\n',
                    encoding="utf-8", newline="\n")
    print(f"Hook instalado en {hook}")


if __name__ == "__main__":
    if "--instalar" in sys.argv:
        install_hook()
        sys.exit(0)
    if "--revisar" in sys.argv:
        prohibidos, nombres = forbidden_values()
        ramas = sys.argv[sys.argv.index("--revisar") + 1:]
        errores = [h for rama in ramas for h in _review_commit(_git("rev-parse", rama).strip(), prohibidos, nombres,
                                                                todo=True)]
    else:
        errores = review_push(sys.stdin.read().splitlines())
    if errores:
        print("ADVERTENCIA: subida bloqueada, hay datos de la empresa:")
        for error in errores:
            print(f"  - {error}")
        sys.exit(1)
    print("Revisión antes de subir: sin datos de la empresa")
