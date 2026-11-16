from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil
import xlwings as xw
from tqdm import tqdm

from despacho_multiobjetivo import parametros
from despacho_multiobjetivo.config import RAW, PROCESSED, EXTERNAL
from despacho_multiobjetivo.datos_empresa import DATOS

CACHE_COTIZACIONES = PROCESSED / "cotizaciones_cache.parquet"

# Nombres de columna reales del Excel crudo de despachos (DATOS["archivo_despachos"]),
# en el orden exacto en que aparecen, mapeados a un nombre limpio en
# snake_case. Se usa reasignación posicional (ver `standardize_columns`)
# en vez de un dict {nombre_viejo: nombre_nuevo}, porque varios nombres
# originales traen espacios finales o problemas de codificación (p. ej.
# "Sede " con espacio, "Direcci�n") que hacen poco confiable hacer match
# por el nombre exacto.
COLUMNAS_LIMPIAS = [
    "mes", "transportadora", "guia", "factura", "cuenta", "sede",
    "fecha_captura", "dias_promesa_entrega", "fecha_entrega", "ciudad_destino",
    "remitente", "nombre_destinatario", "direccion", "unidades", "peso",
    "valor_flete", "documento", "tipo_despacho", "tipo_ruta", "ciudad_sede",
    "latitud_destino", "longitud_destino", "tipo", "toneladas", "sede_flete",
]

# Regiones interiores que, despachadas desde el CEDI Barranquilla (0211),
# se agrupan y consolidan hacia Bogotá como hub de redistribución.
REGIONES_INTERIOR_BAQ = {"CENTRO", "CENTRO_SUR", "SANTANDERES", "EJE_CAFETERO", "SUR_OCCIDENTE"}

# Ciudades del área metropolitana de Bogotá: despachos locales desde Mosquera
# (0111) hacia estas ciudades no aplican para consolidación regional.
BOGOTA_METRO = ["BOGOTA", "MOSQUERA", "FUNZA", "MADRID", "SOACHA"]

# Ciudades del área metropolitana de Barranquilla: despachos locales desde Barranquilla
# (0211) hacia estas ciudades no aplican para consolidación regional.
BARRANQUILLA_METRO = ["BARRANQUILLA", "SOLEDAD", "GALAPA", "MALAMBO"]

# Códigos de los CEDI de origen
SEDE_MOSQUERA = "0111"
SEDE_BARRANQUILLA = "0211"

# Overrides puntuales de ciudad/guía -> región, tomados de la depuración
# manual hecha en la etapa exploratoria (casos que el mapeo por departamento
# no resuelve bien).
CIUDAD_REGION_OVERRIDE = {
    "ALCALA": "EJE_CAFETERO",
    # Homónimos que divipola resuelve al departamento equivocado: en esta base
    # siempre son los municipios de Antioquia.
    "RIONEGRO": "ANTIOQUIA_CHOCO",
    "CALDAS": "ANTIOQUIA_CHOCO",
    # GUAMAL existe en Magdalena y en Meta; el histórico lo escribe así cuando es el del Meta
    "GUAMAL - META": "CENTRO",
}

# Nombres de ciudad del histórico 2024–2026 que no están en divipola → nombre de divipola (revisado con
# Johan el 2026-11-16). Abreviaturas, errores de escritura, nombres cortados y corregimientos. Cuando el
# despacho trae dos ciudades se toma la primera. Queda sin corregir "0" (no se puede saber cuál es);
# "SAN JOSE D" depende de la sede (CIUDAD_CORRECCION_POR_SEDE).
CIUDAD_CORRECCION = {
    "B/QUILLA": "BARRANQUILLA", "B/MANGA": "BUCARAMANGA", "V/CENCIO": "VILLAVICENCIO",
    "B/BERMEJA": "BARRANCABERMEJA", "F/BLANCA": "FLORIDABLANCA", "S/MARTA": "SANTA MARTA",
    "P/DECUESTA": "PIEDECUESTA", "S/DETORRES": "SABANA DE TORRES",
    "SANTAMARTA": "SANTA MARTA", "PERREIRA": "PEREIRA", "IBAGUÉ": "IBAGUE", "TIERRA ALTA": "TIERRALTA",
    "DIBULA": "DIBULLA", "DON MATIAS": "DONMATIAS",
    "SAN JOSE GUAVIARE": "SAN JOSE DEL GUAVIARE",
    "SAN SOJE": "SAN JOSE DEL GUAVIARE",   # mismo transportador (COOTRANSTAME) que los de San José del Guaviare
    "SAN JERONI": "SAN JERONIMO", "PUERTO TEJ": "PUERTO TEJADA", "GUADALAJAR": "GUADALAJARA DE BUGA",
    "CASTILLA L": "CASTILLA LA NUEVA", "BELEN DE U": "BELEN DE UMBRIA", "JESUS MARI": "JESUS MARIA",
    "DAPA": "YUMBO", "ARMERO GUAYABAL": "ARMERO", "GRANADA - META": "GRANADA",
    "PUEBLO NUEVO - ARAUQUITA": "ARAUQUITA",
    "LA GRAN VI": "LA GRAN VIA",           # el de Magdalena (Johan, 2026-11-16); divipola lo resuelve a Magdalena
    "MEDELLIN Y SABANETA": "MEDELLIN", "MEDELLIN Y MONTERIA": "MEDELLIN", "CALI Y YUMBO": "CALI",
    "BUCARAMANGA Y BARRANCABERMEJA": "BUCARAMANGA", "BARRANCABERMEJA Y BUCARAMANGA": "BARRANCABERMEJA",
    "MANIZALES Y PEREIRA": "MANIZALES", "PEREIRA-MANIZALES": "PEREIRA",
}

# Nombres que dependen de la sede de origen (Johan, 2026-11-16): (sede, ciudad) → nombre de divipola.
# 0111 es Mosquera; 0106 y 0212 son las sedes de Medellín (CEDI de Sabaneta)
CIUDAD_CORRECCION_POR_SEDE = {
    ("0111", "SAN JOSE D"): "SAN JOSE DEL GUAVIARE",
    ("0106", "SAN JOSE D"): "SAN JOSE DEL NUS",
    ("0212", "SAN JOSE D"): "SAN JOSE DEL NUS",
}

# Guía → región y guía → ciudad (traslados a PAI con ciudad_destino vacía en el Excel crudo; la ciudad se
# identifica por el PAI del destinatario o la dirección). Los números de guía son de la empresa y no se
# publican: vienen de data/external/datos_empresa.json (ver datos_empresa.py)
GUIA_REGION_OVERRIDE = DATOS["guia_region"]
GUIA_CIUDAD_OVERRIDE = DATOS["guia_ciudad"]

# Tipos de pedido que van completos por una sola vía (x binaria). Los traslados (y traslados
# al gasto) son movimientos internos entre sedes y se pueden partir (x continua)
TIPOS_SIN_PARTIR = ("CLIENTES", "OTROS")

# Movimientos que llegan al CEDI en lugar de salir de él: no entran al modelo
# (en el histórico 2024–2026 las devoluciones vienen en plural)
TIPOS_ENTRADA_AL_CEDI = ("PROVEEDORES", "DEVOLUCION", "DEVOLUCIONES")

# Variantes del tipo de despacho en el histórico → nombre único
TIPO_DESPACHO_UNIFICADO = {"TRASLADO AL GASTO": "TRASLADOS AL GASTO",
                           "CX": "OTROS", "COMEX": "OTROS", "VARIOS": "OTROS"}

# Formatos de fecha del histórico, en el orden en que se prueban (en el Excel de 2026 ya llegan como fecha).
# Con barras se lee día/mes; mes/día solo si día/mes es imposible (ver parse_dates)
FORMATOS_FECHA = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y")

# Razón social en el archivo de despachos → nombre comercial (el que usa el cotizador)
NOMBRE_TRANSPORTADORA = {"COLVANES": "ENVIA"}

# Sedes de la empresa (sede_flete) donde descarga el vehículo troncal de cada hub:
# un traslado a una de estas sedes llega directo, sin tramo hub→destino.
SEDES_LLEGADA_HUB = {
    "BARRANQUILLA": {"0211", "0112"},
    "MEDELLIN": {"0106", "0212"},
    "CALI": {"0107", "0207", "1107"},
    "MANIZALES": {"0101"},
    "PEREIRA": {"0105"},
    "BUCARAMANGA": {"0114"},
    "BOGOTA": {"0111", "0108"},
}

def read_shipments_file() -> pd.DataFrame:
    df = pd.read_excel(RAW / DATOS["archivo_despachos"])
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas del Excel crudo a snake_case (ver COLUMNAS_LIMPIAS),
    por posición y no por nombre, para no depender de los espacios finales o
    problemas de codificación que trae el archivo original."""
    df = df.copy()
    df.columns = COLUMNAS_LIMPIAS
    return df

def clean_column_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    df["valor_flete"] = pd.to_numeric(df["valor_flete"], errors="coerce")
    df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce")
    df["sede"] = df["sede"].astype(str).str.strip().str.zfill(4)
    df["sede_flete"] = df["sede_flete"].astype(str).str.strip().str.zfill(4)
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    df["fecha_captura"] = parse_dates(df["fecha_captura"])
    df["fecha_entrega"] = parse_dates(df["fecha_entrega"])
    if "dias_promesa_entrega" in df:
        df["dias_promesa_entrega"] = pd.to_numeric(df["dias_promesa_entrega"], errors="coerce")   # texto en el histórico
    return df

def parse_dates(columna: pd.Series) -> pd.Series:
    # Lee fechas con formatos mezclados (FORMATOS_FECHA) y las deja en el día, sin hora.
    # Con barras se lee día/mes; solo si eso es imposible (segundo número > 12) se lee mes/día.
    # Lo que no encaja en ningún formato queda vacío (NaT).
    if pd.api.types.is_datetime64_any_dtype(columna):
        return columna.dt.normalize()

    # 1. "p.m." → "PM" para que lo entienda %p
    texto = (columna.astype("string").str.strip()
             .str.replace("a.m.", "AM", regex=False).str.replace("p.m.", "PM", regex=False))

    # 2. Cada valor se queda con el primer formato que encaje
    fechas = pd.Series(pd.NaT, index=columna.index, dtype="datetime64[ns]")
    for formato in FORMATOS_FECHA:
        faltan = fechas.isna()
        fechas[faltan] = pd.to_datetime(texto[faltan], format=formato, errors="coerce")

    # 3. Barras donde día/mes es imposible: mes/día
    faltan = fechas.isna()
    fechas[faltan] = pd.to_datetime(texto[faltan].str.split().str[0], format="%m/%d/%Y", errors="coerce")
    return fechas.dt.normalize()

def normalize_dispatch_type(df: pd.DataFrame) -> pd.DataFrame:
    # Quita espacios sobrantes ("TRASLADOS   " → "TRASLADOS") y unifica variantes (TIPO_DESPACHO_UNIFICADO)
    df = df.copy()
    tipo = df["tipo_despacho"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["tipo_despacho"] = tipo.replace(TIPO_DESPACHO_UNIFICADO)
    return df

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Filas idénticas en todas las columnas: se deja una
    duplicadas = df.duplicated()
    if duplicadas.any():
        print(f"ADVERTENCIA: se quitan {duplicadas.sum()} filas idénticas a otra")
    return df[~duplicadas].copy()

def _has_value(serie: pd.Series) -> pd.Series:
    # True donde el campo trae un dato real (no vacío ni "NO DISPONIBLE")
    texto = serie.astype("string").str.strip().str.upper()
    return (serie.notna() & ~texto.isin(["", "NAN", "NONE", "<NA>", "NO DISPONIBLE"])).fillna(False).astype(bool)

def resolve_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    # Filas con la misma guía, fecha de captura, peso y destino (revisadas con Johan el 2026-11-16):
    # Solo se agrupan filas con guía y peso, y solo si alguna fila del grupo trae factura (sin ninguna, como en
    # flota propia, no hay cómo saber si son repetidos).
    #   E. cada documento distinto es un pedido distinto (la guía puede ser la placa del camión) → se conserva
    #   Si los documentos están vacíos o son el mismo:
    #   B. misma factura, distinto tipo y distinto flete: un despacho con el cobro partido → una fila con el flete
    #      sumado y el tipo del cobro mayor (el peso queda una sola vez)
    #   A. misma factura, distinto tipo y mismo flete: el mismo despacho registrado dos veces → queda el de CLIENTES
    #      (decisión de Johan del 2026-11-16); si ninguno es CLIENTES, el primero
    #   C. el mismo documento facturado en dos meses contables → queda el del mes más antiguo
    #   D. filas sin documento cuando otra del grupo sí lo trae (copia sin factura) → se quitan; si ninguna trae
    #      documento, queda una: la que trae factura, luego la del mes más antiguo
    clave = ["guia", "fecha_captura", "peso", "ciudad_destino"]
    identificables = df["guia"].notna() & df["peso"].notna()
    repetidas = identificables & df.duplicated(subset=clave, keep=False)
    if not repetidas.any():
        return df

    # 1. Qué trae cada fila
    df = df.copy()
    factura = df["factura"] if "factura" in df else pd.Series(pd.NA, index=df.index)
    con_documento, con_factura = _has_value(df["documento"]), _has_value(factura)

    # 2. Decidir grupo por grupo
    quitar, unidos, depurados = [], 0, 0
    for _, grupo in df[repetidas].groupby(clave, dropna=False):
        filas = grupo.index
        if not con_factura[filas].any():
            continue                                                            # sin factura: no se sabe
        documentos = grupo["documento"].astype("string").str.strip()[con_documento[filas]]
        misma_factura = con_factura[filas].all() and factura[filas].nunique() == 1
        if documentos.nunique() <= 1 and misma_factura and grupo["tipo_despacho"].nunique() > 1:
            fletes = grupo["valor_flete"]
            if np.isclose(fletes.min(), fletes.max(), rtol=0.01):              # A
                clientes = grupo.index[grupo["tipo_despacho"] == "CLIENTES"]
                conservar = clientes[0] if len(clientes) else filas[0]
                quitar += [fila for fila in filas if fila != conservar]
                depurados += 1
                continue
            principal = fletes.idxmax()                                         # B
            df.loc[principal, "valor_flete"] = fletes.sum()
            quitar += [fila for fila in filas if fila != principal]
            unidos += 1
            continue
        orden = pd.DataFrame({"documento": grupo["documento"].astype("string").str.strip(),
                              "con_documento": con_documento[filas], "factura": con_factura[filas],
                              "mes": grupo["mes"]}).sort_values(["factura", "mes"], ascending=[False, True])
        if orden["con_documento"].any():
            # E y C: una fila por documento distinto (la del mes más antiguo); D: fuera las que no traen
            conservar = orden[orden["con_documento"]].drop_duplicates("documento").index
        else:
            conservar = orden.index[:1]                                         # D sin documentos
        sobran = [fila for fila in filas if fila not in conservar]
        if sobran:
            quitar += sobran
            depurados += 1

    if quitar:
        print(f"ADVERTENCIA: registros repetidos: {unidos} despachos con el cobro partido se unen y "
              f"{depurados} registrados dos veces quedan en uno ({len(quitar)} filas menos)")
    return df.drop(index=quitar)

def exclude_non_positive_weight(df: pd.DataFrame) -> pd.DataFrame:
    # Peso <= 0: sin carga real. El peso vacío no se toca aquí (lo quita get_shipments)
    sin_peso = df["peso"] <= 0
    if sin_peso.any():
        print(f"ADVERTENCIA: se quitan {sin_peso.sum()} despachos con peso <= 0")
    return df[~sin_peso].copy()

def blank_impossible_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Se borra solo el dato malo, no el despacho
    df = df.copy()
    entrega_antes = df["fecha_entrega"] < df["fecha_captura"]
    promesa_negativa = df["dias_promesa_entrega"] < 0
    df.loc[entrega_antes, "fecha_entrega"] = pd.NaT
    df.loc[promesa_negativa, "dias_promesa_entrega"] = np.nan
    if entrega_antes.any() or promesa_negativa.any():
        print(f"ADVERTENCIA: se borran {entrega_antes.sum()} fechas de entrega anteriores a la captura "
              f"y {promesa_negativa.sum()} promesas de entrega negativas")
    return df

def clean_shipments(df: pd.DataFrame) -> pd.DataFrame:
    # Limpieza común del Excel de 2026 y del histórico (process_shipments y scripts/cotizar_historico.py)
    df = normalize_dispatch_type(df)
    df = drop_exact_duplicates(df)
    df = resolve_duplicate_records(df)
    df = exclude_non_positive_weight(df)
    return blank_impossible_dates(df)

def load_shipments() -> pd.DataFrame:
    return clean_column_types(standardize_columns(read_shipments_file()))

def filter_cedis(df: pd.DataFrame):
    df_filtrado = df[df["sede"].isin([SEDE_MOSQUERA, SEDE_BARRANQUILLA])]
    return df_filtrado

def normalize_city(city):
    # Mayúsculas, sin espacios ni comillas al inicio o al final ("'BARRANQUILLA" → "BARRANQUILLA"),
    # y nombres corregidos según CIUDAD_CORRECCION
    city = city.str.upper().str.strip().str.strip("'\"").str.strip()
    return city.replace(CIUDAD_CORRECCION)

def correct_city_by_origin(df: pd.DataFrame) -> pd.DataFrame:
    # Ciudades cuyo nombre depende de la sede de origen (CIUDAD_CORRECCION_POR_SEDE); va después de normalize_city
    df = df.copy()
    corregida = [CIUDAD_CORRECCION_POR_SEDE.get((sede, ciudad), ciudad)
                 for sede, ciudad in zip(df["sede"], df["ciudad_destino"])]
    df["ciudad_destino"] = pd.Series(corregida, index=df.index, dtype=df["ciudad_destino"].dtype)
    return df

def search_department(df_filtrado: pd.DataFrame, divipola: pd.DataFrame):
    diccionarioMunicipio = dict(zip(divipola["Municipio"],divipola["Departamento"]))
    diccionarioCP = dict(zip(divipola["CentroPoblado"],divipola["Departamento"]))
    df_filtrado = df_filtrado.copy()
    df_filtrado["departamento"] = df_filtrado["ciudad_destino"].map(diccionarioMunicipio)
    df_filtrado["departamento"] = df_filtrado["departamento"].fillna(df_filtrado["ciudad_destino"].map(diccionarioCP))
    return df_filtrado



def exclude_metro_area(df) -> pd.DataFrame:
    # La ciudad se compara ya normalizada: en ene–jun 2024 viene " MOSQUERA" (con espacio) y se colaban
    # los despachos urbanos al área metropolitana
    df = df.copy()
    ciudad = normalize_city(df["ciudad_destino"])
    df = df[~(
        ((df["sede"] == SEDE_BARRANQUILLA) & (ciudad.isin(BARRANQUILLA_METRO))) |
        ((df["sede"] == SEDE_MOSQUERA) & (ciudad.isin(BOGOTA_METRO)))
    )].copy()
    return df

def exclude_inbound_to_cedi(df: pd.DataFrame) -> pd.DataFrame:
    # Recolecciones en proveedores y devoluciones de clientes llegan al CEDI, no salen de él:
    # no hacen parte de la consolidación
    return df[~df["tipo_despacho"].isin(TIPOS_ENTRADA_AL_CEDI)].copy()


def load_carrier_types() -> dict:
    """transportadora (normalizada a mayúsculas/sin espacios) -> tipo
    (PAQUETEO, MASIVO, URBANO, FLOTA PROPIA, MENSAJEROS), según
    data/external/tipo_transporte.xlsx."""
    tipos = pd.read_excel(EXTERNAL / "tipo_transporte.xlsx")
    transportadora_norm = tipos["Transportadora"].str.upper().str.strip()
    return dict(zip(transportadora_norm, tipos["tipo"]))


def tag_carrier_type(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna tipo_transportadora (PAQUETEO, MASIVO, URBANO,
    FLOTA PROPIA, MENSAJEROS) según load_carrier_types, sin excluir
    ninguna fila. Los despachos MASIVO (p. ej. FRACOR, COLTANQUES) siguen
    participando en el modelo: su peso compite por la capacidad del camión
    de la región igual que cualquier otro pedido, y el modelo decide si
    conviene incorporarlos a la consolidación regional o dejarlos con su
    costo histórico pagado."""
    mapa_tipo = load_carrier_types()
    transportadora_norm = df["transportadora"].str.upper().str.strip()
    tipo_transportadora = transportadora_norm.map(mapa_tipo)

    sin_clasificar = df.loc[tipo_transportadora.isna(), "transportadora"].unique()
    if len(sin_clasificar) > 0:
        print(f"ADVERTENCIA: {len(sin_clasificar)} transportadora(s) sin clasificar "
              f"en tipo_transporte.xlsx: {list(sin_clasificar)}")

    df = df.copy()
    df["tipo_transportadora"] = tipo_transportadora
    return df


def map_region(df: pd.DataFrame) -> pd.DataFrame:

    mapeo = pd.read_csv(EXTERNAL / "mapeo_regiones.csv")
    mapa = dict(zip(mapeo["departamento"], mapeo["region_logistica"]))

    df = df.copy()
    df["region_logistica"] = df["departamento"].str.upper().map(mapa)
    df["region_logistica"] = df.apply(
        lambda r: CIUDAD_REGION_OVERRIDE.get(r["ciudad_destino"], r["region_logistica"]), axis=1
    )
    df["region_logistica"] = df.apply(
        lambda r: GUIA_REGION_OVERRIDE.get(str(r["guia"]), r["region_logistica"]), axis=1
    )

    mask_interior = (df["sede"] == SEDE_BARRANQUILLA) & (df["region_logistica"].isin(REGIONES_INTERIOR_BAQ))
    df.loc[mask_interior, "region_logistica"] = "INTERIOR_BAQ"
    return df


def load_regional_hubs() -> dict:
    """region_logistica -> hub (sede intermedia de consolidación)."""
    nodos = pd.read_csv(EXTERNAL / "nodos_regionales.csv")
    return dict(zip(nodos["region_logistica"], nodos["hub"]))


def load_vehicle_capacities() -> dict:
    """vehiculo -> capacidad máxima en kg (parámetro q_v)."""
    cap = pd.read_csv(EXTERNAL / "capacidades_vehiculo.csv")
    return dict(zip(cap["vehiculo"], cap["capacidad_kg"]))


def load_ftl_rates() -> dict:
    """(vehiculo, sede, hub) -> tarifa FTL en COP (parámetro f_vr)."""
    tar = pd.read_csv(EXTERNAL / "tarifas_ftl.csv", dtype={"sede": str})
    return {
        (row.vehiculo, row.sede, row.hub): row.tarifa_cop
        for row in tar.itertuples()
    }

def load_ftl_transit_times() -> dict:
    """(sede, hub) -> días de tránsito del camión troncal sede → sede regional
    (no depende del tipo de vehículo)."""
    tiempos = pd.read_csv(EXTERNAL / "tiempos_transito_ftl.csv", dtype={"sede": str})
    return {
        (row.sede, row.hub): row.dias_transito
        for row in tiempos.itertuples()
    }

def _to_float(valor):
    """Convierte a float normal (Excel/xlwings a veces devuelve Decimal, u
    otras veces texto como "NO HAY RUTA" cuando ninguna transportadora tiene
    ruta disponible). Si no se puede convertir, devuelve None."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None

def _excel_quoter(consultas: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    ruta_temporal = Path(tempfile.gettempdir())/"COTIZADOR_temp.xlsx"
    shutil.copy(RAW / "COTIZADOR 2026.xlsx", ruta_temporal)

    resultados = []
    app = xw.App(visible=False)
    try:
        wb = app.books.open(ruta_temporal, update_links=False, read_only=True)
        sh = wb.sheets["COTIZADOR"]

        for origen, destino, peso, unidades in tqdm(consultas, desc="Cotizando", unit="consulta"):
            sh.range("D5").value = origen
            sh.range("D6").value = destino
            sh.range("D3").value = peso
            sh.range("D4").value = unidades
            sh.range("D7").value = parametros.VALOR_DECLARADO_COTIZACION
            app.calculate()

            transportadora, flete, tiempo_entrega = sh.range("G3:G5").value
            promesa_envia = sh.range("F12").value  # fila de ENVIA; el orden de la tabla es fijo
            resultados.append({
                "origen": origen,
                "destino": destino,
                "peso": peso,
                "unidades": unidades,
                "transportadora": transportadora,
                "flete": _to_float(flete),
                "tiempo_entrega": _to_float(tiempo_entrega),
                "promesa_envia": _to_float(promesa_envia),
            })
        wb.close()
    finally:
        app.quit()
    return pd.DataFrame(resultados)

RESULTADO_COTIZACION = ["transportadora", "flete", "tiempo_entrega", "promesa_envia"]
COLUMNAS_COTIZACION = ["origen", "destino", "peso", "unidades"] + RESULTADO_COTIZACION
LLAVE_CACHE = ["origen", "destino", "peso_llave", "unidades_llave"]
TAMANO_TANDA_COTIZACION = 500   # la caché se guarda después de cada tanda: si la cotización se corta, se retoma

def _with_cache_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["peso_llave"] = df["peso"].round(3)
    df["unidades_llave"] = df["unidades"].round(3)
    return df

def _load_quote_cache() -> pd.DataFrame:
    if CACHE_COTIZACIONES.exists():
        return pd.read_parquet(CACHE_COTIZACIONES)
    return pd.DataFrame({
        "origen": pd.Series(dtype=str), 
        "destino": pd.Series(dtype=str),
        "peso": pd.Series(dtype=float), 
        "unidades": pd.Series(dtype=float),
        "transportadora": pd.Series(dtype=str),
        "flete": pd.Series(dtype=float),
        "tiempo_entrega": pd.Series(dtype=float),
        "promesa_envia": pd.Series(dtype=float),
    })

def quote(consultas: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    if not consultas:
        return pd.DataFrame(columns=COLUMNAS_COTIZACION)

    pedidas = pd.DataFrame(consultas, columns=["origen", "destino", "peso", "unidades"])
    pedidas["_orden"] = range(len(pedidas))
    pedidas = _with_cache_key(pedidas)

    cache = _with_cache_key(_load_quote_cache())
    cache = cache.drop_duplicates(subset=LLAVE_CACHE, keep="last")

    cruce = pedidas.merge(cache[LLAVE_CACHE + RESULTADO_COTIZACION], on=LLAVE_CACHE, how="left", indicator=True)
    faltantes = cruce[cruce["_merge"] == "left_only"]
    a_cotizar = list(
        faltantes.drop_duplicates(subset=LLAVE_CACHE)[["origen", "destino", "peso", "unidades"]]
        .itertuples(index=False, name=None)
    )

    if a_cotizar:
        # Por tandas, guardando la caché después de cada una: lo cotizado no se pierde si Excel o el equipo se caen
        cache_actualizada = cache
        for inicio in range(0, len(a_cotizar), TAMANO_TANDA_COTIZACION):
            nuevos = _excel_quoter(a_cotizar[inicio:inicio + TAMANO_TANDA_COTIZACION])
            cache_actualizada = pd.concat([cache_actualizada[COLUMNAS_COTIZACION], nuevos], ignore_index=True)
            cache_actualizada = _with_cache_key(cache_actualizada).drop_duplicates(subset=LLAVE_CACHE, keep="last")
            cache_actualizada[COLUMNAS_COTIZACION].to_parquet(CACHE_COTIZACIONES, index=False)

        cruce = pedidas.merge(
            cache_actualizada[LLAVE_CACHE + RESULTADO_COTIZACION],
            on=LLAVE_CACHE, how="left",
        )

    resultado = cruce.sort_values("_orden")[COLUMNAS_COTIZACION]
    return resultado.reset_index(drop=True)



def fill_destination_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ciudad_destino"] = df["ciudad_destino"].fillna(df["guia"].astype(str).map(GUIA_CIUDAD_OVERRIDE))
    return df

def fill_units(df: pd.DataFrame) -> pd.DataFrame:
    # Unidades vacías o en 0 (sobre todo despachos MASIVO): 1 unidad por tonelada, mínimo 1
    df = df.copy()
    unidades_estimadas = (df["peso"] / 1000).round().clip(lower=1)
    df["unidades"] = df["unidades"].mask(df["unidades"] == 0).fillna(unidades_estimadas)
    return df

def process_shipments(df: pd.DataFrame | None = None) -> pd.DataFrame:
    # Sin argumento lee el Excel de despachos; con un DataFrame ya cargado (p. ej. el histórico 2024–2026 en
    # parquet, con las columnas en snake_case) le aplica la misma limpieza
    df = load_shipments() if df is None else clean_column_types(df)
    df = fill_destination_city(df)
    df = fill_units(df)
    df = clean_shipments(df)
    df = filter_cedis(df)
    df = exclude_metro_area(df)
    df = exclude_inbound_to_cedi(df)
    df = tag_carrier_type(df)

    divipola = pd.read_excel(EXTERNAL / "divipola.xlsx")
    df["ciudad_destino"] = normalize_city(df["ciudad_destino"])
    df = correct_city_by_origin(df)
    divipola["Municipio"] = normalize_city(divipola["Municipio"])
    divipola["CentroPoblado"] = normalize_city(divipola["CentroPoblado"])
    divipola["Departamento"] = normalize_city(divipola["Departamento"])

    df = search_department(df,divipola)
    df = map_region(df)
    return df

def get_shipments(sede: str, mes_inicio: str, mes_fin: str) -> pd.DataFrame:
    """Pedidos de una sede de origen, en todas sus regiones logísticas, dentro
    de un rango de meses — el conjunto I de la formulación. El mes es la columna
    `mes` del archivo de la empresa (mes contable), que no siempre coincide con el
    mes de la fecha de captura (en ene–mar 2026 difiere en una pequeña parte de los registros),
    tratado como demanda determinística conocida al momento de decisión.
    El modelo decide simultáneamente sobre todas las regiones; la única
    simplificación de esta versión es que la demanda del corte ya se conoce.

    Columnas devueltas: guia, tipo (CLIENTES, TRASLADOS, TRASLADOS AL GASTO,
    OTROS; las recolecciones en proveedor y las devoluciones se excluyen en process_shipments),
    peso_kg, flete_pagado_actual (flete real pagado; vacío en pedidos nuevos, ver
    prepare_unconsolidated_cost), ciudad_destino, ciudad_sede,
    region_logistica, fecha_captura, dias_promesa_entrega, unidades, tipo_ruta,
    fecha_entrega, latitud_destino, longitud_destino, sede, sede_flete
    (sede que paga el flete), tipo_transportadora (PAQUETEO, MASIVO, ...),
    transportadora (la que despachó el pedido, con nombre comercial: COLVANES → ENVIA).
    No filtra por flete: los pedidos sin flete pagado se manejan según el modo.
    """
    df = process_shipments()
    df = df[
        (df["sede"] == sede)
        & (df["mes"] >= mes_inicio)
        & (df["mes"] <= mes_fin)
        & df["peso"].notna()
    ].copy()

    pedidos = pd.DataFrame({
        "guia": df["guia"].values,
        "tipo": df["tipo_despacho"].values,
        "peso_kg": df["peso"].values,
        "flete_pagado_actual": df["valor_flete"].values,
        "ciudad_destino": df["ciudad_destino"].values,
        "ciudad_sede": df["ciudad_sede"].values,
        "region_logistica": df["region_logistica"].values,
        "fecha_captura": df["fecha_captura"].values,
        "dias_promesa_entrega": df["dias_promesa_entrega"].values,
        "unidades": df["unidades"].values,
        "tipo_ruta": df["tipo_ruta"].values,
        "fecha_entrega": df["fecha_entrega"].values,
        "latitud_destino": df["latitud_destino"].values,
        "longitud_destino": df["longitud_destino"].values,
        "sede": df["sede"].values,
        "sede_flete": df["sede_flete"].values,
        "tipo_transportadora": df["tipo_transportadora"].values,
        "transportadora": df["transportadora"].str.upper().str.strip().replace(NOMBRE_TRANSPORTADORA).values,
    })
    return pedidos.reset_index(drop=True)

def add_unique_id(pedidos: pd.DataFrame) -> pd.DataFrame:
    pedidos = pedidos.copy()
    secuencia = pedidos.groupby("guia").cumcount()
    pedidos["id_pedido"] = pedidos["guia"].astype(str) + "_" + secuencia.astype(str)
    return pedidos

def rate_shipments(pedidos: pd.DataFrame, nodos_regionales: dict) -> pd.DataFrame:
    pedidos = pedidos.copy()
    pedidos["nodo_regional"] = pedidos["region_logistica"].map(nodos_regionales)
    con_hub = pedidos["nodo_regional"].notna()

    pedidos["flete_triangulacion"] = 0.0
    pedidos["tiempo_triangulacion"] = 0.0

    # Un traslado a una sede donde descarga el troncal del hub (SEDES_LLEGADA_HUB)
    # llega directo: sin tramo hub→destino, flete_triangulacion queda en 0.
    # Todo lo demás (clientes, o traslados a otras sedes) se cotiza hub→destino.
    es_traslado = pedidos["tipo"].str.startswith("TRASLADOS", na=False)
    sede_llegada_del_hub = [
        sede in SEDES_LLEGADA_HUB.get(hub, set())
        for sede, hub in zip(pedidos["sede_flete"], pedidos["nodo_regional"])
    ]
    llega_directo = es_traslado & con_hub & pd.Series(sede_llegada_del_hub, index=pedidos.index)
    pedidos["llega_directo"] = llega_directo
    a_cotizar = con_hub & ~llega_directo

    pedidos_cotizar = pedidos[a_cotizar]
    consultas = list(zip(
        pedidos_cotizar["nodo_regional"],
        pedidos_cotizar["ciudad_destino"],
        pedidos_cotizar["peso_kg"],
        pedidos_cotizar["unidades"],
    ))
    result = quote(consultas)

    # Basta con superar cualquier flete directo para que el modelo nunca lo consolide;
    # un factor mayor solo empeora la escala numérica del problema para el solver.
    # Anclado a la mayor tarifa conocida (cotizada o pagada): funciona también sin histórico
    big_m = pd.concat([result["flete"], pedidos["flete_pagado_actual"]]).max() * parametros.FACTOR_BIG_M

    sin_ruta = result["flete"].isna()
    print(f"Pedidos sin ruta de triangulación (Big-M aplicado): {sin_ruta.sum()} de {len(pedidos_cotizar)}")

    pedidos["sin_ruta_triangulacion"] = False
    pedidos.loc[a_cotizar, "sin_ruta_triangulacion"] = sin_ruta.values

    pedidos.loc[a_cotizar, "flete_triangulacion"] = pd.Series(result["flete"].values, index=pedidos_cotizar.index).fillna(big_m)
    pedidos.loc[a_cotizar, "tiempo_triangulacion"] = result["tiempo_entrega"].values
    return pedidos

def rate_ltl_direct(pedidos: pd.DataFrame) -> pd.DataFrame:
    pedidos = pedidos.copy()
    consultas = list(zip(
        pedidos["ciudad_sede"],
        pedidos["ciudad_destino"],
        pedidos["peso_kg"],
        pedidos["unidades"],
    ))
    result = quote(consultas)
    # Queda NaN si ninguna transportadora cubre el destino; lo resuelve prepare_unconsolidated_cost
    pedidos["flete_directo_cotizado"] = result["flete"].values
    pedidos["transportadora_directo_cotizada"] = result["transportadora"].values
    pedidos["tiempo_ltl_cotizado"] = result["tiempo_entrega"].values

    # Promesa en días hábiles (sábado = 0,5). Sin ruta de ENVIA se trata como reexpedido.
    pedidos["promesa_envia"] = result["promesa_envia"].values
    # Promesa estricta = promesa de ENVIA, o la de reexpedido si ENVIA no cubre el destino
    pedidos["promesa_estricta"] = pedidos["promesa_envia"].fillna(parametros.PROMESA_REEXPEDIDO)
    return pedidos

def prepare_unconsolidated_cost(pedidos: pd.DataFrame, modo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define flete_sin_consolidar, el costo de despachar directo (u = 1), y
    transportadora_directo (la que haría el despacho directo, para su cumplimiento), según el modo:
    - historico: lo realmente pagado y la transportadora que realmente despachó.
      Se detiene si algún pedido no tiene flete pagado.
    - operacion: la cotización directa sede→destino y su transportadora ganadora. Los
      pedidos sin cotización salen del modelo y se devuelven aparte para revisión manual.
    Devuelve (pedidos para el modelo, pedidos para revisión manual)."""
    pedidos = pedidos.copy()

    if modo == "historico":
        sin_pagado = pedidos["flete_pagado_actual"].isna().sum()
        if sin_pagado:
            raise ValueError(f"Modo histórico: {sin_pagado} pedidos no tienen flete pagado; "
                             "no se puede medir el ahorro contra lo pagado. ¿Querías --modo operacion?")
        pedidos["flete_sin_consolidar"] = pedidos["flete_pagado_actual"]
        pedidos["transportadora_directo"] = pedidos["transportadora"]
        return pedidos, pedidos.iloc[0:0]

    con_pagado = pedidos["flete_pagado_actual"].notna().sum()
    if con_pagado:
        print(f"ADVERTENCIA: {con_pagado} pedidos traen flete pagado; ¿querías correr --modo historico?")

    sin_cotizacion = pedidos["flete_directo_cotizado"].isna()
    if sin_cotizacion.any():
        print(f"ADVERTENCIA: {sin_cotizacion.sum()} pedidos sin cotización directa: salen del modelo para revisión manual")
    revision_manual = pedidos[sin_cotizacion].copy()
    pedidos = pedidos[~sin_cotizacion].copy()
    pedidos["flete_sin_consolidar"] = pedidos["flete_directo_cotizado"]
    pedidos["transportadora_directo"] = pedidos["transportadora_directo_cotizada"]
    return pedidos, revision_manual
