from scripts import revisar_antes_de_subir as revisar

# Valores de prueba inventados: el script real lee los prohibidos de data/ (que no se publica)
NOMBRES = {"ACME LOGISTICA"}
PROHIBIDOS = {"4321987", "555" + "66677788"}   # partidos para no bloquear este archivo


def _hallazgos(texto):
    return revisar.find_sensitive(texto, PROHIBIDOS, NOMBRES)


def test_detecta_el_nombre_de_la_empresa_sin_importar_mayusculas():
    assert _hallazgos("# datos de Acme Logistica en enero")


def test_detecta_numeros_de_guia_de_nueve_o_mas_digitos():
    assert _hallazgos('GUIA = {"' + "7" * 11 + '": "CALI"}')


def test_no_bloquea_las_guias_de_ejemplo_que_empiezan_por_ceros():
    assert not _hallazgos('"00000000001": "EJE_CAFETERO"')


def test_detecta_cifras_con_forma_de_pesos():
    assert _hallazgos("costo 987." + "654.321 en enero")


def test_no_bloquea_montos_menores_de_cien_millones_escritos_con_puntos():
    assert not _hallazgos("# 1 Turbo ($2.100.000) y Mula + 2 Turbos costaría 9.200.000")


def test_no_bloquea_decimales_ni_miles_sueltos():
    assert not _hallazgos("(8.500,0000000001 kg) y 37.000 kg → Mula")


def test_detecta_valores_prohibidos_escritos_de_cualquier_forma():
    for escrito in ("4321987", "4_321_987", "4.321.987", "$4.321.987", "4321987.0"):
        assert _hallazgos(f"tarifa = {escrito}"), escrito


def test_no_bloquea_valores_inventados():
    assert not _hallazgos("TARIFAS = {('Turbo', SEDE, 'MEDELLIN'): 2_100_000}")


def test_no_deja_subir_ramas_original_ni_etiquetas():
    cero = "0" * 40
    lineas = [f"refs/heads/original/main {'a' * 40} refs/heads/original/main {cero}",
              f"refs/tags/revision {'b' * 40} refs/tags/revision {cero}",
              f"refs/heads/main {'c' * 40} refs/heads/main {cero}"]
    errores = revisar.check_refs(lineas)
    assert len(errores) == 2
    assert all("main " not in error or "original" in error for error in errores)
