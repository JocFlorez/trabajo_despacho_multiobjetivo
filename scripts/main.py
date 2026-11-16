import argparse
from datetime import datetime

from despacho_multiobjetivo.config import PROCESSED
from despacho_multiobjetivo.datos import data_processing
from despacho_multiobjetivo.datos.cargar import prepare_orders
from despacho_multiobjetivo.modelos import model_mensual
from despacho_multiobjetivo.metodos import solver
from despacho_multiobjetivo.reportes import report

def valid_date(texto: str) -> str:
    try:
        datetime.strptime(texto, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Fecha inválida '{texto}'. Usa el formato YYYY-MM-DD.")
    return texto

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelo de consolidación FTL/LTL")
    parser.add_argument("--sede", required=True, choices=[data_processing.SEDE_MOSQUERA, data_processing.SEDE_BARRANQUILLA], help="Código de sede")
    parser.add_argument("--mes-inicio", required=True, type=valid_date, help="Fecha inicial (YYYY-MM-DD)")
    parser.add_argument("--mes-fin", required=True, type=valid_date, help="Fecha final (YYYY-MM-DD)")
    parser.add_argument("--ruta-reporte", default=None,
                        help="Ruta de excel de salida (por defecto data/processed/reporte_<sede>_<mes>_<modo>.xlsx)")
    parser.add_argument("--modo", choices=["historico", "operacion"], default="operacion",
                        help="historico: ahorro contra lo pagado (lo usan las pruebas de la tesis, ver config.py); operacion: pedidos nuevos, contra lo cotizado (por defecto: es el modo de uso real)")
    return parser.parse_args()

def default_report_path(sede: str, mes_inicio: str, mes_fin: str, modo: str) -> str:
    # El nombre lleva sede, periodo y modo para que una corrida no sobrescriba otra
    periodo = mes_inicio[:7] if mes_inicio[:7] == mes_fin[:7] else f"{mes_inicio[:7]}_a_{mes_fin[:7]}"
    return str(PROCESSED / f"reporte_{sede}_{periodo}_{modo}.xlsx")

def main(sede: str, mes_inicio: str, mes_fin: str, ruta_reporte: str | None, modo: str) -> None:
    ruta_reporte = ruta_reporte or default_report_path(sede, mes_inicio, mes_fin, modo)
    print(f"Modo: {modo}")
    print("Cargando y cotizando pedidos (directo y a las sedes regionales)")
    pedidos, revision_manual, nodos_regionales = prepare_orders(sede, mes_inicio, mes_fin, modo)
    print("Cargando información de vehículos")
    capacidades = data_processing.load_vehicle_capacities()
    tarifas_ftl = data_processing.load_ftl_rates()
    print("Construyendo el modelo")
    problema, x, u, y = model_mensual.build_model(pedidos, capacidades, tarifas_ftl, nodos_regionales, sede)

    print("Resolviendo el modelo")
    problema = solver.solve(problema)
    print("Creando reporte")
    resultado_pedidos = report.extract_order_results(pedidos, x)
    resultado_vehiculos = report.extract_vehicle_results(y)
    resumen = report.build_summary(resultado_pedidos, problema, y, modo)

    print("Exportando resultados")
    report.export_report(resumen, resultado_pedidos, resultado_vehiculos, revision_manual, ruta_reporte)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.sede, args.mes_inicio, args.mes_fin, args.ruta_reporte, args.modo)