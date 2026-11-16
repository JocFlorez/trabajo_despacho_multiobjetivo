# MODELO MULTIOBJETIVO PARA EL APOYO A DECISIONES DE DESPACHO EN EMPRESAS GENERADORAS DE CARGA

Modelo de consolidación de carga (FTL vs. paquetería) para los CEDI de Mosquera (0111) y Barranquilla (0211).
Se construye por etapas:

1. **Etapa 1 · mensual:** qué pedidos van en camión hacia la sede regional y cuántos camiones se compran en el mes.
2. **Etapa 2 · ventanas de tiempo:** cada pedido escoge el día en que sale (espera máxima E); se minimiza el costo.
3. **Etapa 3 · multiobjetivo:** costo (Z1) y días de retraso ponderados por pedido (Z2) con la frontera
   de Pareto; cada punto se reporta además en % a tiempo y días hábiles de retraso por pedido.

## Estructura

El código está organizado por capas: los datos alimentan las reglas del negocio, las reglas alimentan
los modelos y los modelos se resuelven con distintos métodos.

```
MODELOS/
├── src/despacho_multiobjetivo/   ← paquete del proyecto
│   ├── config.py                 rutas, escenario base (sede, mes, modo, espera) y parámetros del solver
│   ├── parametros.py             parámetros del modelo: cumplimientos, histograma y castigo por retraso, manipulación, Big-M, promesas
│   ├── datos/
│   │   ├── data_processing.py    lectura, limpieza, regiones, exclusiones y cotización
│   │   └── cargar.py             prepare_orders() y load_data(): pedidos cotizados y parámetros
│   ├── reglas_negocio/           reglas del negocio, sin optimización
│   │   ├── calendario.py         días hábiles, festivos, sábado y fecha límite
│   │   ├── opciones_salida.py    días en que puede salir cada pedido, con su espera y su retraso
│   │   └── servicio.py           cumplimiento de las transportadoras y castigo esperado por retraso
│   ├── modelos/
│   │   ├── model_mensual.py      Etapa 1
│   │   ├── componentes.py        variables, restricciones y objetivos de las Etapas 2 y 3 (una sola vez)
│   │   ├── model_ventanas.py     Etapa 2 = componentes + min Z1
│   │   ├── model_multiobjetivo.py Etapa 3 = componentes + Z1 y Z2
│   │   └── version_pyomo.py      la Etapa 3 escrita en Pyomo (validación y comparación de solvers)
│   ├── metodos/
│   │   ├── solver.py             resolver la Etapa 1
│   │   ├── frontera_pareto.py    extremos, matriz de pagos y frontera de Pareto
│   │   ├── resolver_pyomo.py     resolver la versión en Pyomo con distintos solvers
│   │   └── metaheuristicas/      evaluador (Z1 y Z2 sin solver, validado contra el MILP), búsqueda tabú
│   │                             y genético NSGA-II (cruce por región, reparación de camiones)
│   └── reportes/
│       ├── report.py             resultados por pedido y vehículo, y reporte en Excel de la Etapa 1
│       ├── frontera.py           métricas de cada plan de la frontera y gráfica del espacio de objetivos
│       └── metaheuristicas.py    curva de convergencia de una corrida de la tabú
├── scripts/                      ← lo que se ejecuta
│   ├── main.py                       Etapa 1 con reporte en Excel
│   ├── main_multiobjetivo.py         Etapa 3: matriz de pagos, frontera y gráfica
│   ├── frontera_puntos_manuales.py   puntos adicionales de la frontera (zona del codo)
│   ├── validar_ventanas_tiempo.py    Etapa 2 para varias esperas frente a la Etapa 1
│   ├── comparar_pyomo.py             PuLP vs. Pyomo con el mismo modelo
│   ├── comparar_pyomo_semillas.py    la comparación repetida con varias semillas de HiGHS
│   ├── medir_cumplimiento.py         mide en el histórico el cumplimiento de ENVIA y el histograma de retrasos
│   ├── validar_evaluador.py          comprueba que el evaluador de las metaheurísticas da el mismo Z1 y Z2 que el MILP
│   ├── validar_tabu_search.py        búsqueda tabú con datos reales frente al MILP (todo directo y semillas del MILP)
│   ├── comparar_solvers_neos.py      CPLEX y Xpress en NEOS frente a HiGHS, con el modelo anonimizado y escalado
│   ├── frontera_dias_reales.py       frontera en días hábiles reales, % a tiempo y flete por kilo (lee los CSV)
│   ├── validar_genetico.py           NSGA-II sin y con semillas del MILP frente al MILP en los puntos del codo
│   └── comparar_metaheuristicas.py   comparación justa: tabú, genético propio y genético genérico, mismas semillas
├── tests/                        ← misma estructura que src/despacho_multiobjetivo
├── data/                         ← no se versiona (datos de la empresa)
│   ├── raw/                      datos de la empresa
│   ├── external/                 parámetros externos (tarifas, capacidades, regiones, tiempos)
│   └── processed/                resultados y cachés que genera el proyecto
└── pyproject.toml                paquete, dependencias y configuración de pytest
```

## Instalación

Desde la carpeta MODELOS, con el entorno virtual activo:

```bash
pip install -r requirements.txt
```

Instala el paquete `despacho_multiobjetivo` en modo editable (los cambios en `src/` se usan de inmediato) con sus dependencias.

## Cómo correr

El escenario base (sede, mes, modo y espera máxima) se cambia en `src/despacho_multiobjetivo/config.py`;
los parámetros del modelo, para la sensibilidad, en `src/despacho_multiobjetivo/parametros.py`.

Convenciones: carpetas, módulos, variables y comentarios en español; nombres de funciones en inglés.
`scripts/main.py` usa por defecto `--modo operacion` (uso real); las corridas de la tesis usan `historico`.

```bash
python scripts/main.py --sede 0111 --mes-inicio 2026-01-01 --mes-fin 2026-01-31 --modo historico
python scripts/main_multiobjetivo.py
python scripts/frontera_puntos_manuales.py
```

Los resultados quedan en `data/processed/`.

## Pruebas

```bash
python -m pytest
```
