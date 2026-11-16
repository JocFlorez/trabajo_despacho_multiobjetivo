import numpy as np
import pandas as pd

from scripts import construir_series


def test_selecciona_solo_la_sede_y_los_despachos_con_peso():
    despachos = pd.DataFrame({"sede": ["0111", "0111", "0211"], "peso": [10.0, np.nan, 5.0]})
    assert construir_series.select_shipments(despachos, "0111")["peso"].tolist() == [10.0]
