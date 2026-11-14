"""Veri çekme (extract) katmanı. Her kaynak bağımsız bir modüldür ve `update()` fonksiyonu sunar."""
from . import dgcis, eia, epdk, gacc, jodi

SOURCES = {
    "jodi": jodi.update,
    "eia": eia.update,
    "epdk": epdk.update,
    "dgcis": dgcis.update,
    "gacc": gacc.update,
}
