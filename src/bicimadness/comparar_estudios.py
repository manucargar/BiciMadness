"""Compara los estudios (ejecuciones) de un algoritmo: qué estaciones, variables, modo... funciona mejor.

Para cada CSV de datos, ordena las ejecuciones de results/<algoritmo>/ por MAE y hace un T-test pareado de
cada una contra la mejor (sobre el MAE de cada estación en cada bloque). Si p >= 0.05 la diferencia con
la mejor no es significativa: son equivalentes. Solo se comparan ejecuciones del mismo CSV (mismas
estaciones y horas).

Uso:  python src/bicimadness/comparar_estudios.py rna
      python src/bicimadness/comparar_estudios.py rna --datos sel60

Salida: results/<algoritmo>/comparacion_estudios.csv
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from bicimadness.comun import comparar as C  # noqa: E402
from bicimadness.comun.datos import RESULTADOS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("algoritmo", help="carpeta de results/ (rna, lstm...)")
    p.add_argument("--datos", help="solo las ejecuciones de este CSV (sel60, top50...)")
    args = p.parse_args()

    por_datos = defaultdict(list)
    for e in C.ejecuciones(args.algoritmo):
        if not args.datos or C.corto(e["datos"]) == args.datos:
            por_datos[C.corto(e["datos"])].append(e)
    if not por_datos:
        raise SystemExit(f"no hay ejecuciones terminadas en results/{args.algoritmo}/")

    filas = []
    for datos, lista in por_datos.items():
        lista.sort(key=lambda e: e["mae"])
        mejor, _ = C.mae_por_estacion_bloque(lista[0]["carpeta"])
        for pos, e in enumerate(lista, 1):
            t, pv, n = C.ttest(C.mae_por_estacion_bloque(e["carpeta"])[0], mejor) if pos > 1 else (float("nan"),) * 2 + (len(mejor),)
            filas.append({
                "datos": datos, "puesto": pos, "ejecucion": e["carpeta"].name, "modo": e["modo"], "id_estacion": e["id_estacion"],
                "variables": e["variables"], "meteo": e["meteo"], "ventana": e["ventana"], "semanas": e["semanas"],
                "parametros": e["parametros"], "mae": e["mae"], "rmse": float(e["rmse"]), "r2": float(e["r2"]),
                "mae_referencia": e["mae_referencia"], "minutos_total": round(e["segundos_total"] / 60, 1),
                "segundos_entreno_por_bloque": e["segundos_entreno_por_bloque"],
                "p_vs_mejor": pv, "equivalente_a_la_mejor": "" if pos == 1 else ("sí" if pv >= 0.05 else "no"),
                "n_pares": n,
            })
        print(f"\n== {args.algoritmo} | {datos}: {len(lista)} ejecuciones (de mejor a peor MAE) ==")
        C.tabla([f for f in filas if f["datos"] == datos],
                ["puesto", "ejecucion", "modo", "id_estacion", "variables", "meteo", "ventana", "semanas", "mae", "rmse", "r2",
                 "minutos_total", "p_vs_mejor", "equivalente_a_la_mejor"])

    salida = RESULTADOS / args.algoritmo / "comparacion_estudios.csv"
    C.escribir(salida, filas)
    print(f"\nGuardado en {salida.relative_to(RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
