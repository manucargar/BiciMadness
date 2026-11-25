"""Compara algoritmos: el mejor estudio de cada uno (menor MAE) sobre el mismo CSV de datos.

Recorre las carpetas de results/ (rna/, lstm/, gradient_boosting/...), coge de cada una la ejecución con
menor MAE para cada CSV (mismas estaciones y horas) y las compara entre sí: MAE, RMSE, R², tiempo y
T-test pareado de cada par (sobre el MAE de cada estación en cada bloque). Añade la referencia sencilla
"misma hora de ayer" para dar contexto.

Uso:  python src/bicimadness/comparar_algoritmos.py
      python src/bicimadness/comparar_algoritmos.py --datos sel60
      python src/bicimadness/comparar_algoritmos.py --datos sel60 --modo global --id-estacion no
      (solo los modelos globales que no saben qué estación es: RNA, Ridge y LSTM globales)
      python src/bicimadness/comparar_algoritmos.py --datos sel60 --modo global multivariante --id-estacion no
      (RNA y Ridge globales frente al LSTM multivariante)
      python src/bicimadness/comparar_algoritmos.py --datos sel60_filtrada --nombre-salida comparacion_sel60_filtrada

Salidas por defecto: results/comparacion_algoritmos.csv y results/comparacion_algoritmos_ttest.csv
"""
import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from bicimadness.comun import comparar as C  # noqa: E402
from bicimadness.comun.datos import RESULTADOS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", help="solo este CSV (sel60, top50...)")
    p.add_argument("--modo", nargs="+", choices=["global", "estacion", "multivariante"],
                   help="solo las ejecuciones de estos modos (uno o varios)")
    p.add_argument("--id-estacion", choices=["sí", "si", "no"], help="solo las ejecuciones con (sí) o sin (no) id de estación")
    p.add_argument("--meteo", choices=["actual", "anterior"], help="solo las ejecuciones con esta meteorología")
    p.add_argument("--nombre-salida", default="comparacion_algoritmos",
                   help="nombre base de los CSV en results/ (por defecto comparacion_algoritmos)")
    args = p.parse_args()
    if not args.nombre_salida or Path(args.nombre_salida).name != args.nombre_salida:
        p.error("--nombre-salida debe ser un nombre de fichero sin ruta")
    if args.id_estacion == "si":
        args.id_estacion = "sí"

    def vale(e):
        return ((not args.modo or e["modo"] in args.modo) and (not args.id_estacion or e["id_estacion"] == args.id_estacion)
                and (not args.meteo or e.get("meteo") == args.meteo))

    mejores = defaultdict(dict)  # datos -> algoritmo -> ejecución con menor MAE
    for alg in C.algoritmos():
        for e in C.ejecuciones(alg):
            d = C.corto(e["datos"])
            if (not args.datos or d == args.datos) and vale(e) and (alg not in mejores[d] or e["mae"] < mejores[d][alg]["mae"]):
                mejores[d][alg] = e
    if not mejores:
        raise SystemExit("no hay ejecuciones terminadas en results/<algoritmo>/")

    filas, pruebas = [], []
    for datos, por_alg in mejores.items():
        maes = {a: C.mae_por_estacion_bloque(e["carpeta"]) for a, e in por_alg.items()}
        # referencia de ayer: la de la ejecución con más bloques (es la misma en todas, no depende del modelo)
        ref_de = max(por_alg, key=lambda a: len(maes[a][1]))
        candidatos = {**{a: maes[a][0] for a in por_alg}, "referencia_ayer": maes[ref_de][1]}
        actuales = []
        for alg, e in sorted(por_alg.items(), key=lambda x: x[1]["mae"]):
            actuales.append({
                "datos": datos, "algoritmo": alg, "ejecucion": e["carpeta"].name, "modo": e["modo"], "id_estacion": e["id_estacion"],
                "variables": e["variables"], "meteo": e["meteo"], "ventana": e["ventana"], "semanas": e["semanas"],
                "parametros": e["parametros"], "mae": e["mae"], "rmse": float(e["rmse"]), "r2": float(e["r2"]),
                "n_bloques": int(e["n_bloques"]), "minutos_total": round(e["segundos_total"] / 60, 1),
                "segundos_entreno_por_bloque": e["segundos_entreno_por_bloque"],
            })
        r = por_alg[ref_de]
        actuales.append({**{k: "" for k in actuales[0]}, "datos": datos, "algoritmo": "referencia_ayer",
                         "mae": r["mae_referencia"], "rmse": float(r["rmse_referencia"]), "r2": float(r["r2_referencia"]),
                         "n_bloques": int(r["n_bloques"]), "minutos_total": 0.0, "segundos_entreno_por_bloque": 0.0})
        for i, f in enumerate(actuales, 1):
            f["puesto"] = i
        filas += actuales
        for a, b in combinations([f["algoritmo"] for f in actuales], 2):
            t, pv, n = C.ttest(candidatos[a], candidatos[b])
            pruebas.append({"datos": datos, "algoritmo_a": a, "algoritmo_b": b, "t": t, "p": pv,
                            "diferencia_significativa": "sí" if pv < 0.05 else "no", "n_pares": n})

        print(f"\n== {datos}: mejor estudio de cada algoritmo ==")
        C.tabla(actuales, ["puesto", "algoritmo", "ejecucion", "modo", "id_estacion", "variables", "mae", "rmse", "r2",
                           "minutos_total", "segundos_entreno_por_bloque"])
        print("\nT-test pareado sobre el MAE (p < 0.05 -> diferencia significativa):")
        C.tabla([x for x in pruebas if x["datos"] == datos],
                ["algoritmo_a", "algoritmo_b", "t", "p", "diferencia_significativa", "n_pares"])

    salida = RESULTADOS / f"{args.nombre_salida}.csv"
    salida_ttest = RESULTADOS / f"{args.nombre_salida}_ttest.csv"
    C.escribir(salida, filas)
    C.escribir(salida_ttest, pruebas)
    print(f"\nGuardado en {salida} y {salida_ttest}")


if __name__ == "__main__":
    main()
