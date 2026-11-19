"""MAE, RMSE y R² medios por grupo de estaciones (demanda alta/baja, céntrica/afueras, fácil/difícil).

Junta estaciones_seleccionadas.csv con el error_por_estacion.csv que genera model_per_station.py.

Uso:  python src/bicimadness/model_selected60.py
      python src/bicimadness/analyze_groups.py                            (lee results/selected_60/)
      python src/bicimadness/analyze_groups.py --resultados results/otra_carpeta

Salida: <resultados>/errores_por_grupo.csv, una fila por grupo y configuración (red, referencia...).
La media de salidas por hora del grupo ayuda a leer el MAE: con más demanda, más bicis de error.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[2]  # raíz del repositorio
GRUPOS = ["demanda_alta", "demanda_baja", "centrica", "afueras", "facil", "dificil"]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--resultados", default="results/selected_60",
                      help="carpeta con error_por_estacion.csv (rutas relativas: desde la raíz del repositorio)")
    args.add_argument("--estaciones", default="data/processed/estaciones_seleccionadas.csv", help="estaciones con su grupo")
    args = args.parse_args()
    carpeta = PROYECTO / args.resultados

    with open(PROYECTO / args.estaciones, encoding="utf-8") as f:
        estaciones = {r["id_estacion"]: r for r in csv.DictReader(f)}
    with open(carpeta / "error_por_estacion.csv", encoding="utf-8") as f:
        errores = list(csv.DictReader(f))
    configs = [c[4:] for c in errores[0] if c.startswith("mae_")]  # mae_referencia_24h -> referencia_24h

    por_grupo = defaultdict(list)  # grupo -> filas de error de sus estaciones
    for r in errores:
        if r["id_estacion"] in estaciones:
            por_grupo[estaciones[r["id_estacion"]]["grupo"]].append(r)
    grupos = [g for g in GRUPOS if g in por_grupo] + ["todas"]
    por_grupo["todas"] = [r for g in GRUPOS for r in por_grupo[g]]
    media = lambda vals: sum(vals) / len(vals)

    filas = []
    for g in grupos:
        rs = por_grupo[g]
        demanda = media([float(estaciones[r["id_estacion"]]["media_salidas_hora"]) for r in rs])
        for c in configs:
            filas.append({"grupo": g, "config": c, "n_estaciones": len(rs), "salidas_hora": round(demanda, 3),
                          **{m: round(media([float(r[f"{m}_{c}"]) for r in rs]), 4) for m in ("mae", "rmse", "r2")}})

    salida = carpeta / "errores_por_grupo.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    ancho = max(len(c) for c in configs) + 2
    print(f"{'grupo':14}{'n':>3}  {'salidas/h':>9}  {'config':{ancho}}{'MAE':>7}{'RMSE':>7}{'R2':>7}")
    for i, r in enumerate(filas):
        if i and r["grupo"] != filas[i - 1]["grupo"]:
            print()
        print(f"{r['grupo']:14}{r['n_estaciones']:>3}  {r['salidas_hora']:>9}  {r['config']:{ancho}}"
              f"{r['mae']:>7.3f}{r['rmse']:>7.3f}{r['r2']:>7.3f}")
    falta = set(estaciones) - {r["id_estacion"] for r in errores}
    if falta:
        print(f"\nAviso: {len(falta)} de las {len(estaciones)} estaciones no están en {carpeta.name}/ "
              f"(¿se entrenó con los datos de {args.estaciones}?)")
    print(f"\nGuardado en {salida.relative_to(PROYECTO)}")


if __name__ == "__main__":
    main()
