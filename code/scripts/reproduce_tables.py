#!/usr/bin/env python3
"""Recompute paper summaries from the released event, orbit and flight scores."""
import argparse
from collections import defaultdict
import csv
from itertools import product
import json
import math
from operator import itemgetter
from pathlib import Path
import random
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2] / "results"
METRICS = ("log10_mae", "linear_mae", "linear_rmse")
VALIDATION_MODELS = ("adair", "promptir", "datprl", "logunet_m", "gauss_logunet", "physical_sdc")
MATCHED = ("physical_sdc", "fullrange", "physical_clip", "physical_null", "signed_asinh")
CPR_METRICS = ("average_precision", "roc_auc")
CPRO_METRICS = ("median_abs_dlog10_beta", "log10_mae", "mae_beta")
HSRL_METRICS = ("top_mae_m", "base_mae_m", "thickness_mae_m")
ANALYSIS_SEEDS = {
    "AdaIR": (20260718, 20260710, 20260714),
    "LogUNet": (20260713, 20260714, 20260715),
    "Gaussian-LogUNet": (20260710,),
}


def read_csv(name):
    with (ROOT / "evidence" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_macro(rows, metric):
    expected = {(event, band) for event in {row["event"] for row in rows} for band in ("355", "532")}
    if len(rows) != 8 or {(row["event"], row["wavelength_nm"]) for row in rows} != expected:
        raise ValueError("Each simulation result must contain eight distinct event–wavelength cells.")
    return mean(float(row[metric]) for row in rows)


def simulation_tables(out):
    cells = defaultdict(list)
    for row in read_csv("simulation_event_scores.csv"):
        cells[(row["split"], row["model"])].append(row)
    validation = []
    for model in VALIDATION_MODELS:
        row = {"model": model}
        for band in ("355", "532"):
            for metric in METRICS:
                values = [float(cell[metric]) for cell in cells[("validation", model)]
                          if cell["wavelength_nm"] == band]
                if len(values) != 4:
                    raise ValueError(f"Expected four validation events for {model} at {band} nm.")
                row[f"{metric}_{band}"] = mean(values)
        validation.append(row)
    out["main_validation"] = validation
    comparison = {(row["event"], row["wavelength_nm"], model): float(row["log10_mae"])
                  for model in ("physical_sdc", "fullrange") for row in cells[("validation", model)]}
    reductions = [{"event": event, "wavelength_nm": band, "reduction_percent":
                   100 * (1 - error / comparison[event, band, "fullrange"])}
                  for (event, band, model), error in sorted(comparison.items()) if model == "physical_sdc"]
    out["validation_event_reductions"] = reductions
    for split in ("validation", "test"):
        arms = MATCHED if split == "validation" else tuple(model for s, model in cells if s == split)
        reference = {metric: event_macro(cells[(split, "physical_sdc")], metric) for metric in METRICS}
        comparison = []
        for model in arms:
            totals = {metric: event_macro(cells[(split, model)], metric) for metric in METRICS}
            reduction = {f"sdc_reduction_{metric}_pct": 100 * (1 - reference[metric] / totals[metric])
                         for metric in METRICS}
            comparison.append({"model": model, **totals, **reduction})
        name = "representation_ablation" if split == "validation" else "frozen_heldout"
        out[name] = comparison


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_seed_tables(out, scores):
    """Aggregate the analysis populations explicitly reported in the paper."""
    indexed = defaultdict(list)
    for row in scores:
        indexed[(row["backbone"], row["representation"], int(row["seed"]))].append(row)
    expected = {(backbone, arm, seed) for backbone, seeds in ANALYSIS_SEEDS.items()
                for seed in seeds for arm in (MATCHED if backbone == "AdaIR" else MATCHED[:2])}
    if set(indexed) != expected:
        raise ValueError("Scores must contain exactly the reported run and representation combinations.")
    test_cells = {(event, band) for event in ("E02", "E14", "E26", "E48") for band in ("355", "532")}
    for rows in indexed.values():
        if len(rows) != len(test_cells) or {(row["event"], row["wavelength_nm"]) for row in rows} != test_cells:
            raise ValueError("Each run must contain the eight reserved-test event–wavelength cells.")
    display = []
    intervals = []
    for backbone, seeds in ANALYSIS_SEEDS.items():
        arms = [arm for arm in MATCHED if all((backbone, arm, seed) in indexed for seed in seeds)]
        for metric in METRICS:
            summaries = {arm: [event_macro(indexed[(backbone, arm, seed)], metric) for seed in seeds]
                         for arm in arms}
            sdc = mean(summaries["physical_sdc"])
            for arm, values in summaries.items():
                display.append({"backbone": backbone, "metric": metric, "representation": arm,
                                "seeds": ",".join(map(str, seeds)), "mean": mean(values),
                                "sample_sd": stdev(values) if len(values) > 1 else "",
                                "sdc_reduction_pct": 100 * (1 - sdc / mean(values))})
            rows = defaultdict(dict)
            for seed in seeds:
                for arm in ("physical_sdc", "fullrange"):
                    for row in indexed[(backbone, arm, seed)]:
                        rows[(seed, row["event"], row["wavelength_nm"])][arm] = float(row[metric])
            differences = defaultdict(list)
            for (_, event, _), pair in rows.items():
                differences[event].append(pair["physical_sdc"] - pair["fullrange"])
            event_diffs = [mean(differences[event]) for event in sorted(differences)]
            if len(event_diffs) != 4:
                raise ValueError("Paired statistics require four events.")
            draws = [mean(draw) for draw in product(event_diffs, repeat=len(event_diffs))]
            intervals.append({"backbone": backbone, "metric": metric, "mean_sdc_minus_rawlog": mean(event_diffs),
                              "descriptive_95_lo": percentile(draws, .025), "descriptive_95_hi": percentile(draws, .975)})
    out["paired_seed_display"] = display
    out["paired_seed_intervals"] = intervals


def score_groups(rows, group_key, unit_key, expected_units):
    groups = defaultdict(dict)
    for row in rows:
        group, unit = row[group_key], row[unit_key]
        if unit in groups[group]:
            raise ValueError(f"Duplicate scoring unit {unit} for {group}")
        groups[group][unit] = row
    reference = set(next(iter(groups.values())))
    if len(reference) != expected_units or any(set(group) != reference for group in groups.values()):
        raise ValueError(f"Expected the same {expected_units} scoring units for every method")
    return groups


def original_observations(out, reference):
    radar = score_groups(read_csv("original_cpr_orbit_scores.csv"), "model", "unit_id", 6)
    optical = score_groups(read_csv("original_cpro_unit_scores.csv"), "model", "unit_id", 29)
    table = []
    for source in reference["original_observation_macros"]:
        model = source["model"]
        scores = {metric: mean(float(row[metric]) for row in groups[model].values())
                  for groups, metrics in ((radar, CPR_METRICS), (optical, CPRO_METRICS)) for metric in metrics}
        for metric, value in scores.items():
            if not math.isclose(value, float(source[metric]), rel_tol=1e-12):
                raise ValueError(f"Original observation archive differs for {model} {metric}")
        table.append({"model": model, **scores})
    out["original_observation_macros"] = table

    units = sorted(radar["physical_sdc"])
    intervals = []
    for metric in CPR_METRICS:
        delta = [float(radar["physical_sdc"][unit][metric]) - float(radar["gauss_logunet"][unit][metric])
                 for unit in units]
        distribution = [mean(draw) for draw in product(delta, repeat=len(delta))]
        intervals.append({"metric": metric, "sdc_minus_gaussian_unet": mean(delta),
                          "lower_95": percentile(distribution, .025), "upper_95": percentile(distribution, .975)})
    out["original_cpr_sdc_vs_gaussian_intervals"] = intervals


def cpro_intervals(out, original_macros):
    groups = score_groups(read_csv("cpro_paired_orbit_differences.csv"), "arm", "orbit_id", 1270)
    reference_orbits = sorted(groups["fullrange"], key=int)
    metrics = ("log10_mae", "mae_beta")
    contrasts = [(arm, metric) for arm in MATCHED[1:] for metric in metrics]
    columns = [[float(groups[arm][orbit][f"{metric}_control_minus_sdc"]) for orbit in reference_orbits]
               for arm, metric in contrasts]
    distributions = [[] for _ in contrasts]
    rng = random.Random(20260929)
    count = len(reference_orbits)
    population = range(count)
    for _ in range(10000):
        draw = itemgetter(*rng.choices(population, k=count))
        for column, distribution in zip(columns, distributions):
            distribution.append(sum(draw(column)) / count)
    records = []
    for (arm, metric), values, distribution in zip(contrasts, columns, distributions):
        observed = mean(values)
        control = original_macros[("CPro_all1270", arm, metric)]
        sdc = original_macros[("CPro_all1270", "physical_sdc", metric)]
        if not math.isclose(control - sdc, observed, rel_tol=1e-12):
            raise ValueError(f"CPro paired differences disagree with archived macros for {arm} {metric}")
        records.append({"arm": arm, "metric": metric, "orbits": count,
                        "mean_control_minus_sdc": observed, "sdc_reduction_percent": 100 * observed / control,
                        "lower_95": percentile(distribution, .025), "upper_95": percentile(distribution, .975)})
    out["cpro_paired_orbit_intervals"] = records


def expanded_observations(out, reference):
    archived = reference["expanded_observation_macros"]
    original_macros = {(row["cohort"], row["arm"], row["metric"]): float(row["mean"]) for row in archived}
    radar = score_groups(read_csv("cpr_orbit_scores.csv"), "arm", "orbit_id", 1298)
    flights = read_csv("hsrl_flight_scores.csv")
    hsrl = score_groups([row for row in flights if row["arm"] in MATCHED], "arm", "flight_id", 12)
    recomputed = {(cohort, arm, metric): mean(float(row[metric]) for row in groups[arm].values())
                  for cohort, groups, metrics in (("CPR_all1298", radar, CPR_METRICS), ("HSRL_all12", hsrl, HSRL_METRICS))
                  for arm in MATCHED for metric in metrics}
    for key, value in recomputed.items():
        if not math.isclose(value, original_macros[key], rel_tol=1e-12):
            raise ValueError(f"Expanded observation archive differs for {key}")
    out["expanded_observation_macros"] = [
        {**row, "mean": recomputed.get((row["cohort"], row["arm"], row["metric"]), row["mean"])}
        for row in archived]
    cpro_intervals(out, original_macros)

    original = score_groups([row for row in flights if row["cohort"] == "original5"], "arm", "flight_id", 5)
    out["original_hsrl_flight_macros"] = [
        {"arm": arm, **{metric: mean(float(row[metric]) for row in units.values())
                        for metric in ("coverage", *HSRL_METRICS)}} for arm, units in original.items()]
    shared = read_csv("hsrl_fernald_shared_flight_differences.csv")
    shared_groups = score_groups([row for row in shared if row["fernald"] == "fernald50"], "method", "flight_id", 5)
    intervals = []
    for arm, units in shared_groups.items():
        for layer in ("top", "base", "thickness"):
            differences = [float(row[f"{layer}_difference_fernald_minus_method_m"]) for row in units.values()]
            distribution = [mean(draw) for draw in product(differences, repeat=len(differences))]
            intervals.append({"arm": arm, "boundary_measure": layer, "mean_fernald50_minus_method_m": mean(differences),
                              "lower_95_m": percentile(distribution, .025), "upper_95_m": percentile(distribution, .975)})
    out["original_hsrl_fernald50_shared_intervals"] = intervals


def verify_and_merge(reference, recomputed):
    """Check all recomputed values before writing the result tables."""
    for name, rows in recomputed.items():
        expected = reference[name]
        if len(rows) != len(expected):
            raise ValueError(f"Row count differs for {name}.")
        for row, archived in zip(rows, expected):
            if set(row) != set(archived):
                raise ValueError(f"Fields differ for {name}.")
            for key, value in row.items():
                matches = (math.isclose(value, float(archived[key]), rel_tol=1e-12, abs_tol=1e-15)
                           if isinstance(value, (int, float)) else value == archived[key])
                if not matches:
                    raise ValueError(f"Recomputed {name}.{key} differs from the released value: {value}.")
    return {name: [{key: str(value) for key, value in row.items()} for row in recomputed[name]]
            if name in recomputed else archive for name, archive in reference.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new.")
    summaries = {name: json.loads((ROOT / f"{name}.json").read_text())
                 for name in ("simulation", "observations", "robustness", "training_and_cost")}
    simulation, observations = {}, {}
    simulation_tables(simulation)
    paired_seed_tables(simulation, summaries["simulation"]["reported_seed_event_scores"])
    original_observations(observations, summaries["observations"])
    expanded_observations(observations, summaries["observations"])
    summaries["simulation"] = verify_and_merge(summaries["simulation"], simulation)
    summaries["observations"] = verify_and_merge(summaries["observations"], observations)
    args.output.mkdir(parents=True)
    for name, data in summaries.items():
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed and verified {len(simulation) + len(observations)} tables and interval series.")
    print("Retained archived inputs and aggregates where lower-level scores are unavailable.")


if __name__ == "__main__":
    main()
