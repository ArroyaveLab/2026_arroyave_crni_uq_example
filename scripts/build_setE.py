#!/usr/bin/env python3
"""Set E = set D with ORCHID-rule sigmas for the observations not in ORCHID (2026-11-05).

ORCHID rule (orchid-data-next literature/UNCERTAINTY_CALIBRATION.md, version 2026-10-24):
sigma_effective = max(sqrt(max(printed, reading, method)^2 + composition^2 + conversion^2), scatter).
No method is printed in these ESPEI-datasets records (the papers were not read), so the method class is
'unknown': HM_FORM floor 3.5 kJ/mol-atom (calibrated, 290 pairs); ACTIVITY floor 2.9 kJ/mol on RT ln a
(cleaned_below_threshold, 9 pairs). No printed sigma in the files; reading resolution negligible (table
values to 3 digits; figure digitisation << floor); no composition / conversion / scatter terms.
ESPEI weight = ESPEI default sigma (500 J/mol for HM and ACR) / sigma_effective.
Everything else (ZPF, ORCHID records, DFT, hinge cells and settings) is unchanged from set D.
"""
import json
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
D = RUN / "inputs" / "derived" / "setD_twophase_hinge"
E = RUN / "inputs" / "derived" / "setE_orchid_sigmas"
RULE = {"HM_FORM": (3500.0, "HM_FORM / unknown method class floor 3.5 kJ/mol-atom (calibrated, 290 pairs)"),
        "ACR_CR": (2900.0, "ACTIVITY / unknown method class floor 2.9 kJ/mol on RT ln a (cleaned_below_threshold, 9 pairs)")}


def main():
    (E / "datasets").mkdir(parents=True, exist_ok=True)
    (E / "hinge" / "CR-NI").mkdir(parents=True, exist_ok=True)
    shutil.copy2(D / "hinge" / "CR-NI" / "cr-ni-model.json", E / "hinge" / "CR-NI" / "cr-ni-model.json")
    log = {"base": "setD_twophase_hinge (hinge unchanged)", "orchid_rule_sigmas": {}}
    for f in sorted((D / "datasets").glob("*.json")):
        d = json.loads(f.read_text())
        nonorchid = ("pugliese" in f.name) or ("HM_FORM" in f.name)
        if nonorchid:
            sig, basis = RULE[d["output"]]
            d["weight"] = 500.0 / sig
            d["comment"] = d.get("comment", "") + f" | REBUILD set E: ORCHID-rule sigma_effective {sig:.0f} J/mol ({basis}); weight {d['weight']:.6g}."
            log["orchid_rule_sigmas"][f.name] = {"output": d["output"], "sigma_effective_J": sig, "weight": d["weight"],
                                                 "basis": basis, "sigma_printed": None, "method_class": "unknown"}
        (E / "datasets" / f.name).write_text(json.dumps(d, indent=1))
    log["data_flag_not_changed"] = ("pugliese1970activities-3 (1448.15 K): a_Cr = 0.112 at x_Cr = 0.95 between 0.98 and "
                                    "0.998; probable transcription error in ESPEI-datasets; kept")
    (E / "BUILD_LOG.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
