"""The complete experiment matrix behind the paper.

Every run reported in the paper appears here exactly once, keyed by the
directory name it writes into ``results/``. Nothing else was run; nothing run
is omitted. ``scripts/run_all.py`` executes this manifest, ``scripts/tables.py``
reads the results back.

Each entry is ``(script, [extra CLI args], {extra env})``. The base invocation
is always::

    python src/inference_ev{17,18}.py --config <preset> --seed 1

Two things are not expressible as a preset and so appear as overrides:

* ``MDSAM3_NUM_MASKMEM`` -- SAM3's total memory slots (1 conditioning + N-1
  non-conditioning). The paper's "six / four / two slots" are 7 / 5 / 3.
* ``--prune-rate`` -- the rate-matched controls delete the same *fraction* of
  entries as the reliability rule they are matched against, so each control's
  rate is copied from its own reference run's measured deletion rate. Those
  reference rates are recorded in ``RATE_MATCHED_FROM`` below.
"""

# Deletion rates measured on the reference runs, used to rate-match the
# controls. Source: the "RETENTION: pruned N/M" line each run logs.
RATE_MATCHED_FROM = {
    "EV17_C2": 0.6261,       # 745/1190
    "EV17_C2-SRG": 0.4664,   # 555/1190
    "EV17_C4-SRG": 0.4404,   # 983/2232
}

EV17 = "src/inference_ev17.py"
EV18 = "src/inference_ev18.py"

RUNS = {
    # ---- Table 1: main configurations -------------------------------------
    "EV17_C0":     (EV17, ["--config", "C0"], {}),
    "EV17_C1":     (EV17, ["--config", "C1"], {}),
    "EV17_C2":     (EV17, ["--config", "C2"], {}),
    "EV17_C4":     (EV17, ["--config", "C4"], {}),
    "EV17_C4-GG":  (EV17, ["--config", "C4-GG"], {}),
    "EV17_C4-SRG": (EV17, ["--config", "C4-SRG"], {}),
    "EV17_C2-SRG": (EV17, ["--config", "C2-SRG"], {}),

    # ---- Table 2: rate-matched controls for the fixed threshold -----------
    "EV17_C2-random": (EV17, ["--config", "C2", "--prune-mode", "random",
                              "--prune-rate", "0.6261"], {}),
    "EV17_C2-stride": (EV17, ["--config", "C2", "--prune-mode", "stride",
                              "--prune-rate", "0.6261"], {}),

    # ---- Sec 4.7: rate-matched controls for the self-calibrated threshold --
    "EV17_C2-SRG-random": (EV17, ["--config", "C2-SRG", "--prune-mode", "random",
                                  "--prune-rate", "0.4664"], {}),
    "EV17_C2-SRG-stride": (EV17, ["--config", "C2-SRG", "--prune-mode", "stride",
                                  "--prune-rate", "0.4664"], {}),
    "EV17_C4-SRG-random": (EV17, ["--config", "C4-SRG", "--prune-mode", "random",
                                  "--prune-rate", "0.4404"], {}),
    "EV17_C4-SRG-stride": (EV17, ["--config", "C4-SRG", "--prune-mode", "stride",
                                  "--prune-rate", "0.4404"], {}),

    # ---- Sec 4.6: memory capacity sweep (6 / 4 / 2 non-conditioning slots) -
    "EV17_C0_nm7":     (EV17, ["--config", "C0"],     {"MDSAM3_NUM_MASKMEM": "7"}),
    "EV17_C0_nm5":     (EV17, ["--config", "C0"],     {"MDSAM3_NUM_MASKMEM": "5"}),
    "EV17_C0_nm3":     (EV17, ["--config", "C0"],     {"MDSAM3_NUM_MASKMEM": "3"}),
    "EV17_C2_nm5":     (EV17, ["--config", "C2"],     {"MDSAM3_NUM_MASKMEM": "5"}),
    "EV17_C2_nm3":     (EV17, ["--config", "C2"],     {"MDSAM3_NUM_MASKMEM": "3"}),
    "EV17_C4-SRG_nm5": (EV17, ["--config", "C4-SRG"], {"MDSAM3_NUM_MASKMEM": "5"}),
    "EV17_C4-SRG_nm3": (EV17, ["--config", "C4-SRG"], {"MDSAM3_NUM_MASKMEM": "3"}),
    "EV17_C2-SRG_nm5": (EV17, ["--config", "C2-SRG"], {"MDSAM3_NUM_MASKMEM": "5"}),
    "EV17_C2-SRG_nm3": (EV17, ["--config", "C2-SRG"], {"MDSAM3_NUM_MASKMEM": "3"}),

    # ---- Sec 4.7: fixed-threshold sweep ------------------------------------
    "EV17_C2_tau070": (EV17, ["--config", "C2", "--tau-rel", "0.70"], {}),
    "EV17_C2_tau050": (EV17, ["--config", "C2", "--tau-rel", "0.50"], {}),
    "EV17_C4_tau070": (EV17, ["--config", "C4", "--tau-rel", "0.70"], {}),
    "EV17_C4_tau050": (EV17, ["--config", "C4", "--tau-rel", "0.50"], {}),

    # ---- Sec 4.7: SRG quantile ablation ------------------------------------
    "EV17_C2-SRG_q25": (EV17, ["--config", "C2-SRG", "--srg-quantile", "0.25"], {}),
    "EV17_C2-SRG_q75": (EV17, ["--config", "C2-SRG", "--srg-quantile", "0.75"], {}),

    # ---- Sec 4.5: re-injection repaired so it can actually act -------------
    "EV17_C7-store": (EV17, ["--config", "C7-store"], {}),

    # ---- Table 3: EndoVis 2018 cross-dataset -------------------------------
    # C3/C5/C6/C7 are kept because their agreement with C0/C1/C2/C4 is the
    # evidence for the claim that DFS cannot act (Sec 4.4); they are identities
    # of the executed program, so re-running them is the check.
    "EV18_C0":      (EV18, ["--config", "C0"], {}),
    "EV18_C1":      (EV18, ["--config", "C1"], {}),
    "EV18_C2":      (EV18, ["--config", "C2"], {}),
    "EV18_C3":      (EV18, ["--config", "C3"], {}),
    "EV18_C4":      (EV18, ["--config", "C4"], {}),
    "EV18_C5":      (EV18, ["--config", "C5"], {}),
    "EV18_C6":      (EV18, ["--config", "C6"], {}),
    "EV18_C7":      (EV18, ["--config", "C7"], {}),
    "EV18_C4-GG":   (EV18, ["--config", "C4-GG"], {}),
    "EV18_C4-SRG":  (EV18, ["--config", "C4-SRG"], {}),
    "EV18_C2-SRG":  (EV18, ["--config", "C2-SRG"], {}),
    "EV18_C7-store": (EV18, ["--config", "C7-store"], {}),
}

# Runs whose numbers appear in the paper's tables, for the --paper-only filter.
TABLE_RUNS = [
    "EV17_C0", "EV17_C1", "EV17_C2", "EV17_C4", "EV17_C4-GG",
    "EV17_C4-SRG", "EV17_C2-SRG",
    "EV17_C2-random", "EV17_C2-stride",
    "EV18_C0", "EV18_C1", "EV18_C2", "EV18_C4", "EV18_C4-GG",
    "EV18_C4-SRG", "EV18_C2-SRG",
]
