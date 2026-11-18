"""Unit and script-chain tests of fixed-row requantification.

bin/lib_generator.py -> (FeatureFinderMetaboIdent, emulated by hand-built candidates)
-> bin/requantification_qc.py -> bin/gapfill_consensus.py

Every expectation is a literal derived from the hand-built fixture below, never a
value recomputed by the code under test. The scripts are loaded from $REQUANT_BIN
(default: <repo>/bin), so the mutation harness can point the suite at a mutated copy.

Fixture: three samples with dotted names (map indexes 0, 1, 2) and nine consensus rows.
s.A has a handle in every row (no targets); s.B and s.C miss several rows, and each of
their candidates exercises one QC rule (see PLAN). s.D is a sample without any feature
(FeatureFinderMetabo found nothing); it is not in SAMPLES and is added only by the
empty-sample tests (EMPTY_PLAN).
"""

import contextlib
import csv
import difflib
import hashlib
import importlib.util
import io
import itertools
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyopenms as po

BIN = Path(os.environ.get("REQUANT_BIN", Path(__file__).resolve().parents[1] / "bin"))
PROTON = 1.007276466621
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def load(name):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = load("lib_generator")
qc = load("requantification_qc")
gf = load("gapfill_consensus")

SAMPLES = ["s.A", "s.B", "s.C"]
REASONS = [
    "malformed_qc_metadata",
    "existing_signal_overlap",
    "no_peak_at_apex",
    "apex_outside_target_window",
    "model_status_not_valid",
    "low_apex_signal_fraction",
    "shared_candidate_signal",
    "duplicate_feature_identity",
]


# --------------------------------------------------------------------------- fixture
def feature(uid, mz, rt, intensity, traces=None, charge=1, fwhm=6.0):
    f = po.Feature()
    f.setUniqueId(uid)
    f.setMZ(mz)
    f.setRT(rt)
    f.setIntensity(intensity)
    f.setCharge(charge)
    f.setWidth(fwhm)
    f.setMetaValue("masstrace_centroid_mz", list(traces or [mz, mz + 1.0033548378]))
    return f


# original features: uid -> (mz, rt, intensity[, mass traces])
ORIGINALS = {
    "s.A": [
        feature(1001, 100.0, 50.0, 11.0),
        feature(1002, 200.0, 100.0, 12.0),
        feature(1003, 300.0, 150.0, 13.0),
        feature(1004, 200.0005, 160.0, 14.0),
        feature(1005, 400.0, 200.0, 15.0),
        feature(1006, 500.0, 250.0, 16.0),
        feature(1007, 600.0, 280.0, 17.0),
        feature(1008, 598.9966, 281.0, 18.0),
        feature(1009, 500.001, 252.0, 19.0),
    ],
    "s.B": [feature(2001, 100.0001, 50.5, 21.0), feature(2002, 200.0001, 100.5, 22.0)],
    "s.C": [
        feature(3001, 100.0, 51.0, 31.0),
        feature(3004, 200.0005, 160.5, 34.0),
        # monoisotopic m/z 598.9966; its second mass trace sits at 600.0000 (row 107)
        feature(3008, 598.9966, 281.0, 38.0, traces=[598.9966, 600.0]),
    ],
    # no feature at all: its consensus column is empty and every row is a target
    "s.D": [],
}
# consensus uid, m/z, RT, members (sample, original uid)
ROWS = [
    (101, 100.0, 50.5, [("s.A", 1001), ("s.B", 2001), ("s.C", 3001)]),
    (102, 200.0, 100.25, [("s.A", 1002), ("s.B", 2002)]),
    (103, 300.0, 150.0, [("s.A", 1003)]),
    (104, 200.0005, 160.25, [("s.A", 1004), ("s.C", 3004)]),
    (105, 400.0, 200.0, [("s.A", 1005)]),
    (106, 500.0, 250.0, [("s.A", 1006)]),
    (107, 600.0, 280.0, [("s.A", 1007)]),
    (108, 598.9966, 281.0, [("s.A", 1008), ("s.C", 3008)]),
    (109, 500.001, 252.0, [("s.A", 1009)]),
]
# FFMI candidates to emulate: (sample, consensus uid) -> candidate spec
PLAN = {
    ("s.C", 102): {"apex": 100.4, "left": 95.0, "right": 106.0, "area": 777.0},
    ("s.C", 103): {
        "apex": 150.1,
        "left": 145.0,
        "right": 156.0,
        "area": 903.0,
        "masserror": [-1.0],
    },
    ("s.C", 106): {
        "apex": 250.1,
        "left": 245.0,
        "right": 256.0,
        "area": 0.0,
        "status": "3 (left side out of bounds)",
    },
    ("s.C", 107): {"apex": 280.2, "left": 275.0, "right": 286.0, "area": 907.0},
    ("s.B", 103): {
        "apex": 150.2,
        "left": 145.0,
        "right": 156.0,
        "area": 803.0,
        "hull": "spike",
    },
    ("s.B", 104): {"apex": 160.2, "left": 95.0, "right": 170.0, "area": 804.0},
    ("s.B", 105): {
        "apex": 290.0,
        "left": 284.0,
        "right": 296.0,
        "area": 0.0,
        "status": "4 (right side out of bounds)",
    },
    ("s.B", 106): {"apex": 250.0, "left": 244.0, "right": 256.0, "area": 806.0},
    ("s.B", 107): {
        "apex": 280.0,
        "left": 275.0,
        "right": 286.0,
        "area": 807.0,
        "hull": "none",
    },
    ("s.B", 109): {"apex": 251.0, "left": 245.0, "right": 257.0, "area": 809.0},
}
# (sample, consensus uid) -> (status, rejection_reason, failed_checks)
EXPECTED_STATUS = {
    ("s.B", "103"): (
        "rejected",
        "low_apex_signal_fraction",
        "low_apex_signal_fraction",
    ),
    ("s.B", "104"): ("rejected", "existing_signal_overlap", "existing_signal_overlap"),
    ("s.B", "105"): (
        "rejected",
        "apex_outside_target_window",
        "apex_outside_target_window;model_status_not_valid",
    ),
    ("s.B", "106"): ("rejected", "shared_candidate_signal", "shared_candidate_signal"),
    ("s.B", "107"): ("rejected", "malformed_qc_metadata", "malformed_qc_metadata"),
    ("s.B", "108"): ("no_candidate", "no_candidate", "no_candidate"),
    ("s.B", "109"): ("rejected", "shared_candidate_signal", "shared_candidate_signal"),
    ("s.C", "102"): ("accepted", "", ""),
    ("s.C", "103"): ("rejected", "no_peak_at_apex", "no_peak_at_apex"),
    ("s.C", "105"): ("no_candidate", "no_candidate", "no_candidate"),
    ("s.C", "106"): ("rejected", "model_status_not_valid", "model_status_not_valid"),
    ("s.C", "107"): ("rejected", "existing_signal_overlap", "existing_signal_overlap"),
    ("s.C", "109"): ("no_candidate", "no_candidate", "no_candidate"),
}
EXPECTED_MASK = [
    ["0", "101", "observed", "observed", "observed"],
    ["1", "102", "observed", "observed", "requantified"],
    ["2", "103", "observed", "missing_failed_qc", "missing_failed_qc"],
    ["3", "104", "observed", "missing_existing_signal_overlap", "observed"],
    ["4", "105", "observed", "missing_failed_qc", "missing_no_candidate"],
    ["5", "106", "observed", "missing_failed_qc", "missing_failed_qc"],
    ["6", "107", "observed", "missing_failed_qc", "missing_existing_signal_overlap"],
    ["7", "108", "observed", "missing_no_candidate", "observed"],
    ["8", "109", "observed", "missing_failed_qc", "missing_no_candidate"],
]
SUMMARY_HEAD = (
    [
        "sample",
        "consensus_rows",
        "observed",
        "targets_requested",
        "candidates_recovered",
        "accepted",
        "rejected",
        "no_candidate",
    ]
    + [f"rejected_{reason}" for reason in REASONS]
    + ["missing_after"]
)
EXPECTED_SUMMARY = [
    [
        "s.A",
        "9",
        "9",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ],
    [
        "s.B",
        "9",
        "2",
        "7",
        "6",
        "0",
        "6",
        "1",
        "1",
        "1",
        "0",
        "1",
        "0",
        "1",
        "2",
        "0",
        "7",
    ],
    [
        "s.C",
        "9",
        "3",
        "6",
        "4",
        "1",
        "3",
        "2",
        "0",
        "1",
        "1",
        "0",
        "1",
        "0",
        "0",
        "0",
        "5",
    ],
    [
        "all",
        "9",
        "14",
        "13",
        "10",
        "1",
        "9",
        "3",
        "1",
        "2",
        "1",
        "1",
        "1",
        "1",
        "2",
        "0",
        "12",
    ],
]
FILL_UID_S_C_102 = 5813629404252596011  # blake2b-64 of "metaboigniter-requant|s.C|102"

# The sample without features (s.D, last column, map index 3). Its library holds every
# consensus row; four of them get a candidate.
EMPTY = "s.D"
WITH_EMPTY = SAMPLES + [EMPTY]
EMPTY_PLAN = {
    (EMPTY, 101): {"apex": 50.6, "left": 45.0, "right": 56.0, "area": 501.0},
    (EMPTY, 104): {"apex": 160.3, "left": 155.0, "right": 166.0, "area": 504.0},
    (EMPTY, 106): {
        "apex": 250.1,
        "left": 245.0,
        "right": 256.0,
        "area": 0.0,
        "status": "3 (left side out of bounds)",
    },
    (EMPTY, 107): {
        "apex": 280.2,
        "left": 275.0,
        "right": 286.0,
        "area": 507.0,
        "masserror": [-1.0],
    },
}
EXPECTED_EMPTY_STATUS = {
    (EMPTY, "101"): ("accepted", "", ""),
    (EMPTY, "102"): ("no_candidate", "no_candidate", "no_candidate"),
    (EMPTY, "103"): ("no_candidate", "no_candidate", "no_candidate"),
    (EMPTY, "104"): ("accepted", "", ""),
    (EMPTY, "105"): ("no_candidate", "no_candidate", "no_candidate"),
    (EMPTY, "106"): ("rejected", "model_status_not_valid", "model_status_not_valid"),
    (EMPTY, "107"): ("rejected", "no_peak_at_apex", "no_peak_at_apex"),
    (EMPTY, "108"): ("no_candidate", "no_candidate", "no_candidate"),
    (EMPTY, "109"): ("no_candidate", "no_candidate", "no_candidate"),
}
EXPECTED_EMPTY_MASK_COLUMN = [
    "requantified",
    "missing_no_candidate",
    "missing_no_candidate",
    "requantified",
    "missing_no_candidate",
    "missing_failed_qc",
    "missing_failed_qc",
    "missing_no_candidate",
    "missing_no_candidate",
]
# s.D and the "all" row with s.D; the rows of s.A, s.B and s.C are EXPECTED_SUMMARY[:3]
EXPECTED_EMPTY_SUMMARY = [
    ["s.D", "9", "0", "9", "4", "2", "2", "5"]
    + ["0", "0", "1", "0", "1", "0", "0", "0"]
    + ["7"],
    ["all", "9", "14", "22", "14", "3", "11", "8"]
    + ["1", "2", "2", "1", "2", "1", "2", "0"]
    + ["19"],
]
FILL_UID_S_D_101 = 9342968162976763775  # blake2b-64 of "metaboigniter-requant|s.D|101"
FILL_UID_S_D_104 = 2594190472446476450  # blake2b-64 of "metaboigniter-requant|s.D|104"


def consensus(header_order=SAMPLES, filenames=None):
    """The initial consensus; `filenames` overrides column header file names by sample."""
    by_uid = {s: {f.getUniqueId(): f for f in ORIGINALS[s]} for s in SAMPLES}
    cmap = po.ConsensusMap()
    cmap.setUniqueId(7)
    headers = {}
    for map_index, sample in enumerate(header_order):
        header = po.ColumnHeader()
        header.filename = (filenames or {}).get(sample, f"{sample}.mzML")
        header.size = len(ORIGINALS.get(sample, []))
        header.unique_id = 70 + map_index
        headers[map_index] = header
    cmap.setColumnHeaders(headers)
    for uid, mz, rt, members in ROWS:
        cf = po.ConsensusFeature()
        cf.setUniqueId(uid)
        cf.setMZ(mz)
        cf.setRT(rt)
        cf.setCharge(1)
        cf.setIntensity(float(uid))
        cf.setQuality(0.5)
        for sample, fuid in members:
            cf.insert(header_order.index(sample), by_uid[sample][fuid])
        cmap.push_back(cf)
    return cmap


def candidate(target, spec, uid):
    """An FFMI-like candidate feature for a targets.tsv row."""
    mz = float(target["target_mz"])
    apex, left, right = spec["apex"], spec["left"], spec["right"]
    status = spec.get("status", "0 (valid)")
    area = spec["area"]
    f = po.Feature()
    f.setUniqueId(uid)
    f.setMZ(mz)
    f.setRT(apex)
    f.setIntensity(spec.get("intensity", area))
    f.setCharge(abs(int(target["charge"])))
    f.setMetaValue("label", target["target_id"])
    f.setMetaValue("PrecursorMZ", mz)
    f.setMetaValue("model_status", status)
    f.setMetaValue("model_area", area)
    f.setMetaValue("raw_intensity", area - 77.0)
    f.setMetaValue("leftWidth", left)
    f.setMetaValue("rightWidth", right)
    f.setMetaValue("num_of_masstraces", 1)
    f.setMetaValue("masserror_ppm", list(spec.get("masserror", [0.5])))
    sub = po.Feature()
    sub.setUniqueId(uid * 10 + 1)
    sub.setMZ(mz)
    sub.setRT(apex)
    sub.setMetaValue("peak_apex_position", apex)
    sub.setMetaValue("peak_apex_int", 9.0)
    rts = np.arange(math.ceil(left), math.floor(right) + 1, 1.0)
    if spec.get("hull") == "spike":  # signal only in the scan nearest the apex
        intensities = np.where(np.abs(rts - round(apex)) < 0.5, 5.0, 0.0)
    else:
        intensities = 9.0 * np.exp(-((rts - apex) ** 2) / 18.0) + 0.5
    if spec.get("hull") != "none":
        hull = po.ConvexHull2D()
        hull.setHullPoints(np.column_stack([rts, intensities]))
        sub.setConvexHulls([hull])
    f.setSubordinates([sub])
    return f


def store_features(path, features, sample, uid=11):
    fmap = po.FeatureMap()
    fmap.setUniqueId(uid)
    # as FeatureFinderMetabo writes it: the staged mzML name, which the linker copies
    # into the column header
    fmap.setPrimaryMSRunPath([f"{sample}.mzML".encode()])
    for f in features:
        fmap.push_back(f)
    po.FeatureXMLFile().store(str(path), fmap)


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def read_dicts(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def snapshot(cmap):
    """Everything fixed-row insertion must preserve, as plain values."""
    rows = []
    for cf in cmap:
        handles = sorted(
            (
                h.getMapIndex(),
                h.getUniqueId(),
                h.getMZ(),
                h.getRT(),
                h.getIntensity(),
                h.getCharge(),
            )
            for h in cf.getFeatureList()
        )
        rows.append(
            (
                cf.getUniqueId(),
                cf.getMZ(),
                cf.getRT(),
                cf.getCharge(),
                cf.getIntensity(),
                cf.getQuality(),
                handles,
            )
        )
    headers = cmap.getColumnHeaders()
    return rows, sorted((k, str(headers[k].filename), headers[k].size) for k in headers)


def run(script, *args):
    return subprocess.run(
        [sys.executable, "-B", str(BIN / script), *[str(a) for a in args]],
        check=True,
        env=ENV,
        capture_output=True,
        text=True,
    )


def run_chain(
    tmp,
    plan=PLAN,
    header_order=SAMPLES,
    sample_order=SAMPLES,
    reverse=False,
    polarity="positive",
):
    """lib_generator -> emulated FFMI -> requantification_qc -> gapfill_consensus."""
    tmp = Path(tmp)
    po.ConsensusXMLFile().store(
        str(tmp / "initial.consensusXML"), consensus(header_order)
    )
    tables = []
    for sample in sample_order:
        # the linked map: its unique id is the one of its column header
        store_features(
            tmp / f"{sample}.featureXML",
            ORIGINALS[sample],
            sample,
            uid=70 + header_order.index(sample),
        )
        run(
            "lib_generator.py",
            "--consensus", tmp / "initial.consensusXML",
            "--features", tmp / f"{sample}.featureXML",
            "--sample", sample,
            "--polarity", polarity,
            "--rt-margin", "30",
            "--library", tmp / f"{sample}.library.tsv",
            "--targets", tmp / f"{sample}.targets.tsv",
            "--originals", tmp / f"{sample}.originals.tsv",
            "--sample-map", tmp / f"{sample}.sample_map.tsv",
        )  # fmt: skip
        targets = read_dicts(tmp / f"{sample}.targets.tsv")
        qc_args = [
            "--sample", sample,
            "--targets", tmp / f"{sample}.targets.tsv",
            "--originals", tmp / f"{sample}.originals.tsv",
            "--mz-window", "10",
            "--min-apex-signal-fraction", "0.3",
            "--output-candidates", tmp / f"{sample}.candidates.tsv",
            "--output-fills", tmp / f"{sample}.fills.tsv",
        ]  # fmt: skip
        if targets:
            cands = [
                candidate(
                    t,
                    plan[(sample, int(t["consensus_uid"]))],
                    9000 + int(t["consensus_uid"]),
                )
                for t in targets
                if (sample, int(t["consensus_uid"])) in plan
            ]
            store_features(
                tmp / f"{sample}.ffmi.featureXML",
                cands[::-1] if reverse else cands,
                sample,
            )
            qc_args += ["--candidates", tmp / f"{sample}.ffmi.featureXML"]
        run("requantification_qc.py", *qc_args)
        tables += [
            "--sample-tables", sample,
            tmp / f"{sample}.sample_map.tsv",
            tmp / f"{sample}.candidates.tsv",
            tmp / f"{sample}.fills.tsv",
        ]  # fmt: skip
    run(
        "gapfill_consensus.py",
        "--consensus", tmp / "initial.consensusXML",
        *tables,
        "--output-consensus", tmp / "gapfilled.consensusXML",
        "--mask", tmp / "mask.tsv",
        "--provenance", tmp / "provenance.tsv",
        "--summary", tmp / "summary.tsv",
    )  # fmt: skip
    out = po.ConsensusMap()
    po.ConsensusXMLFile().load(str(tmp / "gapfilled.consensusXML"), out)
    return out


def targets_of(sample, header_order=SAMPLES, polarity="positive"):
    return lib.build_targets(
        consensus(header_order), header_order.index(sample), sample, 30.0, polarity
    )


def as_text_rows(targets):
    """Targets as read back from targets.tsv (all strings)."""
    return [
        {k: (repr(v) if isinstance(v, float) else str(v)) for k, v in t.items()}
        for t in targets
    ]


def index_of(sample):
    return qc.build_overlap_index(
        as_text_rows(lib.original_rows(fmap(ORIGINALS[sample])))
    )


def fmap(features, run_paths=None):
    m = po.FeatureMap()
    if run_paths is not None:
        m.setPrimaryMSRunPath([p.encode() for p in run_paths])
    for f in features:
        m.push_back(f)
    return m


def qc_rows(sample, plan=PLAN, extra=None, min_fraction=0.3):
    targets = as_text_rows(targets_of(sample))
    by_id = {t["target_id"]: t for t in targets}
    cands = [
        candidate(
            t, plan[(sample, int(t["consensus_uid"]))], 9000 + int(t["consensus_uid"])
        )
        for t in targets
        if (sample, int(t["consensus_uid"])) in plan
    ] + list(extra or [])
    originals = as_text_rows(lib.original_rows(fmap(ORIGINALS[sample])))
    rows, fills = qc.qc_sample(
        sample, targets, by_id, originals, fmap(cands), 10.0, min_fraction
    )
    return {r["consensus_uid"]: r for r in rows}, fills


def one_status(sample, uid, spec, **kwargs):
    rows, _ = qc_rows(sample, plan={(sample, uid): spec}, **kwargs)
    row = rows[str(uid)]
    return row["status"], row["rejection_reason"], row["failed_checks"]


VALID_C102 = PLAN[("s.C", 102)]


# --------------------------------------------------------------------------- lib_generator
class LibGeneratorTest(unittest.TestCase):
    def test_map_index_from_uid_membership_for_every_sample(self):
        cmap = consensus()
        for map_index, sample in enumerate(SAMPLES):
            self.assertEqual(
                lib.resolve_map_index(cmap, fmap(ORIGINALS[sample]), sample), map_index
            )
        # header order does not matter: membership decides
        cmap = consensus(["s.C", "s.A", "s.B"])
        self.assertEqual(lib.resolve_map_index(cmap, fmap(ORIGINALS["s.B"]), "s.B"), 2)

    def test_map_index_failures(self):
        cmap = consensus()
        # a map without features takes the empty-map branch (EmptySampleTest): it resolves
        # only to an empty column named by its unique id and MS run path
        self.assertEqual(
            lib.resolve_map_index(consensus(WITH_EMPTY), empty_map(), EMPTY), 3
        )
        for name, fm in {
            "empty map naming a column with features": empty_map(["s.B.mzML"], 71),
            "empty map without unique id or run path": fmap([]),
            "empty map naming no column": empty_map(),
        }.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                lib.resolve_map_index(cmap, fm, "s.B")
        cases = {
            "duplicate uid in featureXML": ORIGINALS["s.B"]
            + [feature(2001, 1.0, 1.0, 1.0)],
            "uid not in consensus": ORIGINALS["s.B"] + [feature(2999, 1.0, 1.0, 1.0)],
            "features of two maps": ORIGINALS["s.B"] + [ORIGINALS["s.C"][0]],
            "foreign handle at the index": ORIGINALS["s.B"][:1],
        }
        for name, features in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                lib.resolve_map_index(cmap, fmap(features), "s.B")

    def test_zero_targets_for_a_complete_sample(self):
        self.assertEqual(targets_of("s.A"), [])
        self.assertEqual(lib.library_rows([], "positive"), [])

    def test_targets_of_c_exact(self):
        got = [
            (
                t["target_id"],
                t["map_index"],
                t["consensus_index"],
                t["consensus_uid"],
                t["n_observed"],
                t["charge"],
            )
            for t in targets_of("s.C")
        ]
        self.assertEqual(
            got,
            [
                ("requant_2_102", 2, 1, 102, 2, 1),
                ("requant_2_103", 2, 2, 103, 1, 1),
                ("requant_2_105", 2, 4, 105, 1, 1),
                ("requant_2_106", 2, 5, 106, 1, 1),
                ("requant_2_107", 2, 6, 107, 1, 1),
                ("requant_2_109", 2, 8, 109, 1, 1),
            ],
        )

    def test_target_window_centred_on_handle_span(self):
        t = {x["consensus_uid"]: x for x in targets_of("s.C")}[102]
        # handles at 100.0 and 100.5; consensus RT 100.25; margin 30
        self.assertEqual(
            (
                t["handle_rt_min"],
                t["handle_rt_max"],
                t["target_rt"],
                t["consensus_rt"],
                t["window_min"],
                t["window_max"],
            ),
            (100.0, 100.5, 100.25, 100.25, 70.0, 130.5),
        )
        self.assertEqual(t["target_mz"], 200.0)

    def test_window_centre_is_min_max_midpoint_not_consensus_rt(self):
        cmap = consensus()
        rows = list(cmap)
        rows[1].setRT(100.4)  # consensus RT off-centre
        moved = po.ConsensusMap(cmap)
        moved.clear(False)
        for cf in rows:
            moved.push_back(cf)
        t = {
            x["consensus_uid"]: x
            for x in lib.build_targets(moved, 2, "s.C", 30.0, "positive")
        }[102]
        self.assertEqual((t["target_rt"], t["consensus_rt"]), (100.25, 100.4))

    def test_rt_margin_must_be_positive(self):
        for margin in (0.0, -1.0, float("nan")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                lib.build_targets(consensus(), 2, "s.C", margin, "positive")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            lib.parse_args(
                ["--consensus", "c", "--features", "f", "--sample", "s", "--polarity", "positive",
                 "--rt-margin", "0", "--library", "l", "--targets", "t", "--originals", "o",
                 "--sample-map", "m"]
            )  # fmt: skip

    def test_library_rows_positive_negative_and_charge_two(self):
        base = {
            "target_id": "requant_2_102",
            "target_mz": 200.0,
            "target_rt": 100.25,
            "window_min": 70.0,
            "window_max": 130.5,
        }
        cases = [
            ("positive", 1, "+1", 200.0 - PROTON),
            ("positive", 2, "+2", 400.0 - 2 * PROTON),
            ("negative", -1, "-1", 200.0 + PROTON),
            ("negative", -2, "-2", 400.0 + 2 * PROTON),
        ]
        for polarity, charge, want_charge, want_mass in cases:
            with self.subTest(polarity=polarity, charge=charge):
                row = lib.library_rows([dict(base, charge=charge)], polarity)[0]
                self.assertEqual(row["CompoundName"], "requant_2_102")
                self.assertEqual(row["SumFormula"], " ")
                self.assertEqual(row["Charge"], want_charge)
                self.assertAlmostEqual(float(row["Mass"]), want_mass, places=9)
                self.assertEqual(float(row["RetentionTime"]), 100.25)
                self.assertEqual(float(row["RetentionTimeRange"]), 60.5)
        self.assertAlmostEqual(
            lib.neutral_mass(200.0, 2, -1), 400.0 + 2 * PROTON, places=9
        )

    def test_negative_mode_targets_have_negative_charge(self):
        self.assertEqual(
            {t["charge"] for t in targets_of("s.C", polarity="negative")}, {-1}
        )

    def test_library_rejects_bad_charge_and_range(self):
        base = {
            "target_id": "t",
            "target_mz": 200.0,
            "target_rt": 100.0,
            "window_min": 70.0,
            "window_max": 130.0,
        }
        bad = [
            ("positive", dict(base, charge=0)),
            ("positive", dict(base, charge=-1)),
            ("negative", dict(base, charge=1)),
            ("positive", dict(base, charge=1, window_max=70.0)),
            ("positive", dict(base, charge=1, target_mz=float("inf"))),
        ]
        for polarity, target in bad:
            with self.subTest(target=target), self.assertRaises(ValueError):
                lib.library_rows([target], polarity)

    def test_original_rows_carry_every_mass_trace(self):
        rows = {r["uid"]: r for r in lib.original_rows(fmap(ORIGINALS["s.C"]))}
        self.assertEqual(rows[3008]["masstrace_centroid_mz"], "598.9966;600.0")
        self.assertEqual(
            (rows[3001]["mz"], rows[3001]["rt"], rows[3001]["fwhm"]), (100.0, 51.0, 6.0)
        )
        broken = feature(4000, 1.0, 1.0, 1.0)
        broken.removeMetaValue("masstrace_centroid_mz")
        with self.assertRaises(ValueError):
            lib.original_rows(fmap([broken]))


# --------------------------------------------------------------------------- empty sample
def empty_map(run_paths=("s.D.mzML",), uid=73):
    """A feature map without features, as FeatureFinderMetabo writes it for an empty run.

    The linker copies the map's unique id and its single "spectra_data" entry into the
    column header (unique_id, filename); consensus() gives map index i the unique id 70 + i.
    """
    m = po.FeatureMap()
    if uid is not None:
        m.setUniqueId(uid)
    if run_paths is not None:
        m.setMetaValue("spectra_data", list(run_paths))
    return m


def edit_headers(cmap, **changes):
    """Return `cmap` with column header attributes changed: index -> {attribute: value}."""
    headers = cmap.getColumnHeaders()
    for index, attributes in changes.items():
        for name, value in attributes.items():
            setattr(headers[int(index.lstrip("i"))], name, value)
    cmap.setColumnHeaders(headers)
    return cmap


class EmptySampleTest(unittest.TestCase):
    """A sample without features is resolved from the map's unique id and MS run path.

    Both must name the same single column header, which must be empty. Non-empty maps keep
    the UID-membership rule. In the pipeline meta.id is the mzML base name and every
    upstream file is written as <meta.id>.mzML, so the sample id always equals the header
    file name stem there; the unit tests also cover a sample argument that differs, which
    must not influence the resolution.
    """

    def resolve(
        self, fm, header_order=WITH_EMPTY, filenames=None, sample=EMPTY, **changes
    ):
        cmap = edit_headers(consensus(header_order, filenames), **changes)
        return lib.resolve_map_index(cmap, fm, sample)

    def assert_fails(self, fm, header_order=WITH_EMPTY, filenames=None, **changes):
        """A header that cannot be identified: the failure lists every column header."""
        cmap = edit_headers(consensus(header_order, filenames), **changes)
        with self.assertRaises(ValueError) as caught:
            lib.resolve_map_index(cmap, fm, EMPTY)
        message = str(caught.exception)
        self.assertIn(EMPTY, message)
        for header in cmap.getColumnHeaders().values():
            self.assertIn(str(header.filename), message)
            self.assertIn(str(header.unique_id), message)

    def test_primary_ms_run_paths(self):
        self.assertEqual(lib.primary_ms_run_paths(po.FeatureMap()), [])
        self.assertEqual(lib.primary_ms_run_paths(empty_map()), ["s.D.mzML"])
        self.assertEqual(
            lib.primary_ms_run_paths(empty_map(["a.mzML", "b.mzML"])),
            ["a.mzML", "b.mzML"],
        )

    def test_resolves_to_the_column_of_its_unique_id_and_ms_run(self):
        for order in (WITH_EMPTY, [EMPTY] + SAMPLES, ["s.A", EMPTY, "s.B", "s.C"]):
            index = order.index(EMPTY)
            with self.subTest(order=order):
                self.assertEqual(self.resolve(empty_map(uid=70 + index), order), index)
                self.assertEqual(
                    lib.resolve_empty_map_index(
                        consensus(order), empty_map(uid=70 + index), EMPTY
                    ),
                    index,
                )

    def test_two_empty_samples_and_a_consensus_without_rows(self):
        order = WITH_EMPTY + ["s.E"]
        cmap = consensus(order)
        for uid, path, index in ((73, "s.D.mzML", 3), (74, "s.E.mzML", 4)):
            with self.subTest(path=path):
                self.assertEqual(
                    lib.resolve_map_index(cmap, empty_map([path], uid), "x"), index
                )
                self.assertEqual(
                    [t["consensus_uid"] for t in lib.build_targets(cmap, index, "x", 30.0, "positive")],
                    [101, 102, 103, 104, 105, 106, 107, 108, 109],
                )  # fmt: skip
        # a consensus without rows: the empty map still resolves and has no target
        rowless = consensus(WITH_EMPTY)
        rowless.clear(False)
        self.assertEqual(rowless.size(), 0)
        self.assertEqual(len(rowless.getColumnHeaders()), 4)
        self.assertEqual(lib.resolve_map_index(rowless, empty_map(), EMPTY), 3)
        self.assertEqual(lib.build_targets(rowless, 3, EMPTY, 30.0, "positive"), [])

    def test_every_consensus_row_is_a_target_in_consensus_order(self):
        targets = lib.build_targets(consensus(WITH_EMPTY), 3, EMPTY, 30.0, "positive")
        self.assertEqual(
            [
                (t["target_id"], t["map_index"], t["consensus_index"], t["consensus_uid"],
                 t["n_observed"], t["charge"], t["target_mz"])
                for t in targets
            ],
            [
                ("requant_3_101", 3, 0, 101, 3, 1, 100.0),
                ("requant_3_102", 3, 1, 102, 2, 1, 200.0),
                ("requant_3_103", 3, 2, 103, 1, 1, 300.0),
                ("requant_3_104", 3, 3, 104, 2, 1, 200.0005),
                ("requant_3_105", 3, 4, 105, 1, 1, 400.0),
                ("requant_3_106", 3, 5, 106, 1, 1, 500.0),
                ("requant_3_107", 3, 6, 107, 1, 1, 600.0),
                ("requant_3_108", 3, 7, 108, 2, 1, 598.9966),
                ("requant_3_109", 3, 8, 109, 1, 1, 500.001),
            ],
        )  # fmt: skip
        first = targets[0]  # handles at 50.0, 50.5 and 51.0
        self.assertEqual(
            (first["target_rt"], first["window_min"], first["window_max"]),
            (50.5, 20.0, 81.0),
        )

    def test_run_path_must_match_exactly(self):
        # s.D and a second empty sample s.E: the same base name in two directories
        dirs = {EMPTY: "/data/run1/s.D.mzML", "s.E": "/data/run2/s.D.mzML"}
        order = WITH_EMPTY + ["s.E"]
        self.assertEqual(
            self.resolve(empty_map(["/data/run1/s.D.mzML"], 73), order, dirs), 3
        )
        self.assertEqual(
            self.resolve(empty_map(["/data/run2/s.D.mzML"], 74), order, dirs), 4
        )
        for path in (
            "s.D.mzML",  # base name only
            "/data/run2/s.D.mzML",  # the other directory (unique id still names column 3)
            "/data/run3/s.D.mzML",
            "file:///data/run1/s.D.mzML",  # no normalisation
            "/DATA/run1/s.D.mzML",
            "/data/run1/s.D.mzML ",
        ):
            with self.subTest(path=path):
                self.assert_fails(empty_map([path], 73), order, dirs)

    def test_unique_id_and_run_path_must_name_the_same_single_column(self):
        twins = {EMPTY: "s.D.mzML", "s.E": "s.D.mzML"}
        order = WITH_EMPTY + ["s.E"]
        cases = {
            "unique id of no column": (empty_map(uid=999), WITH_EMPTY, None, {}),
            "unique id of another empty column": (empty_map(uid=74), order, None, {}),
            "run path of another empty column": (
                empty_map(["s.E.mzML"]),
                order,
                None,
                {},
            ),
            "two headers with the run path": (empty_map(), order, twins, {}),
            "two headers with the unique id": (
                empty_map(),
                order,
                None,
                {"i4": {"unique_id": 73}},
            ),
            "two headers with the unique id and the run path": (
                empty_map(),
                order,
                twins,
                {"i4": {"unique_id": 73}},
            ),
        }
        for name, (fm, header_order, filenames, changes) in cases.items():
            with self.subTest(case=name):
                self.assert_fails(fm, header_order, filenames, **changes)

    def test_map_needs_a_unique_id_and_a_single_ms_run_path(self):
        cases = {
            "no valid unique id": (empty_map(uid=None), {}),
            # an unset map id (0) must not match a header whose unique id is also 0
            "no valid unique id, header id 0": (empty_map(uid=None), {"i3": {"unique_id": 0}}),
            "no spectra_data": (empty_map(None), {}),
            "empty spectra_data": (empty_map([]), {}),
            "empty path": (empty_map([""]), {}),
            # FeatureLinker writes an empty file name for a map with several run paths
            "empty path, header file name empty": (empty_map([""]), {"i3": {"filename": ""}}),
            "two paths": (empty_map(["s.D.mzML", "s.A.mzML"]), {}),
            "the path twice": (empty_map(["s.D.mzML", "s.D.mzML"]), {}),
        }  # fmt: skip
        for name, (fm, changes) in cases.items():
            with self.subTest(case=name):
                cmap = edit_headers(consensus(WITH_EMPTY), **changes)
                with self.assertRaisesRegex(ValueError, "s\\.D"):
                    lib.resolve_map_index(cmap, fm, EMPTY)

    def test_the_column_must_be_empty(self):
        # the header claims features although no handle has this index
        self.assert_fails(empty_map(), i3={"size": 1})
        # handles at the index although the header size is 0 (the header of s.B with the
        # unique id and file name the empty map carries)
        self.assert_fails(
            empty_map(), i1={"size": 0, "unique_id": 73, "filename": "s.D.mzML"},
            i3={"unique_id": 99, "filename": "x.mzML"},
        )  # fmt: skip
        # an empty map that names a column with features
        self.assert_fails(empty_map(["s.B.mzML"], 71))

    def test_non_empty_maps_are_still_resolved_by_uid_membership(self):
        cmap = consensus(WITH_EMPTY)
        b = ORIGINALS["s.B"]
        for name, fm in {
            "without run path or map id": fmap(b),
            "naming the empty column": fmap(b, ["s.D.mzML"]),
            "naming another column": fmap(b, ["s.C.mzML"]),
            "with two run paths": fmap(b, ["s.B.mzML", "s.C.mzML"]),
        }.items():
            with self.subTest(case=name):
                fm.setUniqueId(73)
                self.assertEqual(lib.resolve_map_index(cmap, fm, "s.B"), 1)
        # a non-empty map that names the empty column still fails on its UIDs
        fm = fmap(b[:1], ["s.D.mzML"])
        fm.setUniqueId(73)
        with self.assertRaises(ValueError):
            lib.resolve_map_index(cmap, fm, "s.B")

    def test_sample_argument_is_not_evidence(self):
        named = {EMPTY: "S.1.a.mzML"}
        for sample in ("S.1.a", "sD", "s.B"):
            with self.subTest(sample=sample):
                self.assertEqual(
                    self.resolve(
                        empty_map(["S.1.a.mzML"]), filenames=named, sample=sample
                    ),
                    3,
                )
        with self.assertRaises(ValueError):
            self.resolve(empty_map(["S.1.mzML"]), filenames=named, sample="S.1.a")


# --------------------------------------------------------------------------- QC
class QcTest(unittest.TestCase):
    def test_exact_status_of_every_requested_target(self):
        got = {}
        for sample in ("s.B", "s.C"):
            rows, _ = qc_rows(sample)
            for uid, row in rows.items():
                got[(sample, uid)] = (
                    row["status"],
                    row["rejection_reason"],
                    row["failed_checks"],
                )
        self.assertEqual(got, EXPECTED_STATUS)

    def test_accepted_fill_row(self):
        _, fills = qc_rows("s.C")
        self.assertEqual(len(fills), 1)
        fill = fills[0]
        self.assertEqual(
            (
                fill["sample"],
                str(fill["consensus_uid"]),
                fill["target_id"],
                fill["rt"],
                fill["mz"],
                fill["intensity"],
                int(fill["charge"]),
            ),
            ("s.C", "102", "requant_2_102", 100.4, 200.0, 777.0, 1),
        )

    def test_targets_present_but_all_absent(self):
        """No candidate for any target (e.g. the wrapper's empty map, which it writes only
        for FFMI's zero-candidate outcome with a valid extraction; see ffmi_wrapper_test)."""
        rows, fills = qc_rows("s.C", plan={})
        self.assertEqual(fills, [])
        self.assertEqual({r["status"] for r in rows.values()}, {"no_candidate"})
        self.assertEqual(sorted(rows), ["102", "103", "105", "106", "107", "109"])

    def test_masserror_sentinel(self):
        cases = [
            ([-1.0], 1, ("rejected", "no_peak_at_apex", "no_peak_at_apex")),
            ([0.3, -1.0], 2, ("rejected", "no_peak_at_apex", "no_peak_at_apex")),
            ([4.9], 1, ("accepted", "", "")),
            ([-0.9], 1, ("accepted", "", "")),
        ]
        for masserror, traces, want in cases:
            with self.subTest(masserror=masserror):
                spec = dict(VALID_C102, masserror=masserror)
                if traces == 2:
                    t = as_text_rows(targets_of("s.C"))
                    c = candidate(t[0], spec, 9102)
                    c.setMetaValue("num_of_masstraces", 2)
                    sub = c.getSubordinates()[0]
                    c.setSubordinates([sub, sub])
                    rows, _ = qc_rows("s.C", plan={}, extra=[c])
                    got = rows["102"]
                    self.assertEqual(
                        (got["status"], got["rejection_reason"], got["failed_checks"]),
                        want,
                    )
                else:
                    self.assertEqual(one_status("s.C", 102, spec), want)

    def test_malformed_metadata_all_candidates(self):
        def broken(kind):
            t = as_text_rows(targets_of("s.C"))[0]
            c = candidate(t, VALID_C102, 9102)
            if kind in (
                "model_status",
                "leftWidth",
                "num_of_masstraces",
                "masserror_ppm",
                "PrecursorMZ",
            ):
                c.removeMetaValue(kind)
            elif kind == "unknown_status":
                c.setMetaValue("model_status", "valid")
            elif kind == "left>=right":
                c.setMetaValue("leftWidth", 106.0)
            elif kind == "masserror_scalar":
                c.setMetaValue("masserror_ppm", -1.0)
            elif kind == "masserror_empty":
                c.setMetaValue("masserror_ppm", [])
            elif kind == "masserror_length":
                c.setMetaValue("masserror_ppm", [0.5, 0.5])
            elif kind == "no_subordinates":
                c.setSubordinates([])
            elif kind == "no_apex_position":
                sub = c.getSubordinates()[0]
                sub.removeMetaValue("peak_apex_position")
                c.setSubordinates([sub])
            elif kind == "no_hull":
                sub = c.getSubordinates()[0]
                sub.setConvexHulls([])
                c.setSubordinates([sub])
            elif kind == "status0_area_zero":
                c.setMetaValue("model_area", 0.0)
                c.setIntensity(0.0)
            elif kind == "status0_intensity_not_area":
                c.setIntensity(700.0)
            elif kind == "status0_no_raw_intensity":
                c.removeMetaValue("raw_intensity")
            elif kind == "shared_label":
                c.setMetaValue("alt_PeptideRef", "x")
            return c

        kinds = [
            "model_status", "unknown_status", "leftWidth", "left>=right", "num_of_masstraces",
            "masserror_ppm", "masserror_scalar", "masserror_empty", "masserror_length", "PrecursorMZ",
            "no_subordinates", "no_apex_position", "no_hull", "status0_area_zero",
            "status0_intensity_not_area", "status0_no_raw_intensity", "shared_label",
        ]  # fmt: skip
        for kind in kinds:
            with self.subTest(kind=kind):
                rows, fills = qc_rows("s.C", plan={}, extra=[broken(kind)])
                self.assertEqual(rows["102"]["status"], "rejected")
                self.assertEqual(
                    rows["102"]["rejection_reason"], "malformed_qc_metadata"
                )
                self.assertEqual(fills, [])

    def test_invalid_model_with_zero_intensity_is_not_malformed(self):
        for status in (
            "1 (invalid area)", "2 (center out of bounds)", "3 (left side out of bounds)",
            "4 (right side out of bounds)", "5 (width too large)", "6 (width too small)",
            "7 (asymmetry too high)", "8 (asymmetry too low)",
        ):  # fmt: skip
            with self.subTest(status=status):
                spec = dict(VALID_C102, status=status, area=0.0)
                self.assertEqual(
                    one_status("s.C", 102, spec),
                    ("rejected", "model_status_not_valid", "model_status_not_valid"),
                )

    def test_invalid_model_as_written_by_ffmi_without_imputation(self):
        """FFMI (no_imputation) leaves intensity 0 and a positive or NaN model_area."""
        cases = [("2 (center out of bounds)", 1234.5), ("1 (invalid area)", float("nan"))] + [
            (status, 1234.5)
            for status in (
                "3 (left side out of bounds)", "4 (right side out of bounds)", "5 (width too large)",
                "6 (width too small)", "7 (asymmetry too high)", "8 (asymmetry too low)",
            )
        ]  # fmt: skip
        for status, area in cases:
            with self.subTest(status=status, area=area):
                spec = dict(VALID_C102, status=status, area=area, intensity=0.0)
                self.assertEqual(
                    one_status("s.C", 102, spec),
                    ("rejected", "model_status_not_valid", "model_status_not_valid"),
                )

    def test_isotope_trace_overlap_at_the_tolerance_edge(self):
        # half window at 600 Th and 10 ppm: 0.003 Th
        tolerance = qc.half_window(600.0, 10.0)
        self.assertAlmostEqual(tolerance, 0.003, places=12)
        for trace_mz, want in (
            (600.0029, ["4100"]),
            (599.9971, ["4100"]),
            (600.0031, []),
            (599.9969, []),
        ):
            with self.subTest(trace_mz=trace_mz):
                original = feature(
                    4100, 598.9966, 281.0, 1.0, traces=[598.9966, trace_mz]
                )
                index = qc.build_overlap_index(
                    as_text_rows(lib.original_rows(fmap([original])))
                )
                self.assertEqual(
                    qc.find_overlapping_originals(
                        index, 600.0, tolerance, 275.0, 286.0
                    ),
                    want,
                )

    def test_overlap_with_monoisotopic_and_isotope_traces(self):
        rows, _ = qc_rows("s.B")
        self.assertEqual(rows["104"]["overlapping_original_uids"], ["2002"])
        rows, _ = qc_rows("s.C")
        self.assertEqual(rows["107"]["overlapping_original_uids"], ["3008"])

    def test_overlap_interval_and_tolerance_boundaries(self):
        index = index_of("s.B")
        # 2002: m/z 200.0001, RT 100.5. Tolerance at 10 ppm window = 0.001 Th at 200.
        self.assertEqual(
            qc.find_overlapping_originals(index, 200.0005, 0.001, 100.5, 170.0),
            ["2002"],
        )
        self.assertEqual(
            qc.find_overlapping_originals(index, 200.0005, 0.001, 95.0, 100.5), ["2002"]
        )
        self.assertEqual(
            qc.find_overlapping_originals(index, 200.0005, 0.001, 100.6, 170.0), []
        )
        self.assertEqual(
            qc.find_overlapping_originals(index, 200.0012, 0.001, 95.0, 170.0), []
        )
        self.assertAlmostEqual(qc.half_window(200.0, 10.0), 0.001, places=12)
        self.assertEqual(qc.half_window(200.0, 0.02), 0.01)
        index_c = index_of("s.C")
        self.assertEqual(
            qc.find_overlapping_originals(index_c, 600.0, 0.003, 275.0, 286.0), ["3008"]
        )
        with self.assertRaises(ValueError):
            qc.build_overlap_index(
                as_text_rows(
                    lib.original_rows(fmap(ORIGINALS["s.B"] + ORIGINALS["s.B"][:1]))
                )
            )

    def test_apex_window_boundaries(self):
        # target 106: handles at 250 only -> window [220, 280]
        for apex, want in (
            (280.0, "accepted"),
            (280.5, "rejected"),
            (219.5, "rejected"),
            (220.0, "accepted"),
        ):
            with self.subTest(apex=apex):
                spec = {
                    "apex": apex,
                    "left": apex - 5.0,
                    "right": apex + 6.0,
                    "area": 806.0,
                }
                self.assertEqual(one_status("s.C", 106, spec)[0], want)

    def test_apex_signal_fraction(self):
        pts = [
            (147.0, 0.0),
            (148.0, 0.0),
            (149.0, 0.0),
            (150.0, 5.0),
            (151.0, 0.0),
            (152.0, 0.0),
            (153.0, 0.0),
            (154.0, 7.0),
        ]
        self.assertAlmostEqual(qc.apex_signal_fraction(pts, 150.0), 1.0 / 7.0)
        self.assertEqual(qc.apex_signal_fraction(pts, 500.0), 0.0)
        self.assertEqual(
            qc.apex_signal_fraction([(149.0, 1.0), (151.0, 2.0)], 150.0), 1.0
        )
        rows, _ = qc_rows("s.B")
        # hull scans 145..156, signal only at 150; scans within 150.2 +- 3 s: 148..153
        self.assertAlmostEqual(float(rows["103"]["apex_signal_fraction"]), 1.0 / 6.0)
        # the same candidate passes with a minimum below 1/7
        self.assertEqual(
            one_status("s.B", 103, PLAN[("s.B", 103)], min_fraction=0.1),
            ("accepted", "", ""),
        )

    def test_shared_candidate_signal_and_overlap_removed(self):
        rows, _ = qc_rows("s.B")
        self.assertEqual(
            (rows["106"]["rejection_reason"], rows["109"]["rejection_reason"]),
            ("shared_candidate_signal",) * 2,
        )
        # one of the two alone is fine
        self.assertEqual(one_status("s.B", 106, PLAN[("s.B", 106)])[0], "accepted")
        t = as_text_rows(targets_of("s.B"))
        c = candidate(
            {x["consensus_uid"]: x for x in t}["106"], PLAN[("s.B", 106)], 9106
        )
        c.setMetaValue("overlap_removed", "requant_1_999")
        rows, _ = qc_rows("s.B", plan={}, extra=[c])
        self.assertEqual(rows["106"]["rejection_reason"], "shared_candidate_signal")

    def test_integrity_faults_abort(self):
        t = {x["consensus_uid"]: x for x in as_text_rows(targets_of("s.C"))}
        good = candidate(t["102"], VALID_C102, 9102)

        def relabel(label):
            c = candidate(t["102"], VALID_C102, 9102)
            c.setMetaValue("label", label)
            return c

        precursor = candidate(t["102"], VALID_C102, 9102)
        precursor.setMetaValue("PrecursorMZ", 200.01)
        moved = candidate(t["102"], VALID_C102, 9102)
        moved.setMZ(200.01)
        cases = {
            "unknown target": [relabel("requant_2_999")],
            "other sample's map index": [relabel("requant_1_102")],
            "not a requant label": [relabel("T1")],
            "two candidates for one target": [
                good,
                candidate(t["102"], VALID_C102, 9103),
            ],
            "PrecursorMZ differs from target": [precursor],
            "feature m/z differs from target": [moved],
        }
        for name, extra in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                qc_rows("s.C", plan={}, extra=extra)

    def test_target_id_prefix_must_match_the_map_index(self):
        targets = as_text_rows(targets_of("s.C"))
        # target 102 relabelled as if it came from map index 1, while the table says 2
        targets[0] = dict(targets[0], target_id="requant_1_102")
        by_id = {t["target_id"]: t for t in targets}
        c = candidate(targets[0], VALID_C102, 9102)
        originals = as_text_rows(lib.original_rows(fmap(ORIGINALS["s.C"])))
        with self.assertRaises(ValueError):
            qc.qc_sample("s.C", targets, by_id, originals, fmap([c]), 10.0, 0.3)

    def test_read_targets_faults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.tsv"
            rows = as_text_rows(targets_of("s.C"))

            def write(rows, columns=lib.TARGET_COLUMNS):
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    w = csv.DictWriter(
                        handle,
                        fieldnames=columns,
                        delimiter="\t",
                        extrasaction="ignore",
                    )
                    w.writeheader()
                    w.writerows(rows)

            write(rows)
            _targets, by_id = qc.read_targets(path, "s.C")
            self.assertEqual(
                sorted(by_id),
                [f"requant_2_{u}" for u in (102, 103, 105, 106, 107, 109)],
            )
            for name, content, sample in (
                ("duplicate target", (rows + rows[:1], lib.TARGET_COLUMNS), "s.C"),
                ("wrong sample", (rows, lib.TARGET_COLUMNS), "s.B"),
                ("missing column", (rows, lib.TARGET_COLUMNS[:-1]), "s.C"),
            ):
                with self.subTest(case=name):
                    write(*content)
                    with self.assertRaises(ValueError):
                        qc.read_targets(path, sample)


# --------------------------------------------------------------------------- gap fill
def sample_tables(plan=PLAN):
    samples = []
    for map_index, sample in enumerate(SAMPLES):
        rows, fills = qc_rows(sample, plan=plan) if targets_of(sample) else ({}, [])
        samples.append(
            {
                "sample": sample,
                "map_index": map_index,
                "header_filename": f"{sample}.mzML",
                "candidates": [as_text_rows([r])[0] for r in rows.values()],
                "fills": as_text_rows(fills),
            }
        )
    return samples


class GapFillTest(unittest.TestCase):
    def test_fill_uid_is_the_documented_hash(self):
        self.assertEqual(gf.fill_uid("s.C", 102), FILL_UID_S_C_102)
        digest = hashlib.blake2b(
            b"metaboigniter-requant|synth.C|1", digest_size=8
        ).digest()
        self.assertEqual(gf.fill_uid("synth.C", 1), int.from_bytes(digest, "big"))
        self.assertNotEqual(gf.fill_uid("s.C", 102), gf.fill_uid("s.B", 102))

    def test_fill_goes_into_the_requested_row_only(self):
        cmap = consensus()
        before = snapshot(cmap)
        inserted = gf.insert_fills(cmap, sample_tables())
        self.assertEqual(inserted, {("s.C", 102): FILL_UID_S_C_102})
        after = snapshot(cmap)
        self.assertEqual(after[1], before[1])
        for b, a in zip(before[0], after[0]):
            self.assertEqual(a[:6], b[:6])  # uid, m/z, RT, charge, intensity, quality
            if b[0] == 102:
                self.assertEqual(
                    a[6], sorted(b[6] + [(2, FILL_UID_S_C_102, 200.0, 100.4, 777.0, 1)])
                )
            else:
                self.assertEqual(
                    a[6], b[6]
                )  # neighbours (104 at nearly the same m/z) untouched

    def test_verify_detects_any_other_change(self):
        for change in ("original intensity", "row m/z", "extra handle"):
            with self.subTest(change=change):
                cmap = consensus()
                samples = sample_tables()
                snap = gf.snapshot_consensus(cmap)
                inserted = gf.insert_fills(cmap, samples)
                gf.verify_gapfilled(snap, cmap, samples, inserted)  # unchanged: passes
                rows = list(cmap)
                if change == "original intensity":
                    handles = list(rows[0].getFeatureList())
                    cf = po.ConsensusFeature(rows[0])
                    cf.clear()
                    for h in handles:
                        if h.getMapIndex() == 1:
                            h.setIntensity(h.getIntensity() + 1.0)
                        cf.insert(h)
                    rows[0] = cf
                elif change == "row m/z":
                    rows[3].setMZ(200.0006)
                else:
                    rows[2].insert(1, feature(2998, 300.0, 150.0, 1.0))
                cmap.clear(False)
                for cf in rows:
                    cmap.push_back(cf)
                with self.assertRaises(RuntimeError):
                    gf.verify_gapfilled(snap, cmap, samples, inserted)

    def test_insert_fills_integrity_faults(self):
        def mutate(fn):
            samples = sample_tables()
            fn(samples)
            return samples

        c = 2  # index of s.C in SAMPLES
        cases = {
            "fill without accepted row": lambda s: s[c]["fills"].append(dict(s[c]["fills"][0], consensus_uid="103", consensus_index="2")),
            "accepted row without fill": lambda s: s[c]["fills"].clear(),
            "missing target": lambda s: s[c]["candidates"].pop(),
            "duplicate target": lambda s: s[c]["candidates"].append(dict(s[c]["candidates"][1])),
            "consensus index mismatch": lambda s: s[c]["candidates"][0].update(consensus_index="5"),
            "target for an observed cell": lambda s: s[c]["candidates"].append(dict(s[c]["candidates"][1], consensus_uid="104", consensus_index="3")),
        }  # fmt: skip
        for name, fn in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                gf.insert_fills(consensus(), mutate(fn))

    def test_bad_consensus_membership_aborts(self):
        def rebuilt(change):
            cmap = consensus()
            rows = list(cmap)
            change(rows)
            out = po.ConsensusMap(cmap)
            out.clear(False)
            for cf in rows:
                out.push_back(cf)
            return out

        def complete_row(uid, first_handle_uid):
            cf = po.ConsensusFeature()
            cf.setUniqueId(uid)
            cf.setMZ(700.0)
            cf.setRT(290.0)
            for map_index in range(3):
                cf.insert(
                    map_index, feature(first_handle_uid + map_index, 700.0, 290.0, 1.0)
                )
            return cf

        cases = {
            "two handles of one map in a row": lambda rows: rows[0].insert(1, feature(2099, 100.0, 50.0, 1.0)),
            "feature in two rows": lambda rows: rows.append(complete_row(110, 1001)),
            "duplicate consensus uid": lambda rows: rows.append(complete_row(101, 5001)),
        }  # fmt: skip
        for name, change in cases.items():
            # without sample tables, only the consensus membership checks can fail
            with self.subTest(case=name), self.assertRaises(ValueError):
                gf.insert_fills(rebuilt(change), [])
        # the unmodified map and a valid extra complete row pass the same checks
        self.assertEqual(
            gf.insert_fills(
                rebuilt(lambda rows: rows.append(complete_row(110, 5001))), []
            ),
            {},
        )

    def test_fill_uid_collision_is_rejected_not_reassigned(self):
        clash = consensus()
        crow = list(clash)
        # give original 1003 the fill UID of (s.C, 102)
        handles = list(crow[2].getFeatureList())
        cf = po.ConsensusFeature(crow[2])
        cf.clear()
        for h in handles:
            h.setUniqueId(FILL_UID_S_C_102)
            cf.insert(h)
        crow[2] = cf
        clash.clear(False)
        for r in crow:
            clash.push_back(r)
        samples = sample_tables()
        inserted = gf.insert_fills(clash, samples)
        self.assertEqual(inserted, {("s.C", 102): None})
        gf.apply_duplicate_identity(samples, inserted)
        row = {r["consensus_uid"]: r for r in samples[2]["candidates"]}["102"]
        self.assertEqual(
            (
                row["status"],
                row["rejection_reason"],
                row["failed_checks"],
                row["fill_feature_uid"],
                row["exported_intensity"],
            ),
            (
                "rejected",
                "duplicate_feature_identity",
                "duplicate_feature_identity",
                "",
                "",
            ),
        )
        self.assertEqual(gf.mask_rows(clash, samples)[1][1][4], "missing_failed_qc")

    def test_sample_index_checks(self):
        cmap = consensus()
        good = sample_tables()
        self.assertEqual(
            gf.resolve_sample_indices(cmap, good), {"s.A": 0, "s.B": 1, "s.C": 2}
        )
        cases = {
            "two samples on one index": lambda s: s[1].update(map_index=0),
            "missing sample": lambda s: s.pop(),
            "header changed": lambda s: s[2].update(header_filename="s.mzML"),
        }
        for name, fn in cases.items():
            samples = sample_tables()
            fn(samples)
            with self.subTest(case=name), self.assertRaises(ValueError):
                gf.resolve_sample_indices(cmap, samples)
        dup = consensus()
        headers = dup.getColumnHeaders()
        headers[2].filename = "s.B.mzML"
        dup.setColumnHeaders(headers)
        with self.assertRaises(ValueError):
            gf.resolve_sample_indices(dup, sample_tables())


# --------------------------------------------------------------------------- script chain
class ChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_outputs(self):
        out = run_chain(self.dir)
        mask = read_tsv(self.dir / "mask.tsv")
        self.assertEqual(
            mask[0],
            ["consensus_index", "consensus_uid", "s.A.mzML", "s.B.mzML", "s.C.mzML"],
        )
        self.assertEqual(mask[1:], EXPECTED_MASK)
        summary = read_tsv(self.dir / "summary.tsv")
        self.assertEqual(summary[0], SUMMARY_HEAD)
        self.assertEqual(summary[1:], EXPECTED_SUMMARY)
        prov = read_dicts(self.dir / "provenance.tsv")
        self.assertEqual(
            [(r["sample"], r["consensus_uid"]) for r in prov],
            sorted(EXPECTED_STATUS, key=lambda k: (k[0], int(k[1]))),
        )
        self.assertEqual(
            {
                (r["sample"], r["consensus_uid"]): (
                    r["status"],
                    r["rejection_reason"],
                    r["failed_checks"],
                )
                for r in prov
            },
            EXPECTED_STATUS,
        )
        p = {(r["sample"], r["consensus_uid"]): r for r in prov}
        c102 = p[("s.C", "102")]
        self.assertEqual(
            (c102["target_id"], c102["map_index"], c102["consensus_index"], c102["fill_feature_uid"], c102["model_status"],
             c102["ffmi_masserror_ppm"], c102["overlapping_original_uids"], c102["n_observed"], c102["charge"]),
            ("requant_2_102", "2", "1", str(FILL_UID_S_C_102), "0 (valid)", "0.5", "", "2", "1"),
        )  # fmt: skip
        self.assertEqual(
            [float(c102[k]) for k in ("target_mz", "target_rt", "consensus_rt", "window_min", "window_max", "candidate_rt",
                                       "candidate_apex_rt", "model_area", "raw_intensity", "exported_intensity",
                                       "left_width", "right_width", "apex_signal_fraction", "peak_apex_int")],
            [200.0, 100.25, 100.25, 70.0, 130.5, 100.4, 100.4, 777.0, 700.0, 777.0, 95.0, 106.0, 1.0, 9.0],
        )  # fmt: skip
        self.assertAlmostEqual(float(c102["rt_error_s"]), 0.15, places=9)
        self.assertEqual(p[("s.B", "104")]["overlapping_original_uids"], "2002")
        self.assertEqual(p[("s.C", "107")]["overlapping_original_uids"], "3008")
        self.assertEqual(p[("s.C", "103")]["ffmi_masserror_ppm"], "-1.0")
        for key, row in p.items():
            if row["status"] != "accepted":
                with self.subTest(key=key):
                    self.assertEqual(
                        (row["fill_feature_uid"], row["exported_intensity"]), ("", "")
                    )
        # consensus: only the accepted fill was added, into row 102 at map index 2
        before = snapshot(consensus())
        after = snapshot(out)
        self.assertEqual(after[1], before[1])
        want = [
            b
            if b[0] != 102
            else b[:6]
            + (sorted(b[6] + [(2, FILL_UID_S_C_102, 200.0, 100.4, 777.0, 1)]),)
            for b in before[0]
        ]
        self.assertEqual(after[0], want)
        names = [str(dp.getSoftware().getName()) for dp in out.getDataProcessing()]
        self.assertEqual(
            names[-1], "nf-core/metaboigniter requantification gap filling"
        )
        # zero-target sample: header-only library, n_targets 0
        self.assertEqual(read_tsv(self.dir / "s.A.library.tsv"), [lib.LIBRARY_COLUMNS])
        self.assertEqual(
            read_dicts(self.dir / "s.A.sample_map.tsv")[0]["n_targets"], "0"
        )
        self.assertEqual(
            read_dicts(self.dir / "s.C.sample_map.tsv")[0][
                "header_filename_matches_sample"
            ],
            "true",
        )
        for name in (
            "mask.tsv",
            "provenance.tsv",
            "summary.tsv",
            "s.C.targets.tsv",
            "s.C.library.tsv",
            "s.C.candidates.tsv",
        ):
            with self.subTest(file=name):
                self.assertNotIn(b"\r", (self.dir / name).read_bytes())

    def test_gapfilled_file_only_adds_the_fill_and_one_processing_entry(self):
        run_chain(self.dir)
        before = (self.dir / "initial.consensusXML").read_text().splitlines()
        after = (self.dir / "gapfilled.consensusXML").read_text().splitlines()
        ops = difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        self.assertEqual({op[0] for op in ops} - {"equal"}, {"insert"})
        added = [
            line.strip()
            for op in ops
            if op[0] == "insert"
            for line in after[op[3] : op[4]]
        ]
        self.assertEqual(len(added), 5)
        self.assertRegex(added[0], r'^<dataProcessing completion_time="[^"]+">$')
        self.assertEqual(
            added[1:4],
            [
                '<software name="nf-core/metaboigniter requantification gap filling" version="" />',
                '<processingAction name="Quantitation" />',
                "</dataProcessing>",
            ],
        )
        self.assertRegex(
            added[4],
            rf'^<element map="2" id="{FILL_UID_S_C_102}" rt="100\.4\d*" mz="200\.0\d*" it="777(\.0*)?" charge="1"/>$',
        )

    def test_all_targets_absent(self):
        """Empty candidate maps for every sample, as the FFMI module writes them only for
        the recognised zero-candidate outcome; other FFMI failures stop the pipeline."""
        out = run_chain(self.dir, plan={})
        self.assertEqual(snapshot(out), snapshot(consensus()))
        summary = read_tsv(self.dir / "summary.tsv")
        self.assertEqual([r[5] for r in summary[1:]], ["0", "0", "0", "0"])  # accepted
        self.assertEqual(
            [r[7] for r in summary[1:]], ["0", "7", "6", "13"]
        )  # no_candidate
        states = {v for row in read_tsv(self.dir / "mask.tsv")[1:] for v in row[2:]}
        self.assertEqual(states, {"observed", "missing_no_candidate"})

    def test_deterministic_under_reordered_samples_and_candidates(self):
        reference = None
        orders = (SAMPLES, SAMPLES[::-1], ["s.B", "s.C", "s.A"])
        for i, (order, reverse) in enumerate(itertools.product(orders, (False, True))):
            d = self.dir / str(i)
            d.mkdir()
            run_chain(d, sample_order=order, reverse=reverse)
            got = [
                (d / n).read_bytes()
                for n in ("mask.tsv", "provenance.tsv", "summary.tsv")
            ]
            if reference is None:
                reference = got
            with self.subTest(order=order, reverse=reverse):
                self.assertEqual(got, reference)

    def test_column_header_order_drives_the_mask(self):
        order = ["s.C", "s.A", "s.B"]
        out = run_chain(self.dir, header_order=order)
        mask = read_tsv(self.dir / "mask.tsv")
        self.assertEqual(
            mask[0],
            ["consensus_index", "consensus_uid", "s.C.mzML", "s.A.mzML", "s.B.mzML"],
        )
        self.assertEqual(mask[2], ["1", "102", "requantified", "observed", "observed"])
        row = next(cf for cf in out if cf.getUniqueId() == 102)
        self.assertIn(
            (0, FILL_UID_S_C_102),
            {(h.getMapIndex(), h.getUniqueId()) for h in row.getFeatureList()},
        )
        self.assertEqual(read_tsv(self.dir / "summary.tsv")[1:], EXPECTED_SUMMARY)

    def test_negative_mode_chain(self):
        run_chain(self.dir, polarity="negative")
        library = read_dicts(self.dir / "s.C.library.tsv")
        self.assertEqual({r["Charge"] for r in library}, {"-1"})
        self.assertAlmostEqual(float(library[0]["Mass"]), 200.0 + PROTON, places=9)
        self.assertEqual(read_tsv(self.dir / "mask.tsv")[1:], EXPECTED_MASK)

    def test_join_faults_abort(self):
        run_chain(self.dir)
        d = self.dir

        def gapfill(*tables):
            args = []
            for t in tables:
                args += ["--sample-tables", *t]
            return subprocess.run(
                [sys.executable, "-B", str(BIN / "gapfill_consensus.py"), "--consensus", str(d / "initial.consensusXML"), *map(str, args),
                 "--output-consensus", str(d / "g2.consensusXML"), "--mask", str(d / "m2.tsv"),
                 "--provenance", str(d / "p2.tsv"), "--summary", str(d / "s2.tsv")],
                env=ENV, capture_output=True, text=True, check=False,
            )  # fmt: skip

        def tables(sample, cand=None, sample_map=None):
            return (
                sample,
                sample_map or d / f"{sample}.sample_map.tsv",
                cand or d / f"{sample}.candidates.tsv",
                d / f"{sample}.fills.tsv",
            )

        ok = gapfill(*(tables(s) for s in SAMPLES))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        cases = {
            "missing sample": [tables("s.A"), tables("s.B")],
            "duplicate sample": [
                tables("s.A"),
                tables("s.B"),
                tables("s.C"),
                tables("s.C"),
            ],
            "candidates of another sample": [
                tables("s.A"),
                tables("s.B", cand=d / "s.C.candidates.tsv"),
                tables("s.C"),
            ],
            "sample map of another sample": [
                tables("s.A"),
                tables("s.B", sample_map=d / "s.C.sample_map.tsv"),
                tables("s.C"),
            ],
        }
        for name, args in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(gapfill(*args).returncode, 0)

    def test_read_sample_tables_faults(self):
        run_chain(self.dir)
        d = self.dir

        def tables(sample, cand=None, sample_map=None):
            return (
                sample,
                str(sample_map or d / f"{sample}.sample_map.tsv"),
                str(cand or d / f"{sample}.candidates.tsv"),
                str(d / f"{sample}.fills.tsv"),
            )

        samples, _columns = gf.read_sample_tables([tables(s) for s in SAMPLES])
        self.assertEqual([(s["sample"], s["map_index"], len(s["candidates"]), len(s["fills"])) for s in samples],
                         [("s.A", 0, 0, 0), ("s.B", 1, 7, 0), ("s.C", 2, 6, 1)])  # fmt: skip
        lines = (d / "s.C.candidates.tsv").read_text().splitlines(keepends=True)
        (d / "short.tsv").write_text("".join(lines[:-1]))
        (d / "cols.tsv").write_text(
            "".join([lines[0].replace("	status	", "	state	")] + lines[1:])
        )
        cases = {
            "duplicate sample": [tables("s.A"), tables("s.A")],
            "foreign candidate rows": [tables("s.B", cand=d / "s.C.candidates.tsv")],
            "n_targets differs from the candidate table": [
                tables("s.C", cand=d / "short.tsv")
            ],
            "different candidate columns": [
                tables("s.B"),
                tables("s.C", cand=d / "cols.tsv"),
            ],
            "sample map of another sample": [
                tables("s.B", sample_map=d / "s.C.sample_map.tsv")
            ],
        }
        for name, args in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                gf.read_sample_tables(args)

    def check_empty_sample_outputs(self, d, header_order, out):
        """Exact chain outputs with s.D (no features, EMPTY_PLAN) in `header_order`."""
        mask = read_tsv(d / "mask.tsv")
        self.assertEqual(
            mask[0],
            ["consensus_index", "consensus_uid"] + [f"{s}.mzML" for s in header_order],
        )
        position = 2 + header_order.index(EMPTY)
        self.assertEqual(
            [row[position] for row in mask[1:]], EXPECTED_EMPTY_MASK_COLUMN
        )
        # the other samples are unchanged
        others = {s: 2 + header_order.index(s) for s in SAMPLES}
        self.assertEqual(
            [[row[0], row[1]] + [row[others[s]] for s in SAMPLES] for row in mask[1:]],
            EXPECTED_MASK,
        )
        summary = read_tsv(d / "summary.tsv")
        self.assertEqual(summary[0], SUMMARY_HEAD)
        self.assertEqual(summary[1:], EXPECTED_SUMMARY[:3] + EXPECTED_EMPTY_SUMMARY)
        prov = read_dicts(d / "provenance.tsv")
        want = {**EXPECTED_STATUS, **EXPECTED_EMPTY_STATUS}
        self.assertEqual(
            [(r["sample"], r["consensus_uid"]) for r in prov],
            sorted(want, key=lambda k: (k[0], int(k[1]))),
        )
        self.assertEqual(
            {(r["sample"], r["consensus_uid"]): (r["status"], r["rejection_reason"], r["failed_checks"]) for r in prov},
            want,
        )  # fmt: skip
        empty_index = str(header_order.index(EMPTY))
        p = {r["consensus_uid"]: r for r in prov if r["sample"] == EMPTY}
        self.assertEqual({r["map_index"] for r in p.values()}, {empty_index})
        self.assertEqual(
            [(p[u]["fill_feature_uid"], p[u]["exported_intensity"], p[u]["overlapping_original_uids"]) for u in ("101", "104")],
            [(str(FILL_UID_S_D_101), "501.0", ""), (str(FILL_UID_S_D_104), "504.0", "")],
        )  # fmt: skip
        sample_map = read_dicts(d / "s.D.sample_map.tsv")[0]
        self.assertEqual(
            (sample_map["map_index"], sample_map["n_features"], sample_map["n_targets"], sample_map["header_filename"]),
            (empty_index, "0", "9", "s.D.mzML"),
        )  # fmt: skip
        self.assertEqual(len(read_dicts(d / "s.D.library.tsv")), 9)
        self.assertEqual(read_tsv(d / "s.D.originals.tsv"), [lib.ORIGINAL_COLUMNS])
        # consensus: the original rows plus exactly the accepted fills of every sample
        index = header_order.index(EMPTY)
        fills = {
            102: [
                (header_order.index("s.C"), FILL_UID_S_C_102, 200.0, 100.4, 777.0, 1)
            ],
            101: [(index, FILL_UID_S_D_101, 100.0, 50.6, 501.0, 1)],
            104: [(index, FILL_UID_S_D_104, 200.0005, 160.3, 504.0, 1)],
        }
        before = snapshot(consensus(header_order))
        after = snapshot(out)
        self.assertEqual(after[1], before[1])
        self.assertEqual(
            after[0],
            [b[:6] + (sorted(b[6] + fills.get(b[0], [])),) for b in before[0]],
        )

    def test_sample_without_features_is_requantified(self):
        out = run_chain(self.dir, {**PLAN, **EMPTY_PLAN}, WITH_EMPTY, WITH_EMPTY)
        self.check_empty_sample_outputs(self.dir, WITH_EMPTY, out)

    def test_sample_without_features_first_and_reversed(self):
        order = [EMPTY] + SAMPLES
        reference = None
        for i, (sample_order, reverse) in enumerate(
            ((order, False), (order[::-1], True))
        ):
            d = self.dir / str(i)
            d.mkdir()
            out = run_chain(d, {**PLAN, **EMPTY_PLAN}, order, sample_order, reverse)
            with self.subTest(sample_order=sample_order, reverse=reverse):
                self.check_empty_sample_outputs(d, order, out)
                got = [
                    (d / n).read_bytes()
                    for n in ("mask.tsv", "provenance.tsv", "summary.tsv")
                ]
                reference = reference or got
                self.assertEqual(got, reference)

    def test_sample_without_features_and_without_candidates(self):
        out = run_chain(self.dir, PLAN, WITH_EMPTY, WITH_EMPTY)
        mask = read_tsv(self.dir / "mask.tsv")
        self.assertEqual({row[5] for row in mask[1:]}, {"missing_no_candidate"})
        summary = {row[0]: row for row in read_tsv(self.dir / "summary.tsv")[1:]}
        self.assertEqual(
            summary[EMPTY],
            ["s.D", "9", "0", "9", "0", "0", "0", "9"] + ["0"] * 8 + ["9"],
        )
        self.assertEqual(sum(len(cf.getFeatureList()) for cf in out), 14 + 1)

    def test_qc_cli_requires_candidates_when_there_are_targets(self):
        run_chain(self.dir)
        result = subprocess.run(
            [sys.executable, "-B", str(BIN / "requantification_qc.py"), "--sample", "s.C",
             "--targets", str(self.dir / "s.C.targets.tsv"), "--originals", str(self.dir / "s.C.originals.tsv"),
             "--output-candidates", str(self.dir / "x.tsv"), "--output-fills", str(self.dir / "y.tsv")],
            env=ENV, capture_output=True, text=True, check=False,
        )  # fmt: skip
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
