#!/usr/bin/env python
"""Generate the tiny synthetic MS1 dataset used by the requantification tests.

Writes, below --outdir:
    positive/synth.A.mzML
    positive/synth.B.mzML
    positive/synth.C.fill/synth.C.mzML
    positive/synth.C.absent/synth.C.mzML
    positive/synth.C.empty/synth.C.mzML
    negative/synth.A.mzML
    negative/synth.B.mzML
    negative/synth.C.fill/synth.C.mzML

Design (centroided MS1, 0-300 s at 1 Hz, Gaussian peaks with sigma 3 s):
    K1..K4  anchors present in every sample;
    T1      z=1, present in synth.A and synth.B;
    T2      z=1, present in synth.A only;
    T3      z=2, present in synth.A and synth.B.
    synth.C.fill   carries T1 and T3 below the FeatureFinderMetabo noise threshold
                   (apex 9 and 6.3 counts), so FFM misses them and FFMI finds them;
    synth.C.absent has no signal at the T1/T2/T3 m/z at all;
    synth.C.empty  carries K1..K4, T1 and T3 below the FeatureFinderMetabo noise
                   threshold (apex 6.3-9 counts) and no T2, so FFM finds no feature at all
                   while FFMI finds every compound except T2.
Requantification targets are therefore: synth.A none; synth.B T2 (no signal, zero
FFMI candidates); synth.C T1, T2 and T3 ("fill", "absent"), or every row ("empty").
Sample names contain dots on purpose (the pipeline uses meta.id = msfile.baseName).

The output is deterministic (no random numbers, no timestamps). Each file ends with a
newline so the end-of-file-fixer pre-commit hook leaves it untouched.

Usage: python make_synthetic_mzml.py --outdir tests/fixtures/requant_synthetic
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyopenms as po

C13 = 1.0033548378
SIGMA_S = 3.0
FILL_HEIGHT = 9.0
#       name   m/z        RT     height  |z|
ANCHORS = [
    ("K1", 152.0706, 60.0, 4.0e6, 1),
    ("K2", 241.1183, 110.0, 3.0e6, 1),
    ("K3", 329.1592, 170.0, 5.0e6, 1),
    ("K4", 413.2662, 230.0, 2.5e6, 1),
]
T1 = ("T1", 285.1360, 140.0, 2.0e6, 1)
T2 = ("T2", 367.1784, 90.0, 1.5e6, 1)
T3 = ("T3", 401.2021, 200.0, 2.0e6, 2)
NOISE_MZ = (120.0123, 199.9876, 455.5555, 499.0001)
# per-sample compounds on top of the anchors, and a small per-sample RT shift
SAMPLES = {
    "synth.A": ([T1, T2, T3], 0.0),
    "synth.B": ([T1, T3], 0.5),
    "synth.C": ([], 1.0),
}
C_FILL = [
    T1[:3] + (FILL_HEIGHT, T1[4]),
    # a different height on purpose: two candidates with identical raw intensity
    # make FFMI 3.6.0 imputation fail ("Unable to fit linear transformation")
    T3[:3] + (FILL_HEIGHT * 0.7, T3[4]),
]
# a sample without features: every compound below the FFM threshold, T2 missing;
# distinct apex heights (8.55, 8.1, 7.65, 7.2, 9 and 6.3 counts)
C_EMPTY = [
    anchor[:3] + (FILL_HEIGHT * scale, anchor[4])
    for anchor, scale in zip(ANCHORS, (0.95, 0.9, 0.85, 0.8))
] + C_FILL


def spectrum_peaks(rt, compounds, rt_shift):
    mzs, ints = [], []
    for _name, mz, centre, height, z in compounds:
        apex = height * math.exp(-((rt - centre - rt_shift) ** 2) / (2.0 * SIGMA_S**2))
        if apex < 1.0:
            continue
        for k, rel in enumerate((1.0, 0.15, 0.012)):
            mzs.append(mz + k * C13 / z)
            ints.append(apex * rel)
    for j, noise_mz in enumerate(NOISE_MZ):
        mzs.append(noise_mz)
        ints.append(20.0 + 5.0 * ((int(rt) + j) % 3))
    order = np.argsort(mzs)
    return (
        np.asarray(mzs, dtype=np.float64)[order],
        np.asarray(ints, dtype=np.float32)[order],
    )


def build_experiment(compounds, rt_shift, polarity):
    experiment = po.MSExperiment()
    for i in range(301):
        rt = float(i)
        spectrum = po.MSSpectrum()
        spectrum.setMSLevel(1)
        spectrum.setRT(rt)
        spectrum.setType(po.SpectrumSettings.SpectrumType.CENTROID)
        spectrum.setNativeID(f"scan={i + 1}")
        settings = po.InstrumentSettings()
        settings.setPolarity(
            po.IonSource.Polarity.NEGATIVE
            if polarity == "negative"
            else po.IonSource.Polarity.POSITIVE
        )
        spectrum.setInstrumentSettings(settings)
        spectrum.set_peaks(spectrum_peaks(rt, compounds, rt_shift))
        experiment.addSpectrum(spectrum)
    experiment.updateRanges()
    return experiment


def layout():
    """(relative path, polarity, compounds, rt shift) of every fixture file."""
    files = []
    for polarity in ("positive", "negative"):
        for sample, (extra, shift) in SAMPLES.items():
            compounds = list(ANCHORS) + list(extra)
            if sample != "synth.C":
                files.append((f"{polarity}/{sample}.mzML", polarity, compounds, shift))
                continue
            files.append(
                (
                    f"{polarity}/synth.C.fill/synth.C.mzML",
                    polarity,
                    compounds + C_FILL,
                    shift,
                )
            )
            if polarity == "positive":
                files.append(
                    (f"{polarity}/synth.C.empty/synth.C.mzML", polarity, C_EMPTY, shift)
                )
                files.append(
                    (
                        f"{polarity}/synth.C.absent/synth.C.mzML",
                        polarity,
                        compounds,
                        shift,
                    )
                )
    return files


def write(outdir):
    options = po.PeakFileOptions()
    options.setCompression(True)
    written = []
    for relative, polarity, compounds, shift in layout():
        target = Path(outdir) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        mzml = po.MzMLFile()
        mzml.setOptions(options)
        mzml.store(str(target), build_experiment(compounds, shift, polarity))
        with open(target, "ab") as handle:
            handle.write(b"\n")
        written.append(str(target))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    for path in write(args.outdir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
