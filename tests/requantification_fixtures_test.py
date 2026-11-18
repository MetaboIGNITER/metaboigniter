"""The committed synthetic fixtures match their generator and their documented design."""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pyopenms as po

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "requant_synthetic"
PROTON_MASS_U = 1.007276466621


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_mzml", FIXTURES / "make_synthetic_mzml.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def spectra(path):
    experiment = po.MSExperiment()
    po.MzMLFile().load(str(path), experiment)
    out = []
    for spectrum in experiment:
        mzs, intensities = spectrum.get_peaks()
        polarity = spectrum.getInstrumentSettings().getPolarity()
        out.append((spectrum.getRT(), polarity, list(mzs), list(intensities)))
    return out


class FixtureTest(unittest.TestCase):
    def test_committed_mzml_files_equal_generator_output(self):
        generator = load_generator()
        expected = sorted(relative for relative, *_ in generator.layout())
        committed = sorted(
            str(p.relative_to(FIXTURES)) for p in FIXTURES.rglob("*.mzML")
        )
        self.assertEqual(committed, expected)
        with tempfile.TemporaryDirectory() as tmp:
            generator.write(tmp)
            for relative in expected:
                with self.subTest(file=relative):
                    self.assertEqual(
                        spectra(Path(tmp) / relative), spectra(FIXTURES / relative)
                    )

    def test_files_are_small_and_end_with_a_newline(self):
        for path in FIXTURES.rglob("*"):
            if path.is_file():
                with self.subTest(file=str(path.relative_to(FIXTURES))):
                    data = path.read_bytes()
                    self.assertLess(len(data), 5000 * 1024)
                    self.assertTrue(data.endswith(b"\n"))
                    self.assertNotIn(b"\r\n", data)

    def test_polarity_of_each_file(self):
        for path in FIXTURES.rglob("*.mzML"):
            want = (
                po.IonSource.Polarity.NEGATIVE
                if "negative" in path.parts
                else po.IonSource.Polarity.POSITIVE
            )
            with self.subTest(file=str(path.relative_to(FIXTURES))):
                self.assertEqual({s[1] for s in spectra(path)}, {want})

    def test_signal_design(self):
        """T2 has no signal in synth.B; synth.C 'absent' has no T signal, 'fill' a weak one,
        'empty' only weak signals (below the FeatureFinderMetabo noise threshold of 10)."""

        def apex(path, mz):
            values = [
                max(
                    (i for m, i in zip(s[2], s[3]) if abs(m - mz) <= mz * 5e-6),
                    default=0.0,
                )
                for s in spectra(path)
            ]
            return max(values)

        positive = FIXTURES / "positive"
        t1, t2, t3 = 285.1360, 367.1784, 401.2021
        self.assertEqual(apex(positive / "synth.B.mzML", t2), 0.0)
        for mz in (t1, t2, t3):
            self.assertEqual(
                apex(positive / "synth.C.absent" / "synth.C.mzML", mz), 0.0
            )
        fill = positive / "synth.C.fill" / "synth.C.mzML"
        self.assertAlmostEqual(apex(fill, t1), 9.0, places=4)
        self.assertAlmostEqual(apex(fill, t3), 6.3, places=4)
        self.assertEqual(apex(fill, t2), 0.0)
        self.assertGreater(apex(positive / "synth.A.mzML", t2), 1e6)
        empty = positive / "synth.C.empty" / "synth.C.mzML"
        weak = {
            152.0706: 8.55, 241.1183: 8.1, 329.1592: 7.65, 413.2662: 7.2,
            t1: 9.0, t2: 0.0, t3: 6.3,
        }  # fmt: skip
        for mz, height in weak.items():
            with self.subTest(empty=mz):
                self.assertAlmostEqual(apex(empty, mz), height, places=4)
        # the only peaks at or above the threshold are the constant noise traces
        noise = load_generator().NOISE_MZ
        loud = {
            round(m, 4) for s in spectra(empty) for m, i in zip(s[2], s[3]) if i >= 10.0
        }
        self.assertEqual(loud, {round(m, 4) for m in noise})

    def test_library_fixtures(self):
        def rows(name):
            with open(
                FIXTURES / "libraries" / name, newline="", encoding="utf-8"
            ) as handle:
                return list(csv.DictReader(handle, delimiter="\t"))

        fill = {r["CompoundName"]: r for r in rows("fill_synth.C.tsv")}
        self.assertEqual(sorted(fill), ["T1", "T2", "T3"])
        self.assertAlmostEqual(
            float(fill["T1"]["Mass"]), 285.1360 - PROTON_MASS_U, places=9
        )
        self.assertAlmostEqual(
            float(fill["T3"]["Mass"]), 2 * 401.2021 - 2 * PROTON_MASS_U, places=9
        )
        self.assertEqual(fill["T3"]["Charge"], "+2")
        self.assertEqual(
            [r["CompoundName"] for r in rows("absent_synth.B.tsv")], ["T2"]
        )
        self.assertEqual(
            float(rows("outside_rt_synth.B.tsv")[0]["RetentionTime"]), 5000.0
        )
        # the same libraries plus one target far outside the run (0-300 s)
        outside = rows("outside_rt_synth.B.tsv")[0]
        for name, base in (
            ("fill_outside_rt_synth.C.tsv", "fill_synth.C.tsv"),
            ("absent_outside_rt_synth.B.tsv", "absent_synth.B.tsv"),
        ):
            with self.subTest(library=name):
                extended = rows(name)
                self.assertEqual(extended[:-1], rows(base))
                self.assertEqual(dict(extended[-1], CompoundName="T2"), dict(outside))
                self.assertEqual(extended[-1]["CompoundName"], "T2_outside_rt")
        self.assertNotIn("SumFormula", rows("malformed.tsv")[0])
        self.assertEqual(float(rows("invalid_range.tsv")[0]["RetentionTimeRange"]), 0.0)


if __name__ == "__main__":
    unittest.main()
