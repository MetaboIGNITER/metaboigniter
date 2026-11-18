# Test fixtures

## `requant_synthetic/`

A small synthetic LC-MS dataset for the requantification (gap-filling) tests. It is
used by `tests/requantification.nf.test`, by the module test in
`modules/local/openms_featurefindermetaboident/tests/`, and by the Python tests
`tests/requantification_fixtures_test.py` and `tests/ffmi_wrapper_test.py`.

- **Source:** synthetic. `make_synthetic_mzml.py` in this directory generates every
  mzML file with pyOpenMS 3.6.0. The generator is deterministic (no random numbers and
  no timestamps).
- **Regenerate:** `python make_synthetic_mzml.py --outdir tests/fixtures/requant_synthetic`.
  `tests/requantification_fixtures_test.py` checks that the committed files hold the same
  spectra as the generator output and that the signals follow the design below.
- **Licence:** MIT, the same as the pipeline. The data are artificial and contain no
  sample-derived information.

### Design

Each file has centroided MS1 spectra from 0 to 300 s at 1 Hz. Every peak is Gaussian
with sigma 3 s and carries its monoisotopic peak plus two 13C isotope peaks. Every
spectrum also holds four constant noise peaks (20-30 counts) that FeatureFinderMetabo
does not report as features.

| Compound | m/z      | RT (s) | z   | synth.A | synth.B | synth.C (`fill`) | synth.C (`absent`) | synth.C (`empty`) |
| -------- | -------- | ------ | --- | ------- | ------- | ---------------- | ------------------ | ----------------- |
| K1-K4    | 152-413  | 60-230 | 1   | yes     | yes     | yes              | yes                | apex 8.55-7.2     |
| T1       | 285.1360 | 140    | 1   | yes     | yes     | apex 9 counts    | no                 | apex 9 counts     |
| T2       | 367.1784 | 90     | 1   | yes     | no      | no               | no                 | no                |
| T3       | 401.2021 | 200    | 2   | yes     | yes     | apex 6.3 counts  | no                 | apex 6.3 counts   |

In `synth.C.fill`, T1 and T3 sit below the FeatureFinderMetabo noise threshold, so the
initial consensus has no synth.C values for them. FeatureFinderMetaboIdent recovers both
with model status `0 (valid)`.

In `synth.C.empty`, every compound sits below that threshold (K1-K4 at 8.55, 8.1, 7.65
and 7.2 counts), so FeatureFinderMetabo finds no feature at all and synth.C has an empty
consensus column. FeatureFinderMetaboIdent recovers every compound except T2.

The requantification targets are:

- synth.A: none;
- synth.B: T2, which has no signal, so FFMI finds no candidate for it;
- synth.C: T1, T2 and T3 (`fill`, `absent`), or every consensus row (`empty`).

Sample names contain dots on purpose, because the pipeline uses
`meta.id = msfile.baseName`. `negative/` holds the same design (without the `absent`
and `empty` variants) written with negative polarity.

`libraries/` contains FeatureFinderMetaboIdent assay libraries for the module test:

| Library                         | Run on                      | FFMI 3.6.0 result                                                           | Module result                  |
| ------------------------------- | --------------------------- | --------------------------------------------------------------------------- | ------------------------------ |
| `absent_synth.B.tsv`            | `positive/synth.B.mzML`     | exit 8, "Found 0 feature candidates in total."                              | success, empty feature map     |
| `fill_synth.C.tsv`              | `positive/synth.C.fill/...` | exit 0, two features (T1, T3)                                               | success, two features          |
| `outside_rt_synth.B.tsv`        | `positive/synth.B.mzML`     | exit 8 plus "Could not find any points for chromatogram"                    | fails with exit status 8       |
| `fill_outside_rt_synth.C.tsv`   | `positive/synth.C.fill/...` | exit 0, two features, plus "Could not find any points" for the extra target | success, two features          |
| `absent_outside_rt_synth.B.tsv` | `positive/synth.B.mzML`     | exit 8, "Found 0 feature candidates" plus "Could not find any points"       | fails with exit status 8       |
| `malformed.tsv`                 | any                         | exit 3 (missing SumFormula/IsoDistribution columns)                         | fails with exit status 3       |
| `invalid_range.tsv`             | any                         | not run                                                                     | fails in the library check (1) |

The two `*_outside_rt_*` libraries are `fill_synth.C.tsv` and `absent_synth.B.tsv` plus
the target `T2_outside_rt` at 5000 s, far outside the run (0-300 s).

The module turns FFMI's exit status 8 into an empty feature map only for the recognised
zero-candidate outcome of a valid extraction (see the module script). "Every target
absent" is therefore supported only when the chromatogram extraction was valid; a library
outside the RT range of the run, a malformed library or any other FFMI error still fails.

## `ffmi_logs/`

Console logs of FeatureFinderMetaboIdent 3.6.0, captured from real runs, that
`tests/ffmi_wrapper_test.py` replays through a stub FFMI to test the module's
zero-candidate wrapper:

| Log                                   | FFMI outcome                                                     | Module result |
| ------------------------------------- | ---------------------------------------------------------------- | ------------- |
| `absent_zero_candidates.exit8.log`    | no candidate for any target, valid extraction                    | empty map     |
| `outside_rt_range.exit8.log`          | no chromatogram points (library outside the RT range of the run) | exit 8        |
| `degenerate_imputation_fit.exit8.log` | imputation fit failed ("Unable to fit linear transformation")    | exit 8        |
| `unreadable_library.exit3.log`        | library could not be read                                        | exit 3        |
