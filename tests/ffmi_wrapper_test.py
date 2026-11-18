"""The zero-candidate wrapper of modules/local/openms_featurefindermetaboident (AM2).

The module's script block is rendered as Nextflow would render it and run with bash, with a
stub FeatureFinderMetaboIdent that replays FFMI 3.6.0 logs (tests/fixtures/ffmi_logs, captured
from real runs) or variants that break exactly one acceptance condition. Only the exact
"zero candidates" outcome may be turned into an empty feature map; everything else must keep
FFMI's exit status. The FFMI module nf-test runs the real tool on the synthetic fixtures.

$FFMI_MODULE overrides the module path (used by the mutation harness).
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(
    os.environ.get(
        "FFMI_MODULE", ROOT / "modules/local/openms_featurefindermetaboident/main.nf"
    )
)
LOGS = ROOT / "tests/fixtures/ffmi_logs"
LIBRARIES = ROOT / "tests/fixtures/requant_synthetic/libraries"
ABSENT = (LOGS / "absent_zero_candidates.exit8.log").read_text()
RANGE_ERROR = (
    "Error: Unexpected internal error (Empty or uninitalized range object. "
    "Did you forget to call updateRanges()?)"
)

STUB_FFMI = """#!/usr/bin/env bash
out=""
while [ $# -gt 0 ]; do
    if [ "$1" = "-out" ]; then out="$2"; fi
    shift
done
touch ffmi_was_called
cat "$STUB_LOG"
if [ "${STUB_WRITE_OUTPUT:-0}" = "1" ]; then
    echo '<featureMap><featureList count="1"></featureList></featureMap>' > "$out"
fi
exit "$STUB_EXIT"
"""
STUB_OPENMSINFO = """#!/usr/bin/env bash
echo "Version      : 3.6.0"
"""


def render_script(path):
    """The module's `script:` block as Nextflow renders it (meta.id = 'sample')."""
    text = path.read_text()
    body = text[text.index("script:") :]
    body = body[body.index('"""') + 3 :]
    body = body[: body.index('"""')]
    values = {
        "${prefix}": "sample",
        "${library}": "library.tsv",
        "${mzml}": "sample.mzML",
        "${args}": "",
        "${task.process}": "OPENMS_FEATUREFINDERMETABOIDENT",
    }
    for key, value in values.items():
        body = body.replace(key, value)
    body = body.replace("\\\\", "\0").replace("\\$", "$").replace("\0", "\\")
    unresolved = re.findall(r"(?<!\\)\$\{[^}]*\}", body)
    assert not unresolved, f"unrendered Groovy expressions: {unresolved}"
    return textwrap.dedent(body)


class FfmiWrapperTest(unittest.TestCase):
    def run_wrapper(
        self,
        log_text=ABSENT,
        exit_code=8,
        write_output=False,
        library=LIBRARIES / "absent_synth.B.tsv",
    ):
        tmp = Path(self.tmp.name)
        work = tmp / "work"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        shutil.copy(library, work / "library.tsv")
        (work / "sample.mzML").write_text("<mzML/>\n")
        (tmp / "ffmi.log").write_text(log_text)
        (work / ".command.sh").write_text(self.script)
        env = dict(
            os.environ,
            PATH=f"{tmp / 'stub'}{os.pathsep}{os.environ['PATH']}",
            STUB_LOG=str(tmp / "ffmi.log"),
            STUB_EXIT=str(exit_code),
            STUB_WRITE_OUTPUT="1" if write_output else "0",
        )
        result = subprocess.run(
            ["bash", "-C", "-e", "-u", "-o", "pipefail", ".command.sh"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, work

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        stub = Path(self.tmp.name) / "stub"
        stub.mkdir()
        for name, content in (
            ("FeatureFinderMetaboIdent", STUB_FFMI),
            ("OpenMSInfo", STUB_OPENMSINFO),
        ):
            (stub / name).write_text(content)
            (stub / name).chmod(0o755)
        self.script = render_script(MODULE)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_failed_with(self, result, work, code):
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        if code != 1:
            self.assertTrue((work / "ffmi_was_called").exists())

    def test_zero_candidates_becomes_an_empty_feature_map(self):
        result, work = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        featurexml = (work / "sample.featureXML").read_text()
        self.assertTrue(featurexml.startswith("<?xml"))
        self.assertIn('<featureList count="0">', featurexml)
        log = (work / "sample.ffmi.log").read_text()
        self.assertIn("Found 0 feature candidates in total.", log)
        self.assertIn("writing an empty feature map", log)
        self.assertIn("openms: 3.6.0", (work / "versions.yml").read_text())

    def test_successful_run_is_untouched(self):
        result, work = self.run_wrapper(exit_code=0, write_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('count="1"', (work / "sample.featureXML").read_text())

    def test_real_ffmi_failures_keep_their_exit_status(self):
        cases = {
            "outside_rt_range.exit8.log": 8,
            "degenerate_imputation_fit.exit8.log": 8,
            "unreadable_library.exit3.log": 3,
        }
        for name, code in cases.items():
            with self.subTest(log=name):
                result, work = self.run_wrapper(
                    log_text=(LOGS / name).read_text(), exit_code=code
                )
                self.assert_failed_with(result, work, code)
                self.assertFalse((work / "sample.featureXML").exists())

    def test_each_acceptance_condition_is_required(self):
        lines = ABSENT.splitlines(keepends=True)
        found = "Found 0 feature candidates in total.\n"
        left = "0 features left after selection of best candidates.\n"
        self.assertIn(found, lines)
        self.assertIn(left, lines)
        self.assertIn(RANGE_ERROR + "\n", lines)
        variants = {
            # a "Could not find any points" line without the Error: prefix
            "chromatogram without points": ABSENT.replace(
                found, "Warning: Could not find any points for chromatogram X\n" + found
            ),
            "another Error line": ABSENT.replace(
                found, "Error: something else\n" + found
            ),
            "no 'Found 0' line": ABSENT.replace(found, ""),
            "'Found 0' not an exact line": ABSENT.replace(
                found, "Found 0 feature candidates in total. (x)\n"
            ),
            "no '0 features left' line": ABSENT.replace(left, ""),
            "other internal error": ABSENT.replace(
                RANGE_ERROR, "Error: Unexpected internal error (other)"
            ),
        }
        for name, log_text in variants.items():
            with self.subTest(variant=name):
                self.assertNotEqual(log_text, ABSENT)
                result, work = self.run_wrapper(log_text=log_text)
                self.assert_failed_with(result, work, 8)
        with self.subTest(variant="output file exists"):
            result, work = self.run_wrapper(write_output=True)
            self.assert_failed_with(result, work, 8)
        with self.subTest(variant="exit status 9"):
            result, work = self.run_wrapper(exit_code=9)
            self.assert_failed_with(result, work, 9)

    def test_library_is_validated_before_ffmi_runs(self):
        tmp = Path(self.tmp.name)
        header = (LIBRARIES / "absent_synth.B.tsv").read_text().splitlines()[0]
        row = (LIBRARIES / "absent_synth.B.tsv").read_text().splitlines()[1].split("\t")
        bad = {
            "header only": [header],
            "zero range": [header, "\t".join(row[:5] + ["0.0"] + row[6:])],
            "negative range": [header, "\t".join(row[:5] + ["-5"] + row[6:])],
            "infinite mass": [header, "\t".join(row[:2] + ["inf"] + row[3:])],
            "missing retention time": [header, "\t".join(row[:4] + [""] + row[5:])],
        }
        for name, content in bad.items():
            with self.subTest(library=name):
                (tmp / "lib.tsv").write_text("\n".join(content) + "\n")
                result, work = self.run_wrapper(library=tmp / "lib.tsv")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse((work / "ffmi_was_called").exists())
        result, work = self.run_wrapper(library=LIBRARIES / "invalid_range.tsv")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((work / "ffmi_was_called").exists())


if __name__ == "__main__":
    unittest.main()
