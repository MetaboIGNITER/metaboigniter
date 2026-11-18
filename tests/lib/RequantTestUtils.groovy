// Helpers for the requantification nf-tests. nf-test puts tests/lib on the classpath;
// methods defined in a *.nf.test file itself are not visible inside test closures.
import java.security.MessageDigest

class RequantTestUtils {

    static final String OBSERVED  = 'TABLE_OUTPUT/output_quantification_Linked_data.tsv'
    static final String GAPFILLED = 'TABLE_OUTPUT/output_quantification_Linked_data_gapfilled.tsv'
    static final String MASK      = 'TABLE_OUTPUT/output_requantification_mask_Linked_data.tsv'
    static final String PROV      = 'TABLE_OUTPUT/output_requantification_provenance_Linked_data.tsv'
    static final String SUMMARY   = 'TABLE_OUTPUT/output_requantification_summary_Linked_data.tsv'

    // Columns that hold UIDs assigned at random by OpenMS (FFM, linker).
    static final List UID_COLUMNS = ['id', 'feature_ids']

    // Presence pattern of the observed matrix of the synthetic data set, keyed by consensus
    // m/z (4 dp); values are [synth.A, synth.B, synth.C] > 0. It is the same with and without
    // --requantification, in both polarities and with or without --parallel_linking.
    static final Map SYNTHETIC_OBSERVED_PRESENCE = [
        '152.0706': [true, true, true],
        '241.1183': [true, true, true],
        '285.1360': [true, true, false],
        '329.1592': [true, true, true],
        '367.1784': [true, false, false],
        '401.2021': [true, true, false],
        '413.2662': [true, true, true],
    ]

    // stableMd5() of the matrices of the real "-profile test" data (conf/test.config): without
    // requantification, serial and parallel linking, and with --identification (separate MS2).
    // Requantification must not change any of them. Recorded from runs without
    // requantification; the --identification value was identical in two runs.
    static final String REAL_OBSERVED_MD5 = 'd62f29e6a562260e6daf319e929f806a'
    static final String REAL_OBSERVED_PARALLEL_MD5 = '2f529e502f11c5aba6d6098f6c2e0e96'
    static final String REAL_IDENTIFICATION_MD5 = '4fe8ab8aebf0f15437618d34759b4fb2'

    // Gap-filled matrices and the "all" summary row of the real-data requantification tests.
    // Integers only in the summary, so they do not depend on float formatting of an engine.
    static final String REAL_GAPFILLED_PARALLEL_MD5 = '21408f8ce156852cbf10f5066197348f'
    static final String REAL_GAPFILLED_IDENTIFICATION_MD5 = '9b184358a2f1fd7bb80e749ed15963f2'
    static final List SUMMARY_ALL_COLUMNS = [
        'consensus_rows', 'observed', 'targets_requested', 'candidates_recovered', 'accepted', 'rejected',
        'no_candidate', 'rejected_malformed_qc_metadata', 'rejected_existing_signal_overlap',
        'rejected_no_peak_at_apex', 'rejected_apex_outside_target_window', 'rejected_model_status_not_valid',
        'rejected_low_apex_signal_fraction', 'rejected_shared_candidate_signal',
        'rejected_duplicate_feature_identity', 'missing_after',
    ]
    static final List REAL_SUMMARY_ALL_PARALLEL = [57372, 108352, 63764, 40818, 4321, 36497, 22946, 0, 11294, 5980, 0, 18121, 1012, 90, 0, 59443]
    static final List REAL_SUMMARY_ALL_IDENTIFICATION = [57404, 108353, 63859, 41167, 4292, 36875, 22692, 0, 11171, 6060, 0, 18542, 1016, 86, 0, 59567]

    // Area of the Gaussian injected by make_synthetic_mzml.py (height x sigma x sqrt(2 pi)):
    // T1 apex 9 counts, T3 apex 6.3 counts, sigma 3 s.
    static final double SYNTHETIC_T1_AREA = 9.0 * 3.0 * Math.sqrt(2.0 * Math.PI)
    static final double SYNTHETIC_T3_AREA = 6.3 * 3.0 * Math.sqrt(2.0 * Math.PI)

    static List summaryAll(List lines) {
        def row = tsv(lines).find { it.sample == 'all' }
        return SUMMARY_ALL_COLUMNS.collect { row[it] as long }
    }

    static boolean within(double value, double expected, double relative) {
        return Math.abs(value / expected - 1.0) <= relative
    }

    // stableMd5() of the observed matrix of the positive synthetic run (same with and without
    // --requantification; the absent and fill variants differ only below the FFM threshold).
    static final String SYNTHETIC_OBSERVED_MD5 = '3589d054982787fb304875a382d6f24c'

    // The "empty" variant (tests/fixtures/README.md): synth.C has no feature at all, so its
    // column of the observed matrix is empty. Its stableMd5() is the same with and without
    // --requantification and with serial or parallel linking.
    static final Map SYNTHETIC_EMPTY_OBSERVED_PRESENCE = SYNTHETIC_OBSERVED_PRESENCE.collectEntries { mz, present ->
        [mz, [present[0], present[1], false]]
    }
    static final String SYNTHETIC_EMPTY_OBSERVED_MD5 = '4d06cc83d66d7940ed5f003f5d11f5cb'
    static final String SYNTHETIC_EMPTY_GAPFILLED_MD5 = 'a3054cef87ba11d705ee955c212209ef'
    // Apex heights (counts) of the weak synth.C peaks in the "empty" variant; T2 is absent.
    static final Map SYNTHETIC_EMPTY_HEIGHTS = [
        '152.0706': 8.55, '241.1183': 8.1, '285.1360': 9.0, '329.1592': 7.65, '401.2021': 6.3, '413.2662': 7.2,
    ]

    // Area of a Gaussian peak of the synthetic data (sigma 3 s)
    static double syntheticArea(double height) {
        return height * 3.0 * Math.sqrt(2.0 * Math.PI)
    }

    // nf-schema resolves relative msfile paths against the launch directory, which nf-test
    // places below .nf-test/, so the samplesheet is written at test time with absolute paths.
    static String writeSheet(outDir, projectDir, String polarity, String cVariant, boolean reversed = false) {
        def root = "${projectDir}/tests/fixtures/requant_synthetic/${polarity}"
        def rows = [
            "sA,MS1,normal,${root}/synth.A.mzML",
            "sB,MS1,normal,${root}/synth.B.mzML",
            "sC,MS1,normal,${root}/synth.C.${cVariant}/synth.C.mzML",
        ]
        if (reversed) {
            rows = rows.reverse()
        }
        def sheet = new File("${outDir}/samplesheet_${polarity}_${cVariant}.csv")
        sheet.parentFile.mkdirs()
        sheet.text = "sample,level,type,msfile\n" + rows.join("\n") + "\n"
        return sheet.toString()
    }

    static List header(List lines) {
        return lines[0].split('\t', -1) as List
    }

    static List<Map> tsv(List lines) {
        def columns = header(lines)
        return lines.drop(1).collect { line ->
            def values = line.split('\t', -1) as List
            assert values.size() == columns.size() : "ragged TSV line: ${line}"
            [columns, values].transpose().collectEntries()
        }
    }

    static String mz4(value) {
        return String.format(Locale.ROOT, '%.4f', value as double)
    }

    // Numeric cells rounded to 6 significant digits, so that the value does not depend on
    // the last float digits of the container build.
    static String canonical(String value) {
        if (value ==~ /-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?/) {
            return String.format(Locale.ROOT, '%.6g', value as double)
        }
        return value
    }

    // Table without the random-UID columns, numbers canonicalised, rows sorted.
    static List stableView(List lines) {
        def keep = header(lines).findAll { !(it in UID_COLUMNS) }
        def rows = tsv(lines).collect { r -> keep.collect { canonical(r[it]) }.join('\t') }.sort()
        return [keep.join('\t')] + rows
    }

    static String stableMd5(List lines) {
        def digest = MessageDigest.getInstance('MD5').digest(stableView(lines).join('\n').getBytes('UTF-8'))
        return digest.encodeHex().toString()
    }

    static List sampleColumns(List lines) {
        return header(lines).findAll { it.endsWith('.mzML') }.sort()
    }

    static Map presence(List lines) {
        def samples = sampleColumns(lines)
        return tsv(lines).collectEntries { r ->
            [mz4(r.mz), samples.collect { s -> (r[s] as double) > 0 }]
        }.sort()
    }

    static double cell(List lines, String mz, String sample) {
        def rows = tsv(lines).findAll { mz4(it.mz) == mz }
        assert rows.size() == 1 : "expected one row at m/z ${mz}, found ${rows.size()}"
        return rows[0][sample] as double
    }

    // mask rows keyed by the consensus m/z, joined to a matrix through consensus_uid == id
    static Map maskByMz(List matrixLines, List maskLines) {
        def mzById = tsv(matrixLines).collectEntries { r -> [r.id, mz4(r.mz)] }
        def samples = sampleColumns(maskLines)
        def out = tsv(maskLines).collectEntries { r ->
            assert mzById.containsKey(r.consensus_uid) : "mask row ${r.consensus_uid} not in the matrix"
            [mzById[r.consensus_uid], samples.collect { r[it] }]
        }
        assert out.size() == mzById.size()
        return out.sort()
    }

    static List provenanceTriples(List lines) {
        return tsv(lines).collect { r -> [r.sample, mz4(r.target_mz), r.status] }.sort { it.toString() }
    }

    static Map summaryCounts(List lines, List columns) {
        return tsv(lines).collectEntries { r -> [r.sample, columns.collect { r[it] as int }] }
    }

    // Cross-file consistency of the published requantification tables (real data):
    // observed cells equal the observed matrix exactly, requantified cells are > 0 in the
    // gap-filled matrix and empty (0) in the observed one, missing cells are 0 in both.
    static List crossCheck(List observedLines, List gapfilledLines, List maskLines) {
        def problems = []
        def observed = tsv(observedLines).collectEntries { [it.id, it] }
        def gapfilled = tsv(gapfilledLines).collectEntries { [it.id, it] }
        def samples = sampleColumns(maskLines)
        if (observed.keySet() != gapfilled.keySet()) {
            problems << 'observed and gap-filled matrices have different consensus ids'
        }
        if (sampleColumns(observedLines) != samples || sampleColumns(gapfilledLines) != samples) {
            problems << 'sample columns differ between matrices and mask'
        }
        def counts = [:].withDefault { 0 }
        tsv(maskLines).each { m ->
            def o = observed[m.consensus_uid]
            def g = gapfilled[m.consensus_uid]
            if (o == null || g == null) {
                problems << "mask row ${m.consensus_uid} missing from a matrix"
                return
            }
            ['charge', 'RT', 'mz'].each { c ->
                if (o[c] != g[c]) {
                    problems << "row ${m.consensus_uid} column ${c} differs"
                }
            }
            samples.each { s ->
                def state = m[s]
                counts[state] += 1
                def ov = o[s] as double
                def gv = g[s] as double
                if (state == 'observed' && !(ov > 0 && o[s] == g[s])) {
                    problems << "observed cell ${m.consensus_uid}/${s}: ${o[s]} vs ${g[s]}"
                } else if (state == 'requantified' && !(ov == 0 && gv > 0)) {
                    problems << "requantified cell ${m.consensus_uid}/${s}: ${o[s]} vs ${g[s]}"
                } else if (state.startsWith('missing_') && !(ov == 0 && gv == 0)) {
                    problems << "missing cell ${m.consensus_uid}/${s}: ${o[s]} vs ${g[s]}"
                } else if (!(state in ['observed', 'requantified', 'missing_no_candidate', 'missing_failed_qc', 'missing_existing_signal_overlap'])) {
                    problems << "unknown mask state ${state}"
                }
            }
        }
        return [problems.take(20), counts]
    }
}
