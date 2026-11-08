# Track C DF H2/H4/H6 comparison, 2026-11-08

[REPORT](../../../docs/research/track_c_df_truncation_20261108/REPORT.md) and [reproduction](../../../docs/research/track_c_df_truncation_20261108/REPRODUCE.md).

Final results are in `ordering/`. Root summaries are pre-ordering history retained in the raw archive. Scientific work is complete; `STUDY_STOP.json` preserves mandatory STOP.

`RAW_INDEX.json` lists 13,994 scientific raw members with sizes/SHA256. Concatenate the nine `raw.tar.gz.part000` … `part008` in order to restore a standard tar.gz into a fresh directory. `ARCHIVE_PARTS.json` and `ARCHIVE_VERIFICATION.json` verify all part/archive/member bytes. Large INPUT_ARRAYS, scan and trajectory files are archived; selected metadata/results are also directly tracked for browsing.

`SOURCE.json` identifies loaded stage commits. `SOURCE_HISTORY.bundle` preserves original scientific code history through a4e32484254a090eab6d221776a50a1cfd9f0f17 and requires public base c5a3c102199fe97f4d5c21837c059ae2684f5ef2. The publication snapshot has its own commit identity. Private original pickles are neither published nor changed.
