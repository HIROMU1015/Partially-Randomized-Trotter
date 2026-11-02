# README-only PR案（未作成・未適用）

タイトル：`Add an entry point to the Pauli-JW H2/H4/H6 research synthesis`

Baseはmain `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`。
研究branchをheadにしたPRではなく、このbaseから新しい文書branchを作って[README-only patch](pauli_jw_main_readme_entry.patch)だけを適用する案である。
mainが進んでいた場合は新HEADのREADMEと再照合し、研究branch全体のmergeへ切り替えない。

公開済み参照先：

- [総合資料（固定commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e05f061c86ae853e90568abd4842de652c82398/docs/research/pauli_jw_resource_evaluation_summary.md)
- [15条件・保存値/出典CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e05f061c86ae853e90568abd4842de652c82398/docs/research/data/pauli_jw_summary/latest15.csv)

以下がPR本文案。

---

Add a README entry to the published Pauli-JW H2/H4/H6 resource-evaluation synthesis and its 15-condition saved-data table. Readers can reach the research purpose, methods, latest conditional observations, result history, limitations and fixed evidence refs from main.

The patch changes README navigation only. It retains the existing DF screening/UWC notices, does not merge the October research branches, and does not change scientific code, raw evidence, validation statuses or accepted claims. Results remain restricted to finite candidates, hypothetical zero implementation residual and finite price cohorts.

Validation: the README-only patch applies to the recorded main commit in an isolated directory; the linked fixed documentation commit is published; saved table strings and provenance were checked without scientific computation or numerical reanalysis. The pre-existing validation-manifest/schema mismatch is recorded separately and is unchanged.

---

ユーザーがこの案を選んだ場合の担当：Codexがmain起点の新branchでREADMEだけを変更し、差分・URL・main最新SHAを再確認してPRを作成する。mainへのmergeはユーザー判断とする。今回PRは作成していない。
