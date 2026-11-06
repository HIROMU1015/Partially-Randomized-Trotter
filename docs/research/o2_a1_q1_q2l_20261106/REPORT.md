# A1 Q1/Q2-L：固定identity検査でSTOP

2026-11-06 JST、SERVER1。専用branch `codex/o2-a1-q1-q2l-20261106`。
科学SOURCE：[8e002a3815fb4fba709cf090e9c975052f8dd892][SOURCE_COMMIT]。実行前に公開本文4件とlocal treeを照合して固定した。

**承認されたQ1/Q2-Lを一回開始したが、固定preparation identityに一致せず約5.508秒でSTOPした。H hashは一致、partition/preparation/tail hashは不一致。実CDF・ν・bias値は取得していない。科学retry・Q2-T・Q3–Q5への進行はない。**
不一致を新しいHamiltonian、別label、別SOURCE、許容差の変更で回避していない。

## 条件・実行範囲

保存linear H4、1.00 Å、STO-3G、legacy DF rank12、4電子/8qubit、L_D3、m0/q1/δ=t=.02/r1/K2、threshold0、X/Y共通law。
固定NPZ SHA256は `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。
[EXECUTION](EXECUTION.md) と前回の[Q1/Q2-L仕様](../o2_a1_0_connections_20261106/A1_0_SCIENCE_RUN_SCOPE.md)に従った。
Python3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0の既存opt2環境を変更せず利用した。

再利用した `shared_cpu.module` により元のライブラリをprivate CPU namespaceへ読み、既存NPZ loader→LD3 split→threshold0 preparationを一回呼んだ。
Q1で許された小さい軌道基底再構成を含む。physical-sector groundの対角化、RNG、Hadamard simulation、full interrogation build/compile、variance/N/cost評価は行っていない。

## 保存されたidentity比較

| identity | 固定opt2で期待したhash | 今回取得したhash |
|---|---|---|
| H | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` | 同一 |
| partition | `9034397f5bde1f20cb712578d14245f4dac1236413d34642a75d83d867b52086` | `900342ad6262021ff9b919c36ee182a8ed83914a697524e3fae07248d0d082f8` |
| preparation | `50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc` | `f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299` |
| tail | `c51e35da56f3de1646896339f38da3cb40dcb8405a24d781cef134a654bab642` | `890442900e605a262201338f19afdad638302d670c75a9d95371c10b83ea38d8` |

H hash一致だけで同じpartition/law/targetと認定しない。一方、hash差だけで物理的な有意差・科学的破綻があるとも断定しない。
どのfragment/weight/basis数値が違うかは、今回の停止記録だけでは未特定である。

## 原証拠

| 保存ファイル | 役割 |
|---|---|
| [SOURCE.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/v1/SOURCE.json) | 実行前SOURCE commit・code hash・NPZ hash・version・限定scope |
| [IDENTITY_MISMATCH.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/v1/IDENTITY_MISMATCH.json) | 今回の実取得hashと期待hash。CDF評価前の停止点 |
| [STOP.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/v1/STOP.json) | ValueError、no_retry、elapsed、STOPPED |
| [STATIC_SOURCE_DIFFERENCE.json](../../../artifacts/o2_a1_q1_q2l/2026-11-06/v1/STATIC_SOURCE_DIFFERENCE.json) | 停止後に行ったSOURCEのbyte/静的比較だけの記録 |

`CATALOG.json`、`Q1.json`、`Q2L.json`、成功`RESULT.json`は生成されていない。欠測を0やPASSで補完しない。
SOURCE/STOPを上書きせず、全4保存ファイルと実行前SOURCEを保全した。

## 停止後に分かったSOURCE版差と限界

数値処理を再開せず、元WP11 worktreeと現SOURCEのファイルbytes・diff・既存manifestを確認した。
`df_partial_randomized_pf.py` と `df_rte_tail.py` は元WP11とbyte一致したが、`df_partial_s2.py` は異なった。
現在版SHA256は `6821823749b903ccd1dcb57803f27459f2b4fe265ca53175373fe55d00423d9e`、元WP11版は `2a9ff5483762ac7472ba49b0ed10c14e804ae3e2fbfb8a3979b7dce02a0d9444`。
元WP11と同じ版は、既存 `docs/archive/server_collection_20261103/98d1146f889bf36c2744ee90f997f352f9e7c51b/src/trotterlib/df_partial_s2.py.original` に保存されていた。

現版はexplicit ordered partition対応を追加している。ただし既定のweight-ranked経路ではhash policy文字列 `ranked_df_prefix_v1` を維持しており、この追加だけで今回のhash差を説明できたとはしない。
元opt2の10-file compute manifestは `df_partial_s2.py` 自身を含んでいない。前回の10 SOURCE-file一致やcarrierとの一致は、推移的な原実行SOURCE全体の一致を示していなかった。
これを今回の来歴上の不足として追加記録し、以前の保存結果・SOURCE・判定を上書きしない。

今後の原因調査で区別すべきものは、SOURCE cohort差、rank/weightの数値再現性、軌道基底の数値非一意性、hashに含まれるmetadataである。
単なるbyte完全一致を目的化するのではなく、同じsplit/law/finite targetを維持することが目的。原因が未特定のままguardを外したり、原版へ差し替えて科学再実行したりしていない。

## 技術検証と次の判断

科学実行前の人工testは新4件＋既存24件の**28件PASS**。
小さな架空Pauli/lawで上界の包含、SOURCE scalar/normalizer丸め項、入力拒否、外向きpreviewを確認した。分子のCDF/bias評価が成功した証拠ではない。
新しい上界の導出・適用条件はEXECUTIONに残すが、今回その実科学値はない。

次に必要なのは、元SOURCE cohortとrank/weight/basisの再現性を切り分ける限定調査である。
数値比較・payload再取得や再実行を伴う場合は、このSTOPと今回の原証拠を維持した新scopeとして判断する。今回その追加処理は実行していない。
control/baseline/測定予算の未承認提案を採択済みに変更せず、測定可能とも認定しない。

## 保全・公開・mandatory STOP

元main・既存dirty/untracked、他worktree、原科学SOURCE/input/raw/grant/ledger/marker/STOPを保全する。
C1凍結、S2/R1の既存判定、旧one-shot/retry制限、共有venv/package/cache/OS/他ユーザーjob/filesを変更しない。
停止記録をstatus/manifest/概要/当日ノートに登録し、専用branchへ公開して再取得照合する。
**mandatory STOP。科学retry、CDF/moment/bias計算、Q3/Q4/Q5、別Trackへ自動進行しない。**

[SOURCE_COMMIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8e002a3815fb4fba709cf090e9c975052f8dd892
