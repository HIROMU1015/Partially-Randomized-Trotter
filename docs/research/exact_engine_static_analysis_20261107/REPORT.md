# Exact Fraction engine：ボトルネック・高速化方針調査

2026-11-07 JST、SERVER1 `GPUA100-gleap-01`。
**静的調査・設計完了。実装・新規数値計算・benchmarkなし。4 Markdownを公開照合後mandatory STOP。**

## 1. 結論

主な負担候補は、**対角/疎gateにもdense積和を行うFraction state contraction、大整数bit長に依存するgcd/演算/オブジェクト生成、gateごとのmatrix再取得・Fraction化・hash照合**である。support index生成、固定D列/同一trialの回路再build、Born平方和の重複、group末尾待ちも候補となる。

保存752-shot報告ではengine worker区間累計250,491.399秒、compile区間累計261.224秒、約70分で752/752完了、127 workers、当時の人工52件PASSと記載される。**engine秒はCPU timeではなくperf_counterのwall区間和**で、build/hash/state/Born/facade等を含む。Fraction乗算/gcd等の割合・実bit長分布・最適化速度はUNKNOWN。新しい科学的再集計・ランキングはしていない。

最初に試す価値があるのは、**保存matrixのexact零に基づく疎/対角Fraction contraction（O1）を単独検証し、ordered support別index再利用（O2）を追加比較する案**である。同じmatrix・順序・state・Born/thresholdを保てる可能性があり、削除する無駄がSOURCEで特定できる。新実装のcorrectnessは未確認で、既存認証engineを変更しない。

## 2. 成果物

| 文書 | 内容 |
|---|---|
| [EXACT_ENGINE_HOTSPOT_ANALYSIS.md](EXACT_ENGINE_HOTSPOT_ANALYSIS.md) | 処理経路、Fraction/dyadic構造、timingの境界、Born/grid、gate hashと未測定事項 |
| [EXACT_ENGINE_OPTIMIZATION_OPTIONS.md](EXACT_ENGINE_OPTIMIZATION_OPTIONS.md) | 候補10群の機構・差分・再利用・memory/identity・保証・承認範囲、worker scheduling |
| [EXACT_ENGINE_VALIDATION_PLAN.md](EXACT_ENGINE_VALIDATION_PLAN.md) | 最小人工prototype、exact同値性、time/memory、SOURCE保全、次の承認とSTOP |
| 本REPORT | 固定資料、結論、非実施事項、公開・保全・停止の範囲 |

## 3. 実際に確認した指示・固定資料

| 資料 | 版・参照 |
|---|---|
| 正式共通指示 | 元mainの未追跡 `research_project_instructions_unified_2026-10-25.md` を読み取り、特に§8.1/8.2/8.4/10/10.3を確認。SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`。旧版で代用していない |
| repository指示 | ユーザー提示AGENTS、元mainのAGENTS、独立worktreeのAGENTS。PROJECT_MAPを先に読み、研究概要・現状を全体確認。研究目的・証拠状態/manifestも確認 |
| 現行SCV準備 | [560c6092bfb5e71af8eedc4a4be5416922764f98](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/560c6092bfb5e71af8eedc4a4be5416922764f98) の `docs/research/scv224_preparation_20261107/{REPORT,SCV_IMPLEMENTATION_REPORT,SCV_224_SHOT_EXECUTION_DRAFT}.md` と準備STOP |
| 調査SOURCE | 同commitの `src/trottertracks/algorithm_codesign/{a1_engine_connection,a1_n3_connection,scv224_pilot,a1_n3_campaign,a1_n3_pilot_campaign,a1_n3_pilot_activation,a1_n3_resource_admission,shared_cpu}.py`、関連builder/cache、既存人工test |
| engine認証 | [fdf00cee3c87d804845497afc89ac6a71f3b3f59](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fdf00cee3c87d804845497afc89ac6a71f3b3f59) の `artifacts/o2_a1_new_target/2026-11-06/engine_qualification_run01/{REPORT.md,ENGINE_ERROR_CERTIFIED.json,GATE_CERTIFICATES.json,UNIFORM_ERROR.json,EXECUTION_SOURCE.json,STOP.json}`、対応するEXECUTION/CORRECTNESS |
| finite target契約 | `2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab` の `docs/research/o2_a1_new_finite_target_20261106/{A1_NEW_FINITE_TARGET,A1_NEW_SCIENCE_EXECUTION_CONTRACT_DRAFT}.md`。後続のN1/N2/engine認証で取得済みの項目と、旧草案段階の記述を区別 |
| 752-shot保存記録 | [3c8e55a2728f86a1eda17a2e0105050a637e7c86](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3c8e55a2728f86a1eda17a2e0105050a637e7c86) の `docs/research/o2_a1_n3_pilot_admission_20261107/` のREPORT/EXECUTION/人工logと、run02の保存timing/resource/accounting/STOP |
| 補助仕様 | [Python3.12 Fraction仕様](https://docs.python.org/3.12/library/fractions.html)、[CPython3.12.3 SOURCE](https://github.com/python/cpython/blob/v3.12.3/Lib/fractions.py)、[GMP](https://gmplib.org/manual/Rational-Number-Functions)、[gmpy2 mpq](https://gmpy2.readthedocs.io/en/latest/mpq.html)。機能確認のみ、依存採択・installなし |

全資料はGit固定オブジェクト又は自分の独立worktreeから静的に確認した。gate JSONはschemaと各gate名の保存行の閲覧のみで、δ*/bit長/nnz等の再集計をしていない。`git diff`で認証kernel `24d6bdfa11ca5964c83b2639f060f4655ab1d4be`→SCV準備版のengine SOURCE不変、752-shot結果版→SCV準備版の調査対象engine/worker/activation/admission SOURCE不変を確認した。

上記時間・認証の引用対象は全て、保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector、`L_D=3,m0/q1,δ=0.02,r=1,K=2,coefficient_atol=0`、X/Y、N1 actual finite-grid Q/selected full policy、N2保証state、理想gate・保存DFモデルに限定する。δ掃引、実backend/noise、他round/分子、最終PR/RPE総コストの結果ではない。

## 4. Exact性・認証の扱い

保存matrixのexact非零だけを使う疎演算、同一index/不変descriptor再利用、正しいbuffer/Born重複削減は、算術誤差0を維持し得る。dyadic共通分母、GMP/native exactも数学的同値の候補だが差分とcorrectness負担が大きい。どれも今回の速度は未測定。

既存 `b_engine` は同じN2 state、certified local matrix/angle、全positive path・軸・列長cap、exact state/Born比、exact ceil、独立uniform53-bitモデルと固定SOURCEに依存する。**局所matrix認証の再利用可能性と、新SOURCEの認証済み扱いを区別する。** 新exact実装には同値性とSOURCE適合確認が必要。旧証明/lockを上書きしない。

float/GPUは算術誤差0を引き継げない。interval/mixed precisionもroundoff包含、最終norm正下界、Born比、threshold境界、fallback/word消費の追加保証が必要であり、初回には優先しない。

## 5. 再利用・次に必要な承認

旧 `apply_exact/multiply/add`、Born/threshold、certified matrix table、既存engine/N3/worker/SCV人工testをoracle/接続検査へ再利用する。compile metricsのSQLite/LRUとfork/pool再利用は既に存在するため、新高速化として二重計上しない。旧確率/state/測定bitの流用は提案しない。

次回は独立moduleでO1/O2を実装し、**手書き人工入力だけ**で全振幅・p・threshold一致、time/memory/cold-warm条件を検証する明示指示が必要。実H4 Born/実gate・回路build/compileを伴う性能測定は、SOURCE/input/有限call/STOPを具体化して別途明示承認を受ける。新しい承認証明書・grantシステムは作らない。人工結果が出るまで現科学engineは変更しない。

## 6. 非干渉・保全・非実施

- 専用branch `codex/exact-engine-static-analysis-20261107`、worktree `/tmp/exact-engine-static-analysis-20261107`。baseは固定SCV準備 `560c6092…`。元main `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf` をcheckout/reset/stashしていない。
- 元mainの開始時diff/statusを保存し、分離後に既存modified/untracked 2,136ファイルの読み取りhash snapshotも作成した。文書作成後の照合でdiff/status一致・全2,136ファイルのhash一致を確認。元main/固定SOURCE/結果/証明/STOP/grant/ledger/markerへ書込みなし。
- 調査対象のengine/library/test/runnerは編集していない。新規4 Markdownだけを追加する。科学決定・検証結果・新validation pathを追加していないため、研究概要/研究ノート/manifest/既存indexは変更しない。
- 224-shotのlive worktree・output・cache・journal・manager・STOPを開かず、jobへattach/profile/process操作をしていない。worktree一覧に存在すること以上の稼働状態・完了・健全性は**UNKNOWN**。共有venv/package/cache、OS/scheduler/cgroup、他ユーザーへ変更なし。
- 実H4計算、Born、gate作用、実path/build/compile、RNG/sampling/measurement、state solve、engine再認証、科学結果再集計/ランキング、benchmark、人工test、新kernel実装、重要科学レビューは**今回全て未実行**。C1その他旧STOP研究を再開していない。

## 7. 公開照合とmandatory STOP

4 Markdownだけをcommitし専用branchへ通常pushする。remote branch SHAとlocal HEAD、remote commit/root tree、4ファイルのblob SHAとUTF-8本文byte、成果物directory treeを照合する。自己参照になる最終commit SHAは本文へ埋めず、照合結果とともに最終chatで提示する。照合が失敗した場合は公開完了と報告しない。

**静的調査・設計・公開照合が完了した時点でmandatory STOP。** 具体案が得られてもprototype・人工test・performance benchmark・実H4比較・224-shot変更へ自動進行しない。
