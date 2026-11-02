# PR境界拡張結果：研究証拠・論文資料への追加

2026-10-27 JST。公開済み結果の転記、証拠対応付け、文書の静的検査を実施した。新規科学計算、sampling、compile、signal/shot再計算、数値再解析、再ランキング、fit、作図、詳細科学レビューは実施していない。

**公開状態：科学資料8文書の限定版を公開・remote照合済み。完全19文書版はローカル保全。** ユーザーは19文書の公開を明示承認したが、自動審査は承認後の分割読込も拒否した。この制約を迂回せず、既存運用文書9件の本文・追加checker・研究ノートを公開差分から除いた科学資料8文書版を別に公開した。remote SHAは `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`、branchは `codex/pr-boundary-evidence-integration-20261027`。GitHubから8件を実取得し、全content・Git blobが公開用科学資料および取得checkoutと一致した。公開側でも元43行・15証拠行の転記、全基点6,228ファイル不変、固定55リンクを静的確認した。以下の索引・状態表への追記は完全ローカル版での変更であり、公開branchの対応9ファイルは基点のまま。完全ローカル版は別branch `codex/pr-boundary-evidence-integration-local-full-20261027` に保全する。新科学計算0、mandatory STOP。

## 追加先と採択状態

研究索引は固定 `731b3d6f3572c6aa7950a579b68251fec9a5218b`、統合4文書は固定 `6e3436a16b13e539b0faa2a0a3286f6ab3549543` を継承した。それぞれの対応remote branchの先端が指定commitと一致することをGitHub接続で確認した。統合4文書だけを索引branchへ継承し、無関係なbranchはmergeしていない。

- [EVIDENCE_ADDENDUM.csv](EVIDENCE_ADDENDUM.csv)：保存行15件。新3物理候補の探索・独立確認6件、保存R対照5件、H8の既存2件と未成立1件、公開実行・終了要約1件。行数は独立実験数ではない。
- [EVIDENCE_MATRIX.csv](../../../EVIDENCE_MATRIX.csv)：既存43行をbyte prefixとして保存し、別IDの7行を追加した。旧CEMのID・claim・採択状態を改訂していない。
- [RESEARCH_SYNTHESIS.md](../../../RESEARCH_SYNTHESIS.md)、[PAPER_OUTLINE.md](../../../PAPER_OUTLINE.md)、[FIGURE_TABLE_PLAN.md](../../../FIGURE_TABLE_PLAN.md)：元の方針・本文を保持し、今回の追加節だけを末尾へ接続した。
- [研究データ入口](../README.md)と[2026-10-24固定索引](../2026-10-24_track_a_research_data/README.md)に追加への入口を置いた。旧study/claim/paper-data表、CEM v3、ledger v5、paper manifest、原主表・原稿・図表は変更していない。
- [INTEGRATION_MANIFEST.json](INTEGRATION_MANIFEST.json)は今回の参照・掲載案のoverlay。[STATIC_VALIDATION.json](STATIC_VALIDATION.json)は文書技術検査の記録であり、新しい科学的採択台帳ではない。

今回の7項目は **追加保存証拠・掲載提案** である。既存の正式採択済み主張はその限定状態のまま残し、新項目を正式claim、論文中心主張、final evidence freezeへ自動昇格させていない。

## 科学条件と評価量

H4/H6はlinear、隣接1.00 Å、STO-3G、tol-only DF threshold 1e-8、rank10/19。H6は保存weighted-Hermitian completion。T=2.0、outer δ=T/q=2.0、m5_bestの11 signed piece、ABS_TIME_PER_OUTER。H4はprefix3/q1/K2、H6はprefix7/q1/K6。元Hamiltonian・state・reference・scalar・生成順、finite-RTE/PCG64、symmetric_directional、actual adjoint、cosine/+Im sineの測定込みfull Hadamard wrapper、状態準備P=0を維持したとする公開結果を引用する。

公開実行条件はQiskit1.3.0、NumPy1.26.4、basis rz/sx/x/cx、opt1、transpiler seed17、topologyなし。主指標は単価C（compiled RZ/shot）と解析shot Nによる保存Gであり、全energy/QPE/RPEやFT費用ではない。経験的allowance、有限価格標本に条件付く。nは古典的trajectory数、Nは解析shot数で、cos/sinは同じtrajectoryのpaired軸である。

## H4：R16/R64/R128の限定比較

主ε=0.0005の独立確認値を以下に転記する。両軸単価は元保存値が同じであり、両軸同価を新たに仮定して会計したものではない。

|R/K・candidate ID|cohort / n|N_cos / N_sin|C_cos = C_sin|G ± paired SE|固定R64からの保存変化・診断|
|---|---|---|---|---|---|
|16/2 `f85eb7b56ba5131d1eb1`|BOUNDARY_20261027_CONFIRMATION / 745|145036571 / 114372825|34281.51946308725|8892948257881.707 ± 3422683197.6427016|+19.98337% / HIGHER|
|64/2 `45f9fc0a53c1a37c7345`|SEVEN_20261027_CONFIRMATION / 800|101917842 / 84750408|39705.8275|7411817334226.875 ± 8660467438.08822|固定対照|
|128/2 `3a55c1dac3947ae23c76`|BOUNDARY_20261027_CONFIRMATION / 1185|96764648 / 80505864|46492.60675105485|8241768202974.15 ± 10362735296.841389|+11.19767% / HIGHER|

出所：[公開RESULTS.csv][RESULTS]のline31、56、26（headerをline1とする）。変化率は[公開summary][SUMMARY]の`limited_comparisons`からの引用であり、今回新しく計算していない。

R16は単価が下がる一方、保存logB=0.22022101070674038とNが増え、単価節約ではshot増加を補えなかった。R128は保存logB=0.02571967364327578とNが下がる一方、単価増を補えなかった。R64の保存logBは0.05158697006970798。normalizationだけからshotが決まるとはせず、保存bias・u・headroomを含むNを使用した元会計の関係として説明する。

この固定条件では、Rを減らしても増やしてもGが悪化する実例が得られた。**R64の全Rに対する大域最適性、別分子・別Tへの一般化、単価だけによる説明は支持しない。** R32/K2 `d9cb4e923530349e940a` は探索n9、保存G=7498501185789.556、SE=55880659384.189064（line36）のみである。R32対R64の母平均順位を独立確認したとは書かない。

## H6：下限方向R11/R16とINCONCLUSIVE

同じ主ε・prefix7/q1/K6で、以下の保存独立確認を接続する。

|R・candidate ID|cohort / n|N_cos / N_sin|C_cos = C_sin|G ± paired SE|
|---|---|---|---|---|
|11 `0e0e9fb6da33cb6ab2ab`|BOUNDARY_20261027_CONFIRMATION / 229|82807345 / 113255472|176731.54148471614|34650483876245.812 ± 8617501252.260885|
|16 `efbb07c848b303a6abab`|CONFIRMATION / 33（旧保存）|81429322 / 111370753|180032.18181818182|34710218156959.09 ± 31946367216.00101|

[RESULTS.csv][RESULTS] line21/46、[summary][SUMMARY]の保存変化は−0.17209%。新旧双方のSEを含む±2 combined-SE診断は **INCONCLUSIVE**。点Gが低いことと、母平均優位・確定的改善は分ける。旧R16のSEは新候補側だけを精密化しても消えない。探索R11/R16は別cohortで保存し、Cとpoolしない。

R11は既存11-piece allocationで各pieceの最低1 stepを満たす実装上の最小値である。[Phase Aの固定計画][PHASE]に接続し、R8を同じ科学契約で実行可能とは扱わない。K拡張、R16の追加測定、他の探索は今回の作業範囲に含めない。

±2 combined-SEは保存された正規近似感度診断であり、厳密CI、多重比較保証、無条件母平均順位ではない。未観測Taylor次数の条件付き費用・tail boundはunknown/nullのまま保持する。[paired_statistics_and_rare.json][RARE]の記述をゼロ費用へ置換しない。

## H8：再利用できる値と成立していない会計

H8は別のS2研究で、linear1Å/STO-3G/tol-only1e-8/rank28、T=.8、prefix7/q16/r4/K4、δ=.05、P=0。m5のR_outerとは異なる。

|保存範囲|既存結果|今回の扱い|
|---|---|---|
|q16・ε=.001|探索n8、G_exact=40802612537883/2、SE=50209504581.336464|既存探索値として引用可。独立確認へ昇格しない。|
|q16・ε=.0005|独立n48、G_exact=546493203592571/6、SE=144211830223.71762|この精度の既存独立証拠として維持。同じ標本を新証拠として加算しない。|
|n48価格によるε=.001会計|正確なA_signal原記録、両軸N/bias/u/logB/margin/headroomが未回収|新G/SE会計は未成立。q8対q16の新しい独立確認比較も未成立。|

既存値は[Phase A監査][H8AUDIT]と原表（CSVの固定row selector）に対応する。今回の状態は`SOURCE_OR_DATA_MISSING_STOP`。欠測は誤りの証明、費用0、不適格を意味しない。Nの逆算、.0005からのスケール、signal復元、sampling、compileを実施していない。旧[希少RTE研究の正式終了方針][H8STOP]とSTOPを維持する。「新q16確認で改善した」とは記載しない。

## 既存長時間研究との接続

|証拠集合|保存結果|今回変更しない範囲|
|---|---|---|
|[元H4/H6長時間研究][LONG]|40セル中33セルで取得済みPR探索点Gが最良PF未満|原best/accounting/独立59比較、正式採択証拠、原図表|
|[追加7候補の拡張探索集合][SEVEN]|33/40、独立確認は探索順位へpoolしない|元33/40とは別snapshot・別集合として引用|
|[今回のT2限定集合][CELLS]|探索E8/10、独立確認を含む限定C8/10（公開REPORT記載）|40セル全体の再ランキングではない。今回再集計していない。|

来歴は元結果`155529e2…`→[未取得価格スクリーニング][SCREEN]`2e518b86…`→7候補`bb97a93d…`→[境界調査][BOUNDARY]`db1b5bdf…`→Phase A`a4373f51…`→公開境界結果`202d25eb…`と接続する。登録域境界・適格域境界・価格取得域境界を同じものとせず、探索proxyを測定単価や保証下界としない。

DF-native B3の未採択結果は含めない。Pauli-JWのevent数比較や固定H4 control/compiler対照は既存統合資料に保持するが、このDF-nativeのGをPauliの順位へ移さない。旧正式採択、元独立確認、7候補E/C、今回E/Cのcohortを統合標本にしない。

## 論文への配置案

既存のSection4（Long-Time and Higher-Order PF/PR）のパラメータ感度とDiscussionへ短く接続する。MainのFigure3やTable3は差し替えず、SupplementaryのR_outer境界比較表を第一案とする。既存S4（r/K）、S6（C/N/G）、S7（不確かさ）から参照できる。R32は探索n9という脚注を必須とし、H6はcaptionにもINCONCLUSIVEを残す。

表は上記保存C/N/Gを転記するだけで作成可能。caption・索引追加だけでも目的を満たす。新しい数値グラフ、比の再集計、SE計算、fit、renderは実施しない。将来、新旧cohortを一枚の数値図へ再集計するなら別の数値処理として範囲を定める。一論文／二論文、章立て、中心claim、Main/Supplementaryの正式採択は変更していない。

## 実行来歴・静的検査・公開範囲

公開summaryは旧科学実行を2808 paired trajectories、5616軸・compile、201454 RTE events、worker回収済み、controller/finalizer exit0と報告する。これは過去の実行記録の引用で、今回の実行件数ではない。完全実行SOURCE `ad1286021bdc0e5a0b1760064cefb3c1168b24eb` と会計SOURCE `0f2d167f9c928ba94bfad1a167bf4d3f5964ad1a` は公開summaryにある識別子として記載する。完全版・private rawはローカル保全されており、**今回完全SOURCE/rawを再監査・公開していない**。内部絶対パス、非公開seed資料、private raw、共有環境内部情報は新しい公開差分へ転記しない。

既存の`adopt_h46_standard_y4_confirmation`の固定Git/CSV/JSONローダー・append処理と、`integrate_h468_paper_evidence`のCSV/JSON出力を再利用した。JSON小数は文字列で読み、保存値に算術演算を適用しない。旧索引の`--check`は旧branchの許容差分へ固定されているため今回そのまま実行せず、旧機械可読表のbyte不変を今回の検査で確認する。科学runnerはimport・実行しない。

新しい静的検査の再実行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/integrate_pr_boundary_documents.py --check
```

保存値・candidate/cohort・15証拠行・元43行の完全保持・7追加ID・公開15資料のGit blob対応・固定参照・文書リンク・元採択/結果/図表の保全・公開差分の内部情報非混入を確認する。旧データ体系の機械可読表は再生成しない。検査結果は[STATIC_VALIDATION.json](STATIC_VALIDATION.json)を参照する。技術PASSを独立科学レビューや新しい科学的保証と呼ばない。

公開対象は今回の文書差分だけとし、既存の科学結果は固定リンクから参照する。公開結果や完全SOURCEを再pushしない。初回の一括公開用読込は内部metadata混入のおそれを理由に自動審査で停止したため、既存入口・manifest9ファイルもGitHubの固定公開blobへ戻って追加照合した。原文を公開済みbyteとして区別し、新規差分の内部情報非混入を確認する。専用branchの公開とremote SHA/bytes照合後、**mandatory STOP**。次にユーザー判断が必要なのは掲載配置と中心claimの採否であり、今回の文書統合のための追加科学計算は不要。

[RESULTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/RESULTS.csv
[SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/EXECUTION_SUMMARY.json
[RARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/results/paired_statistics_and_rare.json
[CELLS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/results/limited_cells.csv
[PHASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4373f51d7acb5d70fda894b4ad66f8bc781177d/analysis/pr_boundary_phase_a_20261027/H46_BOUNDARY_EXECUTION_PLAN.md
[H8AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4373f51d7acb5d70fda894b4ad66f8bc781177d/analysis/pr_boundary_phase_a_20261027/H8_Q16_REUSE_AUDIT.md
[H8STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[SEVEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/REPORT.md
[SCREEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e518b86e7c052dafb20dece0de7ffa6c7c85f8f/analysis/h46_unpriced_screening_20261026/REPORT.md
[BOUNDARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1b5bdf96c05aa9ff23e9f843c2d3101496c8ed/analysis/pr_parameter_boundaries_20261027/REPORT.md
