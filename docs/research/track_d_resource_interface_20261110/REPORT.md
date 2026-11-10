# Track D × 元PR＋RPE：共通resource interface

2026-11-10 JST。**Strong／Referenceを共通の科学条件・価格型で切り替え、既存の総資源集計へ接続する最小adapterを追加した。固定CDF8の入口は旧結果の読取専用であり、H4のNや13-round費用を新たに計算していない。**

[INTERFACE](INTERFACE.md)／[TEST_RESULTS](TEST_RESULTS.md)／[次の単一PR条件案](NEXT_EXECUTION_PROPOSAL.md)。今回の唯一のalgorithm要素はTrack D Reference/control。前回の[横断調査](../pr_rpe_joint_integration_20261110/REPORT.md)のうちDと元PR/RPEの接続だけを使った。新A Q、Closed P3、他Trackのコード・科学結果は統合しない。前回のα共同選択案も今回実装・実行していない。

## 実装とSOURCE

- [track_d_resource_interface.py](../../../src/trottertracks/track_d_resource_interface.py)：ScientificInputs／ControlProfile／MethodPrices、SavedCDF8Study、aggregate_supplied。
- [読取専用CLI](../../../scripts/inspect_track_d_resource_interface.py)：methodとA/F/Sを選び、保存価格・保存総値を出典付きで返す。
- [人工・read-only test](../../../tests/check_track_d_resource_interface.py)：新12件。既存check_track_d_bc_totalの人工12件も再利用した。

初回SOURCE `0c4950691bcf54685ab1b62a577740f1cd00ad69`、horizon資格修正 `46b7fbbfcc3a0117c77294bf0d7e8446f56baa70`、最終技術検証SOURCE `afc3bc5106773e7b36f4cb189e697b140c20c3a1`。既存接続計画branch `codex/pr-rpe-joint-integration-plan-20261110` を継続した。旧科学SOURCEは変更していない。

## 元pipelineとの接続

| 段階 | 既存SOURCE・担当 | 今回の接続範囲 |
|---|---|---|
| PR構成／分割 | df_partial_s2.prepare_df_partial_s2、rte.make_rte_config | binding/target/horizonをmetadataとして受ける。DF/PF/RTE構成を実行しない |
| PF/RTE／元H誤差 | B1、track_d_original_h_accuracy | 既存operator誤差とstate/law資格を上流に保持。新誤差評価なし |
| 十分N／decoder | rpe_resource_accounting、B4、original_h_accuracy | 固定B4の保存shot表を参照する。一般N計算・旧uniform guardの変更なし |
| method別単価 | 保存D価格・固定A/F/S | 科学条件とControlProfileが一致する場合だけ、method別26価格を返す。新予測・compileなし |
| 総資源 | c1_stage3_accuracy.aggregate_totals | supplied evidenceの集計だけを既存関数へ委譲。今回数値検証したのは人工scalarのみ |

DFLevel5RCompiledCostProviderは実build/compileを呼ぶためimport・呼出しとも不要。rpe_short_round_optimizationは直接q≤4等の既存制約を持ち、このadapterで長q optimizerへ変わったわけではない。

## 固定CDF8とのread-only整合性

対象はLinear H4／1.00Å／STO-3G／Nα=Nβ2・元H、number-CDF8／元順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、13round q1..4096、ε=.00159 Ha／failure=.01、元H厳密ground仮定・準備費除外、既存X/Y、C1 OFF、P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/宣言phase保持。

保存SOURCE：B1 `d3913866e6ade4ab97ac02645fed91b2d5fcfa75`、B4 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`、B6b `7ffe8b281b76cef3a6fd06f43413904d6766d0ec`、総資源 `8828b298975c88a2c0d1fd57ff6ed905233840f5`。総資源の科学実行SOURCEは `b09b7e7abaa3b817c26657974a62c3bd3047af89`。有限検査器c002153aと各既存STOPも保全する。

固定条件では、B6bのideal測定同値性の宣言前提とB4の保存十分Nを参照できる。合計14,738は**保存結果の引用**であり、新規算定・最小Nの認定ではない。A/F/S×両methodの総6資源は、保存result.jsonから該当fieldを選んだ値と完全一致した。これは36個の保存scalarの対応照合で、新しいΣNμ評価ではない。

q1のevent16 cut価格、q2〜32のfinite-cohort直接価格、q≥64の条件付き外挿という資格を保持した。A/Sの学習域、Fの凍結、CI/model-bias未資格、幅9/17という保存metadataを残す。A/F/Sを再fit・再予測しない。CSWAP／χ／basis共有／step融合等を価格へ追加加減算しない。

## 成立範囲と未接続部分

科学条件・state/law/target/decoder/horizon、compiler、cut、phase、C1などが保存domain外なら、価格・N・総資源を返さず未資格にする。未知method/scenarioはエラーで、暗黙fallbackしない。actual laneへ保存ideal Nを認定しない。

aggregate_suppliedは上流で資格付けたaccuracyとshot計画を受ける**算術接続**であり、条件宣言を真と証明するものではない。未知PRの誤差・N・測定同値性・価格を自動生成しない。RZ/CX/各depth/sizeは別資源、logical qubitsはpeak。加重スカラー目的関数や新成功保証は作っていない。

人工／技術回帰は最終24 test method PASS。初回2 failureは全horizon価格資格の橋渡し不足であり、初回SOURCEとlogを保全して明示資格を追加した。初回もH4読取6件はPASS、実H4の科学再実行は0。[詳細](TEST_RESULTS.md)。

## 次の研究判断とSTOP

次に推奨する単一変更は**同じCDF8・LD2・S2/s1/K0/h=.1のr=16**（比較基準r32）。RTE event負担とattenuation/Nの交換を調べる。まず新lawの全13round誤差/N資格を評価し、成立時だけq4/cut2の一固定pathを両方式・X/Yで検証する案である。旧A/F/Sの流用や、局所fixtureから全費用優位を推定することは認めない。[必要な実行範囲](NEXT_EXECUTION_PROPOSAL.md)。

今回のH4新sampling／PF/RTE誤差計算／ground/DF／回路生成/compile/simulation／N再算定／13round再集計／価格fit／q≥64新検証／他algorithm統合／詳細科学レビューは0。原本・SOURCE・結果・grant・ledger・marker・KEEP・STOP、共有環境・他worktreeを保全する。

公開SHA/tree/bytes照合後**mandatory STOP**。次の担当は**ユーザー／GPT**：r16限定科学検証の範囲・主張水準を判断する。**Codex**は新規科学実行を開始しない。
