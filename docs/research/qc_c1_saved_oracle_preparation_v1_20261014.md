# C1固定pilot：保存oracle・selector診断の実行前準備 v1

2026-10-14 JST。基準 `cc992b579290233ab5a7d8408d4a81c6d295f36d`、独立branch `qc-c1-saved-oracle-diagnostic-prep-20261014`。
今回のユーザー指示は、保存データの診断方法を固定し、実装・人工fixture・認可前検査・公開まで進めるもの。
新しい数学探索・文献調査・H/eig/PF/compile/実データ新規集計・GPTレビュー・研究採否は実施しない。
[固定契約](qc_c1_saved_oracle_diagnostic_contract_v1_20261014.md)、[execution plan](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/execution_plan.json)、[GPT判断用資料](qc_c1_saved_oracle_execution_handoff_v1_20261014.md)。
source freezeのcommitとschema-only preflightは、source固定後に本資料へ追記する。登録診断の認可は発行しない。

## 目的・保存sourceの意味

現selectorの未達と、catalog全体の潜在的利益の不存在は別である。
元 `c1_macrocycle_pilot_v1.py:summaries` は全method/hのeligible、conditional_quantum_costと、選択済みmethod内のoracle-h frontierを保存している。
これらの保存fieldを新しいposthocで最小化すれば、未選択macroを含むoracleと保存selectorを比較できる。
今回そのposthocは実行せず、schema・pointerの確認だけを許可されたpreflightで行う。
[元結果](qc_c1_macrocycle_pilot_v1_results_20261013.md)、[未実行案](../../artifacts/qc_c1_residual_math_audit_v1/2026-10-14/posthoc_information_value_plan.json)を維持する。

原scope：非分子2qubit Pauli4H、24macro、6h、通常16/bad-state16context、target10^-5/10^-6 model units、価格3/6。
h=1/(8Λ2^s),s0…5、Λ=fragment係数L1。geometry/basis/DF rank-policy/L_D/固定粒子sectorなし。
branch/overlap/precision eligibilityは保存値のまま使い、threshold/grid/state/価格を増やさない。
元通常16比較の3selector改善0という既存結果と、今回の未選択oracle診断を分ける。

## 固定した指標・分類

C_Mは24macro、C_Bはsingle/subdivision/S4/C2、C_Kは同じC1 catalog(single/subdivision＋macro)の保存eligible最小費用。
macro_ratio=C_M/C_B、method_regret=C_saved_selector/C_K−1。
regretのhは選んだmethod内で既に事後oracle。真値なしのprecision選択の性能を測るものではない。
jointの既存precision choice/nullを別metadataとして残し、新しいPF boundを計算しない。

分類はcatalogとselector/known-controlの二軸。Bとknown-control同等は同時に成立し得る。

| 分類 | 結果前の固定内容 |
|---|---|
| oracleにも5%利益なし | current catalog/grid/保存eligibility/価格でmacroの最良ratioもgateを通らない |
| oracle優位、selector miss | 少なくとも一selectorが既存5%gateを通らない。selector毎にcost/regret/不適格を別記 |
| oracle優位でもknown compositionと同じ | current factoryとcontrolは同じcatalogと生成規則。独立方法差ではない |
| UNDETERMINED | eligible不足/null・丸め境界・一部selectorの未判定。unresolved_partsを保持 |

strict 5%gateは19/20。保存binary64 costをFractionへ写し、tieは保存row順・同費用pointerを全保存。
浮動小数の近傍一ULPだけでgateが跨ぐ場合は丸め感度のUNDETERMINEDとする。
このguardは物理的なcost/eig誤差の保証ではない。exact ratioとguardを両方保存する。

## 実装・独立性

| 種別 | source |
|---|---|
| scope・限界 | `src/trottertracks/representation_exploration/c1_saved_oracle_spec_v1.py` |
| schema/pointerのみ | `c1_saved_oracle_schema_v1.py` |
| metric core | `c1_saved_oracle_diagnostic_v1.py` |
| 独立metric checker | `c1_saved_oracle_checker_v1.py` |
| SHA/runtime/preflight/承認/一回消費 | `c1_saved_oracle_execution_v1.py` |
| guarded runner | `scripts/run_qc_c1_saved_oracle_diagnostic_v1.py` |
| independent audit entry | `scripts/verify_qc_c1_saved_oracle_diagnostic_v1.py` |
| synthetic検査 | `scripts/check_qc_c1_saved_oracle_preparation_v1.py`、`tests/test_qc_c1_saved_oracle_preparation_v1.py` |

checkerはevaluatorをimportせず、sorted-poolとcross multiplicationで独立に再構成する。
共有はschema/宣言scope/hash/authとstdlib。外部研究者の科学再現・immutable CIではない。
元科学source・runner・checker・数学検査は改修・再実行しない。

## 検査とperformance方針

[synthetic34検査](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/synthetic_checks.json)はF_COSTという人工費用表だけ。
5%優位/miss/known重複、strong S4/C2、same-catalog denominator、stable tie、exact/近傍gate、null/欠落/非finite/異常値、
不正ratio/pointer/tag、非最小saved-h、read-only、未承認/draft/旧認可の拒否、sourceSHA拒否、一回消費、出力上限を検査した。
実データの新oracle/regret・旧結果集約・旧数学恒等式をテストへ読み込んでいない。

source静的確認では、dominant workloadは最大8064rowsのJSON/schema/Fraction/table処理。規定1worker/1threadでserial、NumPy/GPU/Numba/量子backendなし。
既存巨大行列・並列PF経路を再起動する必要はない。schema走査とbounded context pool、checkerのcontext内sortを用いる。
syntheticのresource useは記録するが、実データ診断の速度・費用削減やspeedup factorは主張しない。

## identity・認可前検査・実行境界

source/input/runtime manifestsを固定する。input manifestは実データ4優先ファイル、元source20files、元source manifestの25ファイル。
新source16filesにはrunner/checker/schema/test/namespace/固定契約/lock・output schemaを含む。
[既存証拠307files](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/base_evidence_index.json)はbyte/SHAだけで保全し、再認証や新集計を行わない。

認可前preflightはsource/input/runtime/HIROMU1015 remote/SHA/祖先commit、32context/8064rowsのschema/pointerを確認する。
min/ratio/regret/PF bound/eligibilityを計算しない。source未固定、SHA不一致、既存attempt output、無効schemaは拒否する。

future attemptは一回/retry0。最大8064rows/32contexts、1worker/1thread、memory256MiB、output2MiB、log1MiB、eig/PF/compile calls0、wall capなし。
新しいtypeとidentityを持つユーザー明示posthoc承認が必要で、旧940-call認可は拒否する。
[approval target](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/approval_target.json)は未発行の承認対象であり、承認そのものではない。

## 維持する状態

旧global manifestのdescription欠落FAIL、138prior result entries/non-result fields、root既存131files、旧科学結果・source・認可・STOPを保持。
N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、C1 pilot認可消費済み、C4未起動。
準備・remote取得確認後Mandatory STOP。新規保存診断の別明示実行承認、またはGPT科学判断の開始承認を待つ。
同C1の数学探索・一般文献再調査を続けない。

## Source固定・schema-only preflight完了

source commit `657513ad06661aed84950d5907f3304dcdd5f1f7`、新source16files/固定入力identity25filesを固定した。[preflight](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/preflight.json)は32contexts/8064rowsのschema/pointerのみPASS、新しいmin/ratio/regret/eligibility/PF/cost評価0。[resource記録](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/preflight_resource_use.txt)は認可前検査の実測で、診断性能benchmarkではない。
[実entry認可拒否](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/authorization_guard_checks.json)は承認なし・未発行targetを入力処理/attempt作成前に拒否した。旧940-call認可は人工検査でも拒否。
[準備完了と次の承認境界](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/launch_readiness.json)、[具体的identityの承認対象](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/approval_target.json)。承認は未発行、launch_allowed=false。実登録診断は一度も実行していない。

## 公開・remote再取得確認

payload `d910bc78dc10ab0b804e400c2cb613be66ab2972` とsource `657513ad06661aed84950d5907f3304dcdd5f1f7` をGitHubから独立bare cacheへ取得した。40変更blobs/16固定source/25input identity/307既存files/追加69relative linksがbyte/SHA一致。[取得記録](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/publication/receipt.json)。旧138manifest records/non-result fields/global FAILとroot既存131filesを保持。
これは取得一致であり、外部科学再現や実登録診断ではない。source/入力/planは固定、承認未発行・実集計0。今回の準備を完了し、別明示posthoc実行承認またはGPT科学判断開始承認待ちでMandatory STOP。
