# C1最終oracle・selector診断：実行承認とGPT判断への資料 v1

2026-10-14 JST。基準 `cc992b579290233ab5a7d8408d4a81c6d295f36d`。
[準備・検査](qc_c1_saved_oracle_preparation_v1_20261014.md)、[固定契約](qc_c1_saved_oracle_diagnostic_contract_v1_20261014.md)、
[execution plan](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/execution_plan.json)、[未発行承認対象](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/approval_target.json)。
登録データの新診断は未実行・未認可。今回のCodex担当は実装・synthetic・schema-only検査・公開まで。
source固定・preflight完了の具体的identityは公開後の追記とplanで参照する。

## 得る情報・得ない情報

C_M=24macroの保存eligible最小費用、C_B=strong既存対照(single/subdivision/S4/C2)、C_K=same C1 catalog(single/subdivision＋macro)。
macro_ratio=C_M/C_B、method_regret=C_saved_selector/C_K−1。
method内hが既にoracleであるため、運用precision選択までのregretとしない。
既存3selectorの選択を変更しない。保存eligible/cost/precision/nullを保持する。

| 後日の診断結果 | 分かること | 科学判断に残ること |
|---|---|---|
| oracleにも5%優位なし | 現finite catalog/grid/modelではselectorだけの改良で登録利益に届かない | C1一般の不能性・正式終了は証明しない |
| oracle優位、selector miss | catalog潜在性能と選択情報/規則の障害を分離できる | 真値なし選択、測定/情報費用、confidenceを含む運用利益 |
| oracle優位でもknown compositionと同じ | 既知templateの適用tradeoffになり得る | 独立新規性・新生成算法の認定にはならない |
| eligible不足/境界/情報不足 | 未判定理由と対象部分を保存できる | threshold/grid変更や追加pilotの自動承認にならない |

分類の第2と第3は重複し得る。第4はunresolved_partsで一部指標だけ未判定の場合も区別する。
一ULP guardは保存costの丸め感度のみ。科学的不確かさの保証区間ではない。

元pilotは非分子2qubit Pauli4H/39unitary methods/6h,h=1/(8Λ2^s),s0…5、target10^-5/10^-6 model units/価格3,6。
geometry/basis/DF rank-policy/L_D/固定粒子sectorなし。通常16比較の3selectorで登録改善0は元保存事実。
今回は全24macroのoracleをまだ取得していないので、過去のselected未達をcatalog全体の不能性と読み替えない。

## 技術的な準備と固定上限

人工F_COST費用表の34技術検査PASS。実Hamiltonian/比較データをsyntheticテストへ読み込まず、量子solver/PF/compileはzero。
独立checkerはevaluatorをimportせず再構成。元science source/結果/認可/STOPを変更しない。
source16files/input25files/runtime/planをSHAで固定し、登録データはschema-only preflightに限定する。

将来の許可範囲は最大8064rows/32context（通常16＋bad16）、1worker/1thread、memory256MiB/output2MiB/log1MiB、eig/PF/compile calls0、attempt1/retry0。
入力条件・24macro・6h・価格・精度・5%gateを変更しない。wall capなし。
結果を取得する場合は、新しい明示posthoc承認をsource/input/planのidentityへ結び付ける。940-call承認は流用しない。

## 次の境界

Codex実行前準備の公開・remote取得一致を完了したらMandatory STOP。
次に実際の保存診断を行うかは別明示実行承認、研究上の必要性・採否を判断するGPTレビューは別開始承認。
constructor同値性、真値なし選択、情報費用、分子移送、full QPE/RPE費用、独立新規性は診断成功でも未解決。
C1継続/終了/再設計/C4/主研究採択をCodexは代行していない。
この準備後、同C1の追加数学探索・一般的先行研究調査を繰り返さない。
N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、旧global FAIL、既存科学source/結果/認可/STOPを保持する。

## Source固定・schema-only preflight完了

source commit `657513ad06661aed84950d5907f3304dcdd5f1f7`、新source16files/固定入力identity25filesを固定した。[preflight](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/preflight.json)は32contexts/8064rowsのschema/pointerのみPASS、新しいmin/ratio/regret/eligibility/PF/cost評価0。[resource記録](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/preflight_resource_use.txt)は認可前検査の実測で、診断性能benchmarkではない。
[実entry認可拒否](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/authorization_guard_checks.json)は承認なし・未発行targetを入力処理/attempt作成前に拒否した。旧940-call認可は人工検査でも拒否。
[準備完了と次の承認境界](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/launch_readiness.json)、[具体的identityの承認対象](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/approval_target.json)。承認は未発行、launch_allowed=false。実登録診断は一度も実行していない。

## 公開・remote再取得確認

payload `d910bc78dc10ab0b804e400c2cb613be66ab2972` とsource `657513ad06661aed84950d5907f3304dcdd5f1f7` をGitHubから独立bare cacheへ取得した。40変更blobs/16固定source/25input identity/307既存files/追加69relative linksがbyte/SHA一致。[取得記録](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/publication/receipt.json)。旧138manifest records/non-result fields/global FAILとroot既存131filesを保持。
これは取得一致であり、外部科学再現や実登録診断ではない。source/入力/planは固定、承認未発行・実集計0。今回の準備を完了し、別明示posthoc実行承認またはGPT科学判断開始承認待ちでMandatory STOP。
