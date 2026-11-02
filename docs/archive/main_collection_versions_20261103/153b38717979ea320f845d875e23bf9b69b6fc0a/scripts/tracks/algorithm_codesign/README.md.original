> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](../../../docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](../../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](verify_ra_d0_source_review_v3.py) / [v3 tests](../../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](../../../docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](../../../docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../../../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](run_ra_d0_one_shot.py) / [focused verifier](verify_ra_d0_source_review_v2.py) / [v2 tests](../../../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](../../../docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](../../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](../../../docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](../../../docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](../../../docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](../../../docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](analyze_r1p5_saved_attribution.py)、[全summary](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](../../../docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](../../../docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](../../../docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# Track B scripts

## R1 v2 source preparation

[run_r1_rte_reallocation.py](run_r1_rte_reallocation.py)：static plan / separately authorized native-resource run。
[preregistration](../../../docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)、
[source review](../../../docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
plan126 keys、27 focused tests。registered synthesis/cost取得0、R1未認可、mandatory STOP。


## R0.5 technical verification only

[check_rte_reallocation_r05_symbolic.py](check_rte_reallocation_r05_symbolic.py)：
Aを固定したexact equivalence/norm/support比較。自由wordとPauli fixtureを区別する。
[事前protocol](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/result_prior_protocol_v1.json)、
[readout](../../../docs/tracks/algorithm_codesign/rte_reallocation_r05_symbolic_comparison_readout_v1.md)。
一回実施済み、R1 science runnerではない。mandatory STOP。


## R0 fixed exact symbolic checks

[check_rte_reallocation_symbolic.py](check_rte_reallocation_symbolic.py): standalone stdlib Fraction/free-word checker, A80/B9 fixtures and negative witnesses, 15s CPU/256 MiB AS/30s wall cap. [R0 packet](../../../docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md). No sampling/science/circuit operations.


[BS-0.5設計監査](../../../docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)はdocs-only。
新runner/source/testsなし、old science replayなし、mandatory STOP。次scopeの判断はGPTへ戻す。

[SP-1後block合成仕様](../../../docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)はAPI設計と未承認pilot案のみ。
新script/科学runner/testsは作成・実行していない。既存SP-1/0.5はconsumed、追加run未認可。
RUN_READY=false、mandatory STOP、具体domain/辞書/対照/会計はGPT review待ち。以下は既存履歴。

SP-1の[一回結果とGPT handoff](../../../docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)を公開。
science run1／retry0、mandatory STOP、marker consumed。旧runnerは再実行しない。
[audit_sp1_saved_result.py](audit_sp1_saved_result.py)はstdlibだけで保存identity/field/predicateを照合する。
matrix/guard/coefficients/Bernstein/resourceを再計算せず、source/runnerのimport・新合成は0。
[保存audit／evidence manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)、PASS。
追加scienceは未認可。以下のpending/preparation記述は結果前履歴として保持する。

[SP-1 runner](run_sp1_wrapper_pilot.py)は[採用契約](../../../docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)用。
`plan --source-commit <full S>`はsource-bound静的ledgerだけで、coefficient/resource/signal採点を行わない。
`run`は別review後の新S→direct authorization-only Aと明示指示を要求する。現在はpendingで拒否する。
登録science sweep0、59 focused testsはlocal pass。全結果STOP、次stage自動認可なし。
旧SP-0.5一回markerを再利用しない。

[SP-0.5 one-shot結果・GPT handoff](../../../docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)は
PRIMITIVE_TRADEOFF_EXISTS、mandatory STOP。[audit_sp05_saved_result.py](audit_sp05_saved_result.py)は
sourceのalgorithmをimportせず保存fieldを照合する。既存auditは同一性を確認して保持し、未保存時だけfresh出力を作る。
23 keys／16 rowsの科学計算は完了。旧run／PAI／Jの再評価やwrapper pilotを実行しない。


[SP-0.5 runner](run_sp05_synthesis_economics.py)はplan（合成0）と、別承認後のみのrunを分離する。
[結果前source review](../../../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)、
[contract／preparation](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/)。
pygridsynth一つ、catalogue一つ、8 target／23 keys。登録計測未実行。全outcomeでSTOP、retry0。


BM-0.5の限定記号監査は[audit_bm05_symbolic_equivalence.py](audit_bm05_symbolic_equivalence.py)。
standard-library Fractionと抽象非可換wordsだけを使い、degree3までの三経路を比較する。
physical inputs、行列、Hamiltonian、state、science provider、circuitは扱わない。
stdoutのJSONが[保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json)。
本文・scopeは[BM-0.5 packet](../../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
このscriptの追加はscience runnerの実行承認ではない。

既存BF science/recovery runnerの契約・result・STOPは
[Track B index](../../../docs/tracks/algorithm_codesign/README.md)を参照する。
過去のone-shot markerやauthorizationをBMへ流用しない。

## G1 source準備（2026-10-09）

[専用module](g1_decision_packet/README.md)、[runner](run_g1_decision_packet.py)、
[guard専用audit入口](audit_g1_structure.py)、
[source review](../../../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)、
[59 focused tests](../../../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)。
本構造監査・固定8人工LPは未実行。旧v2 controllerを起動せず、旧guard/verifier/binaryをbyte不変で利用する。
別source-bound明示指示の後も一回のみ、全outcomeでSTOPしGPT G1へ戻す。


## G3有限law・known return（2026-10-09）

`g3_finite_law.py`と`g3_return_comparator.py`は各一回消費済み、再起動禁止。
`audit_g3_saved_outputs.py`は保存sequence/count/complete lawと限定winner poolのみの照合。
[結果と制約](../../../docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)。
新solver/synthesis/circuit/matrix/LP/DFをpost-auditへ混ぜず、mandatory STOP/GPT G3を維持する。


## G4 independent certificate / matched CTS

`g4_independent_certificate.py`、`g4_cts_specialization.py`、`g4_matched_cts.py`、
`audit_g4_saved_outputs.py`を追加。
[結果/GPT handoff](../../../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)、
[manifest](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/evidence_manifest_v1.json)。
exclusive A/B markersは消費済み。科学runnerは再実行不可、STOP後の保存値照合だけ。


## G5（2026-10-10）：保存証拠の終了認証

[g5_fixed_dictionary_closure.py](g5_fixed_dictionary_closure.py)：G4 rational primitives/source bindingを共有し、頂点・価格・CTS scalar verifierを独立実装。
fixed x1/4、252 profiles/756 prices、L1-charge digital classまで確認。run1/retry0、G5 marker消費済み。
[scope](../../../docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[handoff](../../../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)、
[tests](../../../tests/tracks/algorithm_codesign/test_g5_fixed_dictionary_closure.py)、
[manifest](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)。mandatory STOP。


## G6 return generator（2026-10-10）

[g6_return_generator_audit.py](g6_return_generator_audit.py)：bounded stdlib formal technical checks。
[G6 handoff](../../../docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md) §19認可範囲のみ。
一束完了、science0、旧run retry0、technical marker消費済み、mandatory STOP。


## G7 budget/control economics（2026-10-10）

[g7_budget_control_economics.py](g7_budget_control_economics.py)：G6 review §12の委譲範囲。
--source-commit FULL_SHAを要求、固定環境/contract/キー/保護hash検査、exclusive marker。
24 native-Rz keys一度のみ、2 inputs×4 arms、conditional T/query/CPU vector。retry0、mandatory STOP。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../../../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。
[audit_g7_saved_outputs.py](audit_g7_saved_outputs.py)は保存metadata/有理再集計のみ。追加合成・matrix・generator呼出し0。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../../../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
[g8_on_demand_provider_budget.py](g8_on_demand_provider_budget.py)：--source-commit FULL_SHA、fixed runtime、fresh exclusive marker。再実行不可。


## G8 after one-shot

`audit_g8_saved_outputs.py` checks saved rational budgets, sequences, source/protected hashes and scope only. The native runner has consumed its marker; do not rerun. [Results](../../../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md).


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../../../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](audit_g9_saved_outputs.py)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](g9_p5_matched_native_v2.py)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](audit_g9_v2_saved_outputs.py)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。


## G10 saved-policy audit / degree-comparison source preparation（2026-10-10）

[G10 source review / GPT入口](../../../docs/tracks/algorithm_codesign/g10_source_review_20261010.md)。保存P5 direct945 bindingsのT再計数と任意proposal固定policy下界を独立確認。
ordinary/partial/closed P3/CTSは保存closed P5と分離、general full/closed P5はこの下界では未分離。原G9分類は不変。
同p/x/3-qubit providerのm3/5/7（17 rows/34 axes）を固定。m5はsaved-only共通policy再会計、
登録P3/P7は未取得。41 off-domain focused tests PASS、固定runtime/未認可拒否を確認。
実synthesis0、G10 marker absent、authorization pending。旧source/result/auth/marker/STOP・Track Aを保持。
[evidence manifest](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/evidence_manifest_v1.json) / [contract](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json) / [future runner](g10_degree_matched_native.py) / [focused tests](../../../tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)。
次は固定source review→別authorization-only child→新one-shot指示。全結果STOP、研究判断はGPT。
