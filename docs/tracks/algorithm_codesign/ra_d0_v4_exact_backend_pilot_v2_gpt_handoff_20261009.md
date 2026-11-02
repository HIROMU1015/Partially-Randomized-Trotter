# RA-D0 v4 Exact Backend Pilot v2：GPT handoff

2026-10-09 JST。Phase A **`GUARD_V2_PASS`**。
Phase B監査判定 **`V4_EXACT_BACKEND_PARTIAL`**。**mandatory STOP。**
branch：`track-b-ra-d0-v4-exact-backend-pilot-v2-20261009`。
基点：`9e6d37fe6e345b402ba507db75f77dbec0855199`。
[全文report](ra_d0_v4_exact_backend_pilot_v2_20261009.md)を固定commitで確認してください。

## 今回閉じたこと

guardの二重PID計上を修正した。
人工14要件を含む211 checksは211 PASS / 0 FAIL。
orphan/killed grandchildのCPU回収、process残存なし、wall/output/RSS、対象外processの除外、retry拒否を確認。
RSSは`UNIQUE_PID_RSS_SUM_CONSERVATIVE`。共有pageとsample間peakの限界を保持する。
CPUはsubreaper＋wait4によるchild scopeで、controller/supervisor自身は含まない。
outputにはcompiler TMPDIR、binary、全新規artifact/logを含め、input除外scopeを実行前固定した。

SoPlex 7.0.0／GMP 6.2.1／Boost 1.74.0をprivate read-onlyで再利用し、harnessを一回だけbuildした。
compile wall19.312 s、child CPU19.269 s、観測RSS合算1,053.109 MiB、output peak30,616,216 bytes。
cap変更、system/Python環境変更、再compileは0。

rational I/Oはecho-onlyで最初にPASSし、16 LPのcoefficient echoも一致した。
23予定のうち16 LPを求解、九optimalのprimal/dualと六infeasibleのFarkasを独立Fractionで認証した。
dual gapは九件全てexact zero。finite upper correctionとdecimal/dyadic gapのFarkasも認証した。
fixed sign=`nu=-raw_A, u=-raw_H`、orientation探索／reconstructionは0。

## 止まった理由と分類の注意

16件目`100_digit_infeasible`（gap `10^-99`）でSoPlex `ERROR (-15)`。
exact input echoは一致したが、primal/dual/Farkasは取得できなかった。
stderrは空で、内部原因はログだけでは特定できない。resource/guard failureではない。
残りmixed-scales一件とB2/B3-shaped六件は**NOT_RUN**。

raw controllerはverifierがERRORを拒否した場合も`V4_EXACT_BACKEND_CERTIFICATE_FAIL`と記録した。
これは分類実装の不具合。raw output/sourceはそのまま保護し、
利用者指示§14 Dと§15 PARTIALに従った**監査判定を`V4_EXACT_BACKEND_PARTIAL`として別記**する。
不正な数学的証明が返ったとは主張しない。現在のrunを修正・継続していない。

## 次にGPTで判断してほしいこと

**現時点のproduction backend採用は保留。**
小型exact LPの機能は実証できたが、必須extreme-gap fixtureを通らず、RA-shaped LPも未実証である。
そのfixtureの必要性・backend scope・追加検証の情報価値・採用方針をGPT側で判断してください。
Codex側でfixture削除、cap/tolerance/solver調整、別backend切替、追加solveをしない。
旧55,275 main／110,550 total callsはv4へ継承せず、総時間も外挿しない。
certified最大variablesは5、B2 28／B3 22 variablesは未実行である。
内部solve range約0.000052–0.000432 s、Fraction verification約0.000085–0.000428 sは小型prefixの記録に限る。

solver calls16、compile1、build/solver retries0。
181 protected paths、前回private source/libraryのidentity、v2 frozen source/contractは不変。
旧研究・前pilot分類、過去STOPを維持。registered LP、production source変更、science/synthesis、
GPU/DF/molecule/NPZ/circuit/quantum matrix/trajectory、IS/CTS、new angle/precision/authorization、旧marker変更は全て0。
**公開後mandatory STOP。新しい方針と実装範囲はGPT側へ戻す。**

## 証拠入口

- [全証拠manifest](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/evidence_manifest_v1.json)
- [181 protected identities](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/input_identity_v1.json)
- [guard contract / freeze](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/guard_v2_contract_v1.json)
- [全211 checks](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/guard_v2_test_results_v1.json)
- [backend source identity](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/backend_identity_v1.json)
- [Phase B実行前contract](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/execution_contract_v1.json)
- [rational I/O](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/rational_io_v1.json)
- [primal](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/primal_audit_v1.json)
- [dual](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/dual_audit_v1.json)
- [Farkas](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/farkas_audit_v1.json)
- [16実行＋7 NOT_RUN](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/synthetic_results_v1.json)
- [raw terminalを保持したoutput](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/raw_phase_b_execution_output_v1.json)
- [failure分類の根拠](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/failure_semantics_v1.json)
- [resource benchmark](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09/resource_benchmark_v1.json)
- [v2 guard](../../../scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py)
- [solver非依存verifier](../../../scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/verify.py)
