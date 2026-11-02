# Track B：G1限定sourceの実行前レビュー資料

2026-10-09 JST。**`G1_SOURCE_PREPARED_FOCUSED_TESTS_PASS_AWAITING_SEPARATE_EXECUTION_INSTRUCTION`**。
local focused tests **59 PASS / 0 FAIL / 0 ERROR**。本構造監査・実backend LPは未実行。

基点は結果前準備`6c91d1b17a70f773080be9315fabd10e926dae44`。
branch：`track-b-g1-source-preparation-20261009`。
利用者の続行指示は、限定audit/controllerの実装、off-domain launch tests、sourceのcommit・pushに対するもの。
今回の資料公開はpacket実行・登録RA-D0・production実装の認可ではない。

## 固定したままの契約

[結果前packet契約](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/decision_packet_contract_v1.json)、
[構造監査契約](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/structure_audit_contract_v1.json)、
[8入力manifest](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/fixture_manifest_v1.json)、
[runtime identity](../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/runtime_identity_v1.json)
を変更していない。入力、期待status、実行順、symbolic/off-domain範囲、classification、資源上限は維持する。
旧contractのPENDING/authorization=falseは当時の準備状態として保持し、このsource記録で実装の現在地を別記する。

source manifestは今回のpublication commitで固定する。自分自身のSHAをmanifestへ書く自己参照は作らず、
別の明示実行指示がその完全commit SHAを指定する。新RA-D0 science authorizationは作成しない。

## 実装

限定sourceは[scripts専用namespace](../../../scripts/tracks/algorithm_codesign/g1_decision_packet/README.md)に置いた。
`src/trotterlib`や既存RA-D0のalgorithm/solver実装は変更していない。

- [独立構造監査](../../../scripts/tracks/algorithm_codesign/g1_decision_packet/structure.py)：sourceの係数式だけをrestricted ASTで読み、一般解をexact rational functionsで導出する。
  4×4 minorの非zero sign、6境界の20 triple、頂点完全性、各degreeのmean、B2断面、precision復元、sampler・数値classの限界を記録する。
  提案頂点は独立列挙の後に照合する。rank・feasibilityの符号はpower/Bernstein coefficient certificateで扱い、数点の代入で一般保証を置き換えない。
- [exact algebra kernel](../../../scripts/tracks/algorithm_codesign/g1_decision_packet/rational_symbolic.py)：stdlib Fractionのみ。
  多項式gcd、rational-function正規化、restricted AST、RREF、minor、符号certificateを使う。sourceのimport/eval、外部symbolic/LP dependencyはない。
- [one-shot controller](../../../scripts/tracks/algorithm_codesign/g1_decision_packet/controller.py)：A→8 echo-only→8 solve/verifyの順。
  呼出し前にkeyを消費し、exclusive marker/ledger、初回失敗STOP、retry=0を接続した。
- [実行入口](../../../scripts/tracks/algorithm_codesign/run_g1_decision_packet.py)：read-only source確認と、別指示後のrunを分離。
  [内部audit入口](../../../scripts/tracks/algorithm_codesign/audit_g1_structure.py)は固定state・source/contract-bound permit・active guard parent・exclusive stage ledgerを要求する。
  単独起動で本監査へ進まない。

非負係数のprecision復元では、zero groupを除算しない。coef convex decompositionと抽出確率を分ける。
exact ideal matchingを数値許容幅K3や別々のLRM decode像へ拡張せず、J1–J3を元B2へ追加しない。
このsourceの存在は6頂点主張が通過した証拠ではない。実際のA判定は将来の一回だけの実行で取得する。

## launchと失敗分類

HEAD/remote完全SHA一致、独立branch、clean worktree、全固定source/input/protected/runtime hashes、
未消費marker、既存resultなしを確認する。実行にはsourceに対応した明示指示の原文を別ファイルで渡し、
本文とhashをmarkerへ保存する。契約やauthorizationを結果後に書き換えない。

ERROR/unknown・exact payload未取得は`G1_BACKEND_ACQUISITION_INCONCLUSIVE`。
non-JSON、readback不一致、malformed payloadはtechnical。整形式で取得した証明の数式違反だけを
`G1_BACKEND_INVALID_CERTIFICATE`とする。正しいboundsでgapが非zeroなら取得未完で、不正証明とは呼ばない。

既存verifierは拒否時にexit 1を返すため、guardの`PROCESS_EXIT_FAILURE`だけで判断せず、
整形式の`PASS=false` verdictを照合する。旧guard STOP/ledgerのraw値は保存し、再呼出ししない。
残存process、resource failure、verifier自身の出力不備は別に扱う。

global wall 1,200秒、A 60秒/256 MiB、backend各30秒/1,536 MiB、output 64 MiB、single solver、retry=0は固定のまま。
guardは旧v2のbyte不変版を利用する。既存binaryをread-onlyで使い、新build/compile/installは0。
exceptional supervisor cleanupはsubreaperで対象grandchildを回収し、起動前から存在したchildを除外する。
CPUは旧wait4 child scopeの測定で、追加CPU hard capを導入したとは主張しない。
RSS共有page・sample間peak・outer controller除外・output samplingの限界を保持する。

全outcomeでSTOP。STOP後はsource/runtime/outputのread-only確認と証拠保存だけを行う。
全8証明が揃う前にprefixから全体PASS、backend採用、B3>B2、科学的negativeを判断しない。
完了後はGPT G1へ戻し、source修正pilot、production、登録optimization、新baselineへ自動進行しない。

## focused testsの範囲

[59 tests](../../../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)は人工text・generic polynomial・
人工2変数LPのhand-specified mock primal/dual/Farkasと模擬backendを使った。
固定8入力をbackendへ渡さず、本`audit()`はtest全体で呼出し禁止のpatchを入れた。

確認したのはexact kernelの一般的操作、source AST parser、proof payload分類、finite-box correction、
echo-before-solve、prefix failure、duplicate key拒否、no-retry、source/remote/dirty/runtime gate、
実際の旧guardによる模擬processのtimeout/exit1/non-JSON、grandchild回収と既存sibling保護である。
CLIのhelp、指示なしrun拒否、standalone auditの拒否も確認した。

[結果](../../../artifacts/track_b_g1_source_preparation/2026-10-09/focused_test_results_v1.json) /
[実行とsource hashes](../../../artifacts/track_b_g1_source_preparation/2026-10-09/focused_test_execution_v1.json) /
[全test log](../../../artifacts/track_b_g1_source_preparation/2026-10-09/focused_test_stderr_v1.txt)。
local実行でありimmutable CI・外部再現ではない。最終commit内のcode bytesはtests時のhashと一致する。
本監査の数学判定、実backendの証明取得、資源改善は未検証のまま。

## source reviewで確認する事項

1. A01–A10の証明義務とsymbolic sign/completenessの実装が一致するか。
2. code/kernelを自己検証したことと本modelの監査結果を混同していないか。
3. payload未取得・malformed・不正証明・非zero gap・guard failureの分類と初回STOPが契約どおりか。
4. 実行順、exclusive key消費、8＋8上限、旧binary/guard/verifier identity、cap/accountingが接続されているか。
5. 旧science/contract/marker/STOPを維持し、結果後に次stageを自動認可しないか。

source確認はCodexの技術範囲。本packet結果を受けた研究判断はGPT G1の範囲である。

## 別指示後のcommand

`S`はこのsource manifestとコードを含む公開commitの完全SHA。
read-only source確認ではaudit/backendを呼ばない。

```bash
/usr/bin/python3 -B scripts/tracks/algorithm_codesign/run_g1_decision_packet.py --source-commit <FULL_S> --verify-source
```

実行はまだ認可していない。後の指示原文には次のsource-bound statementを含め、原文を`/tmp`等の別ファイルへ保存する。

```
source `<FULL_S>` のG1固定契約で、decision packetを一回だけ実行し、終了後はmandatory STOPしてください。
```

その別指示を受けた後だけ、以下を一回呼び出す。

```bash
/usr/bin/python3 -B scripts/tracks/algorithm_codesign/run_g1_decision_packet.py --source-commit <FULL_S> --execute-one-shot --instruction-file <EXPLICIT_INSTRUCTION_FILE>
```

これは指示原文のscope/identity照合であり、研究採択や自動approval判断の仕組みではない。
marker消費後はtechnical failureでも再実行しない。

[source manifest](../../../artifacts/track_b_g1_source_preparation/2026-10-09/source_manifest_v1.json) /
[公開証拠manifest](../../../artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)。
**今回は本監査=0、実LP/echo=0、compile=0、synthesis/science/GPU=0。資料公開後STOP。**
