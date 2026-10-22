# H2 PF次数逆転C：production runnerと科学実行契約

**技術実装・人工検証完了。科学実行は未承認・未実行。実行契約のbindingはBLOCKED。mandatory STOP。**

固定H2・ε_sig=0.0005のFull S2/q4、Full Y4/q2、PR S2/q4、PR Y4/q2だけを扱う。linear H2、隣接1 Å、STO-3G、保存tol-only1e-8/rank3、T=.8、α_axis=.025、P=0、ordinary signed bias、system4/ancilla4、symmetric_directional、cos/sin full wrapper、compiled RZを固定した。q再探索・H4/H6・全候補再ランキングは含まない。

## A. 実装と検証

新入口 `scripts/run_h2_common_production.py` から、明示承認・正式政策・封印済みcontract・未使用grant・exact budgetをすべて検査してproduction backendへ接続する。旧 `prepare_h2_common_native.py --execute` の拒否は変更していない。

固定production SOURCEは [`59e5ddddb9306e9c076cb3d3e34e15264cc3192d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/59e5ddddb9306e9c076cb3d3e34e15264cc3192d)。科学kernel取得SOURCE 4ec26921…、provider c31ef424…、旧provider 217c971d…、旧mock SOURCE bcc05ef0…、前段資料 b18a9ce5…とは役割を区別する。

production backendは保存snapshotの読取・配列receipt照合、DFHamiltonian/state/basis確認、固定OccupationOracle、共通QR preparation、signed-piece compose、Full deterministic/PR finite-RTE、cos/sin wrapper、guarded cost compile、statevector correctness、reference再結合、診断・経験的allowance・shot会計・条件付き順位出力を接続済み。これらの**実科学入力での実行確認はしていない**。今回はcontrol planeのmock/AST検証である。

新productionテスト36件と旧mock回帰36件、計72件PASS。未承認dispatch、不正・使用済み・scope違いgrant、SOURCE/input/runtime不一致、call予算不足、one-shot予約、marker変更、partial結果の誤PASS、直接source loaderと通常importの封印検査、旧execute拒否を確認した。[ログ](technical_tests.txt)、[機械可読結果](test_results.json)。Codex自身の技術チェックであり、独立科学レビューではない。

静的確認でscheduleがprovider c31ef424には存在しないと判明したため、固定acquisitionの `h46_m5_native.c.core` に接続した。旧・新providerの `h6_matched_accounting_v1.py` はGit blob一致を確認し、元allowanceの演算順を再利用する。科学codeをimportして確認したものではない。

## B. 接続仕様と正確なcall計画

[実装境界](IMPLEMENTATION.md)、[callの計数単位と分岐](CALL_CONTRACT.md)、[exact予算](call_recipes.json)、[contractテンプレート](contract_TEMPLATE_BLOCKED.json)を参照する。

| 主な計数単位 | DIAGNOSTICS | COMPARE_REUSE_PR | PR_TWO_NATIVE_CONTROLS（任意） |
|---|---:|---:|---:|
| bootstrap / setup / prepare | 1 / 1 / 2 | 1 / 1 / 2 | 1 / 1 / 2 |
| diagnose / 内部path_signal | 4 / 8 | 4 / 8 | 0 / 0 |
| reference_rebind（保存scalarとの新bias） | 4 | 4 | 0 |
| native_validation / compose | 0 / 0 | 2 / 2（Full） | 2 / 2（PR） |
| piece lowering | 0 | 10 | 10 |
| wrapper / compile / 内部transpile | 0 / 0 / 0 | 4 / 4 / 4 | 4 / 4 / 4 |
| circuit statevector simulation | 0 | 12 | 12 |
| request_factory / 登録RTE event | 0 / 0 | 0 / 0 | 10 / 192 |
| occupation trajectory | 0 | 0 | 2 |
| allowance / accounting / ranking | 0 / 0 / 0 | 4 / 4 / 1 | 0 / 0 / 0 |
| 新full-H reference / quantum shots / q探索 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

内部callを併記しており、列の和を独立実験数と解釈しない。8 path評価は4候補×primary/independent。4 compileはFull2候補×2軸だけ。診断分岐のsampling/compileは0で、PR67 taskを再取得しない。PR任意分岐の10 requestはq4×1＋q2×3、192 eventはq4×32＋q2×32であり、numpy RNG variateの個数ではない。任意分岐は保存seedを用いる再構成でもfactoryを呼ぶためsampling callとして明示承認を要する。

共通referenceは旧FP scalarとcombined empirical diagnosticの再利用案。数値的一致・厳密誤差上界は認定しない。reference再結合は将来の科学callとして独立予約する。新referenceが必要になれば、この3分岐を流用せず定義・必要件数を別契約にする。

Gaussianはadjacent complex QR、Full prefix3/PR prefix1、Y4正負正、signed時間と浮動小数演算順、scalar最後一回を保持。両方式のcompilerへ同じphase guardを適用する。native/occupation/wrapper誤差strict <1e-9、compiled保存state vector差strict <1e-7、phase alignment/state fitなしを提案する。全state operator証明・shot allowanceには転用しない。

新診断はprimary spectral pathと独立matrix Taylor pathで、実際のactionと内部stageを記録する。rawはattenuated vectorで、normと両path差は明示したlog_Bでcorrected単位へ変換する。closureは `|exp(log_B)*raw_signal-corrected_signal|`。旧endpoint normやstage proxyで代用しない。新instrumentationのwork/trace定義は政策採択対象に含む。新u/marginを既存実データで計算したことはない。

保存snapshotとreceiptの**byte hash**を既知private locatorで照合できた。[入力照合](input_byte_checks.json)。配列のロードも原本の再公開もしていない。保存H/state/snapshot bindingはcontractで固定する。[SOURCE索引](SOURCE_INDEX.md)。

## C. 残存BLOCKERと判断対象

[BLOCKER一覧](blockers.json)、[保存PR Cの具体的照合状況](PR_C_reuse_assessment.json)を参照する。

1. ユーザーによるnative、共通適応allowance、旧reference再利用・経験的uncertaintyの正式採択がない。instrumentationの計数単位とu/headroom≤.01の会計guardも含めて採択対象である。
2. 新runnerで実際に使うsource/runtime/import closureの完全な新sealとcanonical path bindingが未確定。旧runtimeのliblzma不一致は未解消の観測として保持し、影響なしとも旧結果無効とも認定しない。旧環境の無関係な全file一致を要求しない。
3. COMPARE_REUSE_PRには2候補の完全なC identity witnessが不足する。既存67 task・134軸のH/state、prep、compiler hash、event digest、cohortを静的に対応できるが、公開events欄はN2/N4/stratum要約でfull event列ではない。new operation-stream、same rotation、使用runtime、scalar/wrapper/loweringとaccepted correctnessの結合を要する。policy名を改名するだけでは再利用不可。
4. PR任意分岐を選ぶ場合だけ、pieceごとの保存seedのsealとその範囲の科学承認が必要。これは診断の前提でも67 task C再利用の一括証明でもない。

テンプレートには固定SOURCE、4候補、物理条件、exact予算を記録した。未確定path/runtime/proofを空欄で保持し、preflightは拒否する。**null件数を推測で置換した契約ではなく、実装分岐ごとの件数は確定、実行bindingは未完という状態**である。今回live grantの発行・予約・消費は0。

最小解消順は、政策判断→使用runtime/input closureの技術封印→DIAGNOSTICSの限定承認→新診断とprep receiptの確認→PR C identityを判定→条件が揃ったCOMPARE_REUSE_PRの承認である。必要な入力・witnessが最初から揃えば、比較分岐を一括承認し診断・Full correctness・会計まで実行可能。途中のgate失敗時は予約を返金せずSTOPする。

PR Cのidentityが異なる場合は、既存event・影響軸を列挙した新契約を作る。全67 task再sampling/compileを自動計画しない。Full旧Cは新common nativeへ自動流用しない。

## D. 保全・公開・STOP

専用branch `codex/h2-common-production-20261023`。GitHub接続APIで同一treeを公開し、公開SOURCEとcontract参照を一致させる。最終HEADのremote一致と主要file取得は公開後に確認する。

変更は新規code・新namespace・新文書と4索引への追記だけ。旧科学SOURCE、raw、grant、ledger、marker、STOP、前段mock実装・前段監査は不変。実H2 build/compile/simulation、signal/reference/sampling、新u/N/G評価、科学比較CのGO、詳細科学レビューはいずれも0。

**mandatory STOP。** 次の担当はユーザー（政策と最小科学範囲の判断）。その後のCodexによるbinding完成・科学実行は別指示/承認の範囲であり、自動開始しない。
