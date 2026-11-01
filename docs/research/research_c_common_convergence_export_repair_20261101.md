# 研究C：共通収束fitのJSON出力修正・保存結果からの集計復旧

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：5ff11540262dfe74446be9ee089b4b0a8e136c67。

**JSON出力不具合を修正し、完了済み2 fitの保存JSONだけから最終集計を復旧した。追加fit／optimizer呼出し／NPZ decode／新objective・gradient・operator評価は0。保存scalarの再集計を1 logical unitだけ実行した。** 復旧した科学的判定はONE_OR_BOTH_UNQUALIFIEDで、両DF予算PASS／両収束資格未達という既存の結果は変わらない。

[新しい有効なresult.json](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/result.json)／[修復記録](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/recovery.json)／[新STOP](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/STOP.json)。元の[partial result.json](../../artifacts/research_c_common_convergence/2026-11-01/run01/result.json)、failed=trueの旧STOP、SOURCE、元25-call journal／raw37件を保持した。旧fitをresume／再実行したわけではない。

## 1. 原因と修正

[前回の実行報告](research_c_common_convergence_execution_20261101.md)では、sector guardのnumpy.float64からbudget比較のnumpy.bool_が生じ、最終JSON exportでTypeErrorになった。さらに例外処理が既に存在するpartial result.jsonをexclusive-createしようとしてFileExistsErrorになった。2 fitと25計算unitはその前に完了していた。

旧SOURCE44を編集せず、[独立repair adapter](../../src/trottertracks/c_common_convergence_export_repair.py)を追加した。

- NumPy scalarを同じ値のPython scalarへ変換する。配列・complex・未知object・NaNを黙って数値へ置換しない。
- JSON全体をファイルopen前に検証する。書込みは同じ出力directoryの一時ファイルへ行い、flush/fsync後にexclusiveなhard-linkで公開する。既存ファイルを上書きせず、失敗時はtempを回収する。
- 既存resultがpartial／validのいずれでも、TECHNICAL_FAILUREの出力は別のresult_failure.jsonへ保存する。同名再openの二次障害を防ぐ。
- 将来の既存paired runner用にprocess-localなjson_exports contextを用意し、既存decide／io.saveだけを差し替えてfinallyで復元する。比較の算式、solver、目的、初期値、収束条件は変更しない。

今回の実復旧は[JSON専用entry](../../scripts/run_research_c_common_convergence_export_repair.py)で実施した。旧paired runner／fitterは呼ばず、旧JSONをloadして元のdecide関数をそのまま一回使用した。過去の失敗を取り消したり、旧SOURCE manifestのhashを新コードへ置換したりしていない。

## 2. 固定した修復範囲

| 項目 | Identity／範囲 |
|---|---|
| 修正実装commit | 929da067d8a29a1b36f54c8cf19642c5aa4e3789 |
| [SOURCE47](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/technical/SOURCE.json) SHA256 | 80b9edffd90a7a58fb581dbf89f77d77d37608d3a1de95113b8262af0b3d1e58 |
| [修復plan](research_c_common_convergence_export_repair_plan_20261101.json) SHA256 | de0d2358320f542943065dc785fb929b798c80e831ecc501a1d36a2830bdc8bb |
| 元科学SOURCE／契約 | e2f7dfd44357510d54ab9c6c9c5032e0072ab376／旧25-call契約、いずれも不変 |
| 入力 | 既存の両arm final_diagnostic／optimizer／tensor／sector JSON、旧比較用scalar。元raw37件と元input14件はhash確認のみでNPZをdecodeしない |
| 新出力 | artifacts/research_c_common_convergence_export_repair/2026-11-01/run01 |
| 追加処理 | saved_scalar_comparison 1 attempt／1 COMPLETE、FAIL0 |
| 新fit／optimizer／新数値kernel | **すべて0** |
| 歴史の計上 | 元25 unitsは消費済みのまま保持。今回の修復範囲に保存scalar1 unitを別記録し、系列累計26 units・新fit累計2 |

ユーザーの修正・再実行指示に基づき、出力だけを復旧する最小範囲をSOURCE/input/planへ固定してから実施した。新しいscientific approval-proof、承認原文保存、grantは作成していない。旧契約の25-call上限やSTOPを編集せず、元fitのretry/resumeも0のまま。今回は修復attempt1で成功し、中断後の再試行は不要だった。

## 3. 検証と実行

[人工・回帰log](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/technical/tests.txt)：新15＋既存common-convergence25＝**40 PASS、2.70秒**。実NumPy guardを元のmock paired25-call E2Eへ注入し、正常／未資格／DF予算未達／表現FAIL／worker FAILの各経路を再確認した。

Scalar型、非対応値、NaN、I/O失敗、temp回収、既存partial保全、別failureファイル、binding復元も検証した。保存JSON復旧の人工E2Eではnp.load／fit／SectorCARを例外に置き換え、どれも呼ばれないことを確認した。実分子配列を使った新テストやfitは行っていない。

固定後のpassive preflightとJSON復旧を次のentryで実行した。Python3.11.1の標準libraryによる集計であり、実復旧はffsim／JAX／operator計算を必要としない。

~~~bash
/home/abe/.pyenv/versions/3.11.1/bin/python -B scripts/run_research_c_common_convergence_export_repair.py \
 --plan docs/research/research_c_common_convergence_export_repair_plan_20261101.json \
 --recover-saved --output artifacts/research_c_common_convergence_export_repair/2026-11-01/run01
~~~

run内wall .005014264024794102秒、process CPU .084399663秒、peak RSS25088 KiB。Wallはrecover内部、CPUは既存finish方式でprocess全体の値である。小さいJSON I/Oとscalar算術だけなのでprocessを増やす必要はなく、CPU／RAM／worker／wallに人工的な上限を追加していない。共有環境／既存venv／他jobへの変更もない。

## 4. 復旧した内容と残る未資格

対象は保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number CDF rank8、元frame順LD2、同じ元qualified x0、固定α=γ=.1。元g補正κ／scalar／旧scales／L-BFGS-B設定／二つの一次基準／DF予算CA/8は不変。PFの時間・δはこのfit集計に非適用。

| 復旧項目 | RC2 | PR-tail |
|---|---:|---:|
| DF予算 | PASS | PASS |
| Solver success | false | false |
| Native gradient∞ | .03752730461485592 | .3785060501566606 |
| Identity-null norm | .005394362795860612 | .036141983071728855 |
| 二つの一次基準／qualified | 未達／false | 未達／false |
| Exact tail λ [Ha] | 8.545687128230103 | .9545432846581517 |

各leaf値は元JSONと完全一致することを確認した。最終statusは元partialのONE_OR_BOTH_UNQUALIFIEDと一致し、qualified_tail_comparisonはnullのまま。保存済みのsector値から元の算式でDF_budget=true／trueが正常なJSONとして出力された。

これは**出力の復旧成功**であり、optimizerの収束成功ではない。両fitは元のnfev上限で終了したまま。強いRC-DF対照との資格付き比較、共通収束達成後のλ差、最適強度、PF/RTE、ground、native、shots、CA／総資源の判断を新たに認定していない。

[Postflight](../../artifacts/research_c_common_convergence_export_repair/2026-11-01/execution_postflight.json)でSOURCE47、旧SOURCE44、旧raw37、旧契約／failed STOP、旧資産213ファイルのGit blob一致、leaf値一致、新raw7件を確認した。公開時にもremote SHA/tree、修復SOURCE、公開本文、有効result.jsonを照合する。

## 5. 次の担当とSTOP

集計型変換と例外経路の修正・人工回帰・保存値復旧は完了した。今回の修復を理由に2 fitをやり直す必要はない。

次はユーザー／研究Cが、両一次基準未達に対する収束政策や検証方針を判断する。Codexは具体的な依頼後に必要な技術準備を行い、新fit・新数値解析・Stage B/Cは別途明示された範囲だけを実施する。既存のmaxfun、strength、rank、初期値を自動変更しない。

**公開照合後mandatory STOP。** 旧結果・失敗履歴を保全し、追加科学計算へ自動進行しない。
