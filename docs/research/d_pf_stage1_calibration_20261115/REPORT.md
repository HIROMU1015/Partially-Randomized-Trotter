# 決定論PF Stage 1 CALIBRATION：実回路価格取得結果

**新規20/20 wrapper PASS、FAIL/UNKNOWN各0、保存4価格再利用、C guard新規20/20受理。mandatory STOP。** 取得したのは24較正価格であり、12モデルのfit・係数凍結・holdout・長時間資格・MSE/N/Gは行っていない。

## 対象とSOURCE

H4 linear 1.00 Å、STO-3G、保存tol-only DF10（tol1e-8）。Full PFはone-body＋DF fragment0..9、TruncatedはB0I p7でone-body＋fragment0..6を元順序で保持。Tはm4=1.6、m5=3.2。actual幅・q・signed weights・physical/configuration IDはC新版CELL_PLANを保持した。

Stage 0 `acea1118a4df4e727e13398ce0f8ae11511c7396` のbranch `codex/d-price-records-c-bridge-20261114` を継続。条件/code SOURCE `91365dcf0610362a96aab101fbe0ec0e78f58af1` を実行前に公開した。Full PFの結果・guardを先に公開し、code byte不変を確認してp7を実行したため、p7の実行HEADはFull PF bound公開commitである。

| route | 実行SOURCE | raw公開 | bound公開 |
|---|---|---|---|
| FULL_PF | `91365dcf0610362a96aab101fbe0ec0e78f58af1` | `1078d23621566728521d8056fead17118244be42` | `a393a65cc2f00cdad596e11b9910bae0394e2407` |
| TRUNCATED_PF | `a393a65cc2f00cdad596e11b9910bae0394e2407` | `a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd` | `2ff582779ee36741407cac2d27e0f7b243088a0d` |

native/wrapperは`af284d853890155bacd19e61850c196bd1d5c660`、providerは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`、C guardは`c628e9e8447c11c984064b922cbce6c52dc72d48`。元callerのprepare/compose/cost_trial/checked_trial/compilerは変更していない。差分は採択済み科学policyをsource/guard metadataに記録する処理と人工test期待値だけ。

## compiled RZ価格

| route | PF・幅 | m4 q / m5 q | m4 X / Y RZ | m5 X / Y RZ | m4取得 |
|---|---|---|---|---|---|
| FULL_PF | 2nd・16h/7 | 7 / 14 | 54,844 / 54,844 | 109,381 / 109,381 | 新規compile |
| FULL_PF | 4th・16h/3 | 3 / 6 | 70,912 / 70,912 | 141,517 / 141,517 | 新規compile |
| FULL_PF | 4th(m5_best)・16h | 1 / 2 | 85,960 / 85,960 | 171,613 / 171,613 | 保存再利用 |
| TRUNCATED_PF | 2nd・16h/7 | 7 / 14 | 38,898 / 38,898 | 77,489 / 77,489 | 新規compile |
| TRUNCATED_PF | 4th・16h/3 | 3 / 6 | 49,924 / 49,924 | 99,541 / 99,541 | 新規compile |
| TRUNCATED_PF | 4th(m5_best)・16h | 1 / 2 | 60,950 / 60,950 | 121,593 / 121,593 | 保存再利用 |

各X/Yは個別にcompileした。等しい表示値から片軸を省略したものではない。新規全20 wrapperは新しいprivate cacheで取得し、cache hitは0。分類はDIRECT_COMPILED_DETERMINISTIC / DETERMINISTIC。S・stochastic trajectory平均・fit価格ではない。

保存M5/m4はFull physical `0cd1527c80f7b9841a83`、p7 `13d1cd4869951cdf1e2c`。元EXPLORATION cohortと元price SOURCEを保持し、再compile・同cohort化していない。C正本855573064の受入recordと完全一致し、元sidecar/guardの参照をC_INPUTS内のsaved_recordsへ保持した。

## 回路・gate・policy

測定付きfull controlled Hadamard wrapper全体、状態準備P0除外。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、H46_LONG_TIME_NATIVE_COMPOSE、NOT_APPLICABLE_NO_RTEを維持。Full scalarはc_H、p7はc_H+s_tailをtotal Tで一回。保存scalarとの完全一致とretained block/fragment順、signed scheduleを既存checked_trialで確認した。

compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。各private cache receiptのcompiler/hash/6 metricsがRESULTのX/Yへ対応することも確認した。

各cellはindex0でlow_control、high_control_leakage、independent_occupation_vector、Hadamard X/Y、compiled_vector X/Yの7 gateを通過した。native/測定は1e-9、compiled-vectorは1e-7という元の数値gateを緩めていない。全数値とfingerprint・6 metricsは各RESULT/PRICES、gate errorの観測最大値は[C_HANDOFF_INDEX.json](C_HANDOFF_INDEX.json)に記録した。保存初期状態に対する作用検査であり、全入力状態の新証明ではない。

採択済みH-bはRELATIVE・観測C分母・厳密1/50、12モデル個別の全PASS。m6を較正/モデル選択へ使わず、FAIL時は自動refit/feature追加/閾値緩和をしない。このpolicyは固定codeと各RESULT/GUARD_CHECKに記録したが、今回は観測holdout価格・予測・判定を取得していない。C guardのPASSは実装/record受入であり、価格モデルの妥当性を意味しない。

## preflight・資源・保全

§10.5のduplicate_preflightを更新し、既知価格表/catalog・別名schedule・元trial/receipt・stage結果・関連user jobを確認した。新規10 cellに追加重複/競合を検出しなかった。各routeの実行直前にも既存runnerが再確認した（元reportを流用したものではない）。未列挙archive/別サーバーまでの網羅は主張しない。

H/state/snapshotの実input bytes/hash・receipt、C保存4件、新版36行CELL_PLANとの対応、SOURCE/native/provider、Qiskit/NumPy、restore import chain、fresh public/privateを確認した。[PREFLIGHT_METADATA](PREFLIGHT_METADATA.json)／[IMPORT_PREFLIGHT](IMPORT_PREFLIGHT.json)。metadataテスト18 PASS、17.230秒。[TECHNICAL_TESTS](TECHNICAL_TESTS.txt)中のfit/HOLDOUTは架空fixtureのみ。

| route | stage elapsed s | user / system CPU s | 最大worker peak RSS MiB | workers |
|---|---:|---:|---:|---:|
| FULL_PF | 335.500 | 983.04 / 26.52 | 782.50 | 5 |
| TRUNCATED_PF | 226.191 | 709.19 / 28.78 | 645.12 | 5 |

stage elapsedは既存runnerのchild開始からrecord構成まで。worker wallはbuild/作用検査/compile等を含み、compile単独時間ではない。RSSは各processの観測peakであり、同時物理メモリ総量ではない。GNU timeのuser/system CPU・wall・RSS・I/Oは[FULL_PF_time](FULL_PF_time)と[P7_time](P7_time)、stdout/stderrは[FULL_PF_log](FULL_PF_log)と[P7_log](P7_log)に保全した。p7の既存moduleのSyntaxWarningは非fatalで、SOURCEを変更して抑制していない。

routeを順序化して関連job guardを保ち、route内は実idle CPU/available RAMに従い5 cellを並列化した。恣意的な低いCPU/RAM cap・共有設定変更・他process停止はない。private cost/cache/QPY/jobは元の新stage namespaceに保持する。旧Stage 0、保存4価格、raw・grant・ledger・marker・STOP、D1/D2/A/B/C branchには変更していない。retryは0。

## Cへの引渡しとロードマップへの影響

24 recordを12 group各m4/m5の2点として既存C_INPUTSに格納した。[HANDOFF_C](HANDOFF_C.md)と[C_HANDOFF_INDEX](C_HANDOFF_INDEX.json)に正確なgroup・固定公開先・元/新SOURCE・保存/新規の区別を示す。Full/p7、公式/幅、X/Yをpoolしていない。

統合担当へ、決定論PF Stage 1の進捗を「取得・公開・guard完了、12モデル未fit」として反映するよう報告する。保存4件の受入不足は解消済み。次はCによる12モデルの実fit・係数凍結で、別途科学実行指示が必要。その後のStage 2 m6 12 compileも未実行。m7〜m12のaffine性/価格資格・全RPE総G・winner/rankingはUNKNOWNのまま。ロードマップ本体は変更せず、本固定REPORTを更新根拠として引き渡す。

各routeのraw公開5ファイル、bound sidecar/guard、実行codeについて、公開commitとのlocal bytes一致を確認してから次工程へ進んだ。最終REPORT公開のremote SHA・tree・主要成果物照合は最終チャット報告に記載する。

**Stage 1取得・証拠公開・C guard束縛完了でmandatory STOP。**
