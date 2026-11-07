# Signed Support 448-shot：SOURCE準備完了・科学未実行

2026-11-07 JST / SERVER1 GPUA100-gleap-01。

**448-shot採択仕様に適合するactivationを追加し、新規42＋既存159＝201人工テストPASS。固定input bytes/native runtime照合PASS。実H4の新科学計算・manager起動は0。公開SOURCE候補の照合後mandatory STOPする。**

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual Q/selected full、N2保証ground、認証済みexact Fraction engine、X/Y。

## 完成した接続

保存結果のhash/identity/schemaにbindしたtyped control復元、h1 train64/軸→軸別一回OLS/Fraction freeze→signed D≤6、全5層(8,128,8,8,8)/軸のfresh paired評価、固定half二分6解析、seed20261110/12domain/24child、global trial namespace、448-shot/call budgetを実装した。

既存signed kernel/OLS/sampler/engine/price/worker/group/会計/ManagedPool/qualified持続managerを再利用し、旧SOURCEは変更しない。新しい数値kernel・sampler・価格器・管理システムは作っていない。旧indicator Driverのbalance/range/quotaをsignedへ移植せず、必要な外側の制御だけを追加した。

短いfile lockでSTART/STOPを線形化し、STOP前のin-flightを識別・回収する。bit先行保存、価格欠測UNKNOWN、partial tail・両train費の保全を確認した。lock所有人工childのSIGKILL後にもSTOPでき、engine/compile本体はlock外で並列に実行できる。

## 証拠とSOURCE固定

| 成果物 | 内容 |
|---|---|
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | 再利用、必要な差分、typed復元、stage/guard/stream、STOP/call/timerと限界 |
| [ARTIFICIAL_CORRECTNESS_REPORT.md](ARTIFICIAL_CORRECTNESS_REPORT.md) | fixture別の検証範囲・未検証事項 |
| [TEST_RESULTS.json](TEST_RESULTS.json) / [log](ARTIFICIAL_TESTS.log) | 201 PASSのtest別結果・command・hash |
| [SCIENCE_SOURCE_PREFLIGHT.md](SCIENCE_SOURCE_PREFLIGHT.md) | 固定448仕様、input/profile、将来のpreflightと実行承認範囲 |
| [SOURCE.json](SOURCE.json) / [PROFILE_LOCK.json](PROFILE_LOCK.json) | SOURCE候補のhash979ファイル、環境、科学未許可・未実行 |
| [SETTINGS.json](SETTINGS.json) / [INPUT_BINDING.json](INPUT_BINDING.json) | 全設定・spawn mapping・byte/identity binding |
| [PRESERVATION.json](PRESERVATION.json) / [STOP.json](STOP.json) | 原本保全照合・今回の技術作業STOP |

新SOURCE候補は**このdirectory/codeを含む公開commit**。正確なSHA・tree/blobはGitHub metadataと最終公開照合の報告で固定する。科学起動時のSPECはremote SHAとgit HEADの一致を必須にし、実際のSHAを保存する。公開後にSOURCEを変更した場合は同じ候補と扱わず、新SHA/差分の固定が必要。

worktree `/tmp/scv-signed-support-448-preparation-20261107`、branch `codex/scv-signed-support-448-preparation-20261107`、base `138d071805430ef1e556655392e4d9f572feba40`。正式共通指示§8.1/8.2/8.4/10/10.3、AGENTS/PROJECT_MAP、概要・現状、固定4文書、既存SOURCE/結果/STOPを確認した。

旧SOURCE/raw/結果/STOP/grant/ledger/marker、元main/dirty/untracked、C1凍結を維持。索引・概要・研究ノート・manifestには技術準備の案内を追記し、旧科学契約・数値結論を書き換えない。並行O1/O2のworktree/process/cacheへ操作していない。

## 未実施・未解決と次の担当

実H4 saved control復元、saved state/bundle decode、科学PRNG、Born/shot、実fit/paired、full build/compile/価格、相関・分散/費用改善は未実施。旧bit・係数・価格の新規数値再解析も0。次の実行でrank/guardが通るか、改善が見えるか、Fraction長時間負荷/peakはUNKNOWN。人工検証を科学的GOや期待RZ改善としない。

次の担当はユーザーによる**このSOURCE・448-shot一campaignへの明示実行許可**、その後SERVER1 Codexが実行前preflightを行う。今回の指示はSOURCE公開固定までなので、科学submit・cron trigger・本番workerを起動せずmandatory STOP。追加shot/campaign、calibration/保証付きproduction、O1/O2差替え、詳細科学レビューへ自動進行しない。
