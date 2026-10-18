# 統合人工回帰：実収集275件

[再実行runner](run_artificial_tests.py)は`/usr/bin/python3`で6 suitesだけを起動し、fresh /tmpへlogと全IDを保存する。
今回実行した同じ収集harnessを配置した。

[machine結果](artificial_test_results.json)へ全test ID、SOURCE SHA、log SHA、PASS/FAIL/ERROR/SKIPを保存した。

| Suite | 実収集/PASS | FAIL/ERROR/SKIP |
|---|---:|---|
| Admission | 69 | 0/0/0 |
| Readiness | 30 | 0/0/0 |
| B3 Driver | 36 | 0/0/0 |
| Cost Preparation | 48 | 0/0/0 |
| F01 Provenance | 26 | 0/0/0 |
| 新host integration | 66 | 0/0/0 |
| 合計 | **275** | **0/0/0** |

既存は重複なく209件、新規は66件。正常reference8ケースはtest内のsubcasesで、件数へ8を重複加算していない。
旧Admission公開machine/logは69+30+36+48=183。旧README/handoffの181、新67は記載不整合だった。
原本は変更せず、今回の件数は実収集と保存logを正本にする。

## 新しい統合coverage

- 実host/Boundary/executeへの接続を通し、with portでnormal/exceptionの両方がSTOPする。
- authorityなし、自己承認bool、digest/source/input/A6/seed/N、store identity、host欠落/path/API/SHA/import root、旧driverとv1 contractを拒否。
- reserve前/insert未commit/commit直後/再照合/detail更新/receipt前後、source/budget/module相当、output前後/correctness前に障害を注入。
- 前trial成功→次sample失敗、X→Y、結果取得前、angle/evidence、partial write、flush、failure保存、child異常exitと親host STOPを確認。
- primary例外をsecondary STOP失敗が隠さず、正常return時のSTOP保存失敗も成功扱いにしない。spent rowで再取得を拒否。
- hostを通す32 thread/32 owned processで各1成功。別digest/claim・同claim別scope、STOP後reuse/retry/resumeを拒否。
- 固定旧SOURCEで生成された8正常ケースのsummary/trace/bit数/R-C-A/phase/weight/word/canonical bytes/chain/query/coarseをhost接続下でも比較。
- mock correctness256を先行し、mock二passのtrace/A-B対応を確認。all-zero coverage不足は最初の16toy trialsで停止し、full-Nへ進まず補充しない。

full-bundle mockのprofileはtest wrapperで16/axisへ置換した人工endpointだけであり、計768toy calls。
科学契約の保存Nや31,710,848予算を変更・実行したものではない。saved budgetはmock行を返し、保存科学値を数値再解析していない。
science moduleのsys.modules混入は各suite終了時0。fake clockで非決定的CPU時刻/PIDを科学的同一性比較へ混ぜていない。

## 実行環境と限界

CPU32、available RAM約49GiB、disk空き371GiB、load約1.05、Python3.10.12/SQLite3.37.2/local ext4を確認。
独立6 suitesを並列実行。SQLite writeは一意制約のcritical sectionとして直列化する。科学scheduler・性能benchmarkは追加していない。
共有OS/venv/cache/mount/他jobを変更せず、owned temp・childだけを使った。
電源断・媒体故障・VFS torn-pageの完全証明、実G7 correctness/実CPU優位/全N実行成功/CIや外部再現ではない。
