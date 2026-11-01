# 研究C：共通収束政策・2-fit／25-call結果と最終JSON技術STOP

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**RC2／PR各1 fit、25/25 logical unitsの計算は完了したが、最終result.jsonのNumPy真偽値の書出し不具合によりTECHNICAL_FAILUREとしてSTOPした。** 両optimizerは関数評価上限で有限終了し、登録した二つの一次収束基準はいずれも未達。DF予算は通過したが、共通収束後の資格付きtail比較は成立していない。

新fit2／optimizer呼出し2、retry/resume0。固定SOURCE・生出力・途中まで書かれたresult.json・failed=trueのSTOPを一切変更せず保全した。追加fit、集計callの再実行、SOURCE修正、出力上書きは行っていない。

## 1. 固定条件・SOURCE

| 項目 | 固定条件／identity |
|---|---|
| 科学実行SOURCE | [e2f7dfd44357510d54ab9c6c9c5032e0072ab376](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e2f7dfd44357510d54ab9c6c9c5032e0072ab376)、無変更で一度実行 |
| 実装SOURCE | 38c2bc02b9578ab6ab7734bb61899b6cbc1733e2 |
| [固定契約](research_c_common_convergence_contract_unapproved_20261101.json) SHA256 | 3650d105f420e4e5e93a0756ceb6df90930a612b2828e74c066616f7ea6c1a3d |
| [SOURCE44](../../artifacts/research_c_common_convergence/2026-11-01/technical/SOURCE.json) SHA256 | 358df00f1390e23866abc7f78355e95ec061a6feb5173278cfcd5aec7e3e1927 |
| 分子／sector | 保存直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36 |
| DF | 通常number CDF rank8、一般実対称J、実直交U、元frame順LD2 |
| 初期値 | 同じ元qualified J/U/x0。Warm-start／旧fit resume／新logm／新Choleskyなし |
| 目的 | 既存ffsim loss f/f*＋RC2 .1P/P0／PR .1(Mε/M*)²。旧strength／scale不変 |
| 元物理規約 | 元g補正κ、scalar一回、DF予算CA/8=.00019875 Ha・既存engineering guard |
| Solver | ffsim0.0.83・L-BFGS-B、ftol1e-15／gtol1e-6／maxcor10／maxls20／maxiter15000／maxfun15000、boundsなし |
| 事後一次条件 | Native parameter gradient∞、Frobenius正規化identity-null投影normとも1e-5。固定engineering marginを適用 |
| 実行出力 | [run01原本37件](../../artifacts/research_c_common_convergence/2026-11-01/run01/) |

[政策・数学・実装の技術報告](research_c_common_convergence_policy_design_20261101.md)に従った。共通input snapshot SHA256=1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db、qualified initial=78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856。元P-H4来歴8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50を保持した。

比較用の旧RC2科学SOURCE150f904f6975868abe09966dda8ab41722e2a7cc、旧PR科学SOURCEe1085c95f5c01b68566e17d543ace5832cb94061、一次診断SOURCE0996a5abb4876c2e450439044c26e7df61dfd04cは別identityのまま。旧終点配列を新たにdecode／再解析せず、旧scalarだけを登録比較内で読み取った。PFの時間窓・δはこのfit比較に非適用。

チャットの明示許可に基づき実行し、歴史的unapproved filename/metadataは変更していない。承認証明・承認原文保存・新grantは作成していない。

## 2. 実行gateとcall実績

| 処理 | Units | 結果 |
|---|---:|---|
| SOURCE44／入力14／両環境preflight | 0 | PASS、fresh run01未使用 |
| 実行時人工確認 | 0 | 新25件PASS。準備時148件を全再実行したとは扱わない |
| Shared snapshot／初期資格 | 1＋1 | PASS、同じ保存初期sector資格を再利用 |
| Sector context／physical sector | 1＋1 | COMPLETE、元gのphysicalを一回構築 |
| Worker load／setup | 2＋2 | 両arm COMPLETE、旧P0／scales保持 |
| Fitter呼出し | 2 | 各arm一度。双方finite、SciPy success=false |
| 初期診断／raw終点export／独立終点診断 | 2＋2＋2 | 両arm COMPLETE、独立value/gradient差0 |
| 終了fidelity／tensor | 2 | COMPLETE、既存unpack整合・構造検査通過 |
| 終了load／sector／一般K対CAR | 2＋2＋2 | COMPLETE、表現gateは両方PASS |
| 登録比較 | 1 | COMPLETEを記録、返却後のJSON exportで失敗 |
| 計算call合計 | **25** | **25 attempts＝25 COMPLETE、call-level FAIL0／inflight0** |
| 最終集計JSON出力 | call外 | **FAIL：numpy.bool_のserialization、例外経路もpartial fileと衝突** |

[Preflight](../../artifacts/research_c_common_convergence/2026-11-01/execution_preflight.json)／[人工log](../../artifacts/research_c_common_convergence/2026-11-01/execution_synthetic_tests.txt)／[親journal](../../artifacts/research_c_common_convergence/2026-11-01/run01/science_calls.jsonl)／[親STOP](../../artifacts/research_c_common_convergence/2026-11-01/run01/STOP.json)。親11＋各worker7×2＝25。Call-level COMPLETEとpipelineの正常完了は異なる。親STOP.failed=trueをfalseへ変更していない。

## 3. 両fitの保存結果

以下は終了JSONからの転記であり、STOP後にgradient、sector norm、差分、最適化、比較callを再計算していない。

| 指標 | RC2 | PR-tail |
|---|---:|---:|
| Solver success／status | false／1 | false／1 |
| nit | 14151 | 13559 |
| nfev／njev | 15001／15001 | 15001／15001 |
| 終了combined objective | .022324453695515206 | .018666104007351476 |
| Gradient∞ | .03752730461485592 | .3785060501566606 |
| Identity-null投影norm | .005394362795860612 | .036141983071728855 |
| Native gradient判定 | NOT_MET | NOT_MET |
| Identity-null判定 | NOT_MET | NOT_MET |
| convergence_criteria_met／qualified | false／false | false／false |
| Tensor Frobenius | 4.632476099919241e-10 | 4.0253274525626814e-9 |
| Tensor max-entry | 8.444651803330381e-11 | 6.810021223963855e-10 |
| Physical sector error [Ha] | 7.117616656640366e-10 | 3.989173927332953e-9 |
| Engineering guard [Ha] | 3.1689528454574227e-9 | 3.2276640612719506e-9 |
| 一般K対raw CAR max-entry | 1.887379141862766e-15 | 2.8033131371785203e-15 |
| Exact tail λ [Ha] | 8.545687128230103 | .9545432846581517 |
| Full ordered J L2 [Ha²] | .9847059304328908 | 今回のPR診断には未保存 |

[RC2 optimizer](../../artifacts/research_c_common_convergence/2026-11-01/run01/rc2/optimizer.json)／[PR optimizer](../../artifacts/research_c_common_convergence/2026-11-01/run01/pr/optimizer.json)／[RC2診断](../../artifacts/research_c_common_convergence/2026-11-01/run01/rc2/final_diagnostic.json)／[PR診断](../../artifacts/research_c_common_convergence/2026-11-01/run01/pr/final_diagnostic.json)／[RC2 sector](../../artifacts/research_c_common_convergence/2026-11-01/run01/rc2_sector.json)／[PR sector](../../artifacts/research_c_common_convergence/2026-11-01/run01/pr_sector.json)。生x/jac/fun、最終J/Uと独立gradientも各workerのNPZに保存した。

両messageは関数・勾配評価上限によるSTOP。nfev=15001は既存SciPy1.14.1がNEW_X後にsf.nfev>maxfunを検査する出口であり、追加fitやretryではない。15000を事後変更したり、さらに反復したりしていない。

**両DF予算は登録比較でPASSしたが、両収束資格は未達。** 途中のresult.jsonには先頭status=ONE_OR_BOTH_UNQUALIFIEDが書かれている。固定decideの分岐では、これは両DF予算を通過した後の状態である。この保存済み分岐記録を根拠に報告し、STOP後にbudget比較を再実行していない。Sector rowのlegacy status=CONVERGEDはoptimizer収束を意味しない。

旧RC2のP=4.410972371783312 Ha²、λ=2.209761871954849 Ha、旧PR λ=1.4424298855756243 Haは旧報告の値である。今回RC2のPは小さくなり、tail λは大きくなった。PRのtail λは小さくなった。しかし新両点は未資格なので、**共通収束達成後にも差が残るかという研究上の問いは未解決**。記述的な終点値を、十分最適化されたRC-DFへの優位、CA、下流PR性能、総資源へ昇格させない。欠けた集計内のλ差や比率を再計算して補完していない。

## 4. 最終JSON不具合と保全

一次原因は、既存rank_assessmentのguardがnumpy.float64で、そのbudget比較がnumpy.bool_になったこと。新paired decideがこれをPython boolへ変換せず、既存io.saveのjson.dumpへ渡した。ファイルはexclusive-createで開かれ、statusとDF_budget先頭まで書かれた後にTypeErrorとなった。例外ハンドラは同じresult.jsonを再度exclusive-createしようとしてFileExistsErrorになった。

[実行traceback](../../artifacts/research_c_common_convergence/2026-11-01/execution_run01.txt)／[failure.json](../../artifacts/research_c_common_convergence/2026-11-01/run01/failure.json)／[partial result.json](../../artifacts/research_c_common_convergence/2026-11-01/run01/result.json)。最後のファイルは**意図的に原本の不正JSONのまま保持**する。新しい正式resultと誤認しないこと。failure.first_failure=nullなのは失敗がcall wrapperの外の出力処理で発生したためであり、エラー不存在ではない。

Plain-floatを使ったmock E2Eではこの実NumPy scalar境界を見落とした。私の実装・テストの不足である。[人工scalarだけの再現](../../artifacts/research_c_common_convergence/2026-11-01/execution_serialization_diagnostic.json)で一次TypeErrorと二次FileExistsErrorを確認した。実H4配列や科学callは使用していない。

必要な最小修正は、科学SOURCEの別版でbudget結果をPython boolへ正規化し、JSON化の検証をファイルopen前／登録call内で行うなど、partial fileを同名exclusive-createする例外経路を直すこと。実NumPy scalar・partial writeの回帰が必要。**今回のSOURCE・契約・run01は変更していない。** 2-fit／25-callは消費済みであり、修正のために旧runnerを再起動しない。保存leaf値だけを使う出力形式修復と、新たな科学評価／fitは分けて扱う必要がある。

## 5. 資源・公開・次の判断

Preflightでaffinity32、MemAvailable56,220,217,344 bytes、load約1.05を確認。既存動的planは独立2 worker、fitメモリ推定794,320,896 bytes/workerを選択した。Tiny BLASのみ各process内1 thread、JAX CPUは既存環境で実行し、role別cache/tmpを使用。人工的なCPU/RAM/worker/wall上限、共有venv／設定／他job変更はない。Worker logのCUDA-enabled jaxlib不在通知に対しinstallや別backend再実行もしていない。

| 記録 | Wall [s] | Process CPU [s] | Peak RSS [KiB] |
|---|---:|---:|---:|
| Parent | 8.295605764724314 | 1.519620175 | 277204 |
| RC2 worker | 6.5698123797774315 | 9.732154299 | 383184 |
| PR worker | 6.624727422371507 | 10.170712286 | 391668 |

値は各processの計測区間で、aggregate peakではない。[資源plan](../../artifacts/research_c_common_convergence/2026-11-01/run01/resource_plan.json)。

[Postflight](../../artifacts/research_c_common_convergence/2026-11-01/execution_postflight.json)でSOURCE44/input14、元契約/runコピー、25 attempts/COMPLETE、8 NPZのheader、raw37件hash、旧SOURCE／結果／STOP等170ファイルのGit blob一致を確認した。新結果を既存研究記録へ追記し、旧SOURCE/raw/QPY/grant/ledger/marker/STOPを保全した。公開時にremote SHA/tree・本文・新rawを照合する。

今回の実行commandは履歴であり、再実行の許可ではない。

~~~bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_common_convergence.py \
 --contract docs/research/research_c_common_convergence_contract_unapproved_20261101.json \
 --execute --output artifacts/research_c_common_convergence/2026-11-01/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
~~~

次の技術課題はCodexによる集計型変換と例外経路の修正・人工回帰。新fitを繰り返さずに保存値の出力だけを修復できる範囲を明示する。ユーザー／研究Cは、両一次条件未達の科学的意味と、今後の収束政策／検証の要否を別途判断する。maxfun増加・solver変更・preconditioner・gauge操作・新fitを自動採用しない。

**技術STOP・原本保全・公開照合後mandatory STOP。** Retry/resume／追加fit／科学再解析／Stage B/C／PF-RTE／ground／native／shots／総資源／重要GPT科学レビューへ自動進行しない。
