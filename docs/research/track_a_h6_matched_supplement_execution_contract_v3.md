# Track A H6精度一致資源比較：補完実行契約 v3

状態：**PREPARATION_ONLY_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION**。
旧v2はSTOP・grant消費済みのまま。新版の実行には、このsource・sealed manifest・再利用証拠へ結び付けた別の一回grantが必要。
H8・追加探索・旧runのretry/resumeは認可しない。phase/total wall capは設けない。

## 1. 採用した判断と証拠

利用者が採用した[GPT独立科学レビュー](track_a_h6_matched_v2_memory_stop_independent_review_2026-10-11.md)に従う。
同じ科学方針の再レビューを行わず、既存証拠を入力にした新しい補完実行を準備する。

| identity | 固定値 |
|---|---|
| 旧science source | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| 旧結果・公開後訂正 | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| 旧status | `H6_MATCHED_RESOURCE_STOP` / `RuntimeError:MemoryError:` |
| 旧科学plan digest | `801e1eea5d09b6fbe60fe66102bc21edb1bd50a0d9aa94ca05a2b0659d1f8cf3` |
| 旧grant | `2157da1d208a04102166fa1ae3c154490546c90f`、消費済み |

[旧結果・監査索引](track_a_h6_matched_accuracy_result_v2.md)、[旧科学契約](track_a_h6_matched_accuracy_execution_contract_v1.md)、
[旧v2資源契約](track_a_h6_matched_accuracy_execution_contract_v2.md)を保全する。
[sealed manifest / 再利用registry / 検証記録](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/README.md)が新版の実行前identityである。

## 2. 科学条件の維持

linear H6、1.00 Å、STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。
DFはtol-only 1e-8、signed generation orderのrank19、cutoff0、採用済みweighted Hermitian projection。
DF再分解、fragment削除、rank fallback、state/solver再実行なし。
B0 prefix5/10/15、B1 full19のS2/global Yoshida S4、B2の既定q/r/K。
92候補、epsilon_sig=0.05/0.01/0.005/0.001、経験的allowance、normalization、shot式、境界guard、選抜とseed規則を維持。
frozen `h6_matched_contract_v1.plan()` のcanonical digestが旧v2と一致することを検査する。

primaryはsymmetric_directionalの測定込みHadamard X/Y wrapper、state preparation除外。
compilerはbasis rz/sx/x/cx、optimization level1、seed17、backend/coupling/layout/routing指定なし、同じQiskit環境。
full wrapperを実transpileしてcount/depthを得る。部品費用の足し算、proxy、外挿で欠測を埋めない。
量子shotは解析会計であり、量子測定をsamplingする実行ではない。
signal精度と化学的energy精度を換算しない。母平均winner、ground-state、数値誤差certificateは認定しない。

## 3. 静的受理と来歴

旧raw 5,220 fileを実行inventoryのhashと旧結果commitのGit blobで照合する。
旧runtime216 sourceも実行manifestのhash・旧source commit・ローカルbytesで一致を確認する。
実行authorization、signal terminal/claim、全92 IDと4精度field、個別signalとsummaryを結ぶ。
B3は84組を全て再利用する。旧168表記は過去の誤記で、原文は改変しない。

cost groupの受理条件は、TASK_COMPLETE、原登録task/整数seed、worker source/8 GiB limit、event digest、両primary軸、
必要なordinary二軸、compiler hash、wrapper fileとの一致、必要なcontrol検査、compile counterと軸数の整合。
Git blobとraw hashの一致を経て受理する。数値処理を再実行して同一性を作らない。

| 保存証拠 | 再利用 | 新しい取得 |
|---|---:|---:|
| signal | 92 record | 0 |
| B3 normalization | 84組 | 0 |
| 探索group | 449 / 原592 | 143 |
| 探索wrapper | 902（898 primary＋4 ordinary） | 286 primary |
| 確認 | 0、未選抜 | 最大8構成×32 trajectory×2軸＝512 wrapper |
| 新wrapper合計 | — | 最大798 |
| 統合wrapper合計 | — | 最大1,700、原上限1,708以内 |

143件は、原447のTASK_STOP、450のdispatch済み未完了、451〜591の未dispatch141件。
448/449は完了側である。missing集合は保存terminalと登録表から決定し、番号範囲だけで推定しない。
旧未完了taskの途中recordは完了groupへ昇格させない。旧中断compile attemptの未保存数はnull、不明と記録する。
統合結果には各groupのold/new source commit、task index/digest、cost result path/hashを保存する。
旧sourceで得た保存値と、新sourceで得る取得値を別来歴として示す。

## 4. メモリ対処と検証の限界

旧447ではcontrol/X/Y検査まで保存された。最初のmeasured wrapper recordより前に失敗したが、allocation tracebackと失敗時VmSize/RSSは未保存。
wrapper構築・fingerprint・transpile・metric計測のどこで発生したかを断定しない。RLIMIT_ASは仮想アドレス空間の制限で、host RAM枯渇の証明ではない。
既存disposable process、streaming fingerprint、bounded cacheを再利用する。

新版では、gate-only evolutionの一つのcircuitを同じH/U/(Sdg)/H/measure wrapperへ移し、composeによる二つ目の大きなcircuitを保持しない。
同じqregs、ordered gate、numeric parameter、global phase、測定registerを保つ。
transpile後は入力circuitを解放してから同じcost metricを測る。
二番目の軸は、同じprepare済みblocks・同じsample済みstepsからevolutionを作り直す。
**samplingはtaskにつき一回、X/Yは同じevent列。** 二度目のnative buildの古典時間は追加される。実行時間短縮率は主張しない。

syntheticでcanonical bytes、unmeasured unitary、compiler hash、全count/depth/size/global phaseを旧経路と照合する。
入力circuitがmetric計測前に解放されることも確認する。
これは実装のbridgeであり、実H6 S4 q8が8 GiBで完走することの実証ではない。
実H6のbuild/compileやメモリprofilingはこの準備では行わない。

新runは原447を単独で最初に処理する。それ自体が登録された一回の取得であり、追加trialではない。
成功・保存後に残142件を原index順、最大2 workerで処理する。失敗したら残りを開始せずSTOP。
メモリ上限、compiler、q、候補を変えて救済しない。必要になれば新しい条件判断を利用者へ戻す。

ResourceTraceをbindし、evolution build、control validation、wrapper build、fingerprint、transpile、metric計測の開始/終了に小型receiptを保存する。
RSS/VmSize/peak RSS、AS soft/hard、instruction数、軸/phase/taskを記録する。
1 MiB emergency reserveを解放し、失敗tracebackは最大32 frame、localsなし、bounded messageで保存する。
未完了workerのcompile前receiptと失敗batchのpeakは保存し、成功batchだけのpeakを全runのpeakにしない。

## 5. 探索完成から独立確認へ

449旧受理＋143新受理のindex集合が原0〜591と完全一致することを先に検査する。
task/replicaと必要な両軸の欠落・重複を拒否する。部分集合ではresource mapと選抜関数を呼ばない。
全探索後、原規則「各精度B2 point G_RZ上位2、ID辞書順tie、union最大8」を使う。
選択集合、探索row digest、592 groupのpath/hash来歴を記録し、確認値を取得する前に固定する。
原confirmation seed規則の32 fresh whole trajectoryを各選択構成で取得する。
探索8と確認32をpoolしない。旧pilot32 wrapperもfresh平均へ混ぜない。
確認値を見て候補・標本数を増やさない。

経験的uの1倍/10倍感度、N、paired SD/SE、G(P)、原order診断を保存する。
未観測order mass、all-zero確率、有限標本の限界を示す。instruction capをpopulation cost boundにしない。
全完了でも`UNDETERMINED_POPULATION_MEAN`を維持し、科学的GOを自動承認しない。

## 6. 資源・呼出し政策

CPU [0,2,5,6]、cost slot [0,2]/[5,6]、各threads1、BLAS/OMP/Numba/Rayon/Qiskit1、Qiskit parallel FALSE。
coordinator AS soft2/hard8 GiB、worker soft/hard8 GiB、aggregate RSS18 GiB、core dump0。
新runのsignal workerは呼ばない。元signal資源設定は科学plan内に維持する。
GPUは使わない。host GPU照会の可否とメモリ現状は準備receiptに記録し、予約やspeedupとは扱わない。
既存reuseが計算量削減の中心で、全候補への再compileは行わない。

phase/total wall cap=null、出力2 GiB、log1 MiB、cost private32 MiB、diagnostic/progress/instruction guardは原値。
原caps compile1712、trajectory832、occurrence6656、control_probe1024等は維持する。
新規完了の最大compile798はregistered143＋確認最大256 groupの両軸から得る。
旧signal・受理cost・新costの保存counterを統合して原capsと照合する。
旧中断の実attempt数は不明のまま別記し、保存completed totalを全歴史のexact attempt totalと呼ばない。
入力生成、DF/state solver、signal再評価は0。新outputの既存directory・既使用grantは拒否する。

## 7. 認可、公開、mandatory STOP

sealed manifestは計画固定だけで、science_authorized/launch_allowedはfalse。
別schema `h6_matched_authorization_v3` のauthorization-only one-shot grantが必要。
新source commit、manifest digest、reuse registry digest、CPU、exclusive output、retry/resume=false、H8=falseへ結び付ける。
旧v1/v2 grantを拒否する。準備ではapproved grantを発行しない。

明示認可後：新grantをcommit/push→独立remoteからsource/manifest/入力/旧証拠hashを確認→新outputへ一回launch。
完了またはSTOP後：raw、partial、missing、実行inventory、static auditor出力、保存証拠の事後hash、grant消費、環境を保存する。
旧rawを上書きせず、今回だけを明示stageして同branchへ公開し、remote再取得でpath/SHAを照合する。
正常完了とSTOPのstatusを別記し、static auditを科学的承認としない。
原447再失敗でもretry/resumeや上限変更を行わず、その診断を公開してmandatory STOP。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=falseを維持し、H8へ進まない。

## 8. 実装の入口

- [metadata契約・grant gate](../../src/trottertracks/resource_applicability/h6_matched_contract_v3.py)
- [旧証拠の静的受理](../../src/trottertracks/resource_applicability/h6_matched_reuse_v3.py)
- [single-owner wrapper・cost bridge](../../src/trottertracks/resource_applicability/h6_matched_memory_v3.py)
- [cost-only port](../../src/trottertracks/resource_applicability/h6_matched_cost_port_v3.py)
- [supervisor・全探索gate・確認](../../src/trottertracks/resource_applicability/h6_matched_execution_v3.py)
- [runner](../../scripts/resource_applicability/run_track_a_h6_matched_v3.py)、[metadata preparer](../../scripts/resource_applicability/prepare_track_a_h6_matched_v3.py)、[static auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v3.py)
- [synthetic/process/injected tests](../../tests/tracks/resource_applicability/test_h6_matched_v3.py)

実H6でのq8 feasibilityと新費用値は未取得。準備検証を新しい科学的費用成果として使わない。
