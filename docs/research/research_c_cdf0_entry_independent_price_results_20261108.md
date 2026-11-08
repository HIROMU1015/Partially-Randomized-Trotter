# 研究C：CDF0対ENTRY・独立2-arm価格検証結果（2026-11-08）

**固定CDF0/r64対ENTRY E2/r16を各fresh128標本で完了し、登録判定は`NOMINAL_MODEL_ENTRY_LOWER`。** 名目95% Welch差区間は正で、半幅/新ENTRY点値は3.9045023%となり、登録5%精度基準を満たした。科学SOURCE・入力・旧結果を保全し、正常STOPした。これは下記条件付きfinite-bit価格proxyの名目IIDモデル結果であり、実CA達成や無条件の資源優位ではない。

## 1. 科学条件と固定対象

Linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36/system256/controlled512。保存number-CDF rank8、元frame順序0..7、LD2、`tail_linear_Z_to_existing_kappa_keep_frame_identity_v1`の線形Z再配置。Partial ordered S2/s1、δ=h=.1、K0、RPE m0..12、候補固有own-DF sector ground oracle、状態準備除外。CA=.00159 Ha、DF reserve=CA/8、βtotal=π/6・βstat=π/12、failure=.01・axis=.01/26は保存政策のまま。

通常CDF0は`CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8`、J/U SHA256 `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`。ENTRY E2は`H4_100_C0_ENTRY_g003_budget64`（C0/γ=.03/64更新）、J/U SHA256 `0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5`。r64/r16は旧selectionから固定し、再選択していない。近似DF Hamiltonian・own-groundは候補ごとに異なる。

Qiskit1.3.0/P0、optimization_level1/seed17、basis rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、既存MPFR160 declared phase、位相fitなし。Actual raw/compiled inverse、head/X/Y shell、fenced serialを維持した。P0全wordの終了・一様数値誤差を新しい実行条件にせず、実行した全有限回路の既存binary64 correctnessを検査した。

## 2. SOURCE・契約・実行方法

[CASE A準備資料](research_c_cdf0_entry_independent_price_precision_readiness_20261108.md)と[固定proposal](research_c_cdf0_entry_independent_price_proposal_20261108.json)を対象に、会話での「実行してよい」を受け一回実行した。承認証明書・新grantを追加せず、準備時点のproposal/technicalの未承認statusとexecution_permission=falseはbyte原本として保持し、明示execute引数で起動した。準備時点の未実行という記述は履歴であり、現在の実行結果は本書と新result-setを参照する。

- 科学コードcommit：`4f73bec3184523207777ce7f09c75c7cdfd3cb31`
- SOURCE SHA256：`934bfff3923609b5add4b9ec33ea2a9f2f3ac9004f41d7768f77251e62fb6f89`（114ファイル）
- proposal SHA256：`e41aaec18752e6a5f72d8ee7122d08b20b818162db367d7677aaa84a90d5af48`
- fresh output：`artifacts/research_c_independent_price_precision/2026-11-08/science/run01`
- 起動：`scripts/run_research_c_cdf0_entry_independent_price.py --execute`

[実行前環境・受動照合](../../artifacts/research_c_independent_price_precision/2026-11-08/execution/preflight.json)にPython3.12.3、全指定package version、CPU専用Aer0.15.1、専用/tmp overlay、空きCPU/RAM/diskを記録した。共有venvや他jobを変更せず、既存spawn/mmap backendで各32件を32 worker処理。科学runnerは約209.30秒。runtime内部のBLAS/worker協調を再利用し、CPU/RAMへ新しい恣意的な上限は設けなかった。既存のinvalid-escape SyntaxWarningだけを記録し、実行例外はなかった。

保存E2 ground1106 byteのhash `6843e56b955a131583950c585287a41f61c08321b48618c1525b83d2361471b4`を含む全52入力を利用した。新ground solve・DF fit/norm・PF/RTE科学精度評価・RPE shots再計算は0。新trajectory reference行列とlaw/normalization/contextはnative correctnessのため既存処理で評価した。

## 3. 固定N・独立性・会計

新master2026110803、PCG64、axis `INDEPENDENT_PRICE_FINAL_ONLY`、arm/r/batch別cohortを使用した。SOURCE digest・保存JU/再配置version/LD2・baseline role・sample IDを実event記録と照合し、新256 seedの重複なし、旧予約namespace1024 seedとのintersectionなし。これはseedの分離検査であり、PCG64 joint-law/IIDの数学的証明ではない。

各armは32×4バッチのfinal128件のみ。旧selection/finalはrestoreもpoolもせず、旧標本の統計再解析は行わなかった。新観測後のN変更・外れ値除去・confidence選択・cohort選択は0。native標本は古典的な回路価格校正であり、量子RPE shot drawは0。

総費用は保存RPE bindingの`W×新kernel平均+保存shell`。rはkernelに内包され、rやNを追加乗算しない。保存shots/W/shellは入力原本と一致する。今回の主推論はideal_exact_nativeの新CDF0 minus 新ENTRY、一family・一比較・名目95%のみ。D比較、secondary推論、ratio CIは行っていない。

## 4. 新結果

以下は第1節の固定H4/CDF8/LD2/δ=.1条件の新128/arm結果。理想finite-RTE law・exact synthesisを仮定する条件付きfinite-bit価格proxyである。

| 項目 | 新CDF0/r64 | 新ENTRY E2/r16 |
|---|---:|---:|
| native標本 | 128 | 128 |
| kernel RZ平均 | 11220.4375 | 5535.9375 |
| kernel平均SE | 68.62378155158004 | 37.78710916236415 |
| 保存W | 20535472 | 13718624 |
| 保存shell RZ | 60375 | 52913 |
| 保存RPE shots | 17250 | 15118 |
| 条件付き総RZ点値 | **230417040484** | **75945497963** |

CDF0/ENTRY点比=3.033978927839241。ENTRYの点値削減率は約67.039982%で、比の信頼区間ではない。CDF0−ENTRY差点は154471542521、名目95% Welch差区間は**[151506248801.9802,157436836240.0198]**、自由度160.7522926316494、区間半幅2965293719.019806。半幅/新ENTRY点値=.03904502305672512≤.05。

既存`c_near_r32_statistics.primary`がSciPy Welchを一度だけ呼び、既存判定順序を適用した。新標本から得たskewness/influenceは次の通り。delete-oneの分母は各armの個別t半幅であり、Welch差区間半幅ではない。

| 診断 | CDF0 | ENTRY | 登録閾値 |
|---|---:|---:|---:|
| moment skewness | -.1214775430 | -.0737253651 | 絶対値≤2 |
| 最大delete-one移動/個別t半幅 | .1049854139 | .1255162355 | ≤.25 |
| 除去標本 | 0 | 0 | 0 |

両診断に警告なし、精度基準PASS、差区間全体が正のため`NOMINAL_MODEL_ENTRY_LOWER`。名目モデル上の結果であり、診断は未観測の裾・coverage・IIDを保証しない。

## 5. Correctness・call監査・原本保全

[実行監査](../../artifacts/research_c_independent_price_precision/2026-11-08/execution/result_audit.json)はattempt2405、COMPLETE2405、FAIL/inflight0、登録kind別予算と完全一致。256件すべてについてraw/reference、compiled/raw/reference、全512入力出力、位相/スカラー、actual raw/compiled inverseとreturn、X/Y複素信号の既存検査を通過した。256 build/compile、512 actual inverse、1024 native full512 decode、shell decode6を含む。gate許容値1e-10（信号2e-10、normalization1e-12、rtol0、phase fitなし）を維持した。

全event数はCDF0=8192、ENTRY=2048。全バッチでreused0/computed32、価格0への故障置換・不適格標本除去は0。新SOURCE114/全入力52/旧STOP/旧科学結果は実行後もhash一致。旧grant/ledger/markerは変更しない。CODE/contractを科学実行中に修正せず、retry/resume0で`REGISTERED_TASKS_COMPLETE`の正常STOPを保存した。

実行後の受動・権限・freshness関連26テストもPASS（18件は対象外）。完了済みrawを保全したまま、テスト側だけで未使用出力を模擬するfixtureへ調整し、既存ディレクトリの再使用拒否を保持した。科学SOURCE/runnerの変更や科学再実行は0。

## 6. 公開rawと再現

[publication案内](../../artifacts/research_c_independent_price_precision/2026-11-08/publication/README.md)から、summary、native256件CSV、全raw2882ファイルのSHA256索引、全テキスト原本1588件のgzip capsule、全binary原本1294件のtar.gz分割12本へ接続できる。binaryはQPY/reference NPZだけでなく保存ground、worker mmap/cache/pickleもbyte原本のまま含む。約1.289 GBのrawを、全binary約49.918 MBとテキストcapsule約1.284 MBへ可逆包装した。

全text原本と全tar memberを再hashし、分割結合したarchive hashも一致した。これは成果物のbyte監査であり、回路・配列decodeやCIの再計算ではない。ローカルrawは保全し、gitには直接参照する主結果と可逆公開packageを保存する。公開commitのSHA/tree・本文・主要SOURCE・raw blobを照合してSTOPする。外部環境での再実行やimmutable CI再現を済ませたとは扱わない。

## 7. 旧結果・主張の限界・STOP

旧CDF0 run02のfinal64対保存ENTRY final64は`UNRESOLVED_PRECISION_TARGET`のまま。旧区間・6.09945%という精度未達を128標本のPASSへ書き換えず、今回を別の正式結果として保存した。過去のENTRY対PR-aware判定も不変で、新しいPR-aware価格評価は0。

本結果は保存候補・異なる近似Hamiltonian/own-ground・固定r・有限最適化履歴に条件付く。通常CDF0とENTRYの費用差を新cohort同士で再確認したが、正則化の因果効果、全候補/全geometryでの優位、真の最小費用、実CA達成、PR全law誤差・ideal-law価格移送・hardware/FT最終総費用は未資格である。価格統計とRPE統計誤差は区別する。

**全登録scope完了、正常mandatory STOP。追加標本・再選択・別候補・全word保証研究・compiler変更・重要GPT科学レビューへ自動継続しない。**
