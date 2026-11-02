# 研究C・C1 Stage 1：保存fitの評価専用科学実行（2026-10-25）

**固定SOURCEの1回の評価を完了し、判定gate37件は全件PASS、FAIL／NOT_RUNは0。** 物理DF誤差2件は非gateの独立した測定であり、PASS数に加えない。新fit／optimizer0、科学call141／最大141、retry／resume0。`STAGE1_REPRESENTATION_PASS` を保存しmandatory STOPした。Stage2/3・回路build/compile・PR/samplingは実施していない。

以前FAILした保存fitのfull256 frame0は、今回の修正済みraw CAR評価で誤差 `7.11493827835666e-15` Hartree、固定式によるallowance `4.428842301497364e-10` Hartreeに収まりPASSした。baseおよびframe0〜7もすべてPASS。旧FAILの記録／行列／STOPは変更せず、今回の新SOURCE・新approvalによる再検証結果として区別する。NumPy／BLAS内部障害の機序の確定や一般的な再現性証明までは主張しない。

## 固定条件と来歴

| 項目 | 実行条件 |
|---|---|
| Repository / branch | HIROMU1015/Partially-Randomized-Trotter / `codex/research-c-c1-eval-only-science-20261025` |
| 実行した公開SOURCE | `bec41ac7dc7e034bd664817acfdcfed858ddadd0`。科学実行前はtracked差分0 |
| 評価entryの実装commit | `88169af81cbc1140c4e51cfc1acf42a66db340b0` |
| 修正library | `2ab9977994afa31cdfffb8629ce57c6efa83a842` の `c1_stage1.py` をbyte一致で再利用、kernel `car_density_batched_matmul_v2` |
| 契約 | [原本](research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)、scope `C1_STAGE1_P_H4_SAVED_FIT_EVAL_ONLY_V1` |
| 契約SHA256 | `f4d3aa81b3a4fd2272144f0c45fd82bb2aa748f4fc59a6f5875e5dd3e0949dfc` |
| 評価SOURCE manifest SHA256 | `5be0fea39b167845343f94985699f31a0a1bb07d085ea2ed9312a1af9b8dede5`（[原本](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/SOURCE.json)） |
| 元fit公開SOURCE / 証拠commit | `da2edb465d5ae6432439fbd1df8e57512ee59c11` / `f65bbba4d1c54658dfffee11a960370089789030` |
| 元fit SOURCE manifest SHA256 | `e7bf161d2ab45a78d772c960ab5c2c41d3ed145bfd0a1193ee280536c1547240`。新評価manifestに置換しない |
| 保存export | [旧fit_export.npz](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/fit_export.npz)、SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 元入力 | 保存P-H4、commit `86b8b002704aced8ca239414080f53793de81658` のsnapshotと同一bytes、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 分子 / basis | 直鎖H4・隣接1.00 Å・STO-3G、charge0／spin0、保存実canonical RHF MO、4空間／8 spin軌道 |
| 物理sector / full | Nα=Nβ=2、36次元。base・各frameと粒子数保存はfull256でも確認 |
| C-DF | 保存ffsim v0.0.83通常C-DF、number表現、元g補正κ保持、一般Kを元の1 frameとして保持、実outer rank8 |
| 元fit条件 | max_vecs8、tol1e-8、L-BFGS-B、options={}。今回の新fit／optimizerは0 |
| X-DF | 同一inputの保存実rank10、元生成df_rank=None／df_tol1e-8。再分解なし |
| allowance | `binary64_operation_scale_guard8_v1`、限定pilotの工学的数値allowance。式・係数・precision無変更、厳密certificateではない |
| L_D / delta窓 | Stage1の表現評価では非該当。PF／回路評価なし |
| fresh output | `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01`、独立worktree `/tmp/prt-c1-eval-only-science-20261025` |

契約原本の準備時点の未承認表示を変更せず、今回の明示承認を[専用approval](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/approval_run01.json)に記録し、SOURCE・契約・元input・保存export・fresh絶対outputへbindingした。[承認内容の記録](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/user_authorization_record.txt)と[実行provenance](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/provenance.json)を保存した。旧grantの再利用や旧metadataの書換えはない。

[byte/schema/history preflight](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/preflight_bytes.json)はPASS。SOURCE32ファイル・元fit来歴9ファイル・入力・exportを公開commitへ照合した。[環境preflight](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/environment_preflight.json)はCPython3.11.1／NumPy1.26.4／SciPy1.14.1ほかeval41 pinsが一致。既存専用venvをread-onlyで使用し、Python interpreterのsymlinkを保持した。新規インストールや共有venv変更なし。科学実行直前の[評価専用人工19 test](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/artificial_regression.txt)もPASS（[JUnit](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/artificial_regression.xml)）。

## 全gateと独立した物理DF誤差

[結果原本](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/result.json)からの転記。追加の行列decode・norm計算・保存旧行列の再解析はしていない。独立raw CARと一般Kの構築経路は既存実装をそのまま使い、演算子共通のmode／bit／spin順序とscalar所有を維持した。

構造gate19件、full base/frame同値性9件、粒子数保存4件、sector Hermitian4件、sector adapter同値性1件、計37件PASS。規定順序どおりfull base/frame0〜7 PASS後にsector同値性と物理DF normへ進んだ。物理／XDF sector行列の構築自体は、固定契約どおりfull検査前に行うが、その時点でDF誤差normを測定していない。

| gate / 測定ID | 誤差 | 数値allowance | 判定 |
|---|---:|---:|---|
| `kappa_Hermitian` | 8.660473972585522e-17 | 3.518379690624931e-11 | PASS |
| `kappa_original_g` | 0 | 1.8829801476580426e-11 | PASS |
| `saved_XDF_kappa` | 6.7235128089760177e-17 | 2.5246248268644952e-11 | PASS |
| `J_symmetric_0` | 0 | 2.051712654016196e-13 | PASS |
| `U_unitary_0` | 6.8893410559850757e-16 | 1.2505552149378008e-12 | PASS |
| `J_symmetric_1` | 0 | 8.4073782974233653e-14 | PASS |
| `U_unitary_1` | 5.222180388738334e-16 | 1.250555214937801e-12 | PASS |
| `J_symmetric_2` | 0 | 4.5050933236495543e-14 | PASS |
| `U_unitary_2` | 8.3329534739503604e-16 | 1.2505552149378008e-12 | PASS |
| `J_symmetric_3` | 0 | 8.6294621177430161e-15 | PASS |
| `U_unitary_3` | 5.9029774974824496e-16 | 1.250555214937801e-12 | PASS |
| `J_symmetric_4` | 0 | 5.2778094751610599e-15 | PASS |
| `U_unitary_4` | 4.1027585493109925e-16 | 1.2505552149378008e-12 | PASS |
| `J_symmetric_5` | 0 | 5.2802847776052027e-15 | PASS |
| `U_unitary_5` | 5.6348443881691634e-16 | 1.250555214937801e-12 | PASS |
| `J_symmetric_6` | 0 | 6.8723733235929233e-15 | PASS |
| `U_unitary_6` | 8.6130883522782668e-16 | 1.2505552149378008e-12 | PASS |
| `J_symmetric_7` | 0 | 3.2482077697973961e-15 | PASS |
| `U_unitary_7` | 1.619157848017643e-16 | 1.2505552149378008e-12 | PASS |
| `full_base` | 3.5527136788005009e-15 | 1.7591898453124768e-11 | PASS |
| `full_frame_0` | 7.1149382783566607e-15 | 4.4288423014973639e-10 | PASS |
| `full_frame_1` | 2.8489394642099187e-16 | 4.6226096402411803e-10 | PASS |
| `full_frame_2` | 2.8614812432396067e-16 | 8.5134544773428704e-11 | PASS |
| `full_frame_3` | 2.2249299369785851e-16 | 2.0766948070759662e-11 | PASS |
| `full_frame_4` | 9.8440948720208623e-17 | 2.2573729687611672e-11 | PASS |
| `full_frame_5` | 8.6120509803879147e-17 | 1.8473140107046585e-11 | PASS |
| `full_frame_6` | 5.5854059129087263e-17 | 6.8000655030491581e-12 | PASS |
| `full_frame_7` | 4.8572257327350599e-17 | 9.9137091717779434e-12 | PASS |
| `raw_N0` | 0 | 5.8684311191036106e-08 | PASS |
| `raw_N1` | 0 | 5.8684311191036106e-08 | PASS |
| `K_N0` | 0 | 9.3281842875445301e-08 | PASS |
| `K_N1` | 0 | 9.3281842875445301e-08 | PASS |
| `physical_Hermitian` | 1.0985550594078784e-15 | 1.7076932369948171e-08 | PASS |
| `CDF_Hermitian` | 8.8975597563602839e-16 | 5.5016541741596346e-09 | PASS |
| `K_Hermitian` | 9.8796505240531269e-16 | 8.745172769572997e-09 | PASS |
| `XDF_Hermitian` | 2.5559231228807706e-16 | 6.9818534062769377e-10 | PASS |
| `sector_adapter` | 4.011807592116179e-15 | 1.1872355786443861e-09 | PASS |
| `sector_DF_CDF` | 7.6660277613925014e-05 | 1.8815490941337448e-09 | 測定のみ（非gate） |
| `sector_DF_XDF` | 3.3751558354542566e-15 | 1.4812598092146554e-09 | 測定のみ（非gate） |

full／sector同値性と物理DF誤差はspectral operator norm、unitary／対称／Hermitian／粒子数交換子は実装に定めた構造Frobenius診断。物理誤差と同値性はHartree、U unitaryは無次元。すべてbinary64、SciPy gesdd、Hermitian射影や位相fitなし。

sector36の一般K adapter同値性誤差は `4.011807592116179e-15` Hartree、allowance `1.187235578644386e-9` HartreeでPASS。物理sectorにおける `||Hphys−HCDF||₂` は **`7.666027761392501e-5` Hartree**、`||Hphys−HXDF||₂` は **`3.3751558354542566e-15` Hartree**。XDF値はこのbinary64計算の丸め誤差程度であり、その桁数を物理精度の保証とはしない。C-DFの圧縮による物理誤差と、adapterの表現同値性誤差を混同しない。

物理DF誤差には絶対合否閾値を設けていない。2件のallowanceは診断の数値スケールであり、物理誤差の採択閾値ではない。C-DF≤X-DFの制約もない。frame0の旧allowanceと今回のallowanceの数値差は、同じ固定式が各評価の数値スケールと差行列のFrobenius項を使うためであり、guard8／norm係数32／演算長／判定式の変更ではない。事後的な閾値調整は0。

## 科学callと保存証拠

[call集計](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/call_summary.json)、[load原本](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/load_calls.jsonl)、[evaluation原本](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/evaluation_calls.jsonl)。

| 計上項目 | 実数 |
|---|---:|
| input load / saved export load | 各1、計2 |
| adapter import / physical build / XDF build | 各1 |
| XDF basis action | 36 |
| raw/K base build | 2 |
| raw/K frame build | 16 |
| full sum / sector restrict | 各2 |
| spectral norm | 12 |
| structural Frobenius | 27 |
| allowance | 39 |
| evaluation subtotal | 139 |
| **合計 / 上限** | **141 / 141** |
| 新fit / optimizer / retry / resume | **各0** |

規則は `77+8R`、保存R8で141。台帳はload2行／evaluation102行だが、まとめて記録するXDF36 action等の `count` を加重集計し139評価単位とする。行数をcall数と取り違えない。旧fitの6計上単位は旧来歴としてのみ保持し、新callに加算しない。新tensor residual／objective／energy・追加解析は0。

[operator inventory](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/operator_inventory.jsonl)は24 NPZ：physical／XDF sector2、raw/K base2、raw/K frame16、raw/K full total2、raw/K sector2。全byte hashを照合した。[JSON・byte監査](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/evidence_audit.json)は全gate、call bound、spectral順序、出力・SOURCEの整合をPASSとした。監査は数値配列をdecodeせず、既に記録した値・台帳・bytesだけを検査する。

## 計算資源・保全・公開

[実行前資源](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/resources_preflight.json)：affinity32 CPU、load約1.21、memory pressure0、GPU driver利用不可。固定実装のframe pair8 worker／BLAS4 thread、immutable CAR bilinear cacheとbatched matmulを利用し、恣意的なCPU／RAM／worker／wall capは置いていない。実行時のSOURCE／アルゴリズム／検証pipelineの変更0。既存resource-only監視wrapperはentry・固定pathとtask cacheだけを今回用に変更して再利用した（[差分来歴](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/launch_wrapper_provenance.json)）。

[GNU time](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/resource_time_run01.txt)はwall3.64秒、user13.48秒／system3.77秒、最大RSS461,460 KiB、swap0。[1秒間隔の監視](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/resource_usage_run01.json)はpollを含むwall約4.005秒、4 samples、aggregate RSS最大414,888 KiB、threads最大72。測定範囲とsample時刻が異なり、thread数を同時使用CPU数とはしない。資源圧迫やtimeoutによる停止ではなく、性能speedupの比較も行っていない。

[保全記録](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/preservation.json)：旧research C artifact740ファイル、固定SOURCE32ファイル、元workspaceの指定ファイルとtracked差分が不変。旧grant／run01/run02／ledger／marker／STOPを上書き・流用せず、今回の専用approvalとfresh outputだけを追加した。現行概要・索引・validation manifestは新結果を追記し、旧entryを保持する。task専用cache／出力／process以外の共有設定・他ユーザーのjob/fileへの変更はない。

証拠を同branchへcommit／pushし、remote SHA、SOURCE、契約、成果物のGit blob／取得bodyを照合する。これは公開保存したローカル実行証拠であり、独立した外部再現やimmutable CIの結果ではない。

## 次の判断とmandatory STOP

[STOP原本](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/STOP.json)はmandatory_stop=true、新fit／retry／resume0、Stage2/3_authorized=false。今回の承認範囲を完了し、公開照合後も追加科学callを行わない。

- **ユーザー／研究C（次の判断）**：P-H4・保存R8のC-DF物理誤差 `7.6660e-5` Hartreeを、将来の全誤差予算と照らしてStage2対象として採用するか判断する。表現gate PASSだけではStage2適格性やXDFに対する総費用優位を確定しない。
- **Codex（別途依頼・承認後の作業案）**：採択した表現／誤差予算でStage2のnative回路・Gaussian convention・build/compile検証範囲とSOURCE・契約を準備する。実科学実行にはその対象の明示承認が必要。

Stage2回路build／compile、Stage3 PR・sampling・shot／総資源評価は未承認・未実施。**mandatory STOP。**
