# 研究C：保存CDF8/XDF8強Dへの線形Z再配置・限定比較の技術準備

2026-11-07 JST。**実H4の新PF/RPE/native/総資源計算は0。** 保存`B2_CDF_relative`と`B3_XDF_spin_relative`へ固定LD2再配置を接続する新adapter、2候補runner、人工検証、未承認契約とSOURCEを準備した。旧SOURCEを変更せず既存の数学・compiler・accuracy・会計処理を呼ぶ。これは技術準備であり、実保存2候補の適格性・価格・優位を新しく確認した結果ではない。

対象の将来条件は直鎖H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、rank8、元frame順、**F2〜F7だけの線形Z→一体head**、全frame ordered symmetric S2/s1/h=.1/m0..12、own-DF ground oracle・状態準備除外、P0/fenced serial/shell込み。比較先は保存ENTRY E2＋PR/r16の条件付き総RZ点推定 **75,621,395,471**。同じ再配置政策の下での固定2候補比較であり、決定論PFに最適な再配置と仮定しない。

## 1. 固定資料・SOURCE・入力の確認

| 固定commit | 使用した資料と役割 |
|---|---|
| `bfc32c92a9c7a028e2d9065b71b4258809f53c0f` | [比較条件報告§2〜10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfc32c92a9c7a028e2d9065b71b4258809f53c0f/docs/research/research_c_strong_d_alignment_20261106.md)。§10の最小案を今回の準備対象にした |
| `3015d4fb342d4d98b58e41d57553e75539a07655` | [再配置E2＋PRの完了結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md)。引継ぎ実装SOURCEは`cfc64b70861a6482dddd042da8a110abebd6ad7a`。E2 final64のscalar JSON・shell・SOURCE/STOPを保全 |
| `417fa459b823509be5c2010c5836e58955088282` | [旧CDF8/XDF8強D](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。実行SOURCE`1aeedd96178a3f50a15bc2701c3c332f4f391b64`、数値実装`969c17d623d56eb645417b281deadb1be4757699`。旧binding v2契約・B2/B3 packet・結果・STOPを保全 |
| `a67036a6baafc00deaa38a1b0b801d313d519ad5` | [既存再配置の技術準備](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a67036a6baafc00deaa38a1b0b801d313d519ad5/docs/research/research_c_h4_relocated_downstream_preparation_20261106.md)。元LD2再配置、scalar、独立CAR、全512検査を再利用 |

元P-H4 snapshotのSHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、公開元`86b8b002704aced8ca239414080f53793de81658`、生成SOURCE`0c329b30a81accb35c212be8defa66fee0dd0f12`。既にGit管理された同じbytesの`artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/inputs/P_H4_original_snapshot.npz`を参照し、重複コピーは不要とした。準備中はコンテナbytesのhashとmember名までを読み、数値配列はdecodeしていない。

CDFは保存`fit_export.npz`（SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`）のκ/constant/J/U、旧CDF sector行列とown-groundを使用する設計。XDFは元snapshotの保存one-body/signed λ/Gの元順prefix8、旧XDF8 sector generatorsとown-groundを使用する。κは保存exportを使い、元gから再導出した値に置き換えない。将来のidentity gateでだけ保存κと元g補正の対応を検査する。

正式な共通運用指示`research_project_instructions_unified_2026-10-25.md`原本、projects/およびworktreeのAGENTS、PROJECT_MAPと既存研究状況を確認した。作業branch/worktreeは`codex/research-c-h4-strong-d-relocation-preparation-20261107`。他branchのSOURCEや実行状態を変更しない。

## 2. CDF8/XDF8の保存表現と再配置係数規約

CDF frameは既存規約

\[
H_t=\tfrac12\sum_{ij}J^t_{ij}\widetilde N_i\widetilde N_j,
\quad s_i^t=\sum_jJ^t_{ij},\quad
L_t=\sum_i s_i^t\widetilde N_i-\sum_i s_i^t I
\]

に従う。`c_h4_relocated_downstream.prepare/raw_generators`を呼び、Δκ=Σtail U diag(s) Uᵀ、Δc=−ΣtailΣs、tailからL_tを差し引く既存処理を再利用した。

XDFは既存`df_squared_diagonal_primitives`とCAR実装の **H_t=λ_t(Σ_a η_a n_a)²** であり、squareの前に1/2はない。exact interleaved spin lift G_t=G_t^sp⊗I₂、η_spin=(η₀,η₀,η₁,η₁,…)の下では、再配置計算用の等価なspatial coreは

\[
J^t_{ij}=2\lambda_t\eta_i\eta_j,
\quad\Delta\kappa_t=2\lambda_t\operatorname{tr}(G_t^{sp})G_t^{sp},
\quad\Delta c_t=-2\lambda_t\operatorname{tr}(G_t^{sp})^2.
\]

これは係数規約の対応であり、別general-J候補へのfit・置換ではない。負λを含む人工例で、このfactor 2付きcoreのnumber-CAR演算子と既存signed squareが一致することを確認した。native側は既存XDFのsquareから作った各blockをそのまま保持し、F2〜F7の`rz`だけを除く。元rotation、`rzz`、`global_phase`、F0/F1は変更しない。

## 3. 既存コード再利用マップ

| 既存API | 今回の接続 |
|---|---|
| `c_h4_relocated_downstream.prepare/raw_generators` | CDFのrow sum、κ/scalar移送、独立number-CAR reference |
| `c1_h4_strong_baseline.prepare_xdf_basis/spin_adapted_eigh` | 元XDF signed squareとexact spin-lift、basis証拠。α/β間成分を射影して消さない |
| `c1_stage2_native.NativeModel/NativeBlock/ordered_schedule` | 一体項＋元順8 frame、対称S2、時間・diagonal primitiveの既存帳簿 |
| `c1_h4_strong_baseline.StrongBuilder` | relative Gaussian、analytic vacuum phase、scalar一回、同basis run結合 |
| `c1_stage1.car_bilinears/raw_frame/raw_base` | full-Fock CAR。referenceをnative gateから生成しない |
| `c1_stage3_reference.reference_kernels`、`c1_molecule_extension.PF_rounds` | 独立full/sector S2 reference、PF13 round、repeated squaring、exact-H spectral reference |
| `c_same_ju_strong_d.ideal_plans` | 保存E2と共通のexact-native仮定lane。新shots solverを作らない |
| `c_pr_aware_df_m2.evaluate_accuracy/fusion_gate` | 全13 roundの合成誤差伝播・数値RPE・費用、relative Gaussianの向き・schedule gate |
| `c1_h4_native_fair.compile_checked/decode`、`c1_h6_compiler_precision` | P0の位相保持、full512 CPU double unitary。既存compilerを変更しない |
| `c_pr_aware_df_stage_b2.shell_assessment/XY_gate` | 保存shellの構造と全演算子、raw/compiledの複素X/Y signal |
| `c1_h4_full_synthesis.segment/propagate` | full512 Frobenius・両norm-growthのlevel B数値lane |
| `c1_stage3_accuracy`、`c1_h6_ca_resources.serial_round_cost` | 共通target、energy、shots、s1のserial会計・集計 |
| `c1_stage3_integrated.Calls/resource_plan` | 既存call journal、実資源に応じた並列度。新しい承認管理基盤は作らない |

`c_update064_strong_d.evaluate`はnumber-ArmのJ/U・prepared LD8等を検査するため、そのままXDFへ渡すことはできない。旧SOURCEへの追加変更や偽のgeneral-J Armを避け、prepared NativeModel＋独立CARを受ける小さな共通orchestrationを新ファイルに置いた。数値・compiler・誤差伝播・会計の本体は上表の既存関数を呼ぶ。

## 4. 必要な最小実装と変更理由

- [adapter/evaluator](../../src/trottertracks/c_h4_strong_d_relocation.py)：CDFは既存再配置、XDFはtrace式と既存square blockのZ除去。両者の独立full-CAR、同じD評価接続を実装。
- [固定2候補runner](../../src/trottertracks/c_h4_strong_d_relocation_execution.py)：byte-only preflight、明示的な数値whitelist loader、private processごとのfresh output/STOP、固定点値判定。
- [CLI](../../scripts/run_research_c_h4_strong_d_relocation.py)：既定はpassive preflight。`--execute`は将来の別科学実行許可後に限る。今回このoptionは使用していない。
- [人工test](../../tests/management/test_research_c_h4_strong_d_relocation.py)：実分子NPZのdecodeを拒否するguard、人工ファイルからのloader/worker E2E、2 processの完了・失敗・STOPを検査。

既存の科学SOURCEファイルを上書きせず追加した。新validation経路の案内として既存4索引へ短いリンクを追加した。科学結果・研究方針は変わっていないため、過去の概要・結果・manifestを今回の新科学結果として更新しない。

## 5. One-body・scalar・identity・phaseの扱い

CDFでは既存再配置が計算するκ/Δκ/ΔcとCAR因子を再利用し、native headのbasis政策だけを旧B2の`legacy_spin_orbital_eigh`に維持した。tailの元U・general-J primitivesとrelative Gaussianは保つ。XDFでは保存one-bodyそのものを使用し、κとの数値対応を検査したうえでΔκ⊗I₂だけを加える。XDFは`exact_spin_lift_spatial_eigh_v1`を保持する。

両表現で、全体constantは「保存nuclear scalar＋追加Δc」を一回だけ使う。元frame identityは元のglobal phaseとして残し、線形Zだけを移す。追加Δcをtail identityの置換と取り違えない。D評価時にLD8で再配置をやり直さず、LD2で一度準備した全9 generatorを全frame ordered S2へ渡す。

basisはuncontrolledの対、中央diagonalはcontrolled、scalar位相はancillaに一回。既存StrongBuilderのanalytic boundary/vacuum phaseとP0 MPFR160位相政策を保ち、観測operatorに合わせるglobal-phase fitはしない。

## 6. 人工Hamiltonianの等価性

4 spatial/8 spin、8 frame、正負の一般Jまたは正負λを持つsquare、非自明なorbital回転を人工生成した。既存CARでfull-Fock256全演算子を作り、scalar込みの再配置前後を照合した。F0/F1、元順F0〜F7、各rotation、tail I/ZZを比較し、F2〜F7のZ消去を確認した。

XDFではさらに、既存square primitiveで作った**再配置前の強builder**と独立CAR S2のfull512 operator一致、およびsigned square↔factor 2付きcoreのfull-Fock一致を検査した。spin間の不正成分、κ違い、λ schema、scalar/frame破損を拒否する。

これらは人工入力の数値・代数実装検査であり、保存H4のHamiltonian等価性gateを実行したことにはならない。将来の科学実行でも保存候補自身について同じgateを通す。

## 7. PF/RPE/native接続

旧CDFの保存sector H、旧XDF8の保存sector generators、候補固有の保存groundを新入力にbindingする。実行時は新CARの元Hを保存Hへ照合し、再配置後Hとの一致、保存groundのnorm/residual/最低energy、既存alias windowを検査する。XDF保存sector順は元のascending Qiskit順で確認する。

**保存ground oracleの再生成は不要だが、reference用のsector36固有分解を各候補一回行う。** 保存ground NPZは最低固有対だけであり、全13 roundのoperator PF誤差に必要な全スペクトルを含まないためである。これは新しいDF fit・物理DF norm・ground候補探索ではない。契約にreference spectral factorizations=2を明記し、隠れた「再利用済み」扱いをしない。

新しい分割からPF kernel、m0..12のPF誤差を評価し直す設計である。旧PF/shots/native metricsを新結果として受け付ける経路はない。CA=.00159 Ha、DF reserve1/8、β_total=π/6、β_stat=π/12、候補ごとのfailure .01、axis .01/26、h=.1/s1を旧強D・E2と共通に固定した。

ideal laneは`ideal_plans → assess_round/assess_energy`でexact-native synthesis、deterministic lawを仮定する。numerical laneは`segment/propagate → evaluate_accuracy`で非零合成誤差・norm-growthを伝播する。両方のnative単価は同じ一つのcompiled D回路で、shots・energy資格と総費用を別に保存する。数値laneが未資格ならoverall STOPとし、ideal点だけを使って優位を認定しない。

費用は両laneとも、`W=Σm,axis N[m,axis]2^m`、`C=W×kernel＋shell`。headとX/Y readoutを各interrogationで一回数える。rやclassical価格標本数をWへ追加乗算しない。E2の64価格標本を取り直さず、保存点値を読み取るだけである。新CI・比区間・同時優位・母平均保証は作らない。

## 8. Full512 correctness

各新D kernelについてraw、compiled、実raw inverse、実compiled inverseの4回路を別々に生成・評価し、**全512入力列と全512出力成分**を比較する。referenceは独立CAR256のcontrolled liftであり、sector36や数個のprobeに縮めない。

全Hamiltonian、full/sector S2、raw/compiled/reference、実inverseのdagger/reference、inverse return、control0/1相対位相、full-Fock/sector外leakage、head/X/Y shell、ground上の複素X/Yを検査する。component1e-10、signal2e-10、normalization1e-12、rtol0、phase fitなし。shellの既存helperが返す全行列を再利用し、重複decodeは除いた。

PF/RPE不適格はnative前に停止する。compiled phase破損はmetrics/価格へ進まず例外・STOPを保存する。数値lane不適格はideal laneと別に残し、比較判定を出さない。失敗済みoutputは再使用せず、retry/resume0とmandatory STOPを維持する。

## 9. 人工・回帰検証結果

最終結果は**unique 138件PASS（新規31＋既存回帰107）、FAIL/ERROR/SKIPは0**。[verification.json](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/verification.json)を原本とする。[新規suite](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/tests.xml)は28件/122.96秒、[STOP補強suite](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/stop_tests.xml)は4件/45.85秒、[既存回帰](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/regressions.xml)は107件/236.47秒。STOP補強suite内の既存preflight再確認1件は重複計上しない。suiteを並行実行したため、この時間の和を作業wallとは呼ばない。

既存回帰は旧強baseline、B3、fair native、H4再配置PR、same-J/U Dの5 suite、107件PASS。新規suiteは人工CDF/XDFの両lane/full512/実inverse/13 round、whitelist loader、固定点値・tie・未資格、2 process E2E、fresh/STOP、passive scope/SOURCE違反を含む。追加のSTOP検査はPF不適格、numerical lane不適格、compiled phase例外を実際のworkerに注入する。

開発時にshell referenceのcomplex128指定とtest用Path guardの扱いを修正した。最終版の検証状況だけを資格とし、修正前の試行を独立PASSとして数えない。人工のdeclared DF bound・保存風referenceは技術fixtureであり、実H4のDF性能や資源優位の証拠ではない。

環境は既存`h6-environments/20261012/native-dependency-v2/venv`を読取利用した。実体はPython3.11.0rc1で、旧研究C記録のPython3.11.1とは異なる。数値・compilerの指定package versionは契約どおり照合した。欠けていたthreadpoolctl3.6.0だけを本worktreeの`.cache/relocation-deps`へ追加し、既存環境を変更していない。Python cacheは無効化、test出力は専用領域とした。この検査を旧環境のbitwise再現や外部独立再現と呼ばない。

## 10. 科学実行規模・資源見積もり

| 対象 | 将来の固定規模 |
|---|---|
| 候補・再配置 | 2候補、元順LD2の一政策のみ。新fit/分解/rank/LD/order探索0 |
| Reference | full-CAR原/新因子・H等価性、保存sector/ground binding、sector36 spectral factorization2 |
| PF | sector kernel2、PF13 round×2。full256 S2 kernel2との照合 |
| Native | 各候補1 kernel build/compile、両実inverseで計4 inverse回路 |
| 全行列decode | kernel4×2=8、shell raw/compiled×3×2候補=12。全512入出力を保持。helperの小さなancilla構造検査等は別 |
| RPE/費用 | 2候補×13 round×2 lane。E2は保存点引用、PR新標本0、新統計0 |
| Logical calls | 候補ごと最大143、2候補で286。これは既存Callsの処理段階数であり、BLAS演算や全library内部呼出し数ではない。rootの固定点値判定は別の一回 |

complex128の512×512 matrixは4 MiB、返却kernel行列8個は計32 MiB。full-CAR bilinear tensorは各process64 MiB、既存plannerのdense workspace見積もりは各候補256 MiBで、Python/Qiskit等のRSSは別である。保存するkernel4行列＋referenceは非圧縮で各候補約17 MiB、QPY/JSON等を加えたraw保存量は**0.1〜0.3 GB程度という見込み**である。

候補間はspawnの2 private process、候補内は4独立decodeを実資源plannerに応じて並列化する。compiler/inverse生成は候補process内で順序を保つ。小行列処理と複数processの競合を避けるためBLASはprocess内1 threadに調整し、共有設定を変更しない。元H・候補固有スペクトルは候補内13 roundで共有し、shellはhelperの返却行列を再利用する。

準備中の読取観測ではaffinity64 CPU、MemAvailable約529 GB、disk free約294 GB。上位cgroupのquotaはこの環境から参照できず、無制限とは認定しない。実H4の再配置Dのwall/RSSは未測定。人工full512 E2Eと処理規模からの粗い見通しは**wall数分〜15分程度、合計peak RAM1〜5 GiB程度**であり、保証・実測値・強制上限ではない。実行前の空き資源・共有競合で並列度を決める。旧800標本の時間から比例推定したものではない。

## 11. 未解決の技術的問題

実H4の保存因子自身をdecode/buildしていないため、将来の実入力gateとPF/RPE/native資格・費用順位は未確認である。これは今回の承認境界であり、人工PASSで補完しない。必要なファイルは保存bytesとして取得済みで、missing inputや未実装adapterを残して実行可能と表示する設計ではない。最終passive preflightと検証の結果はverificationに記録する。

PR側の全law synthesis、implemented-law誤差、ideal-law価格移送、actual CA、FT資源保証は未資格のまま。この強Dの追加比較で取得される資格ではない。新規性・論文化・研究CのGO/NO-GOも判断していない。実H4で不一致・不適格が出たら、別候補や別の再配置政策で自動的に埋めず、部分結果とSTOPを保全する。

## 12. SOURCE・保全・MANDATORY STOP

[未承認契約](research_c_h4_strong_d_relocation_contract_unapproved_20261107.json)は2保存候補、23入力、元H/κ/ground、LD2政策、precision/target/compiler、2 lane、全512 correctness、最大範囲、E2固定点、fresh output、retry/resume0、STOPを固定する。[SOURCE](../../artifacts/research_c_h4_strong_d_relocation/2026-11-07/technical/SOURCE.json)は継承64＋新規4の68ファイル。既存の固定SOURCEファイルは変更していない。契約SHA256は`b011325203f1b6e1f5a69f3a32ed058ff9c097f0bb45645135aea93924527bb6`、SOURCE SHA256は`8323d7c76847b24276b500a80545172fd92719b7a82f6a5b298ca601adec4f53`。最終passive preflightはPASS、実H4数値decode/科学実行0、fresh output未作成を確認した。

fresh outputは`artifacts/research_c_h4_strong_d_relocation/2026-11-07/science/run01`で、準備完了時には存在しない。次のコマンドはpassiveのみであり、科学実行は起動しない。

```sh
PYTHONPATH=src:.cache/relocation-deps PYTHONDONTWRITEBYTECODE=1 \
  /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
  scripts/run_research_c_h4_strong_d_relocation.py
```

| 質問 | 回答 |
|---|---|
| Q1 CDF/XDFの再利用実装 | CDFは既存再配置、XDFは既存square/spin-liftを保持するtrace adapter。共通D評価は既存PF/accuracy/native/会計を呼ぶ |
| Q2 一体/scalar/identity/phase | 同じL_t移送として整合。核scalar＋追加Δc一回、元I/ZZ保持。人工full-Fockとcontrol位相で検査 |
| Q3 XDF固有性 | signed λ、square係数、exact interleaved spin-lift、relative Gaussianを保持。general-Jへの再fit・native全体置換なし |
| Q4 人工全演算子一致 | full256 H、square/core対応、full512 controlled operatorを確認。実H4の新gateは未実行 |
| Q5 PF/RPE/native/会計 | 13 round・同target・2 lane・s1 affine会計へ人工E2Eで接続 |
| Q6 全512・実逆・位相・X/Y | 両実inverseを別decodeしfull512検査。return/相対位相/shell/XY/leakage経路を保持 |
| Q7 E2再計算なし | 保存final64のpointをbyte/JSON binding。新E2価格・CIを作らず共通ideal laneで点値比較する設計 |
| Q8 2 lane | exact-native仮定とlevel B数値合成伝播を別保存。片方の未資格を他方でPASSにしない |
| Q9 PASS/FAIL | unique138 PASS、新規31＋回帰107、FAIL/ERROR/SKIP0。人工・回帰のみ。実保存2候補の新価格PASSは0 |
| Q10 規模・資源 | 2候補、PF26 round、native2、kernel全行列8＋shell12、reference factorization2。見積もりと観測を§10で区別 |
| Q11 固定SOURCE・契約・入力・出力 | SOURCE68、保存入力23、未承認契約、未使用fresh outputを準備。別許可待ち |
| Q12 残件・担当 | ユーザーが公開固定commit/SOURCE/契約の範囲を確認し、実H4の一回の独立科学実行を別途明示許可するか判断。Codexは許可前に実行しない |

旧H4元辞書・mass診断・再配置PR・CDF/XDF強D、H2/H6、1.15 Å STOP、旧grant/ledger/marker、他worktree/process/cache、共有package/OS/schedulerは不変更。新しい承認証明書・grant・承認原文保存は作成していない。技術準備に関する本人の検査であり、独立科学レビューではない。

commit/push後にremote SHA・tree・公開ファイル・SOURCE一致を確認して**MANDATORY STOP**。次の担当は**ユーザー**による固定範囲の科学実行許可の判断。許可後の**Codex**の担当候補は固定2候補の独立run一回であり、今回は未実行。重要GPTレビューや他の新科学計算へ自動進行しない。
