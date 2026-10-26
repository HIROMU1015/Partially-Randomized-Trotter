# H4 CDF8／保存順XDF8：共通precision・native・full512合成誤差の技術準備

2026-10-26 JST。起点 `1f318bb9fc4e412bda647b424278deb80dbc5790`、独立branch `codex/research-c-c1-h4-native-fair-preparation-20261026`。

**実行直前までの技術準備を完了した。新規20＋既存136＝156 unique testsが適切なprocess分離でPASS。実H4の数値decode／build／compile／operator／合成／CA／shots／総費用の新評価は0。公開照合後mandatory STOP。**

- [薄いnative adapter／限定runner本体](../../src/trottertracks/c1_h4_native_fair.py)／[full-space Frobenius・伝播接続](../../src/trottertracks/c1_h4_full_synthesis.py)
- [passive既定CLI](../../scripts/run_research_c_c1_h4_native_fair.py)／[人工test](../../tests/management/test_research_c_c1_h4_native_fair.py)
- [固定未承認305-call契約](research_c_c1_h4_native_fair_contract_unapproved_20261026.json)／[SOURCE・人工原本・byte preflight](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/technical/)
- [前段XDF8の26 PFセル](research_c_c1_h4_xdf8_pf_execution_20261026.md)／[旧CDF8・XDF10精度費用結果](research_c_c1_stage3_deterministic_accuracy_execution_20261025.md)

## 採用条件・公平性

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、full system256／ancilla-last controlled512。保存通常CDF rank8 numberと、OpenFermion1.6.1 eigh由来の保存λ/G元順prefix8を比較する。元g補正κ・核scalar・spin/bit/frame順を維持し、全frame ordered S2、s=1/4、δ=.1、m=0..12、PR split非適用。own-DF ground oracle、状態準備費用除外、CA=.00159 Ha／DF reserve CA/8。

同じrank・DF予算でも両armの物理近似誤差は異なる。今回の単一kernel費用差だけを、共通CAを満たす総資源優位とは扱わない。

今回ユーザーが**両arm共通のH6精度修正、4 kernel＋head/X/Yの3 shell、full512 Frobenius方式を採用**した。literal routeの追加比較・一般相対Gaussian融合は含めない。

| 条件 | 両arm共通の固定値 |
|---|---|
| Gaussian | `nature_jw_exact_zero_lapack_atan2_v1` |
| P0 phase | `qiskit_P0_mpfr160_declared_phase_v1` |
| P0 profile | Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false。backend/target/coupling/layout指定なし、非identity layoutを拒否 |
| basis／角度 | 既存NativeBuilderの隣接**完全同一rotation**共有、math.fsum角度結合、exact-zeroのみ除去 |
| 回路境界 | head／1 h-step kernel／readoutは別compile。後続serial反復は既存all-wire fence規則。kernel内部へ新しいfenceや最適化は加えない |
| correctness | max-entry1e-10、unitarity1e-10、X/Y complex2e-10、rtol0、phase fitting／閾値緩和／compiler fallbackなし |
| 費用 | RZ count/depth、CX count/depth、total depth、circuit size。単一h-step kernelだけを同sで比較 |

旧CDF回路の同一性を無条件には認定しない。**次の限定実行でCDF8も新たにbuild/compileして再資格付けする。旧CDF metrics・shotsやXDF10の結果を新比較へ流用しない。** 旧CDF8/XDF10約27%の条件付き総RZ削減はその登録条件の独立した歴史的結果として保全する。新比較の削減量は未測定。

## SOURCE再利用と実装差分

| 既存資産 | 再利用内容・確認した適合性 |
|---|---|
| `NativeBuilder`／`prepare_cdf` | 保存J/Uから一般Kの中央I/Z/ZZ primitivesとκを準備。元frame順、核scalarのancilla相対位相を維持 |
| `prepare_xdf` | 保存κ・signed λ/G[:8]から既存square primitivesを作る。新DF分解・reorderingなし |
| `c1_h6_native_precision.PrecisionBuilder` | **CDFもXDFもNativeBlock.rotation→同じ `_basis` を通る。** XDF squareを直接Gaussian helperへ迂回させる別経路はない。消去順序／U.T／RZ・XXPlusYYを保ち、exact-zero／zlartg／atan2／単粒子transfer確認をそのまま適用 |
| `basis_request`（新規薄い接続） | 既存cacheを変更せず、同一rotationのcache hitにも各blockのprecision reportを付ける。18 blockすべてのcoverageをfairness gateで確認 |
| `c1_h6_compiler_precision.compiled_wrapper` | 同じP0 pass/gate bodyとMPFR160 declared-phase集計。出力状態を使うphase fitなし |
| `CompileSerialCalls` | P0とactual inverse生成の既知Qiskit共有translator競合をtask内mutexで分離。重い行列処理は並列 |
| `stage1.car_bilinears`／`xdf_car_generator` | 全Fock CARからXDF8 base＋8 frameを独立構築。CDF8 full CAR9行列は保存bytesを再利用 |
| `reference._ordered_exponentials`／`ordered_schedule` | 元factor順とexact Fraction時間のS2 reference。local18 expmをcacheし、s1の半stepへ再利用。新PF誤差評価器は作らない |
| `Operator`／`actual_adjoint`／`readout_segments`／ancilla-Z／signal_gate | 全512入力・出力のoperator、実inverse回路、共有shell、m0 own-ground X/Yを既存処理で評価 |
| `numerical_segment`／`component_check` | 既存のmax-entry/Frobenius・finite・unitarity検査。SVDを含む旧matrix_gate／controlled_gateは新経路から呼ばない |
| `reference_info`／`repeat_bound` | H6で使用した数値norm guardとtelescopingを再利用し、両方の数値行列とshellのnorm growthを保持する最小接続を追加 |
| `Calls`／`resource_plan`／QPY／circuit_metrics | 既存call計上、STOP、資源選択、QPY保存、6費用指標を再利用 |

新コードは対象の読込・dispatch・保存・公平性gateとFrobenius伝播接続。Gaussian合成器、CAR、PF、compiler、state simulator、norm evaluatorを再実装していない。旧SOURCE・固定契約は無変更。

## 固定入力と来歴

| 対象 | 固定するidentity |
|---|---|
| 元P-H4 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。旧commit86b8b002…のbytesをignored `.external/P-H4.npz`へコピー、数値decodeなし |
| CDF8 fit SOURCE／export SHA256 | `da2edb465d5ae6432439fbd1df8e57512ee59c11`／`c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| CDF8独立full CAR SOURCE | `bec41ac7dc7e034bd664817acfdcfed858ddadd0`。raw_base＋raw_frame0..7を保存再利用 |
| CDF8 PF／ground | SOURCE `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f`、既存D s1/s4の26セル、own-ground／alias／DF reserve保存証拠 |
| XDF8 DF／PF SOURCE | `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`／`328f31007ee8069d1f7ba24f91029a3748b59109`。保存sector9因子、own-ground、kernel2、PF26セル |
| H6 precision科学SOURCE | `3d7b56b44832a2a90a8d3e500bf55e856f36da76`。旧修正codeを無変更再利用 |

実行契約に29 SOURCE filesと33 saved inputsのhashを固定した。保存入力は技術準備中byte/hashだけを照合。実行時は既存export loaderで元fit由来metadata／array hashを維持し、原inputとκ／scalar、rank・frame・sector順、保存DF予算・PF/ground来歴を照合する。新physical-minus-DF norm・fit・ground eigensolve・PF52セルの再計算はしない。

SOURCE manifest SHA256：`fc00dfef12de6e642ae3a5f52b7b495ad161fb2c2bc6c70f187af7802d74a464`。

未承認科学契約SHA256：`4a99e76ea879226e077376dd69f12b529f9eb1a48e98dd048143892a59b0e578`。

## full512評価・13round接続と限界

各referenceは独立CARから作るfull256のS2 kernel Wを`diag(I256,W)`としたfull512 operator。局所18因子も同じcontrolled規約で検証する。Gaussian gateのsector射影や有限probeによる代用はなく、全512入力と全512出力を覆う。control0 identity、control1、κ/scalar相対位相、raw/compiled actual adjointと復帰、head/X/Yを確認する。

主量は **ε_F=||V_compiled−U_reference||_F の直接値**。raw/refとcompiled/rawを別gateにし、compiled/refを独立に測る。最大成分差は受理gateでありoperator norm上界ではない。Frobeniusの数値値は表現された数値行列のoperator誤差を上から評価するpilotとして扱い、丸めenclosureや厳密certificateへ昇格しない。full SVDは呼ばない。旧full256/32-thread SVD異常は未解決のまま保全する。

既存reference_infoからraw/P0/referenceと全shellの数値unitarity欠損・norm上界pilotを取り、共通α≥1にまとめる。両方をαで規格化して既存repeat_boundを使い、n=s2^mに対して

`ε_n ≤ α^n min(2, n ε_F/α)`

を使う。head／kernel^n／tailを一度ずつtelescopingし、axis state距離を

`d_b ≤ α^(n+1)(ε_head+ε_b)+α² ε_n`

complex signalを`2 α^(n+2) hypot(d_X,d_Y)`で評価する。α=1の単純式へ無断で置換せず、最大n=16384を保持。これは**既存norm計算値を入力とした数値pilot接続**であり、floatのnorm guard自体が厳密に囲われたものではない。

各DF固有ground上の保存PF operator誤差はcomplex matrix element誤差をfactor1で抑えるため、合成signalと一度だけ足し、その後asinで位相へ変換する。DF誤差（Ha）はここに足さない。52行でπ/12以内なら`POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL`、十分条件超過は`INFEASIBLE_REGISTERED_BOUND`、証拠不足は`UNRESOLVED`。いずれも最終CA認定ではない。

**reference対応の限界**：XDFは再構成full referenceのsector blockと保存kernel2を照合できる。旧CDFの数値kernel自体は保存されておらず、同じhashのfull CAR9因子・順序・時間・既存reference codeによる**同じ解析的PF target**の対応を使う。新旧数値kernelのbitwise同一性やreference形成丸めを証明したとは扱わない。CAR/expm／Operator復元／差とnorm評価／旧PF基準のrounding enclosureは未知のまま保持する。数値受理閾値をそのenclosureへ転用しない。

m0のraw/P0 X/Yは実際のown-ground作用で比較する。全13roundの実X/Y信号を直接再計算する契約ではなく、残りは全operatorからの数値伝播profileである。

## 科学call・実行順序・規模

**上限305 logical units**。H6の298/6535や前段93をコピーせず、37分類の内部処理を契約に列挙した。

| 分類 | 上限 |
|---|---:|
| input/export/evidence10/CDF行列9/sector packet・行列/ground2/kernel2 | 27 |
| identity／model2／CAR1／XDF factor9／sector identity2／fairness | 16 |
| Gaussian18／local reference18・build18・Operator18・gate18 | 90 |
| shell build3／小2×2 structure6／kernel reference4・sector照合2・build4 | 19 |
| P0 compile7／主要raw-P0 Operator14／segment7 | 28 |
| actual adjoint build8／Operator8／check8 | 24 |
| signal reference4／prepare8／axis16／complex gate12 | 40 |
| metrics7／round synthesis52／同s費用比較2 | 61 |
| **合計** | **305** |

基本比較対象は4 kernel＋3 shell＝7回路。正しさ検証のため局所18回路とactual inverse8回路を別計上する。計40 full512 Operator復元と6件のancilla2×2検査、QPY40件、保存operator packet18件（system reference4＋主要raw/P0 14）を予定。Gaussian18はblock別requestで、既存の完全同一rotation cache hitを保ったままcoverageを記録する。

局所保存名の`CDF_Fi`／`XDF_Fi`はblock indexで、i=0がκ＋核scalarのbase、i=1..8が元frame F0..F7に対応する。Gaussian reportのframe_id自体は元のF0..F7を保持する。

入力・表現target・precision公平性→局所18全PASS→共通shellと4 kernel/P0/full operator/adjoint/X-Y→52合成/PF位相profile→単一kernel費用2比較の順。最初のcorrectness FAILで新しい依存callを止め、開始済みjobを回収し、partial QPY/operator/reportとfirst tracebackを保存する。数値NOT_MET／UNRESOLVEDはcorrectness FAILと分離し、他の登録候補を継続。fresh outputは`artifacts/research_c_c1_h4_native_fair/<execution-date>/run01`、retry/resume0、正常・不足・FAILいずれもmandatory STOP。

full512 complex128は1行列4 MiB、full CAR bilinearsは64 MiB。旧H4 Stage2の251秒はSVDを含む別workloadの運用記録で、今回の速度推定／speedupの証拠にはしない。dominant workloadは40件の深い回路Operator復元、7 P0 compiles、局所/full reference。実行時間は実H4未実行のため未測定で、分単位以上になる可能性がある。

現在のpassive資源snapshotはaffinity32 CPU／空きRAM約52.8 GiBで、既存resource_planはlocal18 workers／候補4 workersを選ぶ。小行列・独立jobにprocess-local BLAS1を協調し、compile/inverse生成だけ既知mutexを維持する。workerは実空きRAM/quotaから再選択し、固定CPU/RAM/wall capは置かない。実行時cache/tempは当task内へ分離し、共有環境・venv/cache・他ユーザーjobを変更しない。

## 人工・回帰テスト

新規20件PASS（全体19件＋追加native-basis拒否1件）、既存core109＋旧Stage2 mock独立process27＝136件PASS、**計156 unique tests PASS**。最終I/O/cache・後続接続・native出力guardに対する選択試験もPASSした（重複計上しない）。[機械可読結果・環境](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/technical/validation.json)。

- 人工4-mode/controlled32・rank8のreal NativeBuilder/P0/独立CAR一体試験：305 units、局所18、7 compile、QPY40、operator18、52 profile、actual adjoint、X/Yまで確認。旧Gaussian／fit／physical生成／CA・shot関数は遮断。
- 小角1e-9、π近傍、det±1、signed square／一般K、κ/scalar、向き・順序・位相・compiler異常のSTOP、cache hitを含む18block共通適用を確認。
- 人工full512のsector外column0の残差を保持し、Frobenius／全入出力coverageを確認。SVDを遮断した試験でPASS。数値norm欠損と16384反復、NOT_MET／UNRESOLVED／nullを確認。
- 人工H4-shaped NPZを既存export/packet loaderで実際に読み、27 load、array schema／provenance／candidate identity／ground normalizationを検証。実分子値は使わない。
- 既存回帰：Gaussian precision24、MPFR phase5、native mock27、subspace bound21、deterministic accuracy12、RPE/接続25、XDF8 PF22。
- 初回混在batchは135 PASS／1 FAIL。既存mock fixtureが`ops.py`を偽RZZGateの下で初回importする前提を持ち、実Qiskit試験が先にreal opsを読み込むとmock gate名の検査に失敗する。旧SOURCEを修正せず、core109とmock27を別processにして全PASS。混在FAILの原本も保持し、消去・科学的FAIL扱いはしない。

科学用評価環境は既存read-only `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`（literal path、Python3.11.1）。NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Nature0.7.2／SymEngine0.13.0等9版と既存requirements.lockを固定した。実packet/source/inputのpassive CLIはBYTE_PREFLIGHT_PASS、NumPy/Qiskit未import・科学call0、実run01なし。

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_native_fair.py \
  --contract docs/research/research_c_c1_h4_native_fair_contract_unapproved_20261026.json \
  --input .external/P-H4.npz
```

これはbyte確認のみ。将来の別明示科学許可後だけ、同じCLIへ`--execute --output artifacts/research_c_c1_h4_native_fair/<execution-date>/run01`を追加する。

## 後続CA・shots・条件付き総費用

今回production runnerからshots／energy／total-cost関数は呼ばない。後続では同じ`AccuracyTarget`、`assess_round`、Hoeffding solver、`assess_energy`、`compose_serial_cost`、`aggregate_totals`、`compare_totals`を両armに適用する。人工データではその接続も確認した。

CA=.00159 Ha、DF reserveCA/8、β_total=π/6／β_stat=π/12、候補failure=.01・各round/axis=.01/26、ideal IID Born、own-DF oracle／状態準備除外、13round X/Y、serial native costを維持。合成・PFのsignal/phaseを接続してから共通shotsを新しく算定し、最後にDF norm＋guard（Ha）を一度加えてenergy適格性を判定する。新precisionのmarginを確認せず旧CDF shotsを使わず、旧XDF10のshots／費用／CAも流用しない。同CAが適格な両armだけ総RZ比較へ進め、depthはserial scheduled sumとして扱う。

## 次に必要な許可・mandatory STOP

今回範囲の技術blockerはない。実H4で全gateが通るか、Frobenius伝播が位相予算に有用か、費用差がどうなるかは**未測定**。厳密rounding enclosure・CA保証・状態準備込み／無条件資源優位も未資格。

次は**ユーザー／研究C**が固定公開SOURCEと上記未承認契約を確認し、**最大305 logical unitsのnative/full512合成pilotだけ**を別途明示許可するか判断する。許可後に**Codex**がSOURCEを変えずfresh run01を一度実行し、7主要回路・局所/adjoint・52 profile・native metrics・call/STOPを公開する。承認証明書・原文保存・ID・receipt・新grantは不要。

今回の実H4科学call／新fit／DF-PF誤差再計算／build/compile／合成／CA／shots／総費用はすべて0。旧H2/H4/H6のSOURCE/raw/grant/ledger/marker/STOPを保全し、公開照合後mandatory STOP。詳細科学レビューや後続科学計算へ自動進行しない。
