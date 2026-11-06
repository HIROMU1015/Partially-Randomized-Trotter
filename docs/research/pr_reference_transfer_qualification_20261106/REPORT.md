# 新A：長round費用移送・全support誤差資格の限定検証

実行開始2026-11-06、完了2026-11-07 JST、SERVER 2。基準commitは `79cba2fc4882bc42282f8feee7257a69f7628072`。新規の独立branch `codex/pr-reference-transfer-qualification-20261106` に保全する。

固定された保存Hamiltonian・compiler A・有限bit sampling・浮動小数点モデルの下で、18roundの十分shot数を再計算した。referenceの総CX・RZの上界が、登録Simon–Love実装の対応する下界を下回る。判定は **`CONDITIONAL_ALL18_CX_RZ_REDUCTION`**。全roundのdepth・size・T、別の制御分解まで最適化したbaselineへの優越、元の連続積分からの無条件包含は未資格である。全資源・実機費用の改善とは呼ばない。

科学実行は[STOP](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STOP.json)で終了。以下はCodex自身の検証と計算機支援上界であり、独立レビュー・CI・GPT科学レビューの結果ではない。

## 固定条件とStage判定

Linear H4、1.00 Å、STO-3G、DF12、Partial S2、LD3、h=0.02、finite RTE K=2。元Hの厳密基底状態を各shotで受け取り、その準備費を除く。18roundは q=2^m (m=0,…,17)、rは保存scheduleの1/2/4/8/16/32。エネルギー精度1.5936e-4 Ha、全失敗確率0.05、位相予算0.4 rad。ordinary diag(I,V)、canonical law、trajectory別vacuum位相補正、F/G境界、2-register reference gadgetを固定した。

compiler AはQiskit 1.3.0、native RZ/SX/X/CX、全結合、近似commutation・角度cutoffなし。対照は既存のbasis非制御化・中央制御・D境界融合を用いたSimon–Love型実装。PF、sampling law、gadget、配置、precision候補の探索は行っていない。

| Stage | 結果 | 根拠・限界 |
|---|---|---|
| 1 構造 | 次段へ進行 | 保存48組とQPYを再利用。q1/2だけのaffine外挿はq8で不一致 |
| 2 価格 | CXの全schedule移送を資格化 | 固定compilerの有限context・境界からbaseline価格とreference上界を構成。短holdout一致だけを根拠にしない |
| 3 誤差 | 固定算術モデルで全18round通過 | 全support局所区間上界、telescoping、vacuum位相丸めを含む |
| 4 RPE | CX・RZの条件付き削減 | 新しい整数shot scheduleを両armで算出。depth・size・Tは空欄を維持 |

## 全18roundの結果

[STAGE4_RPE.json](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json)の保存値を示す。資源は各軸・roundの十分shot数と、固定有限bit lawによる1-shot期待価格の積和である。実際にRPE shotをsamplingした結果ではない。

| 指標 | 登録Simon–Love | reference |
|---|---:|---:|
| 十分shot数、X/Y合計 | 8,598 | 8,606 |
| 総CX | [593,162,239,027.1154, 593,162,239,027.1156] | ≤480,289,613,434.8142 |
| 総RZ | [3,735,644,723,605.280, 3,735,644,723,605.281] | ≤3,577,340,249,201.294 |
| qubit数 | 9 | 17 |
| reference専用reset | 0 | 68,848 |
| source CSWAP | 0 | 137,696 |
| 総depth / size / T | 未資格 / 未資格 / 未取得 | 未資格 / 未資格 / 未取得 |

CXのbaseline期待値は構造上のexact値を外向き数値区間に変換した。RZはcontextごとの局所価格区間である。referenceは実価格そのものではなく安全側の上界。上界対下界の不等式が削減を示すのであって、下界を実行可能baseline価格と呼ばない。fraction形式の軸別価格はJSON内に保存した。

両armに同じ固定alphaを用いた。各36軸のalpha=1/720で合計0.05。半径下界ρ、残存統計位相βに対し ε=ρ sin(β)/√2、N=ceil(2 log(1440)/ε²) を192-bit外向き区間で評価した。alpha最適化は行っていない。旧8,598を新方式へ流用していない。baselineは再計算後に同数となり、referenceの最終roundが各軸626→630となった。

全roundのr、半径、β、整数Nは[STAGE4_ROUNDS.tsv](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_ROUNDS.tsv)を参照。最終round q=131072/r=32では、baseline/referenceのraw半径下界は0.5726873161221171 / 0.5722335555306725、β下界は0.38598697336359056 / 0.38519464269240195 rad。保存された同一Hermitian-Hのalias priorとendpoint marginを再利用し、位相capを変えていない。

## Stage 1–2：価格移送の根拠

保存済みq1/2/8・r1の48 paired realizationとnative QPYを読み取り、新たに価格を取り直さなかった。wire frontierによるcritical pathは保存QPYの実DAG depthと一致した。q1/2からの単純affine予測はq8で外れたため採用しなかった。q2/8 fitも診断専用のままで、全round価格には使用していない。

[事前固定](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE2_PREDECLARED.json)した追加条件は (q,r)=(4,1),(32,1),(128,1),(256,2)。各条件で「order2なし」「少なくとも一つorder2」の各2標本、計16 paired realization、X/Y×2armで64 whole-wrapper compile。seed式、stratum確率、各event、compiler trace、RZ/CX/depth/sizeは[raw archive](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/raw_prices.tar.xz)と[対照表](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/PAIRED_NATIVE_PRICES.tsv)に保存した。条件・標本数は結果を見て増やしていない。

表は各paired pathの実価格を掲載する。rare stratumを無加重で混ぜた平均をcanonical価格として用いない。全roundの期待価格の資格はMC推定ではなく、同じfinite-bit floor-CDFの有理確率と下記の構造による。

### 定義ベースcompilerの有限contextと境界

SOURCEのtailには1-site Zと2-site ZZの両方がある。低確率のsingle-Zのみのtrajectoryを落とさない。post-unroll contextはbaselineで「controlled-ZZ/CCXの有無」「CZ productの有無」「Y読出し」の8通り、referenceで「Z productの有無」「Y読出し」の4通りを覆った。全結合のBasisTranslatorはsource gate集合に応じた変換規則を局所nodeへ適用するため、wire数17そのものから新しい規則は生じない。

12 contextをbarrier区切りの局所catalogueで評価した後、同じ12 contextを最終X/CX cancellation直前の境界観測付きで取得した。これは境界資格の追加観測であり、価格holdoutや物理条件を増やしたものではない。初回と補足の両SOURCE・rawを保存した。

baselineについて、Gaussian frame、D複合chunk、CRZ/CRZZ/CZと読出しの各prefixに、全operand上で先頭に露出するnative X/CXがないことを、最終cancellation**前**に確認した。chunkを跨ぐX/CX消去が開始できないので、chunk内価格を接続できる。source側の固定inverse passについても、basis XX+YY/RZの逆対やCRZ/CRZZを勝手に消すpassではないことを照合した。referenceではproduct Z等の追加消去が起こり得るため、対応価格は上界とした。

局所CX数はXX+YY=2、RZZ=2、CRZ=2、登録controlled-RZZ=14、CZ=1、CSWAP=8。contextで不変だった。baseline Dの固定chunkはleft_all=528、left_no0=456、right_all=634、right_no0=546、boundary_control=88。したがってD部分のCXは1090q+72。referenceはeven q≥2でD+CSWAPが984q+272（q1は1184）となる。CSWAPの128 CXも含む。

各RTE occurrenceのbasis b確率をp_b、order2確率をp₂とすると、期待basis-run数は

`Σ_b [p_b + (r(1+2p₂)−1)p_b(1−p_b)]`。

これはiidのcomponent draw数がr+2N₂となることから得られる。q個のoccurrenceに加算する。各basis runの実frame/inverse価格、Z/ZZ rotation、product support多重度を使い、basis共有を無視したgate加算は行わない。Fのactual inverseは順序を反転するが各basisのrun数を変えない。

[CX証拠](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE2_CX_CERTIFICATE.json)では旧48+新16組の両軸、128件でbaseline exact一致・reference上界包含を確認し、新64 whole compileのcontextも全て含まれた。これは有限contextと境界の構造的適用範囲を伴う検査であり、holdoutの経験則だけによる全q/r保証ではない。

RZは最後のpassがX/CXしか除去しない点を利用し、context内の最小/最大局所RZ数を接続した。baselineのRTE位相Pは各occurrenceにorder2が一つでもある場合に存在し、期待個数はq[1−(1−p₂)^r]。scalar/correction P、X/Y wrapper、reference product Z、16 CSWAPも計上した。[RZ検査](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/RZ_PATH_BOUND_CHECKS.json)でも同128件が包含された。

depthはlaneのcritical pathを含む実whole native DAGで測った。新holdoutでもCX/depthの減少が観測されたが、r4/8/16/32を含む全roundのcritical-path式は資格化していない。局所depthの和やaffine fitで空欄を埋めなかった。sizeも全round未資格。

### baselineの強さに関する限界

ここでいう強baselineは**固定登録SOURCE内**のbasis非制御化・中央制御・D融合を指す。その中央controlled-ZZはQiskitの `RZZGate(...).control()` に由来し、このprofileでは14 CXへ展開される。ZZ parityの計算・CRZ・parity解除という別分解なら、理想論では4 CXで表せる。その別分解を実装・compile・誤差資格化した比較は今回行っていない。この静的な回路恒等式から、代替baselineの実native価格・総RPE価格を新たに認定しない。

したがって登録対照に対する今回の不等式を、あらゆる中央制御実装への優位性や世界的新規性へ拡張しない。特にRZ削減はこのloweringの影響を受ける。GPT／ユーザーが次の比較対象を判断すべき論点である。

## Stage 3：全support誤差と算術条件

旧r1 SOURCEの6代表構成はF/G/baselineがliteral一致。変更したr2 adapterは登録q256/r2/common/j0の最初のoccurrenceだけを検査し、全512次元controlled operatorとdiag(I,U)のFrobenius差5.6552194205362074e-15だった。[ADAPTER_CHECKS](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/ADAPTER_CHECKS.json)。これは数値correctnessであり、長round上界の根拠とは別である。旧suiteは再実行していない。

既存 `track_c_compiler_a_18round.py` の区間行列評価を再利用し、新CSWAP・Z、保存basis角度とactual inverse、正負のscheduled r/orderのCRZZ、X/Y wrapperを扱った。445個の異なる局所loweringを192-bit外向き区間で評価した。[局所上界](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_LOCAL_INTERVALS.json)。RZZ/CRZは実loweringのCX–RZ分解とbinary half-angleを照合して解析的誤差0とした。P→RZは任意の補正角を保つsymbolic変換を確認した。

CRZZの角度は元SOURCEの `math.atan` に一致させ、全contextで確認したaffineな±half-angle templateへ実角度を代入した。catalogueの有限数値変種はlowering規則の確認用であり、新しいprecision/angle探索ではない。

局所Frobenius上界≥operator normをtensor拡張し、全trajectoryのgate多重度でtelescopingした。joint局所gate全体に一つだけ共通phaseを許し、controlled分枝ごとの独立fitはしていない。全17qubit空間への上界であり、粒子数sectorへの射影やreference leakageの無視を行わない。理想reset/測定は論理操作として扱い、価格では追加resetを明示した。

### trajectory vacuum位相の丸め

既存 `vacuum_phase` のnaiveな浮動小数点加算を変更せず評価した。片側のstep数をn=max(1,q/2)、u=2^-53、γ_N=Nu/(1−Nu)とする。RZ subtraction数上界はN=(336+49r)n+24。basis frameとinverseの厳密なRZ和相殺、Dの正味対角driftを用いて、厳密部分和の絶対値Mを抑え、片側の丸めをγ_N Mで評価した。単に総gate角の絶対値を足し続けるモデルは使用していない。両側と最後の和も含む。

Qiskit 1.3のglobal phaseはbinary64のτ=2π近似値で `rem_euclid` する。[一次SOURCE](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/circuit_data.rs)の `set_global_phase` / `add_global_phase` を確認した。理想2πとの差を無視せず、正規化一回につき

`γ₄ (|x|+2τ) + (|x|/τ+2)|τ−2π|`

を加える。これは丸め最近接のbinary64、overflow/underflowなし、remainderと符号調整の標準的丸めモデルを仮定した上界である。保存SOURCEのD倍角・power-of-two scalar aggregationも照合した。OS/libm全体を形式検証した結果ではない。この算術仮定は結論の条件であり、GPTレビューで検討すべき事項として残す。

最終roundのvacuum位相生成誤差上界は0.00045377757276880644 rad（naive和0.00045372585145842106、controlled SOURCEとのnormalization差5.172131038537467e-08）。片側N上界124,780,568、M上界16,375.943495244212。RTE eventの旧phase diskは一度だけ継承し、この追加誤差と区別した。未知E₀・gap・基底状態の古典記述を補正位相生成へ与えていない。

### raw信号への接続と二重計上防止

旧 verified SOURCE/finite-RTE operator diskを2√2倍し、旧event phase・finite-bit proposal diskと同じraw信号単位で合算した。旧compiler diskは**置換**し、新full joint operator diskの2√2倍とreferenceの位相生成diskを加えた。位相diskは|exp(iδ)−1|≤|δ|でraw信号誤差へ抑えた。PF phaseは別に残す。

nominal radiusの保存下界からこのtotal diskを引き、atan2によるarcsin相当の偏角上界を引いて統計予算を算出した。最終roundのbaseline/reference total raw diskは3.6784162689939868e-06 / 0.000457439007713711。全36 arm/roundで正半径と正統計予算が残った。[全support会計](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json)。保存Hermitian行列モデル・理想native gate・独立finite bits・厳密ground供給の条件を超える保証はしていない。

## SOURCE、実行、保存先

共通入力pickleのSHA256は `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。旧pilot公開済み `common.pkl.xz` を基準commitから読み、新しい専用cacheに展開した。新SCF・DF fitting・ground solveは0。HamiltonianとPFの再生成、旧runner/STOPの再開はない。

| SOURCE | 使用した機能 |
|---|---|
| `src/trottertracks/pr_reference_transfer_qualification.py` | 入力読取、Stage1、差分adapter検査、固定16 paired価格 |
| `src/trottertracks/pr_reference_transfer_catalogue.py` | 有限12 context、native lowering、最終消去前の境界観測 |
| `src/trottertracks/pr_reference_transfer_cost_bounds.py` | CX構造証拠、finite-bit有理期待価格 |
| `src/trottertracks/pr_reference_transfer_error.py` | 全support区間評価、位相算術、18round誤差会計 |
| `src/trottertracks/pr_reference_transfer_accounting.py` | RZ包含検査、十分整数shot、総G、STOP |

再利用したcoreは `df_partial_s2.py`、`df_partial_s2_repeated.py`、`df_rte_qiskit.py`、`rte.py`、旧 `pr_reference_interference_pilot.py` のbuilder/wrapper/fusion、`track_c_compiler_a.py`、`track_c_compiler_a_18round.py` の区間helper、既存SOURCE保証/alias/finite-bit結果。旧scientific SOURCEは変更していない。

実行は上表の関数をStage順に明示的に呼び出した。旧runnerを流用してSTOPを解除する入口は作っていない。呼出順・修正経緯・executed SOURCE hashは[SOURCE.json](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/SOURCE.json)。現在の保存runを再実行する指示ではない。

[成果物README](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/README.md)から入力・環境・raw・native QPY・catalogue・誤差・RPE・STOPへ辿れる。局所区間計算後にmpmath interval asin非対応を修正し、キー集合が一致する保存区間結果を再利用して算術会計だけ継続した。初回失敗を成功結果として採用していない。価格の再取得や区間jobの重複実行はしていない。

実affinity64、初期MemAvailable約112GBを観測し、価格は独立16 worker、catalogueは12 worker、局所区間は独立件数とaffinityに応じたworkerで並列化した。上限を新しく設けず、共有設定・他ユーザーの環境・processは変更していない。packageは既存専用venvを読み取り使用した。

## 未資格事項と次の判断

- 固定登録SOURCEに対するCX/RZ期待総費用の不等式であり、実機topology/routing/noise/wall time、総T、全round depth/size、あらゆるPF/PR系への優越ではない。
- 元の連続積分・DF近似から保存Hermitian Hへの無条件包含、算術環境全体の形式保証は継承・実施していない。上記の固定入力・理想native・binary64モデルを明記して評価する必要がある。
- GPT／ユーザー：有限contextとcancellation境界の論証、位相丸めモデル、固定baselineの強さを科学的にレビューし、この条件付き結果の採用可否と次に必要な比較対象を判断する。
- Codex：追加作業は未承認・未着手。別制御分解の実装・価格・誤差資格、depth/T資格、条件拡張へ自動進行しない。

**Stage 1〜4の承認範囲を完了しmandatory STOP。旧新A pilot、Track C/C2、新B、P-D、BF/BM、PR-3等のSTOPを維持し、新C/SERVER 1へ干渉しない。**
