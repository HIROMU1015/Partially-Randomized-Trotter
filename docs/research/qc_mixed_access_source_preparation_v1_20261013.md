# 量子化学mixed-access feasibility v1：source・環境・実行前準備

2026-10-13 JST。親計画公開commit `839410511da2dfde46b20b1775964c4f15313a57`、独立branch `qc-mixed-access-feasibility-source-v1-20261013`。
ユーザーの「続きの作業を行って大丈夫です」は、直前に提示したrunner/checker/環境・登録外synthetic・source固定・公開の技術準備を認可する。分子生成と科学batchの承認は引き継がない。

[固定科学計画](qc_mixed_access_feasibility_plan_v1_20261013.md)、[科学契約](../../artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json)、[実行計画](../../artifacts/qc_mixed_access_source_v1/2026-10-13/execution_plan.json)、[source索引](../../artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json)、[準備状態](../../artifacts/qc_mixed_access_source_v1/2026-10-13/launch_readiness.json)。旧契約のnull欄・旧readinessは当時の履歴として変更せず、このversionで技術準備を接続する。

## 到達点と次の認可

新しいmodule、runner、独立saved checker、hash付きdependency lockを用意し、登録外synthetic **22 tests PASS**。実分子H4生成・SCF/AO2MO・登録PF誤差評価・compile/T synthesisは0。PySCFのimportとAPI signatureだけ確認した。
source固定後のread-only preflightと静的保存チェックを保存する。これは局所技術検証で、CI・分子feasibility・外部科学再現の証拠ではない。

次の担当はユーザーによる固定H4一回の実行承認。現在はMandatory STOP。新しいGPTレビューは技術準備だけを理由として要求せず、新科学結果から継続・backend投資判断が必要になったときに開始承認を得る。
constructorの独立新規算法としての非採択、N1 STOP/N2 HOLD/N3 CLOSED、旧source/result/認可/STOPを保持する。ground-energy計算全体の総費用は未評価。

## 実装と数式の対応

| ファイル | 役割 |
|---|---|
| `src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py` | CAR normal ordering、励起group、JW、full DF、Gaussian QR、row更新、7 PF、Taylor reference、SVD witness生成 |
| `qc_mixed_access_resource_v1.py`（同directory） | 固定logical template、S1/S2精度、shared theta、UNKNOWN込みsymbolic frontier、κ補助model |
| `qc_mixed_access_execution_v1.py`（同directory） | stdlib preflight・source/env/承認hash照合・一回marker |
| `qc_mixed_access_pipeline_v1.py`（同directory） | 認可後だけlazy PySCF入力、seal、2spawn workers、caps、raw seal、監査、resource表、STOP/失敗保存 |
| `qc_mixed_access_checker_v1.py`（同directory） | 独立basis-state CARと保存norm witness検証。SVD/eigh/PySCF呼出しなし |
| `scripts/run_qc_mixed_access_feasibility_v1.py` | `--preflight`または別承認書を伴う`--execute`。暗黙の科学実行なし |
| `scripts/verify_qc_mixed_access_feasibility_v1.py` | 保存raw監査のみ |
| `scripts/check_qc_mixed_access_synthetic_v1.py` | このversionの登録外testsとsource binding記録 |
| `scripts/check_qc_mixed_access_preparation_v1.py` | source/契約/旧証拠/manifest保全の静的照合 |
| `tests/test_qc_mixed_access_feasibility_v1.py` | 独立行列・異常入力・認可拒否・2worker・8mode mock adapter tests |

chemist tensor `(pq|rs)`から、
`sum h_pq a†_(pσ) a_(qσ) + 1/2 sum (pq|rs) a†_(pσ) a†_(rτ) a_(sτ) a_(qσ) + E_nuc`を生成する。
modeは`2pα,2p+1β`、bit0=mode0、ladderは右から作用。creator昇順・annihilator降順へ整数CARで整理する。singleのdensity-assisted項だけ同一transitionへ厳密にまとめ、`g0+sum g_l n_l`を使う。
literal zero以外の項を消さず、Hermiticity、粒子数保存、元Hと励起再構成を検査する。

DFは空間ERIを`ns² × ns²`へreshapeして全固有factorを保持する。補正one-bodyは`h_ps - 1/2 sum_q (pq|qs)`。
各factorに`1/2 λ (sum_i η_i n_i)²`を使い、nuclear scalarは別に残す。4 spatialなら16 factorsと補正one-body、small-matrix eigh18 calls（cap66内）。reporting threshold1e−12はrank表示にだけ使う。
全factor係数のmax差≤1e−10、full-Fock matrix差のFrobenius上端≤1e−8。後者をoperator-normの保守的上端としてDのmodel予算から`|T|`倍を控除する。外向き丸めのcertificateではない。

Gaussian frameはspatial QR/Givensを両spinへ展開し、row対をfermionic parity付きで更新する。独立determinant-minor frameと照合した。basis/inverseはDFのQPE controlの外側へ置く費用設計。
P2はcanonical JWのPauli単語のS2。E2/E4、D2/D4は固定palindrome/Suzuki。D4では可換scalarを外へ出し、E/R同様に合法な隣接exact-exponential fusionを適用する。

Rのpair上では`A+B_j=dbar I+Δ Z/2+b X`。
`C_j(t)=exp(itA)exp(-it(A+B_j))`の平均phaseはこのblock内で相殺し、outer Aとnuclear phaseは保持する。R2は`P(h/2)P(-h/2)†`、R4は同じSuzuki composition。
`C(-t)`を`C(t)†`へ置換しない。計算はfull-Fock256上で行い、ν4の70 input columnsと全256 output rowsを保存する。個々のPauliをsectorへ投影しない。global-phase最小化なし。

## precision・資源会計の固定詳細

元計画のPF配分0.4/0.6/0.8、Rの残余三等分・対照の二等分、κ3/6/12を変更しない。PF誤差は元Hへのideal matrix差で、未知functionを無料の量子oracleとしない。
科学batchのT1/4、r倍増1…1024、7法・154 queries・91 distinct seeds以下も変更しない。次の2rと全大きい登録rが通る場合だけeligible。1024のみ通る例はboundary censored。

Rの全正負occurrenceから`S1_j=r sum_l |t_jl|`, `S2_j=r sum_l t_jl²`を取得する。
coefficient予算の半分をouter位相へ、残りを各groupのdetuning・振幅へ同額`εD_j=εb_j=εcoef/(4M)`で割り当てる。
前段監査と同じ十分式：

\[
f_{D,j}=\max(0,\lceil\log_2[b_{max,j}(z_j+1)S2_j/(4\epsilon_{D,j})]\rceil),
\quad
f_{b,j}=\max(0,\lceil\log_2[(z_{g,j}+1)S1_j/(2\epsilon_{b,j})]\rceil).
\]

ゼロ分子はlogを評価しない。共通fraction bitsはこの二つのmax。sign/integer/guardは丸め後の`|Δ0|+sum|d|`と`|g0|+sum|g_l|`を覆い、word全体≤128。
sum detuning error≤sum `bmax ηΔ S2/2`、sum amplitude error≤sum `ηb S1`。これはbounded coefficient roundingの設計条件で、function内部のatan2の一様感度を仮定しない。
各groupのbits、共通S1/S2、配分をκに依存しないprecision catalogへ保存し、group descriptorと接続する。

outer/baseline係数・Gaussian角度の一様roundingでは、全descriptorの絶対値和S、数K、r×stage数Lを使う`1024(1+S)²(1+|T|)LK`の保守的係数guardを採用する。
8 occupations以下のDF phaseはdegree3で、その偏微分はこのguard内に収まる。basis-frame angleはunitary telescoping、単/二density・Pauli・affine項は同じguardに含む。
この係数word設計は数値errorに合わせてfitせず、実backendのrounding・compile保証は未実施。R function angle bitsは全四回転（中心pairは五回転）を覆う`ceil(log2(16π r W/εfunction))`。modulo4π・zero/quadrant/time/cleanupの実装費用はF_remainingに残す。

| template | 記録するlogical primitive envelope |
|---|---|
| controlled `n_i` | 3 RZ＋2 CX |
| controlled `n_i n_j` | 上記＋AND compute/erase2 Toffoli、known ancilla1 |
| controlled Pauli weight k | parity＋CRZ、2 RZ＋2k CX、basis Clifford≤4k。identityはancilla phase1 RZ |
| excitation pair k=2/4 | encoding/decode2(k−1) CX、parity2p CX＋2CZ、valid/QPE flag Toffoli≤2/8。各非zero g coefficientは2 RZ＋2CX、density coefficientにはAND往復2 Toffoli |
| Gaussian Givens | parity付きpair回転2 RZ＋(4＋2p)CX、一frame。inverseも計上、control外。sign ZはClifford |
| constant-load affine sum | 前段の`z(4w−6)` Toffoli、`sum(10w−14＋4 h_l)` CX、`z(4w−12)`X、定数load/clear≤2w X。Δとgへ同じ方式 |
| relative block | 四可変回転で8a RZ/CX/Toffoli。θは一度生成し±へ共有。signed/quadrant処理・function compute/eraseはUNKNOWN |

R2中心の隣接同一group `C(t)C(-t)†`は、二つの中間Ryが相殺する。
`Rz(-tΔ) Ry(θ) Rz(4tρ) Ry(-θ) Rz(-tΔ)`の五回転と一組の係数/function compute/eraseへまとめる。R4も各R2中心に同じ共有を与える。
このknown sharingは2×2独立テストで確認し、PF数学定義を変更しない。Rを一般semigroupとして融合しない。basis sharing等の大域最適性は主張しない。

全Toffoliは7T unitary convention、RZは未合成の`T_RZ(η)`。一次frontierはsymbolic differenceをη付きで残す。既存対照のeligible全r/配分から最小の同じmodel費用を選ぶ。
κによる`ceil(κ log2(1/η))`は補助hypothesis。実合成の達成値・厳密all-angle上界ではない。known native serial depth envelopeとknown workspaceを別記し、Rのtotal depth/workspaceとF_remainingをUNKNOWNのまま保持する。
追加state Q=0は指定bitwise設計の条件。共通ground-state準備・測定・QPE成功率はこのinterrogation費用の外で、total ground-energy費用未評価。
positive frontierは条件付き余白、negativeはこの指定envelope/hypothesisの結果であり、全手法の不能判定にしない。

## checkerの独立性と限界

保存error E、σ、right singular vectors Vの組を使い、checkerはSVD/eighを追加しない。
`δ=||V†V-I||F`, `R=||E†E V-V diag(σ²)||F`, `g=||E†E||F`から、

\[
||E||_2\le\sqrt{(1+\delta)\sigma_{max}^2+R\sqrt{1+\delta}+g\delta}.
\]

下端は第一vectorのRayleigh比`||Ev0||/||v0||`。σ順序・有限性・shape・residual・宣言normと数値guardを検査する。unitary/Gram residualの浮動小数照合であり、有理/interval proofとは呼ばない。
checkerはraw h/ERIから独立basis-state ladder loopでHを構成し、保存HとDF係数再構成を照合する。154のcoverage、70columns、256rows、file ID・seal SHAを検査する。
保存checkerは各PF trajectoryを独立に全再実行するものではない。primitive・7 formulaの独立dense-exponential/別scheduler比較はsynthetic側で担保し、実分子referenceは登録実行内のPadé/Taylor相互照合で担保する。この監査範囲を外部科学再現と呼ばない。

## tests・性能・環境

[synthetic report](../../artifacts/qc_mixed_access_source_v1/2026-10-13/synthetic_report.json)は22 PASS、source SHAを記録。
手書き2/3 spatial integralsとsigned density factors、8mode Hadamard frame、8modeのmock input adapter＋登録外T=.01/.02・r1/2/4を使う。geometry/basis/DF rank policy/L_D/化学精度は適用なし。
42の保存probeを2spawnで作り独立saved auditし、小fixtureでは別schedulerのdense referenceと全7法のerror columnsを比較。Rの負時間/真adjoint/zero branch、全位相、CAR、DF correction、漏れ、censoring、word bounds、未知費用、改変・schema/認可/二回目の拒否を試験した。
開発中のreal→complex配列処理とNumPy scalarのJSON保存不備を修正し、最終sourceで回帰した。失敗したmock testの期待値も修正した。分子結果の修正・再試行は0。

[environment](../../artifacts/qc_mixed_access_source_v1/2026-10-13/environment.json)、[import/API確認](../../artifacts/qc_mixed_access_source_v1/2026-10-13/import_check.json)、[requirements lock](../../artifacts/qc_mixed_access_source_v1/2026-10-13/requirements.lock)。CPython3.12.3、numpy1.26.4/scipy1.14.1/PySCF2.8.0/h5py3.11.0/setuptools75.8.0、Linux x86_64 wheel SHA固定。
一時venv `/tmp/prt-qc-feasibility-source-v1-venv-20261013` を用意し、pip check PASS。/tmpは永続環境ではないので、消えた場合は同lockを`pip install --require-hashes -r .../requirements.lock`で再構築しpreflightする。
API根拠：[PySCF RHF](https://pyscf.org/pyscf_api_docs/pyscf.scf.html)、[AO2MO restore/kernel](https://pyscf.org/pyscf_api_docs/pyscf.ao2mo.html)。実際の呼出し仕様は導入した2.8.0 source/signatureを確認した。最新versionへ自動追従しない。
標準RHFのmax_cycle50は通常iteration loop。engineの標準extra convergence diagonalizationを残し、`conv_check`もinput sealへ記録する。SCF diagonalizationをFCI0と混同しない。

hostは32 affinity CPUs、約67.1GB total RAM、GPU driverはnvidia-smi exit9。CPU2workers/BLAS・OpenMP1threadを選択。
dominantなfull-Fock assemblyをrow更新、Gaussian parity kernel、vectorized pair、seed cache13h/method、binary matrix powerで処理する。method単位の独立jobが両Tを受け、同じhのideal seedだけ共有する。r依存word/合成配分の費用をseedから流用しない。
source/比較条件を変えたspeedup倍率の主張は行わない。局所suiteの実時間とpeak RSSはreportに記録。科学run時はmain/worker timer、前処理timer、.2秒のaggregate RSS sampling、countsとreservation一致を保存する。

## 将来実行とSTOP

source/env/契約を固定してから、別承認書にユーザーの承認参照、scope ID、payload commit、source manifest SHA、contract SHA、lock SHA、唯一のoutput path、attempt1/retry0を記載する。
現在の[承認template](../../artifacts/qc_mixed_access_source_v1/2026-10-13/authorization_DRAFT_NOT_AUTHORIZED.json)はDRAFTで、runnerが拒否する。承認書を自動で有効化しない。
preflightはread-only。executeは承認照合後にmarkerを作りattemptを消費し、その後だけPySCF importと分子構築へ進む。source byte/固定commit/HIROMU1015 remote identityも照合する。
input SHA封印後に同一packの派生処理を行う。raw filesをselection前に封印、saved独立audit後にresource/selectorを出す。非収束・numerical guard・caps失敗も保存してSTOP、別namespaceで自動retryしない。

科学scopeはH4直線1Å/STO-3G/RHF8mode/ν4、full DF・L_Dなし、固定7法/T1,4/r1…1024のみ。FCI/ground/gap、量子compile、T合成、shots/PRは0。
4 reference matrices、SVD≤200（PF154＋coupling1、saved checker0）、91seeds、power products770≤2000、primitive≤4M、resource rows≤1386、RSS4GiB/output64MiB/log4MiB、wall/CPU capなし。
raw error/witnessは約56.3MB（decimal）の上限設計で、dispatch前に既存出力＋予測witness＋4MiB metadataを照合する。precision catalogはκ間で重複保存せず、resource JSONはcompact serialization。新登録input/pack SHA・host状況は認可後の実行時に記録する。

旧global manifestの`relevant_commits[7].description`欠落FAILは保持。旧122 result setsと全non-result-set fieldsを変更せず、この準備のentryだけ追加する。scopeチェックPASSとglobal manifest PASSを混同しない。

**この準備の完了で分子の優位性を判定しない。現在は固定一回の科学実行承認待ち、Mandatory STOP。**
