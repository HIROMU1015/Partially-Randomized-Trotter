# 量子化学THRIFT型mixed-access：最小feasibility計画 v1

2026-10-13 JST。基準commit `b575ff1a4e659857367a266cb59acb958ba446fd`、独立branch `qc-mixed-access-feasibility-preparation-20261013`。
[前段技術監査](qc_relative_block_method_cost_audit_v1_20261013.md)、[今回の判断記録](../../artifacts/qc_mixed_access_feasibility/2026-10-13/research_decision.json)、[結果前契約JSON](../../artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json)。

**ユーザーが伝達した完了GPT判断を採用する：現constructorは独立新規算法として非採択。次は既知THRIFT型mixed-accessの化学実装に、詳細backendへ投資する余地があるかを調べる。** 新規性の再監査・新GPTレビューを行わない。
目的は元の有限基底電子Hamiltonianのground-energy低コスト化を維持する。現constructorの非採択を、THRIFT全体の無価値・全分子での不能へ拡張しない。
今回は計画・静的source/仕様確認・公開だけ。分子生成、係数数値評価、eig/SVD、時間発展、compile、T合成、syntheticを実行していない。Mandatory STOP、N1 STOP/N2 HOLD/N3 CLOSED。

## 1. このpilotで答える問い

同じH、同じ作用精度で、追加したcoherent算術・制御・clean uncomputeを回収する**費用の余白**があるか。

\[
Q_R+r_RG_R<Q_B+r_BG_B,
\quad G_R=G_{R,\mathrm{known}}+\sum_j m_jF_{j,\mathrm{remaining}}.
\]

F_remainingはfunctionのcompute/eraseの合計に、まだ会計できないzero/sign、time scale、guard、workspace cleanup等も含む。未実装部分を0と扱わない。
一つのblockの安さ、弱いbaselineへの勝利、理想のα²scalingだけを成功条件にしない。step実測、構造量、baseline/known overheadから、許容remaining費用の境界を出す。
この段階では正の余白が得られても、functionがその価格・精度で実装できるとは認定しない。最終QPE・ground-energy全費用の優位も評価しない。

## 2. 対象を一入力へ縮小する

新規に生成する場合の唯一の対象は **H4直線鎖、隣接1.00 Å、STO-3G、電荷0、spin0、実RHF canonical MO、4 spatial/8 spin orbitals、ν=4、frozen coreなし**。
座標はz=(-1.5,-0.5,0.5,1.5) Å。単一RHF job、conv_tol=1e−10、max_cycle=50、初期guess=minao、標準DIIS、symmetryによるorbital削除なし。非収束なら入力変更・Newton/UHF等の自動再試行を行わずTECHNICAL_STOP。
同geometryは既存developmentで使われており、blind/独立移送ではない。H2は二準位の特殊な成功を増やすだけになり得るため今回含めない。距離・basis・orbital localization・rankをscanしない。
既存rank12 snapshotを自動流用しない。元integralと定数・mode順・DF再構成の完全なidentityを先に証明できた場合だけ、**別の明示した入力契約改訂**で再利用できる。今回のdefaultは一つのfresh pack生成を将来認可する案。

fresh packのSHAは生成前には存在しない。認可済みinput生成後、h/g/E_nuc/MO/mode順をserializationしてSHAを封印し、それ以降のPF・費用セルは同じpackだけを読む。結果による軌道・geometry選択は禁止。
SCFは一体Fock行列の対角化を含む。DFのfactor/basis対角化も将来のscopeに明記し、これらを「固有値計算0」とは呼ばない。多体FCI・Hのground/gap対角化は不要、将来pilotでも0。
spin orderは(0α,0β,1α,1β,…)、bit0はmode0、fermionic符号はordered ladderの右からの作用。chemist tensor→normal ordering→Hermitian monomial＋随伴のprefactor/反対称化・scalar補正を保存する。

## 3. 構造量の取得方法（今回は未取得）

| 量 | 結果前に固定する取得手順 | 留保 |
|---|---|---|
| M_raw / M_eff | normal ordering後のreal Hermitian offdiagonal項をlexicographic canonical keyへ合算。支持が同じsingleのdensity-assisted項だけをg_j(s)=g0+Σg_l n_lへ厳密group | doubleの任意group化・結果後のgroup探索なし。literal zero以外を無料で捨てない |
| z_j / z_g,j | Δ0とd_jl=Σ_support J_li(u_i−v_i)、振幅gの係数を生成し、literal nonzero数を数える | analytic/symmetry zeroと浮動小数のsmall値を別記。small判定で項を除去しない |
| range / parity | |Δ0|+Σ|d|、|g0|+Σ|g_l|、JW mask length、support/valid flagを記録 | 全算術枝を覆うword/guardに使用 |
| coupling | λ_A=Σ|ε_i|+Σ|J_ik|、λ_B=Σ_j(|g0|+Σ|g_l|)、元係数・scalarを分ける。補助としてν-sector BのSVD norm、Aのdiagonal rangeを取得 | λ_B/λ_Aはupper同士の比であり、真のnorm比やαの証明ではない。identity shiftでαを作らない |
| DF rank | full supplied factorization、one-body/scalar correctionsを保持。全factor・内部固有値を保存。L_storedとthreshold付きL_numerical、ρ_stored/ρ_numericalを別記 | reporting threshold=1e−12は表示分類のみ。truncationへ使わない。全factorを使って元H再構成を照合 |
| classical preprocessing | integral、normal ordering、descriptor、DF、basis metadata、input保存の時間/RSS/bytesを別timerで記録 | 量子gateへ無断換算しない |

係数抽出H=A+ΣB_j+E_nuc、DF再構成、spin/mode、Hermiticity、粒子数のidentityを将来の入力監査で照合する。binary64で係数差max≤1e−10、full matrix差norm≤1e−8を予定する。
これは有限precisionの再構成検査で厳密等式certificateではない。DF再構成差の時間依存作用誤差を係数/model予算から差し引く。許容値・予算を満たさなければDFを弱い対照へ置換せず全比較をTECHNICAL_STOP。
8modeのreal single/double＋上記single groupではM_eff≤28+210=238、reserve cap256。Pauli keys cap65536、DF factor cap64。cap超過を理由に項を捨てない。

## 4. 比較集合と同機能条件

| 固定ID | ideal PFの定義 | resource会計 |
|---|---|---|
| P2 | canonical JW Pauli termsの対称S2、identityは別phase | Pauli parity compute–controlled rotation–erase、合法な隣接fusion |
| E2 / E4 | [A,B1,…,BM_eff]のexcitation/density S2及びSuzuki S4 | 同じexact single grouping。linear g(s)のdirect controlled Rxも許し、不要なatan2/hypotを課さない |
| D2 / D4 | corrected one-body Gaussian＋全DF squared fragmentのS2/S4 | basis/inverse、diagonal、scalar。U† D Uのbasisはcontrol外へ置けるので、全basisに不要なQPE controlを課さない |
| R2 / R4 | known compressed relative blockを用いるTHRIFTの対称2次／同じSuzuki4次 | 前段の共通Δ近似、shared θ、constant-load、parity/flag/control/逆算。独立candidate rowを重複追加しない |

S2は指定term listのhalf stagesを順方向・逆方向に置くpalindrome。S4はs=1/(4−4^(1/3))でS2(sh)² S2((1−4s)h) S2(sh)²。
Rのseed P(h)=e^{-ihA}∏_j C_j(h)、R2(h)=P(h/2)P(−h/2)†、R4は同じS4 composition。[THRIFT Proposition2 Eq.13](https://arxiv.org/html/2403.08729)との対応で、負時間とadjointの順序を区別する。source IRに行列積とchronological順の両方を保存する。
[Trotter errorの一次資料](https://arxiv.org/pdf/1912.08854)はcommutatorを用いる比較の根拠。今回は全nested-commutatorの巨大列挙やPF係数fitを追加せず、下の有限gridで同じ直接作用誤差を測る案にする。

**P2のindividual Pauli stageは粒子数を保存しない場合がある。** ν-sectorへ各stageを投影して時間発展させない。
全方式のideal matrixをfull-Fock dimension256で作り、Qνの70入力columnsについて

\[
e_{m,T,r}=\|(S_m(T/r)^r-e^{-iTH})Q_\nu\|_2
\]

を測る。出力rowsも256とし、sector外leakageを含める。粒子数保存法も同じmetric。これはground state情報を使わず、ν=4の全入力を対象にする。controlled errorはI⊕Uとして同じnormであり、512行列を毎回作る必要はない。global/ancilla-relative phaseは捨てない。
同じH、同じscalar、precision・control・native modelを使い、全方式に可能なgroup/fusion/係数圧縮/precision配分を与える。global Aを各mixedで再生成する弱いTHRIFTだけを対照にしない。

## 5. 結果前の最小gridと精度・selector

T={1,4} atomic time、r={1,2,4,…,1024}、7 method rows、operator target ε_U=1e−3。geometry/basis/DF policyは§2–3、L_Dなし、PRなし。
**1e−3はunitary作用誤差であり、1mHaのground-energy達成を意味しない。** 最長T=4もQPE全horizonを検証するものではない。
ideal PF-error queriesは7×2×11=154が上限。rを追加・補間したりfit Cから未実行rを認定しない。
精度配分は各方式へ同じPF fractions a={0.4,0.6,0.8}を与え、PF予算a ε_U、残りをRはcoefficient/function/synthesisへ等分、Bはcoefficient/synthesisへ等分する。未使用予算、DF residual、将来のinput model誤差を明記する。これが全配分の大域最適だとは主張しない。
Rの理想PF matrixではfunctionsを古典的に評価できるが、量子oracleを無料で使えたというresource結果へ変換しない。

各(T,m,a)で、判定margin±1e−8を置く。εPF以下で、次の2rを含む**全ての大きい登録r**も以下のrだけをeligibleにする。r=1024のみ通過はBOUNDARY_UNCONFIRMED。非単調・aliasによる孤立通過を使わず、grid外保証にも拡張しない。
実行時のreferenceはSciPy scaling/Padéと独立scaling/Taylorを各Tで照合し、差≤1e−10を要求する。Taylorはscaled norm≤1/2、degree≤128、squaring≤32、tail設計とnumerical residualを記録する。
unitarity/Hermiticity/各primitive assembly/phase/sector leakageの検査も行う。reference guard失敗、margin内の曖昧な最良候補はNUMERICAL_UNDETERMINEDとし、精度やgridを自動追加しない。
binary64照合は局所数値検証であり、全roundoffの外向きcertificateではない。最良rをrigorous quantum-error保証と呼ばない。

## 6. functionへ残る費用を出す

各eligible r/aに対し、全正負stageのS1_j=Σ|t_jl|、S2_j=Σt_jl²から前段監査の係数word式を適用する。word sign/integer/fraction/guardを分け、w/a_angle cap128を置く。overflow・zero/sign・g近似の費用を残す。
primitive ledgerをclassical descriptorsから作り、Toffoli、RZ、CX/CZ、Clifford、logical depth envelope、known workspaceを別々に保存する。Qiskit circuit生成・transpile、実T synthesisをこのpilotには含めない。
constant-loadのz(4w−6) Toffoli、相対四回転のbitwise envelope、controlled Pauli/励起/Givensの同じ固定templateを使い、exact design count・upper envelope・UNKNOWNをタグで分ける。routingはall-to-all logical modelに固定し、物理実機の深さ勝利を認定しない。
残るfunction workspace/depthもUNKNOWN。各jの未会計部分はF_remainingへまとめ、後で二重請求しない。共有できるangleでも、他の励起がspectatorを変えた後の全step cacheを無料にしない。

logical Tに換算する主式は7×Toffoli+Σ T_RZ(η)。T_RZ(η)は合成未実施なのでUNKNOWNの関数として残す。
補助として固定したhypothesis κ={3,6,12}に対し **T_hat_RZ(η;κ)=ceil(κ log2(1/η))** のmodel sensitivityを出す案にする。
[Ross–Selinger v3](https://arxiv.org/abs/1403.2975)のtypical leading scalingを参考にした**仮定モデル**で、κ値の選択・省略定数は本pilotの仮定。全angleの厳密上界、実合成count、指定ηを実際に達成した証拠ではない。全方式へ同じκ関数を与え、per-RZ ηは各total synthesis budget/rotation数から設定する。
モデルを使わないsymbolic frontierを一次出力とし、κ点のmodel結果を補助に分ける。resource rows上限154×3配分×3model=1386。

B∈{P2,E2,E4,D2,D4}の各eligible r/aを公平に選べるようにする。Bの最小全cost envelopeをC_BASEとすると、Rの一step remaining予算は

\[
\mathcal B_{\rm remaining}
=\frac{C_{\rm BASE}-Q_R}{r_R}-G_{R,\rm known},
\qquad \sum_jm_jF_{j,\rm remaining}<\mathcal B_{\rm remaining}.
\]

Q_mはphase-gradient等の量子初期資源。今回のbitwise設計はそのstateを使わないので追加Q_m=0という**選んだ設計の仕様**とする。共通state preparationを比から省いたことを明記し、別方式のQを0にしない。
均一remaining費用ならF_remaining< B_remaining/W、W=Σm_j。compute一方向Fを使う場合だけ2WF< B_remainingへ変える。単なるr_B/r_Rの最大化でなく、全known costとprecisionを含む最大余白のR2/R4/r/aを有限集合から選ぶ。
upper envelopeを代入してB_remaining≤0でも一般的不能の証明にはならない。Rknownを楽観的に置くなら、未会計値はF_remainingの中に残し、示したceilingを達成費用としない。baseline upperを使う場合も、その価格仮定への条件付き結果と表示する。
古典P_m、prep、ground overlap、測定repeat、QPE powers、statistical/model errorは未評価のまま別項。短いTでのmodel余白をground-energy全費用利益へ読み替えない。

## 7. 取得する結果と停止判定

一次出力はcoefficient identity、M_eff/z_j/group/parity/range/DF ranks、154以下のideal PF errors、reference/phase/leakage監査、eligible/censored理由、partial resource ledger、symbolic remaining-budget frontier。
κ別cost/余白は補助model結果。全raw rowsをselectorの前に封印し、単一選択とtie-break（cost、r、order、IDの順）を記録する。negative/unknown rowsを捨てない。

| 結果分類 | 何が判明したか | 結果後の扱い |
|---|---|---|
| POSITIVE_CONDITIONAL_FUNCTION_BUDGET | 登録H/T/精度・baseline・price model内でremaining費用へ正の余白 | 実backendの可能性・追加投資はGPT判断。量子総costのGOではない |
| NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL | 全eligible登録R構成で余白が非正 | 現model/scopeでの負のfeasibility材料。一般的不能ではない。追加scanせずGPTへ |
| UNDETERMINED_BOUNDARY_OR_COST | eligible rなし、grid端のみ、数値境界、未確定価格が結論を変える | 何を知ると判断が変わるかを保存。自動追加計算なし |
| TECHNICAL_STOP / RESOURCE_STOP | input identity、再構成、source/env、cap、phase/assemblyが不成立 | partial/失敗を保存。科学的不成立と分け、retry0でSTOP |

全結果を保存・独立監査・公開した後Mandatory STOP。N1/N2/N3や別候補を起動しない。重要な新結果の科学解釈はGPTへ戻す。完了した新規性判断を再開しない。

## 8. 実行上限と性能準備

| 項目 | 将来認可する最大値 |
|---|---|
| molecular input / RHF | H4一入力、一job、SCF50 cycles、attempt1/retry0 |
| DF | 一decomposition、外側64以下・内部全factor、explicit小行列eigh≤66。SCF内部solverは別計測 |
| ground/gap/FCI | 0 |
| reference | 2T×2実装=4 matrices、dimension256 |
| PF grid | 154 error probes、個別診断を含むnorm-SVD≤200 |
| seed/power workload | distinct ideal seeds≤91（7method×union13h）、dense power products≤2000、primitive applications≤4,000,000 |
| structural/resource | M_eff≤256、Pauli keys≤65536、DF factors≤64、word/angle bits≤128、resource rows≤1386 |
| host memory/output | 2workers・各1thread、aggregate RSS4GiB、output64MiB、log4MiB |
| quantum circuit/compile/T synthesis/PR/shots | 0 |
| wall/CPU cap | なし。tool polling timeoutは科学limitと別 |

現在のread-only host確認はCPU32、available RAM約55.35GB。準備はstdlibだけで、科学packageをimportしていない。小matrixなのでCPUを予定しGPUを使わない。
dominantはterm/stage assemblyとPF matrix power/norm。励起pairとPauli maskをvectorized row updatesにし、全columnsをbatch処理、density phaseをrow scaling、DF basis descriptorをinput-SHAでcacheする。
同一hのideal seedはT間で共有し、powersはbinary exponentiation。precision/cost rowsはr/a依存なのでseed cacheから無料で共有しない。7method jobsを2workersへ配り、BLAS/OpenMP/MKL/OpenBLAS/Numba/Rayon各1threadでoversubscriptionを避ける。
source固定前に新しい技術pipelineへ必要なfocused synthetic/reference照合を行う条件を定める。旧13件の再実行を完了要件へ追加しない。新技術testsは登録Hを使わず、固定入力での数値probeはscience authorization後に限る。speedup倍率を主張しない。

## 9. 準備の完了と認可境界

この文書で対象・比較・grid・指標・予算・STOPを固定した。旧H2/H4案は履歴のまま保持し、起動契約へ流用しない。
**planはscope承認に提示可能だが、まだlaunch-ready sourceではない。** 科学runnerは未実装、source SHA・dependency lock・input SHAはnull。今回の依頼は新算法実装ではなく計画作成であり、これを実行済み又は凍結runnerと称しない。
現在のvenvにはPySCFが未導入。将来の通常技術実装ではisolated環境、module/runner/checker、対象外synthetic、optimized/reference一致を閉じ、source/envを固定する。[launch readiness条件](../../artifacts/qc_mixed_access_feasibility/2026-10-13/launch_readiness.json)。その作業だけで新GPT科学レビューを挟まない。
ユーザーの明示承認は **H4 input生成＋上記有限PF/SVD/DF数値＋model/symbolic費用分析の一回scope** を対象とする。認可後でもsource/env/preflight未完了ならlaunch gateを閉じたまま通常技術準備を行う。source固定を記録し、認可scopeを変更しない。入力封印後のslotは事前固定の条件だけで実行する。
現在は科学実行認可なし。GPTによる再レビューも不要・未開始。次はユーザーによる限定scopeの実行承認。旧source/results/auth・global manifest既存description欠落FAILとMandatory STOPを保持する。
[計画metadata checker](../../artifacts/qc_mixed_access_feasibility/2026-10-13/audit_tools/check_plan_metadata.py)はslot数・JSON・source/証拠SHA・旧manifest・追加リンク・未認可gateだけを検査する。[保存検査結果](../../artifacts/qc_mixed_access_feasibility/2026-10-13/static_preparation_check.json)。科学sourceをimport/実行せず、旧13件も再実行しない。局所metadata PASSとglobal manifest既存FAILを区別する。
