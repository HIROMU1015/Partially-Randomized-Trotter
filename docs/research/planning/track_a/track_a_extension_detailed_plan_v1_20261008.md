# Track A拡張研究：詳細計画と妥当性レビュー v1

作成日：2026-10-08  
既存成果の固定点：`HIROMU1015/Partially-Randomized-Trotter` / `4c23453c541700c6a41ba71fc5ec9323b53858d6`  
基準となる数値結果：`5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`（PM-2）、`194cc604b90c56a0e7e949b91b064a4bcfc846da`（PM-1）  
位置付け：研究計画の提案。旧成果・正式判定を変更しない。新しい科学計算・GPU利用・リポジトリ変更の実行認可ではない。

## 0. 推奨判断

大枠の方向は妥当である。既存Track Aを消したり、H4の限定結果を一般結果へ読み替えたりせず、別の拡張フェーズとして、次を並行する二本の主RQに置く。

- RQ-R（resource estimate）：指定したtask・実装classにおけるPRの具体的な資源は、サイズ・精度によってどう変わるか。
- RQ-P（predictive reliability）：同じtaskと費用単位に対応付けた解析モデルは、資源量・精度適格性・構成選択をどこまで予測できるか。

モデルの失敗、PRの勝利、新しい方式逆転を研究成功の必要条件にしない。モデルが正確なら、未使用条件でどの範囲まで信頼できるかを報告する。PRが不利なら、その条件と支配費用を報告する。片方のRQで差が小さいことを、他方のRQの計算中止理由へ直結させない。

最も大きい変更点は、単純な「H4の候補gridをH6/H8へコピー」ではなく、比較対象の対応付け、サイズ対応した信号評価、候補集合の公平性、測定規則依存性、未使用サイズでの評価を先に設計することである。

## 1. 既存証拠・今回確認・提案の区分

### 1.1 既存証拠

既存Track Aは、H4 linear、STO-3G、DF rank12、T=0.8、指定した二次DF-prefix PF・canonical finite-RTE、Qiskit1.3.0 opt1で、状態準備を除くfull measured Hadamard wrapperを評価した。PM-2はdevelopment218構成とM2元5構成に対する保存値解析で、ε_signal=0.005〜0.1の302表示点を用いた。[R1]

M1/PM-1の精度適格性は対称軸配分とHoeffding型十分shot式の条件であり、原理的な精度限界ではない。費用はcompiled RZ等の指標であって、Clifford+T合成済みのT資源や物理実行時間ではない。[R1,R2]

### 1.2 今回確認した実装上の事実

`pr2_matched_accuracy_m1_execution.py` の `_dense_block_operators` は全2^n_qubits空間の密行列を作り、fragmentごとの行列と固有分解を用いる。`_random_signal_record` は有限Taylor多項式の平均作用を直接評価しており、信号平均をrandom trajectoryのMonte Carlo平均だけで推定しているわけではない。[R3]

同ファイルの `_target_signal` は、保存状態のRayleigh値・残差を確認し、固有状態としての位相信号を作る。任意の近似状態にこの式をそのまま適用してよいという実装ではない。[R3]

`df_gpu_statevector.py` にはGPU statevector helperがあるが、既存M1の全科学経路をそのままH8へ移行できると確認したわけではない。また同moduleにはimport時のCUDA関連ライブラリ探索・preloadがある。文書だけの監査段階ではimportせずsource textとして読む。[R4]

### 1.3 今回の非実施範囲

分子データ、NPZ/NPY/pickle、state、旧runtime/cache/registryは開いていない。新signal、ground-state solve、trajectory、circuit build/compile、test再実行、GPU操作は行っていない。全リポジトリ監査や全保存ledgerの独立再検算ではない。

以下のgeometry、精度点、候補生成規則、MC標本数、閾値は推奨案である。利用者のCPU/RAM/GPU/VRAMと許容計算時間が未提示なので、実行上限の最終値やwall timeは未確定である。

## 2. 研究としての最終成果

### 2.1 中心の成果物

1. 同一task・同一費用定義の解析モデルと直接compileを対応させた検証表。
2. H4/H6/H8での、精度に条件付けたPR・discard・決定論・random-dominantの資源台帳。
3. 解析モデルの費用誤差、誤った適格判定、選択regretと、モデル取得費用の評価。
4. bias、normalization、1-shot費用、合成policy、測定規則を分けた原因分析。
5. 未使用サイズでのモデル検証と、成否に依存しない報告。

### 2.2 中心claimの目標形

> 既知の部分ランダム化法について、同じ有限時間信号と実装条件に対応付けた資源モデルを、直接評価した信号bias・normalization・回路費用で検証した。サイズ・要求精度による資源的利益と、モデルを使った構成選択の信頼性を、未使用条件および明示した数値・統計的不確かさのもとで定量化した。

これを新アルゴリズム、新しい一般的最適化原理、全化学系への最適性と呼ばない。H4/H6/H8は有限サイズの傾向を検査する点であり、3点から漸近計算量を証明しない。

### 2.3 エネルギー推定との接続を曖昧にしない

第一の完成範囲は、明示した有限時間のsignal resource studyとする。ただし化学精度との関係を後回しの言葉だけで済ませず、AX-0でenergy-task接続仕様を作る。

仕様には、目標ΔE、成功確率またはRMSE、初期overlap、エネルギー窓、時刻列、alias復元、PF/RTE/統計/合成誤差の扱い、最大回路費用と総費用を入れる。仕様を作ることと、その全RPEを計算することを分ける。

原論文のQPE総資源へ数値として直接比較するclaimを選ぶなら、この接続と同じ合成費用の評価は必須である。有限signalの研究で閉じる場合は、その比較を達成済みとは書かない。

## 3. 原論文との比較を三層に分ける

原論文は詳細な資源モデルと化学QPE評価を持つ。とくにSec. VII、Appendix E.1.b/E.3.cはDF側の費用、Appendix DはTrotter誤差の数値評価を扱う。[W1]

### 3.1 M-paper：原論文の忠実な再現

原論文のtask、表現、切断、誤差規則、control、合成法、ancilla、費用単位を維持して、公開データの選んだ代表点を再現する。著者公開の図・データarchiveがあるので、その対応版と生成元をまず点検する。[W2]

小数何桁までの再現を求めるかは、元データの丸め・乱数・取得可能な入力に合わせて事前に決める。原データにない値を図から推測して厳密再現としない。

### 3.2 M-native：同じ実装に対応した解析会計

現行DF-native finite-RTEと同じsample単位、scalar phase、basis-change、controlled diagonal、wrapper、有限cutoffで解析的費用を作る。

原論文が前提とするbasis間相対変換の合成、内側rank切断、random residual表現、等角rotationの合成戦略等が現行sourceと違う場合、その差は「モデル予測の誤差」ではなく、まず「実装・表現の違い」に分類する。

特にAppendix E.1.bの任意角rotation countを、Qiskit出力の全rz_countへ無条件に同一視しない。Clifford角を含むRZ、Givens内部の回転、制御化の回転、任意2-qubit gateとCXの違いを対応付ける。

直接対応しない原論文の項目はN/Aとし、M-nativeへの変更を式ごとに記録する。M-nativeは原論文の忠実な再現ではなく、明示した適用版である。

### 3.3 M-calibrated：H4だけで校正したモデル

必要な場合だけ、M-nativeの有限個の係数をH4で校正する。特徴量はN、L_D、内側rank、q、R、K、作用数、既知のbasis遷移等、評価前に得られるものに限定する。

同じ候補のdirect compile後のgate数、未知系のtrue bias・実測costをpredictorへ入力しない。原論文型モデル、未校正M-native、校正後モデルを別々に報告する。

H4 1.00/1.30 Åは既に観測済みなので、両方とも開発情報として扱う。H4内部のcross-validationを新しいblind evidenceとは呼ばない。

## 4. 評価量と費用の切り分け

### 4.1 taskと評価変数

instance j=(原子数N, geometry d, basis, Hamiltonian/状態の定義)。候補x=(method,L_D,q,R,K,PF family,implementation policy)とする。

\[
z_j(T)=\langle\psi_j|e^{-iH_jT}|\psi_j\rangle,\quad
b_{j,x,a}=|\nu_{j,x,a}-z_{j,a}|.
\]

\[
N_{j,x,a}=\left\lceil\frac{2B_{j,x}^{2}\log(2/\alpha_a)}{(\epsilon_{sig}/\sqrt2-b_{j,x,a})^2}\right\rceil,
\quad
G^{ref}_{j,x}=\sum_a N_{j,x,a}C^{direct}_{j,x,a}.
\]

ここでrefは「指定した実装・十分shot規則でのbenchmark参照」であり、最小の物理実行資源という意味ではない。

### 4.2 費用モデルだけの検証

NとBとbを共通にし、Cだけを変更する。

\[
G^{cost\ model}_{j,x}=\sum_a N^{ref}_{j,x,a}C^{model}_{j,x,a}.
\]

これは測定回数を真値参照で固定した条件付き検証である。未知系で運用可能な全資源predictorだとは呼ばない。

### 4.3 誤差・測定モデルも含む検証

同じCを使ったN-predictedとN-referenceの比較を別に行い、その後に全体を接続する。4通りを保存する。

| shot | 1-shot費用 | 意味 |
|---|---|---|
| N-pred | C-model | 全予測 |
| N-ref | C-model | 費用モデルだけの影響 |
| N-pred | C-direct | 誤差・測定モデルだけの影響 |
| N-ref | C-direct | benchmark参照 |

Bの上界と有限cutoffのBを差し替える場合も、別の軸として保存する。差を全部「compileの効果」としない。

C_gsに基づくenergy biasとfinite-time complex-signal biasは別である。C_gs δ^pをそのままsignalのaxis biasへ入れない。operator-norm/state-action error、固有ベクトル重み、対象state、Tを含む正当な変換が必要である。

### 4.4 boundとpredictionを分ける

各モデル値に `rigorous_upper_bound / heuristic_prediction / exact_accounting_under_policy` 等の意味を持たせる。

正しい上界がdirect値より大きいこと自体は失敗でない。上界のslack・過剰予算・候補の保守的除外を評価する。逆に、予測が精度不適格候補を適格と判定した場合は、単なる費用誤差より先にaccuracy failureとして扱う。

### 4.5 selection regret

\[
x_M=\arg\min_{x\in\mathcal X:\ predicted\ feasible}G^M(x),\qquad
Regret=\frac{G^{ref}(x_M)}{\min_{x\in\mathcal X:\ reference\ feasible}G^{ref}(x)}-1.
\]

x_Mがreference-ineligibleならregretに都合のよい数を入れず、accuracy failureを記録する。予測側の適格集合が空ならabstention。比の分母の対象集合を必ず表示する。

直接compileしていない候補があるとき、その候補を含む全gridに対するregretや最適性は認定しない。subset regretはsubsetと明記する。

### 4.6 主要な出力

絶対資源、同一候補のsigned相対費用誤差/log比、過小評価、method別resource ratio、モデル選択regret、適格性の誤判定、候補取得の古典費用を報告する。相関係数だけでモデルの有用性を判断しない。

新しいmaterialityを使うなら、例として選択regret 10%を事前登録できる。ただし連続量も全件報告し、10%という値を科学法則や旧M2判定の延長として扱わない。

## 5. ベンチマーク設計

### 5.1 系サイズとgeometryの推奨案

中心はH4/H6/H8、STO-3G。新規計算の中心はH6/H8であり、H4 geometryを大量追加する案は優先しない。

| 系 | 1.00 Å | 1.30 Å | 情報上の役割 |
|---|---|---|---|
| H4 | 既存218候補を校正へ利用 | 元5構成は既使用、広域mapはない | development |
| H6 | 最初のサイズ検証 | 構造依存の確認候補 | 予測を固定後に評価 |
| H8 | 未使用サイズの検証 | サイズ×構造の確認候補 | 最終確認用 |

二つの距離を全新サイズで評価する6-instance案が、サイズとgeometryを混同しにくい目標設計である。予算を絞るならまず全サイズ1.00 Åのサイズ系列を完成させる。H4=1.00 Å、H6=1.30 Å、H8=別距離のように同時に軸を変えない。

H6を見てモデルを変更した場合、H6は変更後モデルのdevelopmentへ移る。変更前の予測・結果は保存し、H8を未使用確認にする。H8を見た後の再fitを確認済みモデルとは呼ばない。

### 5.2 高精度側を最初から含める

主な表示anchor案は ε_signal={0.05,0.01,0.005,0.001,0.0001}。旧範囲との接続と、その50倍厳しい側までを含む。必要に応じて対数曲線を保存値から描くが、表示点数を新しい検証数として数えない。

T=0.8を主比較に残す。同じ固有状態・正しいphase branchを仮定すると、化学精度ΔE=0.0016 Haに対応する十分な単一信号誤差の目安は sin(TΔE)≈0.00128、CA/10では約0.000128である。これらを参照線として表示できるが、RPE各roundの要求精度やenergy保証達成と同一視しない。

ε=0.0001で候補が不適格でも、shotだけ増やして救済しない。q/R/Kの登録範囲が不足したのか、候補のbias floorなのか、数値参照の誤差なのかを分ける。

### 5.3 時間方向

T=0.8だけでは短時間依存を排除できない。目標scopeにfinite-time一般性を含めるなら、結果前に4T=3.2等の長い側の一点を限定診断として用意する案を推奨する。これは原論文QPEの長時間全体を検証したことにはならない。

全候補×全Tの直積は最初から要求しない。同じalgorithm familyをTに合わせた事前候補生成規則で評価するか、固定構成のtime-transferを試すかを分けて宣言する。

### 5.4 stateとHamiltonianの契約

サイズごとに数値的に収束した同じ物理セクターのground-state参照を主benchmarkとする案が自然である。H4はground-like参照、H8はHFのように途中でstateの意味を変えない。state preparationはprimary外として、共通P感度とその限界を保持する。

DF表現を全サイズでrank12へ固定しない。既存H4 rank12は旧証拠のidentityであり、他サイズでは同じ生成・切断規則から得たrank Lを使う。geometryでrankが必ず不変とも仮定しない。

full molecular Hamiltonian、DFで表したtarget Hamiltonian、B0でdiscardしたHamiltonianを区別する。DFの表現誤差を測らず「full chemical Hamiltonianに対する保証」としない。fragment順、縮退tie規則、scalar/identity処理、orbital/mappingを固定する。

異なるサイズで全エネルギーの位相が変わるとaxis bias配分も変わり得る。エネルギーoffsetと測定phase conventionを同一規則で扱い、結果を見て有利なphaseへ回さない。

## 6. 候補集合の公平性

### 6.1 prefix

H4のL_D=3をH8へ単純移送しない。L_D/Lの分率gridまたは累積fragment weightで候補生成規則を作る。例えば分率0,1/8,1/4,1/2,3/4,1と、その丸め近傍を用意する案がある。

B0とB2へ同じ内側prefix集合を与え、B1=L、B3=0を必ず保持する。小さいLなら全prefixを使うことも検討する。近接prefixを結果後に追加する運用ではなく、±1の近傍を含める規則を最初から持たせる。

この候補生成規則がweightに基づく場合、そのweightをexact RTE λ_Rと混同しない。実際のλ_R、内側rank、sampled supportは別に保存する。

### 6.2 qと総RTE steps R

二次PFではR=q r、τ=λ_R T/R。canonical有限分布なら

\[
B=\beta_K(\lambda_R T/R)^R.
\]

したがってr≤32だけを全qへ共通に課すと、q=8,r=32にはR=256を与える一方、q=1,r=256を除外してしまう。これがq選択へ探索境界の影響を持ち込む。

新計画はqとRを主座標にする。qは例えば1,2,4,8,16,32,64を初期候補、Rは2冪系列とし、R/qが正の整数である組を使う。K=2,4に高精度用K=6を加える案を初期候補とする。

最大q、最大R、Kの必要範囲は、最厳精度、入力λ_R、有限Taylor残差、計算予算からAX-1/2で固定する。上限は無制限に拡張しないが、特定methodだけ先に打ち切る設計にもせず、boundary-limitedを明記する。

### 6.3 全gridと直接評価subsetを混同しない

比較予定の解析モデルで安い候補だけを選び、同じモデルの選択能力を評価してはいけない。

推奨は、有限の直接比較集合X_directを結果前に固定し、そこでは全件の必要な信号と直接費用を得ること。より広いX_modelのモデル予測は探索的補足とする。

全件が予算を超える場合は、method、prefix、q、R、K、境界を覆うmodel-independentな層化subsetを固定する。そこで得るregretはX_direct内に限る。信頼できる下界で除外を正当化できた場合だけ、より広い集合の結論へ進める。

誤差上界がεを超えることは、その候補の実際の不適格性を証明しない。特にdiscard/PF相殺を取り得る候補を、緩い上界だけで実証比較から除外しない。

### 6.4 高次PF・強い合成

二次PF内のmodel auditと、PRが有力な決定論法より有利というclaimを区別する。後者を目標にする場合は、高精度側で少なくとも一つの事前固定した有力な高次deterministic PFを比較対象にする。

本計画では高次partial-RTEを同時開発しない。強い合成の確認も、旧実装policyと同値性を確認した既知policyを一つ比較する単位に限定する。B0/B1だけでなく適用可能なB2にも同じ改善を与える。

原論文の回路と現行回路が違う場合には、この合成対応確認は「追加で強いbaselineを入れる任意作業」ではなく、そもそも同じモデルを検証しているかの要件となる。

## 7. 測定規則・合成精度

### 7.1 二つの測定会計

旧結果への接続として、対称軸配分・α_real=α_imag=0.025を維持した会計を残す。

ただし拡張研究で広い実用的な資源順位を論じるなら、非対称軸配分の感度を一度検査する案を推奨する。これは新signal/compileなしで可能な古典的最適化で、例えば

\[
\min_{s_R,s_I>0}\;2B^2\log(2/\alpha)\left(\frac{C_R}{s_R^2}+\frac{C_I}{s_I^2}\right),
\quad (b_R+s_R)^2+(b_I+s_I)^2\le\epsilon_{sig}^2
\]

を用いる。整数shotへの切上げと実現した誤差配分を保存し、全methodへ同じ規則を与える。これはHoeffding型十分条件内の改善であり、最適推定器の発明ではない。

主・副どちらの会計を採用するかは結果前に決め、順位を見て選ばない。元のPM-2判定は変更しない。

### 7.2 logical gate指標とfault-tolerant指標

第1層は旧RZ-work/CX-work/回路深さ等の比較。任意角rotation、Clifford角、basis-change/control/readoutを識別する追加metadataを収録する。旧CSVに角度がなければ、既存値だけでnon-Clifford rotationやT数を捏造しない。

第2層は、指定したgate synthesis policyと誤差予算下のT count/T depth/logical ancilla等。これは第1層からの単一換算係数では得られない。等角rotationの共有、合成精度、補助状態の準備・償却も明記する。

normalization補正後は、raw信号の実装誤差ηが最大でBηへ増幅され得る。したがって「全候補に同じraw gate errorを与えれば公平」とは限らず、補正後のsignal accuracyの中へ合成誤差を配分する。

物理runtime・surface-code qubits等まで今回一度に求めない。原論文のToffoli総資源と直接比較する場合には、合成・task・誤差単位が対応した第2層が必要である。

## 8. 大規模化のための数値エンジン

### 8.1 密行列経路をそのまま拡張しない

STO-3G H_Nを2N qubits、complex128とすると、全空間の行列1枚のメモリは次のとおり。値は次元からの算術計算であり、実測peakではない。

| 系 | 全空間次元 | complex128行列1枚 | Nα=Nβ=N/2セクター次元 |
|---|---:|---:|---:|
| H4 | 256 | 1 MiB | 36 |
| H6 | 4096 | 256 MiB | 400 |
| H8 | 65536 | 64 GiB | 4900 |

fragment行列・固有vectors・作業領域を複数持つ現行経路は、行列1枚よりさらに重い。H8の主経路として全空間密行列は禁止する設計を推奨する。

### 8.2 代替経路

数値参照とtail作用は、保存対称性に対応したセクターまたはmatrix-free作用で計算する。各primitiveがNα/Nβを保持するか確認し、その条件でのみ上表のセクターを使う。

DF fragmentのexp作用はorbital rotationと対角phase作用を利用し、full matrix exponentialを構築しない。一般のHermitian作用にはKrylov/expm_multiply型を候補とする。SciPy1.14.1のexpm_multiplyはlinear operatorの指数作用を直接求められる。[W8]

有限RTE平均は、正規化tail Hbarに対する

\[
P_{K+1}(-i\tau\bar H)v=\sum_{k=0}^{K+1}\frac{(-i\tau\bar H)^k}{k!}v
\]

をmatvec再帰で評価できる。これをr回、外側PFの間に作用させる。元実装のfinite polynomialと演算順・phase・正規化を一致させる。

GPU statevectorでrandom circuitを大量sampleして信号平均を得ることは、有限分布平均の直接作用と別経路である。高精度biasをcostの32 samplesから推定するような置換はしない。

### 8.3 correctnessと数値誤差

H4で既存dense経路と新state-action経路の、raw/corrected signal、B、exact-tail、scalar phaseを比較する。小型synthetic例ではoperator同値性も調べる。H6で時間・メモリの少数profileを行ってからmain campaignを認可する。

高精度側は、参照state、DF表現、Krylov/多項式作用、roundingの数値誤差を管理する。例えば最小signal精度の1%未満を数値誤差予算として仮置きできるが、適格境界のheadroomより誤差幅が大きい候補は判定不能とする。数値再現幅を厳密certificateと呼ばない。

任意stateを扱う場合はz(T)を実際に評価する。小さいRayleigh residualだけでground-state重みや最小固有値を保証したことにしない。

## 9. cost samplingとモデル選択の統計

MCの対象は1-shot circuit costの期待値。必要なquantum shotsと、古典的なcost用trajectory数は別変数として保存する。

同じcandidate内では同一trajectoryをRe/Im、解析会計、直接compile、policy比較へ対応付け、差のSEにも共分散を保持する。candidate間を独立seedにするかcommon random numbersにするかを固定し、その仮定と集計を一致させる。

初期32 trajectoryは旧証拠と接続する実行案であり、十分精度を保証しない。method間の主結論が不確かさで変わる候補だけ、事前規則で独立追加batchを一回実施する案を採る。例としてfresh+96を上限にできるが、r/Kの小差を解像するために全件へ適用しない。

有限Kでもrare Taylor orderが32本に出ない場合がある。既知のorder確率を記録し、重要なrare stratumを層化するか、未観測確率質量×保守的費用上界の寄与を別に評価する。観測されなかったため確率0・費用0とは扱わない。

点最小の選択とSEを同じ標本で行うことによるselection biasを認識する。主比較の候補をfreezeした後の独立cost標本で確認するか、empirical ledger regretとして保証の限界を明記する。family minの単純な±2SEを同時95%CIと呼ばない。

ε方向の大量表示点は同じcost標本の再重み付けであり、独立確認数は増えない。Hamiltonian単位、事前に固定したprimary精度点、sample batch単位で不確かさを報告する。

## 10. 計算量・予算

1 instanceでrandom candidate数M_R、deterministic candidate数M_D、cost用標本数nなら、基本wrapper数は

\[
W=2nM_R+2M_D.
\]

例としてM_R=300、M_D=40、n=32なら19,280 wrapper。これは予算規模の例で、最終候補数ではない。新規4instanceなら77,120 wrapper。追加policy/time/confirmationを行えば、その部分の費用が増える。

εの表示点を増やすだけではwrapper数は増えない。高精度で新q/R/Kが必要な場合にのみ新費用取得が発生する。

wall timeをH4の旧実測秒からqubit比だけで外挿しない。H6で短・中・長回路を覆うprofileを取得し、median・高分位のcompile秒、worker peak RSS、signal matvec数、I/O量を測る。

CPU worker数はcore数だけでなくRAM制約から決める。GPUの有用性はstate-action側で測り、通常のtranspilationの高速化を当然視しない。新エンジンが別環境を要するなら旧venvを上書きせず別環境にpinする。

古典的なモデル構築・真値参照・compileの費用と、推定する量子資源を別表にする。「安価なpredictor」を主張するなら、その入力取得費用も含める。

## 11. ステージとbarrier

旧A0等と混同しないよう、拡張シリーズをAXと呼ぶ。

| stage | 内容 | 新規science | 完了条件 |
|---|---|---|---|
| AX-0 | 文献式・公開データ・実装対応、RQ、task、metric、候補生成、energy接続仕様 | なし | 何が忠実再現／適用版／別実装かを明示 |
| AX-1 | H4保存結果で費用・誤差・選択の切分け、未校正／校正モデル、精度coverage監査 | 保存値解析を中心 | H6前にprediction/model versionをfreeze。差の大小をGO条件にしない |
| AX-2 | サイズ対応エンジン、小系整合性、H6少数profile、予算固定 | 限定あり・別認可 | 数値・memory・時間・sourceのgate通過 |
| AX-3 | H6の固定campaign | あり | 精度適格性・費用・model residual・regretを結果によらず報告 |
| AX-4 | H8の未使用サイズ検証、事前固定した構造/time/policy確認 | あり | 凍結予測を対照に評価。変更したmodelには新versionを付ける |
| AX-5 | 統合論文、達成scopeの固定。必要ならenergy版の実行判断 | 集計・文書 | RQ-R/RQ-Pの両方について成功・失敗・不確定を閉じる |

H6でB2が負けたこと、modelがよく当たったことは、それだけではH8中止理由にしない。H8の進行条件は比較成立性、実行可能性、未解決の研究上の問いが残ること。

stageをさらに小さな承認往復へ分割し続けない。主要barrierは、計画固定、本計算前の技術・予算freeze、H6後のversion確認、最終統合とする。

## 12. STOPと不適格を混同しない

- correctness failure：Hamiltonian/phase/state/sampling/費用意味論の不整合。科学的結論を出さず停止する。
- reference uncertainty：数値誤差が要求精度やheadroomに対し大きい。判定不能として明示する。
- computational limit：時間・RAM・wrapper上限。実行不能とmethod不可能を分ける。
- candidate-boundary limit：最良候補がq/R/K/prefixの端へ張り付く。登録範囲の結果として報告し、範囲外最適性を主張しない。
- scientific negative result：PRが不利、モデルに予測誤差なし、方式逆転なし。正しく評価できた結果として報告する。

大規模campaignでは、旧pilotの「一回launch・一切resume不可」を無条件にコピーする必要はない。新契約で許可する場合に限り、source/input/policy/candidate/seedを固定したtask単位の中断再開を使える。完了済み結果はimmutableに再利用し、未完了attemptは履歴を保持する。結果を見てseedやcandidateを差し替えるretryは禁止する。

旧M1/M2/PM-1の認可、registry、cacheは再利用しない。新branchまたは新系列directoryで、文献版・環境・モデル・入力・予測・結果を結合する。

## 13. 新規性の評価

部分ランダム化、DF残差処理、hybrid simulation、測定費用との競合は既知である。[W1,W3,W4] 原論文は既に水素鎖のサイズ依存やQPEを扱う。H6/H8へ進むこと自体が「先行研究より大きい系」を意味しない。

拡張版の差分候補は、同一task・metric・実装へ対応した解析モデルの校正／検証、費用と誤差の寄与分離、未使用サイズでの構成選択損失、精度・測定規則・合成構造を含む具体的な適用範囲である。

モデルが誤る場合だけ価値があるのではない。未使用条件で一定のresource誤差とselection regretに収まるという定量的な確認も、外挿や設計に役立ち得る。ただし、観測した範囲を超える保証とはしない。

新しい校正式や説明モデルを提案できる場合も、既知の単純式への代入・パラメータfitだけを新しい一般アルゴリズムと呼ばない。strong synthesisやimportance samplingとの重複を限定する。[W4,W6]

今回の確認は最接近論文と関連一次資料の重点確認であり、全関連文献の不存在証明ではない。SPRINT/GRADE等、旧原稿で挙げた広い統合設計はAX-0の詳細照合に残す。今回、その公開本文の再取得は完了していない。

## 14. 論文としての着地点

### 標準目標

「水素鎖の有限時間PRに関する、実装整合した資源評価と解析モデルの予測信頼性」。H4開発、H6/H8確認、高精度側、適用可能な既知合成を含む比較条件、数値・MC不確かさを明示する。単一H4の実装報告から、有限サイズの体系的な数値研究へ拡張する。

### 条件付きの上位目標

同じenergy-error定義と合成policyでRPE全体へ接続し、原論文のQPE資源と比較可能にする。これを選ぶなら、単一時間の結果では完成としない。全時刻の参照取得やcompileが難しい場合は、直接検証範囲と外挿範囲を分け、最終予算への外挿不確かさを示す。

### 縮小して閉じる場合

比較は正しいが、サイズ・精度を広げても既知trade-offの再現と測定条件依存の順位以外が残らないなら、旧原稿と併せたbenchmark/resource noteとして閉じる。これはB2が負けたための罰ではなく、得られた情報に合わせたscopeである。

### 新しいモデル差がある場合

具体的な誤差原因の特定や、少数の構造量で予測が改善しH8へ移送できる等の結果が出れば、その内容を中心にする。都合のよいgeometryを後追加して改善を作らない。

## 15. 直近の実務指示案（まだ実行認可ではない）

AX-0では、文書・source text・公開一次文献・保存CSV/JSONだけを使う。

作るもの：

1. `track_a_extension_research_contract_v1.md`：RQ、scope、評価量、判定条件。
2. `paper_to_native_crosswalk_v1.csv`：原論文の式/仮定、現行の対応、忠実再現/適用/別実装/N/A。
3. `benchmark_and_candidate_rules_v1.json`：N/d/T/ε、prefix/q/R/K規則と未確定上限。
4. `prediction_and_truth_separation_v1.md`：H4開発、H6/H8評価、model version、予測保存時点。
5. `scalable_signal_backend_audit_v1.md`：dense経路依存、matrix-free候補、sector保存性、必要test。
6. `compute_budget_template_v1.json`：candidate数・MC・wrappers・CPU/RAM/GPU/VRAM・wall・storageを別field。
7. `energy_task_bridge_spec_v1.md`：有限signalとchemical-accuracy energyの境界、必要条件、未実行項目。

元原稿とM1〜PM-2を変更しない。新H6/H8作成、NPZ/state読込、GPU helper import、signal/trajectory/compile、既存science runnerの試行は行わない。

AX-0を確認した後、AX-1の保存値解析とAX-2の必要な技術pilotをまとめて設計する。各小作業ごとに研究方針をゼロから変えず、予め固定したRQと終了条件に照らす。

実行予算を確定するために必要な利用者情報は、CPUコア数・RAM、GPU機種/枚数/VRAM、並列利用上限、許容wall/総計算時間である。未提示の期限、人数、投稿先は仮定しない。

## 参照

### リポジトリ一次資料

[R1] `docs/pr2_pm2_precision_resource_result_validation.md`, commit `4c23453c541700c6a41ba71fc5ec9323b53858d6`。

[R2] `docs/manuscripts/track_a_resource_study_v0_1.md`、`docs/research/track_a_post_pm2_claim_evidence_map.md`、同commit。旧scopeと評価規則は会話に提示された本文も参照した。

[R3] `src/trotterlib/pr2_matched_accuracy_m1_execution.py`, 同commit。今回確認：`_dense_block_operators`, `_eigendecomposition`, `_target_signal`, `_random_signal_record`。

[R4] `src/trotterlib/df_gpu_statevector.py`, 同commit。今回確認：冒頭のCUDA探索/preloadとGPU simulator設定。統合の完了を示す資料ではない。

### 外部一次資料

[W1] Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026), DOI: 10.1103/ynxb-p2xq, arXiv:2503.05647。今回重点：Sec. VII、Appendix D、E.1.b/E.3.c。journal PDFの該当頁を画像でも確認。

[W2] Günther et al., *Figures for arXiv:2503.05647 Phase estimation with partially randomized time evolution*, Zenodo record 15387187。公開archiveの存在・説明・previewを確認。archive内容の再実行はしていない。

[W3] Hagan and Wiebe, *Composite Quantum Simulations*, Quantum 7, 1181 (2023), DOI: 10.22331/q-2023-11-14-1181, arXiv:2206.06409。

[W4] Cugini, Atif and Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。net-costとsampling変更の既知範囲を確認。

[W5] Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2。QPE/物理資源taskとの重複確認。

[W6] Simon and Love, *Halving the Cost of Controlled Time-Evolution*, arXiv:2511.13855。出版日や最新journal名の確定を本計画の根拠には用いない。

[W7] Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024), DOI: 10.22331/q-2024-06-13-1371, arXiv:2212.07957。

[W8] SciPy 1.14.1 Reference, `scipy.sparse.linalg.expm_multiply`。matrix exponentialのoperator action、linear operator入力、traceAの性能上の注意を確認。
