# N1・N3再設計とN2補助修正：限定scope v1

2026-10-11 JST。状態は `DESIGN_PREPARED_SCIENCE_NOT_AUTHORIZED`。
本書は新しい設計・数学的条件・将来の比較範囲を具体化した準備資料であり、実装済みsourceや実行認可ではない。
[machine contract](n13_redesign_limited_scope_v1_20261011.json)と一組に読む。

## 1. 受領した方針と今回の到達点

利用者提供の[完了済み科学レビュー](n13_redesign_inputs/n123_feasibility_scientific_review_2026-10-11.md)を受領した。
同じ結果のGPT再レビュー・付録script再実行・旧batch再実行は行わない。
基点は `ae7434452450e80ec007d65417335e18a063fa1e`、旧science sourceは
`ed335a007bee6966a37f3026ac188534e526b5b9`。旧source・入力・結果・監査は凍結したまま保存する。
独立worktree/branchは `n13-redesign-preparation-20261011`。

今回許可された開始点は限定作業の具体化である。利用者の
「今回の『次の作業は』という質問だけでは、新しい科学batchの実行承認とは扱いません」
を適用し、新selector評価、LP solve、対角化、compile、科学的synthetic検証は開始しない。
実施した確認は文書・sourceの静的読解、保存済み候補の件数確認、入力copy/hash、資源inventory、準備資料の整合性だけ。

優先順はN1/N3の設計・保証、N2の固定S修正。中心テーマは未採択。
科学実行は一回限りの下記packageを別途source固定・実行指示へ結び付ける。
成功するまで条件を変えて実行する契約にはしない。

## 2. 共通の比較・情報契約

N1/N2は固定時間complex Hadamard信号、N3は基底エネルギー縮約幅であり、異なるtaskを総合順位にしない。
N1はJW4、ν=2、sector dimension6、signed synthetic DF rank2、geometry/化学basisなし、
DF-prefix `L_D`は該当せず全fragment決定論S2。
T=.6、q={1,2,4,8}、delta={.6,.3,.15,.075}、モデル予算εH=.02、εcomplex=.04、二軸失敗率合計.05を維持する。
旧三contextの角度・係数・状態X0X1を変えない。N2の回路scanは0。
N3はJW6、ν=2、初期active4、density .3 n0 n1の固定模型、エネルギー幅目標δgoal=.004。

selectorが参照できるのは係数、係数行列の小さい代数演算、証明付き上界、固定compilerの回路費用と規則だけ。
full/active many-bodyの真の最低値、gap、真のoperator誤差、真のPF誤差は選択後のauditorに隔離する。
係数から作る試行状態のRayleigh上界は許す。全active Hilbert空間の固有問題/SDPへ置換して効率性を主張しない。
selector recordをhashで閉じてからtruth diagnosticsを作り、APIも別にする。

finite binary64係数は `as_integer_ratio()` で正確な有理数として読める。
意図されたdecimal係数、binary64で生成された入力、三角関数を含む理想回路を混同しない。
数学的な不等式、丸めを囲った機械certificate、binary64の残差診断を別fieldに保存する。
残差がtolerance以下でも保証の誤差予算を0にしない。

## 3. N1：全回路費用と保証付き誤差を使うselector

### 3.1 有限候補と対照

旧公開resultの `generator_candidates` にある44 factor候補を今回の有限domainとする。
planted_local_gaugeとdense_group_frameは各factor9件、separated_spectrumは各4件。
pairの直積は81/81/16、合計178。これは保存recordの件数であり、新しい構成評価ではない。
[selector専用入力](../../artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json)には
一体係数・候補frame/spectrum/familyだけをcopyし、旧many-body Hamiltonian行列、rows、真のmodel/PF誤差、
旧binary64 residual diagnosticsを除いた。新runnerのselectorは旧full resultをloadせず、この入力のhashを検査する。
元のproduct-boundによる候補生成filterを通ったdomainに限定するため、鋭いboundで初めて追加可能になる候補は今回は増やさない。
同じスペクトルでの新center、partition、角度、精度grid追加は0。

N1 classは{exact,cluster}、有限shifted対照は{exact,shifted_cutoff}、zero対照は{exact,zero_cutoff}。
family間を混ぜた組合せは混合対照として明示し、N1単独の効果へ帰属しない。
全pairにgauged/ungaugedの同一近似対を置き、同じq・compiler・bound会計で扱う。
native exactと元H whole-Pauliを必ず含める。新N1のRZ/CX選択に対して同じ近似H whole-Pauliも後段対照に置く。
whole-PauliのPF順序は別の公式なので、自分自身の係数PF上界を計算する。
同じ近似H whole-Pauliは機構比較であり、選択前に全pair分をcompileしたglobal候補とは呼ばない。

### 3.2 モデル誤差の数学と有限精度

共通unitary基底Qで比較する理想factorでは、レビューが導出した

\[
e_\ell=|w_\ell|\max_{|S|=\nu}
\left|\left(\sum_{i\in S}\eta_i\right)^2-
\left(\sum_{i\in S}\widetilde\eta_i\right)^2\right|
\]

を使い、εH≤Σeℓとする。ν=2なので各factor6組の係数和。
一般fillingでの全subset列挙をpolynomial算法とは呼ばない。

実装ではfactorのexact候補のungauged Gaussian回路からQの一体作用を定義する。
元ηにはこのexact候補の `ungauged_eta`、ηtildeには近似候補の `ungauged_eta` を使う。
同じfactorのungauged frameが同じ列順序であることを入力identityで確認する。
Qは理想gate角度でunitary。元のsorted `eta` と列並べ替え後の `approx_eta` を直接引き算しない。
元gとQ diag(η) Q†の行列entry-L1残差上界r0、および実際のgauged/ungauged回路factorと
Q diag(ηtilde) Q†の残差上界r1を、小さい4×4係数行列と区間sin/cos演算から囲う。
s0=max|ΣSη|、s1=max|ΣSηtilde|、d0=νr0、d1=νr1と置くと安全な追加項は

\[
e_{\ell,\mathrm{cert}}=e_\ell+
|w_\ell|\{d_0(2s_0+d_0)+d_1(2s_1+d_1)\}.
\]

理由は ||dΓ(R)||ν≤ν||R||≤ν||R||entry-L1 と
||F²−G²||≤||F−G|| (||F||+||G||)。full-Fock normやground truthは不要。
さらに、係数から指定したdiagonal phaseと実際に発行されたbinary64 gate角度との差、controlled scalar phaseを
各gateの位相差上界として加える。未認証のeigh/frame residualをtoleranceで消去しない。
区間certificateを組めなければ `STOP_N1_ROUNDING_CERTIFICATE_UNAVAILABLE`。

### 3.3 PF上界とshot込み目的

Hermitianな実装fragment列Hjに対し、Rj=Σk>j Hkとして

\[
\alpha_2=\sum_j\left(
\frac{\|[R_j,[R_j,H_j]]\|}{12}
+\frac{\|[H_j,[H_j,R_j]]\|}{24}\right),\qquad
b_{\rm PF}\le \frac{T^3}{q^2}\alpha_2.
\]

は既知の二次S2誤差上界である。交換子をPauli係数辞書で正確にまとめ、
各normを係数L1で上界化する。identityは交換子から除き、controlled phase費用/位相誤差には残す。
Pauli roundoffも区間で囲う。BCHの先頭項や旧保存PF真値を上界へ置換しない。
適用する積順序と残差分割は実際のcircuit列に一致させる。
出典は[Childs et al., §5.1, Proposition 16, Eq. (152)](https://arxiv.org/html/1912.08854v2#S5.SS1)。

理想構成のbias予算は b=TεH,cert+bPF+bangle。b≥εcomplex/√2なら不適格。
compile後まで数学的に保証すると表現するには、別途nativeと理想構成の誤差囲いb_nativeを加える必要がある。
これはgate係数・角度からの回路意味の認証であり、真のHamiltonian/PF誤差のoracleではない。
通常のbinary64全列residual検査だけの場合は `ideal_gate_certificate_with_numerical_compile_validation` と明記し、
compiled回路の厳密な機械certificateやhardware精度を主張しない。native認証が得られないことをPF真値で救済しない。
整数shotは

\[
N_\mathrm{safe}=\left\lceil
\frac{2(\log80)_{\rm upper}}
{((\epsilon_{\rm complex}/\sqrt2)_{\rm lower}-b_{\rm upper})^2}
\right\rceil,\qquad
G_{m,\mathrm{cert}}=N_\mathrm{safe}(C_{X,m}+C_{Y,m}).
\]

RZ/CXを別々に最小化し、secondary metric、model bound、q、canonical candidate idの順にtieを解く。
費用はordinary control diag(I,U)、scalar phase、X/Y readoutを含むfull measured wrapper。
旧N1のwrapperと同じく固定準備X0X1も費用・意味検証に含める。
これは一般の状態準備算法の費用評価ではない。normalization=1。
ceil境界は区間128bit、必要なら256bitの一段だけ精密化し、なお未確定なら上側shotを採る。

exactも同じq集合とPF会計で選ぶため、domain内の適格exactに対する
Gcert(selected)≤Gcert(exact)は保証できる。しかし**真の誤差を使う実費用がexact以下という保証ではない**。
exactの全qが係数PF上界で不適格なら `UNRESOLVED_PF_CERTIFICATE`。
保証に失敗したexactを誤差0の適格候補として扱わず、旧PF診断やq追加で救済しない。

### 3.4 結果と停止

選択後にだけ6次元sectorのmodel/PF/元H信号誤差を診断し、保証の包含とceil会計を照合する。
certificate順位、真値診断の費用順位、同じ近似Hでのframe削減を別表にする。
selectorのAPIへdiagnostic fileを渡す試験は拒否する。
誤差費用込みで新候補を選べたか、既知有限対照との差があるかを報告する。
exactへ戻った場合も正常な研究結果。新規性や研究テーマは自動認定しない。
全不適格/改善なし/対照以下でも保存してSTOPし、同型gridを追加しない。

## 4. N3：結合方向を残す係数certificate

### 4.1 最初の適用class

number-preserving one-body hと非負density相互作用Σi<j Jij ni nj、固定νを対象とする。
Pは一つの外部軌道dを空にしたactive空間、Qはnd=1。
負density、一般二体exchange/pair hopping、外部軌道2本以上へこの式を流用しない。
初期active4の2外部段階は既存の係数scalar保証を保ち、保証不能なら旧係数coupling規則で軌道を戻す。
外部1本になったときだけ本構造certificateを試す。特定のmode idやactive数5を採否に使わない。
全軌道に戻った場合はδ=0のexact fallbackであり、縮約成功ではない。

### 4.2 係数から作る量

active一体行列をhA、外部結合vectorをvとする。HのP/Q blockを(A,B;B†,C)と書く。
v_i=h_{i,d}という規約でB†=a_d†Σi conjugate(v_i) a_i、したがって

\[
BB^\dagger=d\Gamma_\nu(vv^\dagger).
\]

これはQに一つの外部軌道だけを含むclassのCAR恒等式であり、最大結合normへ潰さない。
Mλ=hA−λIが正定値であることを有理数LDLで証明し、

\[
\kappa=v^\dagger M_\lambda^{-1}v,\qquad L=\nu\lambda
\]

を係数線形solveから取得する。v=0は別branch。
rank-one Schur不等式Mλ≽vv†/κを二次量子化し、active density VA≽0を加えると

\[
BB^\dagger\preceq\kappa(A-LI),\qquad A\succeq LI.
\]

一体行列のGershgorin下界rとdiagonal span s=max h_ii−min h_iiから
λ∈{r+ρs:ρ=0,1/20,1/10,3/20,1/5}の五点を先に固定する。
SPD不成立の点は棄却し、結果を見てgridを足さない。s=0では重複除去、SPDが証明できなければ未判定。
同じgridで証明した最大の一体下界ℓから
c=hdd+(ν−1)ℓを得る。非負外部/active densityは差し引かない。
C≽cIであり、真のdiscarded最低値は使わない。

Uは係数から評価するnormalized試行状態のRayleigh上界。
ν=2の全active determinantのdiagonal値と、一体hopで直接接続した各pairの
(|D_i>−sign(Hij)|D_j>)/√2 の期待値 (Hii+Hjj)/2−|Hij| の最小を採る。
今回実数係数に限定し、fermionic signを係数から付ける。
最大active5、10 determinants以下の固定小試行集合であり、真のactive基底状態・固有値を求めない。
Uは上界として粗くても有効。結果を見て試行集合を追加しない。

### 4.3 採否条件と証明

δ>0に対してc>U、t=κ/(c−U+δ)≤1、δ≥t(U−L)を有理数で検査する。
すべて係数由来である。このときactive最低値aとfull最低値E0について
0≤a−E0≤δを保証できる。

証明だけでa=λmin(A)と置く。aはselectorの入力/出力fieldに含めない。
z=a−δとするとC−z≽(c−U+δ)I>0。よって

\[
\begin{aligned}
A-z-B(C-z)^{-1}B^\dagger
&\succeq A-a+\delta-t(A-L)\\
&=(1-t)(A-a)+\delta-t(a-L)\\
&\succeq\{\delta-t(U-L)\}I\succeq0.
\end{aligned}
\]

Schur補元からH≽zI、Pの変分原理からE0≤a。
既知の[Feshbach–Schur法](https://arxiv.org/html/2105.02058v1#S1)に沿った十分条件であり、
この短い証明自体を新規定理と呼ばない。研究上の候補は、小さな係数行列で結合方向を保つ構成・適用条件にある。
新規性・有用性は限定検証後に評価する。

正式採否はδgoal=1/250に対する上の有理数条件を直接検査する。
floating eigensolver、真のgap、未知aによるroot/最適λ選択は0。
κとLDLの分子/分母は最大4096bits、超過は `UNRESOLVED_ARITHMETIC_CAP`。
v=0でも外部に低い状態がある可能性を残し、c≥Uの確認なしにδ=0としない。
条件を満たさなければ軌道を戻すか未判定を保存し、真の誤差が小さいからacceptしない。

### 4.4 将来の限定検証

元模型のhoppingとdensityを固定し、以下の4 inputsだけを結果前に宣言する。

| input | diagonal e | 変更 |
|---|---|---|
| original_gapped | (−.2,−.1,.1,.25,6,8) | 元入力 |
| original_small_gap | (−.2,−.1,.1,.25,.4,.6) | 元入力 |
| strong_external_coupling | original_small_gap | h35=.16、h45=.20（元の4倍） |
| near_external_energy | original_small_gap | e5=−.1 |

対照は旧scalar係数保証、同じU/cを使うnorm-only係数保証、今回の方向付き保証。
一外部軌道ではrank-one CARからβ²=v†vも係数で正確に得られる。
同じU/cに対するnorm-onlyの採否はc>Uかつδgoal≥β²/(c−U+δgoal)。
これを方向付き条件と比較することで、trial上界・外部下界だけの改善と方向情報の追加効果を分ける。
旧scalarのclass係数も有理数で計算し、二外部軌道の採否は各d_j>Uかつ
δgoal≥Σj β_j²/(d_j−U+δgoal)を直接検査する。binary64 fixed-point値をcertificateへコピーしない。
各class下界とblock off-diagonal上界の証明をsource固定前に確認できなければ、当該baselineは未認証と明記する。
各inputでactive4から最大6へ進む同じ拡張規則を使う。δgoal、λgrid、trial-state規則は共通。
改良scalarだけで十分なのか、方向情報が追加の採否を変えるかを分ける。
強結合・低外部エネルギーの二件は反証候補であり、真の誤差が目標を超えると事前断定しない。

selector record固定後、最大15次元ν=2 sectorからa/E0/Cの診断を取り、
係数L/U/c/κ、BB†≼κ(A−L)、採択幅のsoundnessを独立に照合する。
レビューの「小gapでも5 modesで十分」はdiagnostic既知結果であり、accept条件や成功宣言ではない。
保証違反は即 `STOP_CERTIFICATE_SOUNDNESS_FAILURE`。改善なしは正常な限定負結果として保存する。
active solver cost、状態準備/QPE成功率、全時間発展誤差Tδ、総量子資源改善は本packageの主張に含めない。

## 5. N2：固定Sだけのweighted entry-L1修正

固定4×2のSについてKの対称独立成分を変数とし、J−SKSᵀのupper triangle残差に補助変数tijを置く。
−tij≤(J−SKSᵀ)ij≤tij、tij≥0の下で
Σi tii+2Σi<j tijを最小化する標準LPへ置換する。
Frobenius最小化をentry-L1最小化の代わりにしない。

レビューの同じfixed S・同じ摂動Jを一件だけ扱い、LSとweighted-L1の採否を比較する。
意図された有理数fixtureと保存binary64 fixtureを区別し、後者にdecimalの1/500を無条件コピーしない。
前者の1/500は受領した既存数学結果で、今回は再solveしていない。
将来のLP出力はprimal residualを再計算し、dual certificateも有理数回復/検査する。
大きなS探索、旧820候補のLP再scan、charge回路再compileは0。

成立条件は、固定Sの疎性、charge数、register bit幅、compute–uncompute費用とdiagonal phase数を整理する。
native RZ/CXからfault-tolerant T/Toffoliやcatalyzed HWPの優劣を推論しない。
将来FT比較をする場合は既知HWP/directに同じsynthesis精度・workspace・catalyst準備償却を与える別契約が必要。
既知adderの交換やLPの採否修正だけを新算法の資源改善と呼ばない。

## 6. 実装・最小検証・性能の準備条件

将来moduleは `src/trottertracks/representation_exploration/n13_redesign.py`、runnerは
`scripts/run_n13_redesign.py`、testは `tests/test_n13_redesign.py`、保存監査は
`scripts/verify_n13_redesign.py` とする。これらは今回まだ作成・実行していない。
旧moduleやfrozenジョブを書き換えず、新sourceとsource-bound planを先に固定する。
科学結果は `artifacts/n13_redesign/2026-10-11/` 以下の新runへ分離する。

数学条件の最小検証には、CAR符号、ν=0/1/2、zero coupling、非対角active、
SPD不成立、attractive density拒否、境界δ、scalar位相、外部低エネルギー、
oracle入力拒否、LP対称重み、roundoff containmentを含める。
これは実装成立の検証条件であり、実行済みtests数ではない。
科学truthの照合はsource/selector固定後の限定4/6-mode scopeだけ。

今回read-only inventoryはonline CPU32、MemAvailable約51.8 GiB、nvidia-smi executableあり、
/dev/nvidia*なし。GPU backendの利用可能性は確認できないのでCPUを第一案とする。
旧run3のwall38.32秒・worker peak約482MiBという記録と静的sourceから、N1のcompile/全列作用検査を主負担と見込む。
速度向上率は計測していない。
独立N1 pair/q/axisをspawn最大2 workers、BLAS/OpenMP/Numba/Rayon各1、Qiskit parallel FALSE、
transpile num_processes1、seed20261011、optimization_level1、native rz/sx/x/cx、
`qubits_initially_zero=False`を全対照に適用する。task id由来seedはprocess配分に依存させない。
小さい一体/sector/Pauli辞書はimmutable cacheで再利用し、dense truthをcacheへ混ぜない。
compiled IR reuseには入力係数・frame・全角度・q/axis・builder/source・compiler/version/settingsの同一性が必要。
gate countだけ同じという理由で旧IRを流用しない。旧実行で保持したphase/workspaceの絶対作用条件を保つ。

上限はN1 factor44、pair178、gauged/ungauged最大356、q4、axis2、
wrapper compile最大3072（pair系2848＋元H whole-Pauli24＋選択近似whole-Pauli48を含む余裕）、
各circuit system4+control1、native20000 gates、N3 scientific inputs4・三比較法・各active段階最大3、
計36 certificate records以下、N2 fixed S一件。
同じfactor basisの独立再認証が必要なら最大44件を別counterとし、compile総計3116以下。
worker AS4GiB、aggregate8GiB、各JSON/IR shard≤32MiB、総出力512MiB、log16MiB。
大きなIRを一個のpretty JSONへ詰めずtask別shardとhash indexへ保存する。
λ五点、trial determinants10以下、Pauli係数辞書256項以下、rational4096bits。
phase/total wall-time・CPU-time capは設けない。tool poll timeoutは科学実行上限ではない。
launch直前に他ジョブを含む資源を再確認し、optimized native actionを別referenceと最小条件で照合する。

## 7. STOPと引継ぎ

1. source/丸めcertificate/identity付きplan未固定、実行指示なし：`DESIGN_PREPARED_SCIENCE_NOT_AUTHORIZED`。
2. boundが作れない・全q不適格・適用class外・上限超過：原因と未判定を保存して停止。truthで救済しない。
3. soundness、ordinary control、scalar phase、workspace、元H誤差の検査失敗：failureと完了checkpointを保存して停止。
4. 改善あり/なしの全terminal結果：source、一次結果、auditor、比較条件を保存し、GPTへ新結果の判断を返す。
5. 一回実行後の条件追加・λ/q/精度/角度再調整・N2 scan・分子/H12/長RPE/Track A/B統合は0。

旧GPTレビューは完了として扱う。新結果のレビューは結果取得後だけ。
中心仮説null、next_scientific_stage_authorized=false、mandatory STOPを保持する。
将来のcommit/pushはHIROMU1015 remoteだけを使い、公開後に新blobをremoteから再取得する。
本準備時点で新science source、科学結果、immutable CI、外部再現、新規性認定は存在しない。
