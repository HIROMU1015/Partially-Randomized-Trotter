# Track B研究再評価とGPT分析チェックポイント
作成日：2026-10-09
位置付け：GPTによる研究判断・新しい数学的整理。実行authorizationではない。

## 0. 結論
RA-RTEを現在の候補として限定継続する。ただし「汎用的なexact LP基盤を完成させ、その後で研究価値を考える」という順序にはしない。
次の一つの判断マイルストーンを、(A)固定P3候補表の構造簡約の独立監査、(B)既に提案した最大8件のbackend coverage closure、とする。
両者が終了した時点、または重大な反例・技術停止が発生した時点でGPTレビューG1へ戻す。v4 production実装へ自動進行しない。

今回の最も重要な新しい導出：
固定7 prototypeの理想degree-matching classは3次元多面体であり、既存ordinary/PTSC-K0/Aと追加3構成の計6頂点の凸包として表せる。
これは費用を評価した結果ではない。登録LPを求解せず、sourceの係数式に基づいて導いた構造的結果である。
数値許容幅付きの元K3全体や、階層量子化後の像の同値性は意味しない。

## 1. 対象と証拠の区別
評価の最新基点は
`7f9062975d9f2b09f12cda6e83e7de1e830beac4`
（exact backend pilot v2）に固定する。これより後の未提示成果は前提にしない。

- 既存証拠：固定commitの報告・source・保存監査。
- 外部文献：一次論文・著者公開版で確認した既知内容。
- 今回の導出：下記P3構造簡約と、研究上の判断の整理。独立監査前。
- 今回の提案：次の作業の範囲とGPTレビュー点。既存STOPや消費済みmarkerを解除しない。

## 2. ここまでの結果の意味
### B-F
固定family/development条件ではFとLのfinite最良が同じSuzuki5。
さらに共通対照のnative S2が安かった。原INCONCLUSIVEと事後BF-Aは保持する。
高次PF最適化一般を否定した結果ではない。

### B-M
固定nested列の三次BCH係数はcompact再帰と同値。
既知法にも同じaggregation/reuseを認めると、現adapterの独立差はなかった。
同値な構成を別名で新methodにしないという教訓である。

### SP/BS
合成とsamplingの交換は存在し得るが、primitiveの利益はwrapper全体へ自動的に残らない。
SP-1の外側coinはactual finite RTEと同じではない。
BSの現candidateは、同情報・同辞書・同処理のgeneric sparse/operator LCUと独立差が未定義だった。
「generic LCUに含まれるから無価値」ではなく、「具体的な入力・処理・保証が同じなら独立差はない」と解釈する。

### R0/R0.5
一般Hermitian involutionに対する非負adjacent-degree family、有限平均保存、全奇数次数のnormalization最小構成がある。
優先性と任意LCUでの最適性は未確定。PTSC-K0は同I0の重要な対照。
I1のcollected CTSが強い場合がある。小さいtoyでPauli展開を使わない実装を評価したことは、実問題のI0アクセス優位を意味しない。

### R1/R1.5
固定2-qubit P3、distinct-basis controlled、x={1/8,1/4}、pygridsynthの3精度での資源forecast。
Aは登録3資源Paretoに残るが、全精度・全資源で最良ではない。
例：x=1/4, native 1e-3ではA/PTSC-K0のB²比0.99457に対しG_T比0.82349。
native費用とbiasから定めたshotsが同時に作用した。量子実測やwhole-wrapper最適compileではない。

### RA-D0 / T0–T0.2 / v4
v3の登録solverは最初のB2候補で停止し、B2/B3比較は0件。
T0.2は保存1点の事後的な実行可能lawを示したのみ。
v4数学監査は条件付き十分条件を支持したが、保守的innerの空集合は元classの空集合を意味しない。
backend v2は9 primal/dual、6 Farkasを認証し、10^-99 gapで証明未取得。最大認証サイズは5変数で、RA-shaped例は未実行。

## 3. 研究RQを三段階に分ける
### 大きなRQ
PRの内部時間発展アルゴリズムを改善し、同じ物理的な目標精度・成功条件で資源を削減できるか。
Taylorに限定した一般優越性は前提にしない。
Hamiltonian前処理等のPR外経路は別チャットとする。

### 現在の方法RQ
同じ有限Taylor operator meanを保存しながら、実装可能なeventと精度の配分を設計する方法に有用性があるか。
canonical sampling、有限表、single blockは現在の研究範囲であり、普遍的な制約ではない。

### 直近のRA-D0 RQ
同じ保存表、同じconfidence規則、同じ資源制約で、degree-local classが既存3表現の混合classにない資源点を作れるか。
これは狭いmechanism testであり、最終PR優位の証明ではない。

## 4. 理論的な改善余地の整理
R0のclass最適Aを含む限り、同じ非負adjacent-direction classでcanonical normalizationをAよりさらに下げることはできない。
複数angle/precision列を許しても、偶奇成分vectorの三角不等式でB>=sqrt(E²+O²)となる。
したがってB3で狙うものは「Aより小さいB」ではなく、
Bを少し増やしてでもnative費用・bias・shots・capsの組合せを改善すること。

R0にはP3で
B_pair-B_A=x^4/9+O(x^6)
という小step展開がある。総時間をs分割したnormalizationだけの利得は小stepで弱まる。
これはRA-RTEのnative-resource利得全体の上界ではない。

## 5. 新しい構造的導出：固定P3の6頂点
### 5.1 入力prototype
x>0、t=(1,x,x²/2,x³/6)、rho=(x+x³/6)/(1+x²/2)とする。
列は正規化されたunitaryそのものではなく、sourceに保存された正規化前のdegree coefficient vectorである。

- v_O0=(1,x,0,0)
- v_O2=(0,0,x²/2,x³/6)
- v_P2=(0,0,x²/2,0)
- v_P3=(0,0,0,x³/6)
- v_A0=(1,rho,0,0)
- v_A1=(0,x-rho,(x-rho)/rho,0)
- v_A2=(0,0,x³/(6rho),x³/6)

各非負倍率gamma_gについてV gamma=tを考える。
これはv4の構造変数ではgamma_g=(sum_p u_gp)/yに相当する。
unitaryの係数は理想normをc_gとすればw_g=c_g gamma_gである。gammaは抽出確率ではない。

mu=(x²+2)/(x²+6)、lambda=1-mu=4/(x²+6)。
x>0では0<mu<1である。

### 5.2 自由度の消去
s=gamma_O0、r=gamma_A2、b=gamma_O2と置く。
degree 0,1,2,3を順に合わせると、

gamma_A0=gamma_A1=1-s,
gamma_P3=1-r-b,
gamma_P2=mu+(1-mu)s-mu*r-b.

よって全解は
gamma=(s,b,mu+(1-mu)s-mu*r-b,1-r-b,1-s,1-s,r)
と書ける。

非負条件は
0<=s<=1,
r>=0,
b>=0,
r+b<=1,
b+mu*r<=mu+(1-mu)s.

有界3次元多面体となる。

### 5.3 全頂点
0<mu<1のとき、頂点は次の6つ。

| 名前 | (s,r,b) | 非zero prototype倍率 |
|---|---|---|
| ordinary | (1,0,1) | O0=1, O2=1 |
| PTSC-K0 | (1,0,0) | O0=1, P2=1, P3=1 |
| A | (0,1,0) | A0=A1=A2=1 |
| J1 | (0,0,0) | A0=A1=1, P2=mu, P3=1 |
| J2 | (0,0,mu) | A0=A1=1, O2=mu, P3=1-mu |
| J3 | (1,1,0) | O0=A2=1, P2=1-mu |

証明の確認：
6つの境界平面から3枚を選ぶ交点を分類すると、上記以外の一意交点は
(s,r,b)=(0,0,1),(1,1/mu,0),(mu/(mu-1),0,0)
であり、いずれも0<mu<1で非負制約を破る。
残りは重複頂点、平行、または独立性が不足する境界である。
有界多面体なので全体はこの6頂点の凸包となる。

今回の自己検算では、記号xでV gamma=tを6構成と一般parameterizationについて確認し、
抽象muの境界交点も列挙した。登録xでの費用評価・LP optimizationは実施していない。

### 5.4 既存B2の位置
ordinary/PTSC/Aをtheta_O,theta_P,theta_Aで混ぜると、
s=theta_O+theta_P、r=theta_A、b=theta_O。
従ってB2は
r=1-s, 0<=b<=s
という三角形の断面に対応する。

B3は、この「低次側でAを使う量と、高次側でAを使う量を連動させる」条件を外している。
大きなgroup変数空間に見えていた追加自由度は、固定P3のgroup matchingに限れば一つの追加連動解除として解釈できる。

### 5.5 Precisionを含めた表現
gamma_gp>=0、sum_p gamma_gp=gamma_gとする。
gammaを6頂点の混合に分解した後、各groupについて既存のprecision share
pi_gp=gamma_gp/gamma_g
を各頂点の同groupへ共通に付与すれば、元のgamma_gpを復元できる。
gamma_g=0ならそのgroupの全gamma_gp=0とし、shareを定義しない。

従って、理想group matching＋任意precision分配の段階では、
「6完成構成の混合＋group別precision配分」によってdegree-local表現を記述できる。

ここでの凸混合は係数分解の混合である。完成構成の混合係数theta_rをそのまま回路抽出確率にするとは限らない。
各構成のnormalizationをB_rとすると、canonical samplingを維持する実際の構成選択確率は
theta_r B_r / (sum_s theta_s B_s)
であり、全体の補正重みはsum_s theta_s B_sとなる。
theta_rで構成を選び、その都度B_rで補正する別推定器との二次モーメントの同値は仮定しない。

重要な限定：
- 6つの固定precision点を評価するだけで全資源最適化を解いたことにはならない。
- caps/confidenceによって混合点が最適になり得る。
- 量子化後の像の一致は示していない。B2/B3で別々にLRMすればlawは一致しないことがある。
- 元K3の数値mean許容幅にある微小な非構造点まで表現したとは主張しない。
- B2へ追加3構成を勝手に入れて既存primaryを変更しない。
- J1–J3は本資料で導いた構成名であり、既存法・新規性確立済みalgorithm名ではない。
- x=0は別のidentity caseである。
- 高次数や別のangle dictionaryへの6頂点保証ではない。

### 5.6 研究上の意味
B3が既存3表現の混合に勝っても「全てのwhole-ensemble mixtureを超えた」とは言えない。
勝因は追加3構成に由来すると整理できる可能性がある。
これは研究を否定するのではなく、新しい自由度の所在と小型の構成familyを明示する。
新規性はこの3構成の名称ではなく、その構成法、一般化、取得費用、性能域に求める。

## 6. 新規性の暫定評価
既知：
- PTSCなどのidentity配分・Euler pairing・LCU補正。
- CTSによるTaylor項の集約とstochastic unitary構成。
- Sparse Probabilistic Synthesisにおける有限辞書の凸最適化。
- Resource-Optimal ISにおける固定ensembleのcost×second-moment最適化。
- continuous TE-PAI等のmean-channel保存とangle-dependent tradeoff。

候補として残る差：
- 一般involutionに対する有限operator meanを、具体的なdegree-local構成で保存すること。
- 全奇数次数のrestricted normalization optimumと構成。
- phase-awareな実装表、有限confidence、資源制約を整合させた設計手順。
- native implementationで本当に有用な適用域を説明し、古典取得費用まで数えること。

未確立：
- publication priority。
- 強いIS/CTS対照後の資源差。
- actual PR wrapperへの効果。
- 対象familyを超える普遍的最適性。

注意：channelの平均一致とoperator first momentの一致は別である。
例としてIと-Iは同じsystem channelだが、controlled実装の干渉信号は異なる。
従ってchannel-level既知法は無視もしないが、無料でcoherentタスクへ移植できるとも仮定しない。

## 7. 科学的判定を強くする
### Positive
共通task/capsで
U3<L2
なら、該当queryの追加自由度の利益を認証できる。
一つのqueryの証拠を一般的な優位へ拡張しない。

### No witness
U3<L2が見つからないだけでは、利益の不存在を意味しない。
内側生成器の保守性、未認証点、finite query不足、loose lower boundを分ける。

両元classで実行可能性があり、妥当なB2 upper U2と元B3 classに対するlower L3があれば、
0 <= G2* - G3* <= U2-L3.
これが小さければ「残り得る改善幅が小さい」と言える。
B3 innerの最小値をL3として使用してはいけない。
数値/理論幅が大きければ、結論は未判定であってno-go theoremではない。

### Application
single-blockでの削減率をそのままPR全体へ転記しない。
固定周辺費用CF、random部分費用CR、全体shots Nとすれば
G_new/G_old=(N_new/N_old)[(1-f)+f(CR_new/CR_old)],
f=CR_old/(CF+CR_old).
Nが変わる場合は共通の決定論的回路費用にも効果が及ぶ。
multi-blockのnormalization、bias伝播、controlled phase、準備/測定、basis遷移を別途会計する。

## 8. 次の一まとまりのCodex作業（提案）
### A. 構造簡約の独立監査
本資料の6頂点・precision復元・適用範囲を記号的に監査。
登録LP solve、新synthesis、登録費用再採点は行わない。
反例があれば保存し、勝手に元sourceへ修正を採用しない。

### B. Backend coverage closure
既に提案した未実行B2/B3型6 fixture＋結果前固定の高精度2 fixture、最大8 LPの範囲。
既存backend・guard・失敗記録を保持し、既知10^-99 gap failureを消さない。
100桁の係数と10^-99のinfeasibility gapは別の性質として扱う。
残り6件の30-bit程度の係数だけでv4高精度問題へ適合すると断定しない。
controllerの取得失敗/不正証明の分類を分離した仕様を実行前固定する。
新production、新authorization、registered B2/B3はまだ進めない。

この二つを別々に何度も往復するより、G1へ戻す一つのresearch-decision packetにまとめる。
現在の数値条件や旧結果を上書きしない。実行には別途この範囲の明示指示が必要である。

## 9. GPT分析チェックポイント
### G0（今回）
研究RQと成果の再評価、構造簡約の提案、限界と次の判断単位を固定した。

### G1（次の必須レビュー）
時点：上記A/Bの終了、または重大な反例・proof acquisition failureでの停止。
Codex上限：数理監査＋最大8人工LPまで。productionへ進めない。

GPTが決めること：
1. 6頂点簡約は正しいか、どのclassまで等価か。
2. 旧v4をそのまま実装するか、構造を使って簡素化するか。
3. SoPlexは元入力の認証可能な候補生成器として使えるか。既知failureをどう扱うか。
4. 次の最小research query set、比較法、数値精度、資源上限は何か。
5. 数値基盤の整備をさらに続ける情報価値があるか。
6. 科学的な優位を主張するために、どの既知法との差を最初に取るか。

出力：採用model、未確定仮定、source開発範囲、次実行の判定基準。実行許可は別。

### G2（source/実行契約レビュー）
時点：承認されたsourceとoff-domain testsの完成。登録LP実行前。
GPTが確認：
- input→solve→decode→独立certificateの一貫性。
- B2 upper/lowerとB3 upper/lowerの対象class。
- source/inputs/markersと予算freeze。
- query coverage不足とtechnical unknownの分類。
- 実用的効果の評価方法をB3結果前に決める。
- 最初は旧anchor集合を起点とした有限batch。全737点へ自動拡張しない案を検討。
出力：小さい登録比較のGO/NO-GO、明示的な実行authorizationの必要範囲。

### G3（最初のB2/B3比較後）
時点：最初の承認batchの終了。coverage/IS/新tableへ進む前。
GPTが確認：
- strict witnessの有無。
- 効果の絶対量/比率、どの資源座標か。
- J1/J2/J3やprecision配分などの勝因。
- U2-L3による残余headroomと未判定幅。
- 合成列依存、bias上界依存、技術未完率。
- 次にIS等の強い対照へ進む情報価値。
出力：続行、縮小、設計修正、停止のいずれか。

### G4（強い対照とtransfer後）
時点：採択された強いbaseline比較および小さい未使用条件/PR wrapperへのtransferの終了。
GPTが確認：
- 固定ensemble＋ISで説明されるか。
- CTS等と同じ情報/targetで比較できているか。
- 正しいcoherentタスクに対して効果があるか。
- full-wrapper費用、shot、取得費用込みで利益が残るか。
- 論文の中心claimと、必要な追加実験を最小化できるか。
出力：論文構成、残すclaim、削るclaim、最小completion plan。

### 臨時レビューのtrigger
数学的反例、target/accuracy/caps/accessの変更が必要、保存証拠改変の恐れ、backend取得失敗が再発して予定milestoneに到達できないとき。
承認済み範囲内のoff-domainバグ修正・テスト追加・文書修正は、作業契約が許すならCodexでまとめてよい。
旧consumed runのno-retry規則は維持する。

## 10. 論文着地点
### A. Algorithm/theory paper候補
一般の有限平均保存family、restricted最適性、native-aware設計の具体手順と取得計算量、有用性の証明または強い比較。
一般LPを利用したというだけでは足りない。

### B. Methods/design study候補
Normalization optimumと実資源optimumの違いを、固定taskで再現し、精度/回路/shotの交換と有用域を説明する。
2-qubit一表だけで広い優位を主張しない。

### C. 限定theory/technical result
大きな改善がなくても、restricted theoremや改善余地の上界が明確なら保存する。
これだけで独立論文に十分と断定しない。方法優位の物語を無理に付けない。

BF/BM/SP/BSの全履歴を主論文へ列挙する必要はない。
それらは候補絞り込みの記録であり、中心成果の証拠と開発履歴を分ける。

## 11. 参照資料
固定repoはすべて HIROMU1015/Partially-Randomized-Trotter。

- BF-1回収報告：commit 6d2645a09440f50e5b869ef42a1b73a1b625a1af,
  docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md
- BM-0.5：d55de044b8e956ba6292209a94bb081014dfdae2,
  docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md
- SP-1：9d2bb1fa439748b02084bd9fbc9b10a705328f8a,
  docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
- BS-0.5：5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d,
  docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md
- R0/R0.5：672d6bc667eaa7b9ca4979b012f1530499d701b8 / 61dd534567fda5c7348fdc688814089eb26a3561
- R1.5：af3d014d0a0cfcbbd25bb544f6544652fec92942,
  docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
- v4数理監査：beb82427d202f479cc2ba954480d73a51941e322,
  docs/tracks/algorithm_codesign/ra_d0_v4_mathematical_audit_20261009.md
- backend v2：7f9062975d9f2b09f12cda6e83e7de1e830beac4,
  docs/tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_gpt_handoff_20261009.md
- prototype式：同最新commit,
  src/trottertracks/algorithm_codesign/ra_d0/table.py

今回確認した外部一次資料：
- Günther et al., Phase Estimation with Partially Randomized Time Evolution,
  PRX Quantum 7, 020332 (2026), arXiv:2503.05647.
- Zeng et al., Simple and High-Precision Hamiltonian Simulation by Compensating Trotter Error with Linear Combination of Unitary Operations,
  PRX Quantum 6, 010359 (2025), arXiv:2212.04566.
- Peetz, Smart, Narang, Quantum Simulation via Stochastic Combination of Unitaries,
  arXiv:2407.21095v2, Methods IV.2 and Supplementary Note 3.
- Koczor, Sparse Probabilistic Synthesis of Quantum Operations,
  PRX Quantum 5, 040352 (2024), arXiv:2402.15550v2.
- Cugini, Atif, Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms,
  arXiv:2603.13495v1 (2026), Theorems 1–2.
- Dai, Hasselgren, Kiumi, Structure-Aware Variance Reduction for Unbiased Randomized Hamiltonian Simulation,
  arXiv:2606.23544v1 (2026), continuous TE-PAI and angle tradeoff.
- Nakaji, Bagherimehrab, Aspuru-Guzik, High-Order Randomized Compiler for Hamiltonian Simulation,
  PRX Quantum 5, 020330 (2024), arXiv:2302.14811.

文献との差分は限定的な確認であり、全引用網・全バージョンに対する優先性の確定ではない。

## 12. 今回実施した操作の範囲
固定repo資料と一次文献の読み取り、上記のsymbolic degree-matching恒等式と境界交点の自己検算、本資料作成のみ。
登録LP solve、RA-D0 runner、新synthesis、quantum matrix、分子/DF計算、repository変更、authorization作成は行っていない。
自己検算は独立Codex監査ではない。
