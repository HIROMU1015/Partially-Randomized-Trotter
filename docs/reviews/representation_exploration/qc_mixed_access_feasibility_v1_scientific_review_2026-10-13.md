# 固定H4 mixed-access feasibility v1：科学レビュー・backend投資判断

- 作成日：2026-10-13（日本時間）
- 開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`qc-mixed-access-feasibility-run-v1-20261013`
- 公開ref：`ac930c0e0d255a187dc21917104e5cb5b4d09bcb`
- 固定結果payload：`ae3975de24bff6ab6aec3494b522851db2a8f3dd`
- 科学source：`102e6b31401356d6eea775169041ebdf2cfa6ba0`
- 実行承認・preflight：`214f9ab2d7d3e0b64b287f14f49795c99a1a8244`
- 入力SHA256：`f345af12281836731c3122bb2a4c0c2b389d7c90ded7ba16976d3cfeeb7bd951`
- 運用指示：更新済み `research_project_instructions_gpt_codex_review_approval_2026-10-12.md`。研究全体の俯瞰を含む本文を読んだ。
- 今回の実施：固定GitHub資料・source・保存集計・監査記録の読取り、前回レビューとの比較、一次文献の限定確認、解析的な解釈、研究判断と本記録の作成。
- 今回の非実施：新しい分子入力、SCF/DF/固有値/SVD、PF生成・評価、保存データの新しい数値集計、最適化、T合成、compile、synthetic再実行、原run再実行、Gitへの書込み。

## 0. 結論

**現行のbitwise可変角回転・occupation算術templateを前提に、詳細な可逆関数backendへ追加投資することは勧めない。現H4 feasibility系列を、限定結果を保持して一区切りにする。**

ただし、これは次の主張ではない。

- THRIFT一般、mixed-access一般、量子化学への適用全般の不可能性。
- あらゆる回路最適化の後もH4で必ず負けるというlower-bound証明。
- 基底エネルギー推定の全費用での敗北。
- 既知手順を使った応用・実装研究は価値がないという判断。

今回支持する二つの結果は、同時に保持する。

1. 固定H4、指定した作用誤差・有限gridでは、理想THRIFT型PFにstep数削減が見られた。
2. 登録した回路template・誤差配分・T価格仮定では、未知function費用を追加する正の余地が、全273 eligible frontierに存在しなかった。

したがって、**「理想PFとしては改善したが、その改善を現実装設計の費用で購入する見通しは得られなかった」**という限定的な科学結果として固定する。

次のCodex作業は、既存資料への判断・claim範囲・再開条件の追補と、証拠索引の維持である。今回のレビューを理由に、別分子、r追加、価格変更、function実装、T合成、保存データの追加数値解析を自動開始しない。文書反映の完了だけなら、同じ科学レビューの反復は不要。

## 1. 研究全体と前回判断からの変化

利用者が指定した中心目的は、量子化学Hamiltonianの基底エネルギー計算を低コスト化することである。PR・PF・THRIFTは手段であり、PRとの結合やスペクトル強度への対象変更は必須ではない。

これまでの重要な経緯は次のとおり。

- N1では部分回路の改善が誤差・shot込みの固定taskの改善にならなかった。N2は保留。これらの限定結果を全構成の不能性へ広げない。
- N3では保証理論・条件族分離・有限prototype結果を保持しつつ、現検証系列を終了した。保証幅の改善だけで最終taskの改善としなかった。
- 旧THRIFT toyでは条件付きmixed演算子を構成できたが、強いS2対照を入れると固定8条件の費用優位は残らなかった。これは小さいIsing構成の結果で、今回の分子入力と異なる。
- 今回のrelative-block constructorは、既知のTHRIFT相対積・occupation対のSU(2)・可逆算術へ構成的に対応し、独立した新方式としては非採択となった。ただし性能は未評価であった。
- 直前の科学レビューでは、「新方法差がないから化学応用も無価値」とせず、詳細backendを作る前に、実分子の構造と許容function費用を限定的に調べる方針を採った。
- 今回、その一分子の判別結果が得られた。

前回から変更するのは、**費用余地を調べるための限定継続から、現templateのbackend投資を優先しない判断へ進むこと**である。前回の方法差判定を再審査して覆したのではない。

なお、他Trackの成功・停止状態は、その固定範囲の履歴としてのみ参照した。本レビューで最新進捗を横断的に再調査したり、他Trackの優先順位・契約・結論を書き換えたりしていない。過去の多数の否定的結果を「今回も失敗するはず」という証拠へ流用していない。

## 2. 取得資料と検証水準

### 2.1 今回読んだ資料

| ID | 内容 | 使用目的・読取り範囲 |
|---|---|---|
| R1 | 固定H4結果報告 | 入力・PF grid・step数・費用・限界。ツールの表示範囲の主要本文を読んだ |
| R2 | GPT引継ぎ | 研究判断事項・結果payload・公開記録への対応 |
| R3 | `feasibility_contract.json` | 結果前条件、価格仮定、目標精度、入力・計算上限 |
| R4 | `qc_mixed_access_feasibility_plan_v1_20261013.md` | 何を判別するpilotか、比較・投影・費用の設計 |
| R5 | `source_manifest.json` | 科学sourceと固定16 filesの参照 |
| R6 | 固定 `qc_mixed_access_resource_v1.py` | stage費用、word、angle、precision、価格、eligibility、frontierの式 |
| R7 | 固定 `qc_mixed_access_math_v1.py` | DF、stage構成、fusion、relative作用、reference/normの該当部分（230–450行） |
| R8 | `descriptive_summary.json` | 指定部分の保存step/error行、best remainingのsymbolic差、known総T最小、未知項 |
| R9 | `result.json` | 原分類、実行内容、ground-energy/compile未評価 |
| R10 | `independent_audit.json` | 独立CAR・保存norm witness監査の記録と限界 |
| R11 | `saved_execution_integrity.json` | 154 probes、1386 rows、273 frontierの保存算術・identity監査記録 |
| R12 | 直前GPTレビューMarkdown | 前回の投資判断と限定feasibilityの目的。会話に添付されたローカル原本を読んだ |
| L1–L2 | THRIFT、単一RZ合成の一次資料 | mixed-accessの費用条件とT価格仮定の意味を限定的に確認 |

### 2.2 今回行っていない独立検証

全raw NPZ行列の再取得・再生成、全154 PFの再評価、1386行の再集計、273 frontierの再計算、実際の回路・合成費用の取得は行っていない。`resource_evaluation.json` の全大規模payloadを新たに走査して最良値を再選択したわけでもない。

そのため、本レビューは**公開された一次結果・source・保存集計・監査記録を基礎とする科学的解釈と投資判断**であり、元の数値実験の独立再現ではない。NPZ本文を使う新数値再現は今回の判断に必須とはせず、その未実施を隠さない。

R10は全PF trajectoryの独立再生成ではない。R11は保存算術とbindingの監査であり、回路templateの最適性やT価格の実現可能性を証明しない。公開byte照合は外部環境での科学再現やimmutable CIとも別である。

## 3. 固定scopeの確認

対象はH4直線鎖、隣接1.00 Å、STO-3G、実RHF canonical MO、4 spatial/8 spin modes、粒子数ν=4、charge0/spin0、frozen coreなし。geometryは既知developmentで、blind transferではない。[R1–R3]

DFは16 factorsと補正one-bodyを全保持。1e−12でのnumerical rank10は報告用であり、16→10への切捨ては行われていない。既存rank12/19の別実験と同じHamiltonian・rank費用とみなさない。L_Dなし、PRなし。

方法はP2（JW Pauli）、E2/E4（excitation）、D2/D4（full DF）、R2/R4（known compressed THRIFT）。T=1,4 atomic time、r=1,2,…,1024の倍増grid、目標作用誤差ε_U=0.001。

誤差は

\[
 e_{m,T,r}=\left\|\left(S_m(T/r)^r-e^{-iTH}\right)Q_\nu\right\|_2
\]

で、70個のν=4入力columnsと全256出力rowsを使う。個々のPauli stageをsectorへ投影していないため、sector外への作用を落としていない。global phaseの最小化もしていない。これはcontrolled evolutionの位相を勝手に捨てない比較として適切である。[R3,R4,R7]

ただし、ε_Uは無次元の作用誤差で、0.001 Haやground-energy 1 mHa達成とは異なる。FCI・ground/gap diagonalization・QPE成功率の評価はない。

PF予算fractionは0.4/0.6/0.8、残余はR側でcoefficient/function/synthesis三等分、baselineでcoefficient/synthesis二等分。DF model residualと数値marginも登録どおり控除。これは公平に定義された一つの配分規則だが、全配分の最適化ではない。

eligibleには次の2rと全ての大きい登録rの通過を要求し、1024だけの通過を無条件採択しない。有限gridでの孤立通過を抑える設計であって、grid外収束の証明や全r最適性ではない。

154点、1386費用行、273 frontierは、一つの分子を異なるmethod/T/r/配分/価格で評価した数である。独立な273分子や統計的成功・失敗率へ置き換えない。

## 4. 理想PFの結果は肯定的に保持する

同じPF fraction0.6（予算0.0006）の最小eligible stepを転記する。[R1,R8]

| 方法 | T=1のr | operator error | T=4のr | operator error |
|---|---:|---:|---:|---:|
| P2 | 8 | 5.57418e−4 | 64 | 2.69999e−4 |
| E2 | 8 | 5.66599e−4 | 64 | 2.74722e−4 |
| E4 | 2 | 5.16830e−5 | 8 | 4.77571e−5 |
| D2 | 16 | 2.23571e−4 | 64 | 3.85259e−4 |
| D4 | 2 | 5.11161e−5 | 8 | 5.37890e−5 |
| R2 | 8 | 1.64213e−4 | 32 | 3.94818e−4 |
| R4 | 1 | 2.30620e−4 | 4 | 1.89449e−4 |

R4は同次数E4/D4に対して両Tで半分のr。R2はT=4でE2より半分、T=1では同r。この改善は元Hに対する理想PF作用誤差の比較として保持する。

ただし次を主張しない。

- step数低下＝gate数低下。
- 全手法の一stepが同じ数の指数演算・回転を含む。
- 別のT・精度・基底・分子でも同じ比率。
- この二つのTで小α漸近則を検証した。
- R4とE2のような異次数比較から次数別の新しい優位性を認定した。

R4には高次compositionの段数・負時間があり、R2の中央は隣接する逆RYを共有する設計が入っている。step数の表と実装単価を必ず別に扱う。[R6,R7]

## 5. 費用結果は何を意味するか

### 5.1 登録modelの結果

primaryはsymbolic T frontierであり、T_RZ(η)はUNKNOWN。数値額は

\[
 t_\kappa(\eta)=\left\lceil\kappa\log_2(1/\eta)\right\rceil,
 \quad \kappa\in\{3,6,12\}
\]

という仮説価格で計算されている。κはこの資料では**価格係数**であり、以前のN3でのLMIパラメータ等と同一ではない。[R3,R6]

| T | 価格κ | baseline最小 | baseline仮説T | R known総T最小 | R known仮説T |
|---|---:|---|---:|---|---:|
| 1 | 3 | E2/r8/fraction0.6 | 612,208 | R2/r4/fraction0.8 | 19,906,056 |
| 1 | 6 | E2/r8/fraction0.6 | 1,123,408 | R2/r4/fraction0.8 | 36,163,544 |
| 1 | 12 | E2/r8/fraction0.6 | 2,145,808 | R2/r4/fraction0.8 | 68,678,520 |
| 4 | 3 | E4/r8/fraction0.4 | 2,905,776 | R4/r4/fraction0.4 | 102,453,120 |
| 4 | 6 | E4/r8/fraction0.4 | 5,353,136 | R4/r4/fraction0.4 | 185,302,656 |
| 4 | 12 | E4/r8/fraction0.4 | 10,247,856 | R4/r4/fraction0.4 | 351,001,728 |

表は保存集計の転記で、今回算出した実測値ではない。[R1,R8]

共通fraction0.6でのstep表と、全登録fractionから選んだ費用最小表は別である。T=1の費用最小R2がr4であるのはfraction0.8を選んだためで、前節のR2/r8と矛盾しない。新規方式だけに事後的な追加配分を与えた結果ではない。

また、known総Tが最小のR行と、remaining budgetをrで割った値が最大のR行は異なる。例えばκ3の最大remainingはT=1でR2/r8/fraction0.4、T=4でR2/r32/fraction0.4である。異なる行の分子・分母を混ぜた改善率を作らない。

### 5.2 不明な関数部分だけを安くしても、現会計は逆転しない

登録された行をθとし、

\[
 C_R^{\rm hyp}(\theta)=K_R^{\rm hyp}(\theta)+F_\theta,
 \qquad F_\theta\ge0
\]

と書く。K_Rは登録templateのknown会計額、Fは未会計function・cleanup等の追加額。固定の比較baselineをBとする。

全eligible θでK_R^{hyp}(θ)>B^{hyp}なので、**同じK_Rを会計として維持する限り、非負のFをいかに小さく選んでも勝てない。**

ここでF=0を実装できたと認定しているのではない。Fを未知の非負変数として残した、不等式の帰結である。したがって、「sqrt/atan2の実費用を測れば、それだけでpositive budgetになるかもしれない」という理由で現設計のbackendを完成する投資は正当化しにくい。

### 5.3 しかし、これは回路不能証明ではない

K_Rは実現済み最適回路の下界ではなく、設計上界・保守的envelope・価格仮説を含む。baseline側も同様にT合成実測ではない。

一般に

\[
 \overline C_R>\overline C_B
\]

という二つの上側設計値の比較から、

\[
 C_R^*>C_B^*
\]

は導けない。また、価格仮説はrigorous synthesis upper boundでもない。[R3]

負frontierは、**今回の登録モデルで、backendを追加する余地を示せなかった**という結果である。これを最適回路・THRIFT一般・全量子化学・最終energyのNO-GO定理に拡張しない。

本レビューのNO-GOは「現設計への追加投資を推奨しない」という研究配分判断であり、数学的な一般不能判定ではない。

## 6. sourceから分かる費用差の構造

### 6.1 90個のtransitionを、同じ単価では処理していない

H4の厳密group後M_eff=90、各detuningのliteral nonzero係数数z_j=4…6と報告されている。通常励起PFも同じsingle/density-assisted groupを使用する。R側だけのgroup化利益は与えていない。[R1,R3]

通常のexcitationは、既知係数やアフィン振幅の直接controlled rotationを使える。一方、R側はoccupation依存detuning・場合によって振幅の算術、angle registerを使う複数の可変回転、flag・parity・逆算を含む。

通常のR actorでは4回転、中央の共有pairでは5回転を用いる。angle bit数aに対し、sourceのbitwise部分は、回転数をn_rotとして

\[
 N_{RZ}=2n_{rot}a,
 \qquad N_{Toff,enable}=2n_{rot}a
\]

などを加える。通常actorでは各8aに相当する。detuning/group加算には(4w−6)(z+z_g)も加わる。[R6]

これが全相対block・全composition段・全rに掛かる。さらに各RZの許容誤差ηは、synthesis budgetを全RZ occurrence数で割っている。**回転本数が増える効果と、各回転へ要求する精度が厳しくなる効果が同時に入る。**

これはstatic sourceから読める機構であり、各費用の割合を今回新たに集計したものではない。

### 6.2 「88bitの算術を全R blockに課したから負けた」とは言えない

報告ではcommon word最大88、function angle最大40、Rのgroup input word22…31。sourceはR/RPAIRではgroup固有wordを実際のstage会計へ渡しており、common word88をそのまま全detuningへ適用していない。[R1,R6]

一方、angle wordは保守的な一様の設計であり、各angleの特殊値や全ての小角最適化が実際のT合成で価格へ反映されたものではない。ここを最適性と解釈してはいけない。

### 6.3 保存symbolic行からの限定的な追加解釈

T=1の最大remainingを与えた固定行対には、次の値が保存されている。[R8]

| 項目 | baseline E2/r8/fraction0.6 | R2/r8/fraction0.4 |
|---|---:|---:|
| 7×総Toffoli | 94,192 | 6,813,632 |
| 総RZ multiplier | 6,816 | 334,432 |
| η_RZ | 2.9342723004694837e−8 | 5.980288967562912e−10 |

以下は**本レビューでの条件付き代数的解釈**であり、新しい行を評価した結果ではない。

同じ非負価格関数τ(η)を使い、精度を厳しくすると価格が下がらないとする。この行対ではRのToffoli成分が大きく、RZ数も多く、要求精度も厳しいので、価格係数κを単に別の共通単調価格へ変更するだけでは、同じ行対のknown費用は逆転しない。

ただし、この議論は当該固定行対についてである。全273行に対する価格非依存な支配性を今回証明したものではない。実合成はangleにも依存し、同じτ(η)だけで記述できるとは限らない。special-angleの低費用を含む別回路、全体合成、phase-gradient等の変更はこの単調価格仮定の外である。

したがって、「κ3/6/12の偶然だけでnegativeになった」とみなす根拠は弱いが、「任意の実T合成でnegative」とも言えない。

### 6.4 DF対照のrank変更だけでは、Rの費用問題を解消しない

保存価格最小baselineはT=1でE2、T=4でE4であり、D2/D4ではない。したがって、「DFを16全保持したため弱い比較だった」という懸念だけで、今回Rのpositive headroomを回復できるわけではない。

DFを改善してbaseline最小値が下がれば、固定R側のremainingはむしろ小さくなる。ただし、これはbaseline集合の最小値についての論理で、rank切捨てを実行した結果ではない。DF近似・model誤差を変える場合は新契約が必要。

## 7. 誤差・精度・公平性に関する評価

### 7.1 理想PFと有限精度回路を分けた設計は妥当

正負時間・adjoint・operator phase・leakageを含むideal matrix比較と、有限精度回路の予算は別に保存されている。現結果を「functionまで実装済みの量子回路の誤差」とは読まない。

binary64のPadé/Taylor照合やnorm witnessの保存監査は、今回の局所的な数値整合性を支える。全roundoffの外向きcertificateではない。margin内の曖昧値が0であることも、全PFの厳密数学保証への昇格ではない。[R1,R10,R11]

### 7.2 精度配分は公平に登録されているが、完全に最適化されていない

全法へ同じPF fractionsを与え、Rには追加function予算を留保する規則は、機能の違いを反映した明示的な設計である。単にRが三分割、Bが二分割だから不公平だと即断しない。

一方、配分は有限な候補で、per-angle・per-group・per-stageの最適化全体を尽くしていない。長時間時の誤差評価、より強いPF、nonuniform精度、境界fusionなど、合理的な別設計もあり得る。ただし、未評価の改善を無料で差し引かない。

### 7.3 baseline追加が必要な場面と不要な場面

新たな優位性や「最適PFより良い」というclaimを目指すなら、より強い公式や適用可能な既存最適化の比較が必要になり得る。

一方、現Rのknown費用が登録E2/E4に対して既にpositive headroomを持たず、現設計のbackendへ進まないという判断をするために、全ての強いbaselineを実装する必要はない。baselineを増やすことで最小baselineが高くなることはない。

## 8. Ground-energyへの接続で、まだ分かっていないこと

今回のmetricは固定ν-sectorの全入力に対する作用誤差であり、基底状態だけに特化したものではない。基底energyのためには保守的な要求になり得るが、その意味で今回の比較が誤りというわけではない。

固定T=1,4の作用誤差0.001を、単純に0.001/T Haへ変換してground-energy性能を認定しない。固有位相branchの対応、alias区間、準備状態との重なり、controlled powers、推定器の時刻列・統計保証を確認していない。任意phaseを除去しない現在の評価は、その接続に必要な一要素を保持しているだけである。

特に、ここでground-state-only指標へ変更すれば有利になるかもしれないという理由だけで、結果後の再評価を次作業にしない。そうした変更は独立した科学的問いであり、baseline・情報取得・truth使用の境界を含む別契約が必要。

本レビューは、ground-energy全体についての不可能性を判断していない。同時に、後段を未評価にしていることを、現templateの実装を続ける自動的な理由にもしない。

## 9. 一次文献との対応と新規性

### 9.1 THRIFT

Bosse et al., Nature Communications 16,2673 (2025) の公開本文を確認した。mixed propagatorの効率的な実装と、その近似誤差がα²の利点を壊さないことが適用条件として記されている。Fermi–Hubbardではmixed項の費用によって強い通常PFが有利な領域も示されている。[L1]

この文献は今回のH4数値を裏付ける実験ではない。参照するのは、**理想誤差の改善とgate costの改善を分けるという既知の構造**である。Hubbardの境界や倍率をH4へ転記しない。

### 9.2 RZの価格

Ross–Selingerの原論文abstractは、典型的な場合に3 log2(1/ε)+O(loglog(1/ε))という近似合成の挙動を述べ、条件付きの最適性・計算量を扱う。[L2]

この事実は、任意の全回転についてceil(3 log2(1/η))が厳密な実現上界・実測値になることを意味しない。angle依存、定数、exact Clifford等も含まれる。今回のκ3/6/12はあくまで登録された感度仮定である。

### 9.3 今回は新規性の全面再監査ではない

現constructorの独立新方式としての非採択は前回判断を保持する。今回のH4実験はその新規性を新たに証明したものでも、全構成の既報性を証明したものでもない。

既知法の化学応用研究にも価値はあり得る。しかし、現在の一分子・有限grid・保守的templateによる否定的feasibilityだけで、独立論文の主要貢献が成立したとは認定しない。理論・構成・入力・比較基盤・今回の限定結果は、研究記録や将来の方法比較の資料として保持する。

本レビューでは最新文献全体・全library実装・全可解構成を網羅的に検索していない。方法・費用の新しい寄与が存在しないと一般断定しない。

## 10. 追加投資と代替方針

| 選択肢 | 判断 | 理由 |
|---|---|---|
| 現bitwise templateのsqrt/atan2等を完成する | 推奨しない | 現会計のknown部分に既にpositive budgetがなく、Fを埋めるだけでは研究判断を改善しない |
| 同じH4で価格・配分・rを追加scan | 標準の次作業にしない | 未取得点があるだけでは継続理由にならず、今回の有限結果を探索で救済し続ける危険 |
| H2/H6・距離・basisを直ちに追加 | 推奨しない | favorableな結果を探す前に、今度は何の構造で費用差が変わるかを説明する必要がある |
| 保存値の費用帰属をさらに細分化 | 一般的な完了条件にしない | sourceと保存symbolic行で投資判断に必要な主要構造は把握できる。新しい数値診断を自動要求しない |
| 現成果を固定して系列を区切る | **推奨** | ideal PFの改善と登録modelの負headroomを明確に残せる |
| 具体的な別アクセス・回転・共有構成を検討 | 条件付きの将来候補 | 操作列、精度、補助資源、既存対照を含めてknown費用を変える根拠が必要 |
| 別のground-energy向け算法へ優先配分 | 研究目的上の選択肢 | ただし第二候補を本レビューで自動採択しない |

### 10.1 再開に必要な具体性

再開には、新しいprimitiveの発明を一律に義務付けない。既知の可逆回転・共有法でも、今回の障害をどのように変えるかが具体的なら検討に値する。

例えば、基底変換やphase-gradient/QVRの再利用、線形Zの直接生成、occupation演算の共有、transition groupの再構成などは、概念としては候補になり得る。ただし、どれも今回の成功案として採択していない。準備state、control、sign、uncompute、precision、他のB_jによるspectator変更まで含む同機能設計が必要。

**既知費用Kを変えずFだけを安くする案と、K自体を変える案を区別する。** 前者は今回のモデルでの負headroomを回復しない。

同じscheduleとbaselineのままknown費用をΔKだけ減らし、追加function費用F_newを持つ構成へ変えるなら、

\[
 K-\Delta K+F_{\rm new}<B
 \quad\Longleftrightarrow\quad
 \Delta K>K-B+F_{\rm new}
\]

が必要である。これは既存式からの算術的条件で、新しい回路の実現性を証明しない。step数・精度・推定器も変えるなら、その変更を含む全taskを比較する。

入力側を変える案なら、単なる「別分子なら勝つかも」ではなく、少ないgroup数、少ないoccupation依存係数、既知に安いmixed access、より大きいstep削減など、根拠を持つ機構を先に示す。未来の成功例の存在を否定はしないが、根拠なしの系列延長は勧めない。

## 11. 次のCodex作業

次は**既存の結果・引継ぎ・研究概要への結論追補と成果固定**に限ることを勧める。

反映する内容は次のとおり。

1. 理想THRIFT型PFのstep削減を限定成果として保持する。
2. `NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL` を原statusのまま保持する。
3. 現templateでのbackend投資を優先しない、現feasibility系列を一区切りにする、という本レビューの研究判断を追補する。
4. 価格仮説・envelope・有限grid・一分子・作用誤差taskの限定を明記し、全THRIFT/最適回路/ground-energyの不能にしない。
5. 再開時に必要な具体的構造差・同機能費用・科学的問いを記録する。
6. source/result/auth/raw/入力/旧STOPを変更せず、既存の証拠リンクを保持する。

新しいparser・全proof replay・数値再集計を文書反映の完了条件にしない。既に存在する監査・receiptを参照し、Codexの通常の文書・metadata・remote確認を一つの作業にまとめる。

GitHubへ反映する実作業はCodexで行う。本レビューはリポジトリのstatusを直接書き換えていない。利用者が本判断を採用して反映を指示した範囲で進める。

文書反映の完了だけで、同じ科学レビューを要求する必要はない。将来、現結論を変える具体的な矛盾・新証拠・別構成案があるときに、明示開始承認後の重要レビューへ戻す。

## 12. Claimの最終分類

| Claim | 状態 |
|---|---|
| 固定H4の指定gridでR4がE4/D4より少ないeligible rを得た | 支持（ideal PF、指定作用誤差） |
| 登録価格κ3/6/12の全eligible frontierでremaining budgetが負 | 支持（保存算術監査の範囲） |
| 現会計を保ったまま非負function費用を追加しても有利にならない | 条件付き代数的帰結として支持 |
| unknown functionが高かったことが敗因 | 不支持。function実費用は未取得 |
| 任意価格・任意回路でも全候補が不利 | 未証明 |
| THRIFTを量子化学へ使うことは不可能・無価値 | 不支持 |
| 原目標ground-energy 1mHaを達成した／失敗した | 未評価 |
| 現constructorは独立新算法である | 前回の非採択を維持 |
| 本一分子結果だけで独立論文の主要claimが完成した | 未認定 |
| 現templateの詳細backendへ直ちに投資すべき | 本レビューでは推奨しない |

## 13. 参考資料・追跡可能な正本

### 固定GitHub資料

共通prefix R1–R5,R8–R11,R13：
`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac930c0e0d255a187dc21917104e5cb5b4d09bcb/`

- **R1** `docs/research/qc_mixed_access_feasibility_v1_results_20261013.md`
- **R2** `docs/research/qc_mixed_access_feasibility_v1_review_handoff_20261013.md`
- **R3** `artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json`
- **R4** `docs/research/qc_mixed_access_feasibility_plan_v1_20261013.md`
- **R5** `artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json`
- **R8** `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json`
- **R9** `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/result.json`
- **R10** `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/independent_audit.json`
- **R11** `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/saved_execution_integrity.json`
- **R13** `artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/resource_evaluation.json`（今回全payloadを取得・再評価せず、R1/R8/R11の保存集計・監査とR6を参照）

科学sourceの共通prefix：
`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/102e6b31401356d6eea775169041ebdf2cfa6ba0/`

- **R6** `src/trottertracks/representation_exploration/qc_mixed_access_resource_v1.py`
- **R7** `src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py`

**R12：前回科学レビュー**

会話に添付されたUTF-8原本 `qc_relative_block_method_cost_scientific_review_2026-10-13.md` を読んだ。前回判断は同レビューの結論、費用式、限定feasibilityの目的、再開境界を参照。今回GitHub原本の全文を新たに再取得したとは扱わない。

過去系列は、会話に提供された固定結果・closureとR12の振り返りを継承した。別Trackの最新状態を新たに監査していない。

### 外部一次資料

- **L1** Bosse et al., *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nature Communications **16**,2673 (2025), published 26 March 2025.  
  https://www.nature.com/articles/s41467-025-57580-5  
  今回は公開HTMLのmixed-access費用、Theorem 1、近似mixed精度条件、高次composition、Hubbardへの限定を確認。全supplementの再監査・分子への文献結果転用は行っていない。
- **L2** Ross and Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*, arXiv:1403.2975.  
  https://arxiv.org/abs/1403.2975  
  今回は公開abstractの典型的T-count・条件・最適性の範囲を参照。全角度でのcost表や本実装の達成価格を確認したものではない。

## 14. 最終的な実行境界

本レビューの承認は新しい科学実行の承認ではない。原one-shotは消費済み、retry0を保持する。

N1 STOP、N2 HOLD、N3 CLOSED、各別Trackのsource/result/authorization/STOPは維持する。global manifestの既存description欠落FAILは今回の結果の科学的意味や個別保存監査とは別に保持し、無断修正を指示しない。

**最終判断：理想PFのstep改善は成果として残す。しかし、現登録templateでの正のfunction予算が得られなかった以上、そのtemplateを完成するbackend投資は進めず、現系列を一区切りにする。全THRIFT・量子化学mixed-accessの一般的不可能性は主張せず、再開は具体的にknown費用または必要stepを変える構成・適用条件の根拠が得られた場合に検討する。**
