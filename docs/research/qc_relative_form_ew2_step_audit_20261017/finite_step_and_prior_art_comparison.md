# 有限stepの公平な比較・先行研究・実用条件

2026-10-17 JST。[数学的監査](mathematical_audit.md)で成立を確認したEW2と交換子対照の比較。科学的な数値代入・forecast・新しい実験契約の作成は行っていない。[判断用ハンドオフ](decision_handoff.md)と[証拠索引](evidence_index.json)を参照。

## 1. 比較するtaskと情報を揃える

両側とも、同じ元H・対象sector・実装可能なPSD F_j・固定順序S2・絶対原点L、同じground copies access、同じdとα_spec、同じQPE読出し精度ε_readとα_read、同じモデル/合成/state/certification failureの会計を用いる。powerは固定S2の整数powers。native precision、状態準備、前処理の償却回数を片側だけ有利にしない。

両方式でL,U,f_j、任意に追加取得する構造情報を同じ価格で利用できるものとする。相対側はK_SOS、対照側は取得可能なCbar2≥C2を使う。多体normの真値やground vectorは無料入力ではない。よりtightな認証情報・可換性・既知scalar除去は両側に許す。将来の比較に用いる具体的な証拠取得予算・精度配分は今回未採択である。

## 2. 有限stepの条件と厳密な記号的比較

Λ,W,d>0、0<α=α_spec<1とし、数学的監査のη*、h0、h1、`h_R=max(h0,h1)` を使う。同監査式(7a)は `a=min(32Λ²,K_SOS)` を使ったh_Rの閉式で、有限剰余も保持する。`h_R Λ<1/4`、`η(h_R)=η*<1`。これはRF0/RF1/EW2の**十分証拠が認めるstep**であり、真のPF誤差に対する最大stepではない。

| 証拠 | 有限stepの十分条件 | 領域・意味 |
|---|---|---|
| 相対形式＋EW2 | `η(h)²/(1−η(h))≤αd²/[W(W+d)]` | `h>0, hΛ<1/4, η(h)<1`。η=min(η0,η1)を保つ |
| 原レビューのlog対照 | `h² Cbar2/(2−exp(hΛ))≤√α d` | 同じ証明は `hΛ<log 2`。`3h²Cbar2/2≤√αd` は1/4領域での追加の粗い十分条件 |
| 実時間・既知centeringを使った対照 | `b_C(h)=h² Cbar2/(1−βh)≤√α d`、`β=Λ/2` | `βh<1`。相対側の比較点ではK_c=K、同じenergy-window。RF領域外は既知centerでbranchを処理 |

最後の行は、同じF_j上のmidpoint scalarを証明で差し引くだけで得られる。[数学的監査 §5.3](mathematical_audit.md)に積順序・branchを示した。元のH、物理的な絶対位相、元のcontrolled S2を黙って変えるものではない。

Cbar2>0、`s=√α d` とおく。b_Cは0<h<1/βで単調増加し、0から無限大へ動く。この対照のenergy-window証拠の最大十分stepは

\[
h_C=\frac{\sqrt{s^2\beta^2+4\overline C_2s}-s\beta}{2\overline C_2}
=\frac{2s}{\sqrt{s^2\beta^2+4\overline C_2s}+s\beta},
\quad0<\beta h_C<1.
\tag{C1}
\]

従って、**この強化した交換子証拠に対して**相対形式の十分stepが大きい条件は、厳密に

\[
\boxed{h_R>h_C
\quad\Longleftrightarrow\quad
\overline C_2>\frac{\sqrt\alpha\,d\,(1-\beta h_R)}{h_R^2}.}
\tag{C2}
\]

等号は同じstep、逆向きはこの相対証拠のstepが大きくないことを意味する。分母は `βh_R<1/8` から正である。h_Rは認証値W,Λ,K_SOSとd,αだけで決まり、真のeを必要としない。RF0だけを使うならh_Rをh0に置いた(C2)でも、その候補stepがh_Cを上回る十分条件となる。数値例・分子での実現性は確認していない。

(C2)は `3/2` に粗く置いた対照に勝つ条件ではない。実時間unitarity、同じ情報でのcentering、対照の広い領域を許した比較である。ただし最もtightなinverse-log評価や他の有効な証拠をすべて凌ぐ条件ではない。わざと粗いCbar2を選び、(C2)を満たすようにしてはいけない。

ここでのh_R/h_Cは**energy-window証拠上のstep**。実際に採用可能なstepには、共通のreadout/unwrap・実装・入力精度条件も必要である。それらが両stepで成立し、別の支配的制約がないときにのみ、(C2)を実行stepの差へ接続できる。条件未取得を「実装可能確認済み」と扱わない。

## 3. 一つの対照に勝つことと最良の利用可能な保証に勝つこと

比較に利用できる証拠kごとに、その正当な全領域でのadmissible setをA_kとする。使用できる既知保証の全体は `A_known=union_k A_k`。相対側の条件が(C2)を満たしても、別のA_kが同じまたは大きいhを許せば、既知保証全体へのstep優位にはならない。

| 追加で残す経路 | 正当な利用条件 | 今回閉じていない点 |
|---|---|---|
| Childs operator誤差→QPE回路安定性 | 同じpowersでの全base適用数Q、`TV≤Qh³ Cbar2`、理想G-QPEのalias/readout保証 | Q・成功率・合成精度を共通仕様に戻す必要がある。RFの1/4半径を機械的に課さない |
| unitary state residual→phase集合 | `||(S2−exp(−ihe))g||≤h³ Cbar2` と円周距離・branch条件。energy窓ならsine閾値・支持を保持 | scalar近似や未知の真eを証拠の無料入力にしない。適切な支持情報が必要 |
| Mizuta–Kuwahara / Hejazi低energy保証 | PSD局所分解、locality/degree/group条件、低energy支持、有限時間の漏洩会計 | 原groundは低energy stateだが、有用なcutoff・係数・情報取得費を今回インスタンス化していない |
| より鋭いlog/構造付き対照 | branch、既知bandwidth、可換性、同じ取得費を満たすもの | b_Cを最良定数と主張しない。確認したK0.5にも別のinverse-log保証が記録されている |

したがって **(C2)は具体的な条件付き比較成果であり、最良既知法への無条件優位ではない**。各A_kの形・取得費・共通readoutまで閉じる作業を行っていない。対照の上界があるhで不合格だったことだけで、別の大きなhも不可能と結論しない。

## 4. Xiの検算：小stepの診断に限定

`η(h)=a h²+O(h⁴)`、a>0、Cbar2>0、W>0、hΛ≪1とする。RF0はa=32Λ²、RF1はa=K_SOSで四次剰余が従属する場合。min保証で両leadingが正ならa=min(32Λ²,K_SOS)。

両確率上界のleadingは

\[
B_R=\frac{a^2h^4W(W+d)}{d^2}\,[1+O(h^2)],\qquad
B_C=\frac{\overline C_2^2h^4}{d^2}\,[1+O(h\beta)].
\]

従って同じαとdで、飽和点でもη*≪1かつ高次項が従属する領域（例えば後述のd/W→0）を比較すると、leadingのstep比は

\[
\frac{h_R}{h_C}\sim
\sqrt{\frac{\overline C_2}{a\sqrt{W(W+d)}}}.
\tag{C3}
\]

d/W→0、係数・W・αを固定して両stepが小さくなる極限では

\[
\Xi=\frac{aW}{\overline C_2},\qquad
\frac{h_R}{h_C}\longrightarrow\Xi^{-1/2}.
\tag{C4}
\]

aはenergy²、Cbar2はenergy³なのでXiは無次元である。Wや係数が精度・系サイズに依存する場合、それを固定定数に隠さない。同じbase-query設計ならqueryのleading比は逆向きだが、native費の比とは限らない。

αの配分が異なれば(C3)に `(α_R/α_C)^(1/4)` が掛かり、窓配分が異なればそのdの比も変わる。比較では同一に揃える。単純化した3/2をleading定数として残し、人為的に有利なXi閾値を作らない。b_Cのh→0の係数は1である。

| 条件 | Xiをそのまま使えない理由 |
|---|---|
| Cbar2=0 | 対照は全実hでS2が厳密であることを証明する。Xiは定義不能 |
| a=0 | 二次leading仮定が消える。RF1の四次剰余を保持して有限条件へ戻す |
| W=0 / Λ=0 | E0が既知の自明case。量子資源改善の例にしない |
| dがWと同程度以上 | √[W(W+d)]をWへ置き換えられない。EW2の正確なη*を使う |
| ηが1に近い / hΛが1/4に近い | `(1−η)^{-1}` と有限log剰余が支配し得る |
| 強い低energy対照や構造capが使える | 比較分母Cbar2だけでbest-known stepは記述できない |

元レビューのε^{-3/2}依存は、d,ε_readをεに比例配分、Wと正の係数・confidenceを固定した小ε/W域のS2十分設計として整合する。両側同じ指数であり、それ自体は改善ではない。中央値等のconfidence増幅を使うなら両方式へ共通に許し、独立state copies・反復・readout費用を含める。共通certificateの誤りは反復で消えない。

**Xi<1は有限stepの必要十分条件でも研究GOでもない。** 今回Xi、h_R、h_C、shot数、native費を実データで計算していない。

## 5. 先行研究との限定対応

| 固定原文 | 実際の確認範囲 | 技術的な対応・限界 |
|---|---|---|
| [Childsら 1912.08854v3](https://arxiv.org/pdf/1912.08854v3) | §5.1 Eqs.(146)–(152)、Proposition 16、印刷p.39画像 | 元レビューのC2・係数・順序と一致。実時間保証。今回のlog/energy-window接続は標準操作からの直接的な帰結で、同論文にEW2と同じ表記があるという引用ではない |
| [Mizuta–Kuwahara 2504.20746v1](https://arxiv.org/pdf/2504.20746v1) | Supplement S2 Theorem S1、Eqs.(S5)–(S21)の証明、S2末とS3の区別 | 任意の射影Πを許す。以下のrank-one特殊化は有効。locality・extensiveness・PSD局所項の条件を保持する |
| [Hejazi–Zini–Arrazola 2402.10362v1](https://arxiv.org/pdf/2402.10362v1) | main設定・低energy誤差、補足retained/leakage Eqs.(33)–(51)、fermion条件 | 群内可換local項、項norm、degree、群数・低energy cutoffと漏洩が必要。even fermionic項にもmode degreeを保持。任意のDF平方へ有利な定数を自動転用しない |
| [K0 `58d86727…` §4–5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md) | 係数辞書、B/行和、scalar剰余・証拠条件 | K_SOSと生成計算量は既存。RF1の相対有限剰余、EW2、今回の比較とは証拠層が異なる |
| [K0.5 `01bdc56a…` §3–4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md) | inverse-log/QPE/residual/C2/低energy条件 | gap-freeな対照は既存。既知比較を意図的に弱くせず、同じ情報で使う |
| [過去GPT `dbe51f18…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md) | gap-free残差、state/branch、A2未採択という来歴 | 以前の粗い比較点を、今回の強い対照すべてに対する優位へ格上げしない |
| [高次PF evidence map `b4a7a820…`](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md) | U1/U3/U7と直接の重複記載のみ | 有限剰余の適用、情報取得費、truth-freeな幅を資源判断へ戻す問いは既存。RF0/RF1/EW2と同一の証明が当該mapにあるとは確認しなかった。全repositoryの未検討証明ではない |

### Theorem S1のrank-one特殊化

原文の条件を満たすH_0,…,H_qと単位vについて、`Π=|v><v|` とすると `ΠH_lΠ=〈H_l〉_v Π`、`||ΠCΠ||=|〈C〉_v|`。各H_lが選択したF_jならF_j≤Gなので `Δ_v=〈G〉_v` を用いられる。従って既知定理の直接的な特殊化として

\[
|\langle v|[H_q,[\ldots,[H_1,H_0]]]|v\rangle|
\le q!(2k g_{\rm ext})^q\langle v|G|v\rangle.
\tag{C5}
\]

g_extはextensivenessであり励起gapではない。qが偶数なら対象交換子はHermitianなので、全vectorの形式不等式にも移れる。例えば同じlocality条件と一様k,g_extを持つ三項について、Dの辞書へ(C5)を足した行列形式の上界も得られる。したがって先導交換子の**全vector性だけは新規性を支えない**。

この適用は、各fragmentのPSDな局所分解を必要とする。sector内でだけPSDの場合、元のfull-space local項がそのまま仮定を満たすとは限らない。sectorへ射影・zero extensionして全supportのPSD項として扱う粗い特殊化は可能だが、localityの良い定数を失う。局所fermion monomialへ展開してPSD性を失ったものを同じ入力にしない。

(C5)だけではlogの全高次係数・剰余の可算和や収束領域は得られない。BCH係数と全次数の評価、解析的branchの管理が別途必要である。前回RF0のCauchy証明は一つの既存project証拠だが、P1から同じ定数・領域・証拠取得価格で完全に還元できたとは本監査で結論しない。低energy**時間発展**定理の固定spectral cutoffを、任意vectorの期待値で置換してよい、という意味でもない。

EW2はcongruence・spectral calculus・非負momentの短い帰結。指定文献に同一式を見つけなかったことからpriorityを認定しない。相対log全statementの既知性は未確定、交換子対照の接続は既知operator保証からの直接系、という別々の結論を維持する。

## 6. 証拠生成・実装・総費用

| 必要量・費用 | 認証可能性と不足情報 |
|---|---|
| L,U,W | 実際のPSD分割に対応する下界、同じ元Hの上界。Gram PSD、係数恒等式、normal ordering、sector ideal・丸め/モデル残差を検証する。一般SOSのLを別DF分割へ流用しない |
| f_j,Λ,K_SOS | sector normの上界と固定辞書。K0 raw経路のO(M³)演算・O(M)行和保存量は既存。係数だけで定まることと安いtight証拠は別。今回は評価しない |
| Cbar2 | 係数・可換性・fragment構造等から取得したnorm上界。多体norm真値は無料にしない。双方同じ取得情報・計算予算に揃える |
| shift・可換limit | PSD shiftはW,ΛとRF半径を悪化させ得る。C2はscalar shift不変。対照のknown centeringを許す。全fragment可換ならC2=0、一方RF0/粗いRF1は正の剰余を返し得るため一様支配はない |
| 状態・QPE | ground copiesの単価・品質、powers中の全base適用、controlled scalar/Gaussian/対角平方、合成誤差、readout、反復を含める。stateをqueryごとに再準備しない設計ならQ倍で課金しない |
| 証拠の価格・償却 | certificate生成・検証、古典回路生成、bit長・memory・再利用回数は未取得。強いstep保証でも追加古典費用が節約を上回り得る |
| 既知区間で終了 | W≤2εなら区間中点で目標精度を満たす。量子優位の例にしない。認証の古典費用は残る |

同じ資源座標gについて量子側は、固定費用に `Σ_a n_a[Q_a C_S2,a^(g)+P_a^(g)+R_a^(g)]` を加える形で整理できる。T/CX/RZ/depth、古典時間、memoryは別座標。換算規則なしに合算しない。hが大きくなってqueryが減っても、synthesis単価・状態準備・confidence・前処理込みの総資源改善が自動的に出るわけではない。

残る具体的な技術課題を一つ挙げるなら、**可換性・fragment構造を保持した有限相対剰余を、取得費を説明できる認証情報で作り、(11)および適用可能な低energy対照のenvelopeへ比較すること**である。これは未着手・未採択の候補であり、今回新しい保証の探索・実装・実験契約準備へ進まない。

**Mandatory STOP。** 数式の条件と、実例・費用の未取得を区別して保存する。
