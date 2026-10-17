# 先行研究・実用条件の対応表

2026-10-17 JST。[数学的技術監査](mathematical_audit.md)と[判断用ハンドオフ](decision_handoff.md)の補足。限定的な一次文献照合であり、網羅的priority調査でも新しいGPT科学レビューでもない。

## 1. 原文の取得範囲

指定のarXiv **v1 PDF**を取得し、本文・補遺の下記範囲を確認した。PDFのbytes/SHA256は[証拠索引](evidence_index.json)に固定する。原レビューが読んだHTMLと、本監査で読んだPDFの全式が同一であるとは認証しない。出版版・他版・関連論文全体は今回の確認範囲外である。PDFを科学計算の入力には使っていない。

| 原文 | 確認した定理・証明箇所 | 適用条件と既知の対象 |
|---|---|---|
| [Mizuta–Kuwahara, *Trotterization is substantially efficient for low-energy states*, 2504.20746v1](https://arxiv.org/pdf/2504.20746v1) | main setup、Theorem 1 Eqs.(13)–(15)、step/cost Eqs.(19)–(20)、Supplement S2 Theorem S1 Eqs.(S5)–(S24)の証明、S3 Theorem S3・Corollary S4と関連評価 | PSDのk-local項、g-extensiveな相互作用、群内可換な分割と低エネルギー支持。射影交換子のenergy一次依存、漏洩を含む有限時間PF保証と費用が既知。励起gapの入力を要求しない。 |
| [Hejazi–Zini–Arrazola, *Better bounds for low-energy product formulas*, 2402.10362v1](https://arxiv.org/pdf/2402.10362v1) | main Eqs.(2)–(13)、補足の漏洩 Eqs.(23)–(32)、retained component Eqs.(33)–(51)、fermion拡張の条件 | commuting local termsの群、項norm・support・有界degree。低エネルギー射影下の誤差と漏洩を分ける。fermion拡張にもeven項・mode degree条件があり、PSD shiftによるenergy cutoff悪化を無視できない。 |
| [Maxwellほか, *Practical Estimation of Trotter Error for Hamiltonian Simulation*, 2606.30738v1](https://arxiv.org/pdf/2606.30738v1) | III.Aの漸近定理、III.B.1–3の対称BCH・再帰・compact係数、Appendix A Theorem 1証明 Eqs.(A1)–(A27) | fixed leading-errorの漸近的なdephasing/摂動と、二次PFの先導交換子の簡潔な計算法。実際のPDFの節番号はIII.A/Bである。有限hの相対log剰余の定理と同一視しない。 |

P3のAppendix A Eqs.(A19)–(A26)は `γ=min_(m≠n)|λ_n−λ_m|` を分母に使う。rank-one dephasing表示を縮退固有空間へ無条件に適用せず、非縮退または縮退blockの扱いを確認する必要がある。本監査ではこの証明を、uniformなgap-free有限h certificateとして輸入しない。これは当該論文全体の誤り認定ではない。[P3 Appendix A](https://arxiv.org/pdf/2606.30738v1)

## 2. 何が直接既知で、どこが未確定か

以下の「未確定」は未発見からの新規性認定ではなく、今回示せた論理対応の限界である。節1の文献の主張と、以下の独立な比較・推論を区別する。

| 対象statement | 本監査で確認できた対応 | 直接還元と呼ぶために不足するもの |
|---|---|---|
| PSD分散、二端のenergy依存、log/Cauchy/Markov | 既知の数学的道具。正値性によるenergy一次の交換子評価も先行研究にある | これらの道具の利用自体を新規性にしない |
| RF0：全vectorの `−η0 G≼K−G≼η0 G` | 本監査§2で有限任意PSD fragmentについて直接証明した | 指定3論文の射影時間発展誤差等から、この同じ全vector・log・明示領域・定数のstatementがそのまま出る完全な定理対応は未提示 |
| 低energyの有限時間PF誤差・gap不要の十分保証 | P1/P2で既知。したがって「gap-free有限step保証が存在するか」一般を未解決課題として再提出しない | 低期待値 `〈G〉` を持つ任意vectorと、固定cutoff以下に支持されたvectorは同じでない。cutoffを期待値で置換しただけでは全vectorのLoewner評価にはならない |
| RF1のD・辞書 | K0に既存。対称BCHと先導係数のcompact再帰もP3に既存 | 三次係数や計算法に新規性を置けない。今回検算した差は全vectorへの読み替えと `ρ4 G` の有限剰余である |
| EW：ground上のK測定が出力窓へ入る確率 | 相対形式からのMarkovの初等的帰結。単一固有枝・励起gapは不要 | EW単独の独立新規性を主張しない。既知のPF定理がEWへ必要な相対形式をどの情報・定数で供給するかが比較点 |
| fixed-powers QPEとstep/query | 通常QPE・union bound・scalar復元。旧K0.5もgap-free QPE安定性を区別済み | output task、state、branch、合成・読出し誤差まで揃えた最良既知保証との包含・総費用比較は未完了 |

P1のlocalityを「DFは非局所だから無関係」と除外してはいけない。一般PSDを全supportの項とする粗い特殊化は可能である。一方、その置換ではk・g・群数等が悪化し得る。各DF平方が別の軌道basisで対角になるだけで、共通の小support・群内可換性・安価な局所分解が与えられたことにはならない。

P2の射影誤差ではretained成分と漏洩を一緒に会計する必要がある。射影normだけを全空間のlog誤差へ無条件に置換することはできない。本監査は「P1/P2からRF0を導くことが不可能」とも主張しない。指定された照合で、同一statementへの完全な還元証明は確定しなかった、という結論である。

## 3. K0・K0.5・過去レビューとの来歴

| 固定資料 | 再利用する確立済みの記録 | 今回との差・保持する限界 |
|---|---|---|
| [K0 `58d86727…` §3–5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md) | raw κ、非負B、行和K_SOS、生成計算量、ground上の先導係数評価 | 全vectorの検算を数学的監査§3に示した。旧K0のscalar剰余・gap付き固有枝保証を新しい相対剰余と混同しない |
| [K0.5 `01bdc56a…` §3–5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md) | gap-freeなoperator/QPE保証と、明示的な強対照Childs S2 boundが既知であること | RF0と旧粗い残差だけの比較で優位を決めない。既存の実用的SOS分割生成則は未取得 |
| [過去GPT `dbe51f18…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md) | gap不要vector残差、branch/overlapとの区別、未採択A2という判断の文脈 | 過去の推奨や粗い比較点は最新保証の有用性を認定しない。review原文の取得と、その全引用文献の再監査は別 |

K0.5 §3.4に保存された比較点は、同じ順序で

\[
C_2=\sum_{j<M}\{\|[F_j,[F_j,R_j]]\|/24+
                     \|[R_j,[R_j,F_j]]\|/12\},\qquad
\|S_2(h)-e^{-ihG}\|\le |h|^3C_2.
\]

これは実時間のoperator boundであり、複素logの1/4半径を条件にしない。`t/h`反復では `|t|h²C2` のtelescoping上界を使える。QPE全体のcontrolled powersと成功確率に戻すか、branch条件付きの逆log評価に戻す必要がある。RF0のenergy-windowと、一stepのoperator normだけを直接比べて勝敗を付けない。ここで引用した対照は固定K0.5の記録であり、Childs論文全体を今回新しく調査したとは主張しない。

## 4. 実用性の静的会計：未取得を0にしない

| 情報・処理 | 必要な証拠 | 成立しても残る費用・障害 |
|---|---|---|
| `H=LI+ΣFj`、sector、PSD | 元のHamiltonianとの係数恒等式、logical不変sector、certified PSD。近似DFなら元Hとのモデル差を別に台帳化 | 一般SOSのLを、異なる実装可能分割へ転用できない。負符号DF項をshiftするとL・W・Λが変わる |
| LとU | 同じPSD分割の下界、検証可能な変分上界等。GramのPSD・normal ordering・sector ideal・丸め残差の証拠 | SDP出力や保存binary64だけは厳密証拠ではない。生成・証拠検証・bit長の費用は未測定。古典ground解を無料入力にしない |
| fj、Λ | 各Fjの対象sectorでの認証norm上界、scalarと単位の一貫性 | 強くshiftされたdense平方ではΛが膨らみ得る。Λ既知と安いtight Λは別 |
| K_SOS | 固定のκ・順序・同じfjからの上向き評価。sqrt・加算のroundoffも認証対象 | K0のraw経路はO(M³)演算、行和だけならO(M)保存量。固有値やground φは不要だが、正しい有限精度計算と証拠取得費用は必要。今回計算していない |
| hと比較保証 | RF0/RF1の共通領域、η<1、EWの予算。強い交換子・低energy対照と同じtaskで比較 | 定数32・512とhΛ<1/4が支配し得る。WがΛ程度以上なら低energyの利点は自明でない。有用なhが得られる実証なし |
| exact ground copies | 対象sector・元Hについてのstate品質。近似なら別のtrace-distance等の証拠 | 準備費用を含める。低いvariational energyだけでgapなしのoverlap保証を作らない |
| controlled S2とpowers | orbital/Gaussian変換、対角平方、logical encoding、scalar phases、powerごとの合成誤差 | direct DF実装が利用可能でもtotal cost未評価。query数はnative gate数・depthではない。既知scalarを消したuncontrolled等価性だけでは不十分 |
| readoutと成功確率 | 既知energy support、log/alias、各固有値への一様な読出し失敗保証、state/モデル/合成の別予算 | branch上の成功保証を平均複素信号やRPEの保証へ転用しない。精度・失敗率の数値配分は未採択 |

`W=0` または `Λ=0` は自明caseとして分離する。目標絶対誤差εに対し `W≤2ε` なら既知区間の中点で足りるため、量子計算を必要とする理由は別に必要である。その区間を認証する古典費用は消えない。

ある資源座標gでの記号的比較は

\[
C_{\rm total}^{(g)}=C_{\rm cert}^{(g)}+C_{\rm setup}^{(g)}
 +\sum_aN_a[Q_a C_{S_2,a}^{(g)}+C_{{\rm prep},a}^{(g)}+C_{{\rm read},a}^{(g)}].
\]

古典時間・T・CX・depth・memoryは別座標。換算規則なしに加算しない。提案側の追加cert/setup費用が量子側の節約を上回れば総費用の利点はない。state、原点、sector、精度・成功率、入力access、preprocessingの償却条件、controlled/synthesis条件を両対照で揃える。数値パラメータ・改善率・分子入力・benchmark閾値は今回設定していない。

## 5. 判定に残ること

中心証明の技術的成立は[数学的監査](mathematical_audit.md)で閉じた。**新規性の完全な直接還元／非還元、非局所DFでの有用な定数、証拠取得費込みの強対照への優位は未確定**である。このため分類は `VALID_WITH_UNRESOLVED_NOVELTY` とし、`VALID_BUT_KNOWN` とも `VALID_WITH_POTENTIAL_METHOD_DELTA` とも確定しない。

未知であること自体を新アルゴリズムの根拠にしない。少なくとも本監査に、独立した有利な分割生成規則や実証済みの費用削減手順はない。これ以上の文献探索・保証改良・数値確認は本作業の完了後に自動実施しない。**Mandatory STOP。**
