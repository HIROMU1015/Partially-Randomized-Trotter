# 一方向判定の3候補と、成立する範囲

参照IDは[source_catalog.json](source_catalog.json)。既存の式、SOURCEからの直接的な関係、本調査の条件整理を分ける。以下は新定理の採択でも、既存の科学契約の変更でもない。`G`は指定した一つの資源座標の費用であり、CPU秒・T・CX・RZ・bytesを無換算に足さない。

## P-A：適格な対照上界と候補下界による安全除外

**判定と対象。** 同じtarget、state/readout、誤差、confidence、native scope、費用policyの候補dと対照cについて、cの適格性が認証済みで、`L_d ≤ G_d`、`G_c ≤ H_c`が認証され、`L_d > H_c`なら、dはその費用目的の最小候補から除外できる。複数候補の同時保証を要する場合は、全certificateのfailure配分も共通familyへ含める。重複区間はtieではなくABSTAIN。これはW2C_CONDITIONSのENCと一般的な上下界比較そのものである。

**入力と取得費用。** 対照の適格性と上界は、既存の同条件certificate、または独立の誤差・provider費用保証が必要。保存された点推定G・empirical allowanceを認証上界へ読み替えない。候補側は以下の既存cost–moment下界を使い得るが、全係数・全価格の取得は局所query一回とは異なる。Tierは情報表のD/L/E/N/Sに跨り、Mだけでは通常閉じない。

### A1：固定辞書・固定十分予算policyの下界

正のevent係数a_e、実proposal q_e>0（Σq_e≤1）、非負の受理event費用c_eを固定する。zero trialを全trial分母へ含める。理想化した共通表記で

\[
M_2(q)=\sum_e a_e^2/q_e,\quad w(q)=\sum_e q_ec_e,\quad
M_2(q)w(q)\ge (\sum_ea_e\sqrt{c_e})^2.
\]

二axesで共通のNとevent価格を使い、十分予算policy `N=ceil(ell[2M2_plus/s²+4W_plus/(3s)])`、`M2_plus≥M2`、`s=epsilon-b`、`0≤b<epsilon`、非負のrange上界を固定し、二axesの期待費用を`G=2Nw`と定めた場合、

\[
G\ge {4\ell\over\epsilon^2}(\sum_ea_e\sqrt{c_e})^2.
\]

これはG6_NOTE §8の既存下界でsをepsilonへ弱めた直接的な条件整理。**物理的最低shot数・全推定法の下界ではない。** G7/G10の固定ell・bias/finite-bit規約を変更せず、具体予算値は今回取得しない。有限bitには実際のtarget係数とproposalを入れる。理想係数・理想Zをdigital法へ無断代入しない。

全eventを避ける弱い版もある。全受理eventに同じ座標の認証下限c_e≥h≥0があり、係数質量B=Σa_eの認証下限B_minusがあるなら、`G≥4 ell h B_minus²/epsilon²`。G6ではempty parentのd_emptyがB_fullの下限になり、その局所root区間から得られる。しかし、**Uの下端はB_fullの下限ではない**。Uは別量の上界式である。digital係数には別途係数近似の下側保証を要する。h=0しか分からなければこの版は0となり、正の対照上界を用いた除外に役立たない。

hがstate準備費なら、zero後に準備を遅延できる実装と全受理eventへの課金を確認する。一般RZ/Tに正の最低価格があるとは仮定しない。古典setup/query費用は別座標であり、取得済みの選択費用と実行費用の二重計上も避ける。

### A2：全辞書を避けられる既存の限定的な非劣条件

W2Bは同じ固定SOURCE/root/precisionでFullとOrdinaryの十分全trial数`N_F≤N_O`を示す。W2Cはさらに共通digital order/label proposal、raw reject/coin thinning、child非依存の非負構造価格を前提に、期待accepted数・固定helper/skeleton等の論理座標の非劣を示している。これは全signal・全native価格を取らずに使える**既存の条件付き一方向関係**である。

ただし同じfinite operator targetのensemble比較であり、異なるPF次数/q/partitionの全PRを除外する命題ではない。child/angle依存native価格、古典追加処理、hard cap、比較不能なP5 budgetまで結論を移さない。既存G7の条件付きstrict結果も同じscopeの既存成果として保持する。

**根拠・差分・実用性。** G6_NOTE §§8/10、W2B、W2C_CONDITIONS、既知cost-aware IS。法別simulationを減らせる可能性はあるが、必要な対照certificate・h・B_minus・全費用区間が既に得られる場合に限る。下界取得の全event列挙/compileが直接評価を上回らないという証拠は今回ない。既存IS/ENC/論理非劣を超えるR2-CSの新帰結は特定できない。

**反例・失敗条件・残責務。** 上界対上界の大小、未適格の対照、違うtarget/policy、境界fusionによる価格変化、単位差、取得漏れでは除外不可。P_U反例から真のB_fullやGの順序は出ない。未解決は低費用の有用なcertificate取得であり、具体的な新取得機構はない。新しい数値検証計画は作らない。

## P-B：全signal前の適格性の十分条件とABSTAIN

**判定と対象。** 正規化した状態rhoに対し、同じaxis targetの期待値との差をbとする。全pathについて独立に正当化した上界`b≤b_plus`と、全trial estimatorのmoment/range上界、正当なconfidence条件があり、`b_plus<epsilon`なら、`s=epsilon-b_plus>0`として既存の十分予算policyによる**当該誤差契約の認証**が可能。費用budget上限がある場合はその認証も別に必要。この一般条件は、DIAGNOSTICSのempirical allowanceや現在のCOMPARE受入条件を変更するものではない。

**finite targetで既に可能なこと。** G6_NOTE Eq.19は、同じfinite P_mへの係数近似と各eventのstrict controlled実装誤差を扱う。全event共通の誤差上限が認証できれば、その質量上界で和を抑え、全event列挙を避けられる。W2Bの共通bias・positive-margin guardも既存。native errorの宣言だけでは実装達成の証拠にならない。

**exponential/PR targetへ戻す場合。** `m=K+1`（Kは偶数paired cutoff）であり、G6のmとPF次数、P5のdegree、PRのKを同じ記号に潰さない。||R||≤1なら、既知のTaylor tail

\[
e_m(x)=\sum_{n=m+1}^{\infty}{|x|^n\over n!}
\]

は一stepのoperator誤差上界になる。RTE_CODE.step_taylor_truncation_residual_boundもK+2からのtailを使う。複数stepでは、exact unitary U_jとの誤差上界e_jから、非unitary積の増幅を含め`||A_J…A_1-U_J…U_1||≤prod_j(1+e_j)-1`を用い得る。deterministic PF誤差b_PF_plus、input誤差、全pathのfinite-bit/native誤差を別途正しく合成する必要がある。単純な一pieceのtailだけでPRの適格性を認定しない。

**上界不通過は不適格の証明ではない。** `b_plus≥epsilon`は、そのcertificateで正の残余幅を出せないことを表す。実際のbが大きいとは限らず、結果はUNDETERMINED/ABSTAIN。真の下界`b_minus≥epsilon`が独立に成立して初めて、正の残余幅を要求する当該policyで不適格と判定できる。固定policyの保守bias予算でs≤0となる技術的拒否も、物理的な実誤差下界とは区別する。

**入力・Tier・費用。** scheduleはM、lambda/p/fragmentはD（未保存ならD_new）、G6係数はL。b_PF_plusに必要なcommutator/行列や厳密norm certificateはMではない。state/reference/真のheadroomを使うならS/O、native達成証拠はN。位相taskでは非零の信号半径の下側保証も必要。FR_AVAILABLEの有用性失敗を、Oracle半径で置換して救済しない。今回、実分子への新しい上界評価・半径取得は0。

**根拠・新規性・反証条件。** Taylor remainder、norm積、既存G6/W2B/FR、一般のbias付き濃度保証の適用。安価な係数だけではQ間関係・state・PF配置を尽くさない。記号上、同じp=(1/2,1/2)でもQ2=−Q1ならR=0、Q2=Q1ならR=Q1となり、P_mと指数の誤差は同一とは限らない。この例は一般involutionの情報不足を示すだけで、固定分子DFのno-goではない。新しい安価で有用なb_plus/b_minus/半径certificateは特定できず、R3との差分はない。

**不足する検証。** 強く安価な全path bias certificateという未解決責務はあるが、新機構がないまま数値pilotで埋める問題ではない。既存の数値を再評価する計画は推奨しない。

## P-C：law-level同値classによる取得の共有

**対象・前提。** 同じDF/input binding、tail/p/lambda、T、paired cutoff K、law/SOURCE・数値規約を固定したS2で、整数R=qrが同じなら、`tau=lambda*T/R`、R個の独立microstepのraw lawとnormalizationを共有できる。有限bitではprecision・丸め経路・実proposalも鍵に含める。独立sampling法の一致を、同一seedを使う相関sampling許可へ変更しない。

**帰結・取得情報。** 同じtau/Kの分布・係数tableやG6 kernel/P5群等をcold取得一回で共有できる。kernel係数はp/x/m、rootやdyadic lawはprecision/SOURCEを含む。符号を外に保つ係数質量の共有が可能でも、signed controlled phaseを出力から消さない。scopeはM+D+Lの情報再利用で、qの変更に伴うdeterministic挿入順序、signal/bias、headroom、N、native境界fusion、C/Gの再利用は含まない。費用receiptのidentityも別契約である。

normalizationの等式は数学上の`B=b(tau)^R`である。`q*(r*log(b))`等のbinary64評価順序を変えれば丸めが変わり得る。数学的な共有可能性を保存receiptのbit一致へ拡張せず、実使用SOURCE・演算順序・丸め/区間規約のbindingを別に確認する。

**高次PFの失敗条件。** piece係数a_j、整数配分r_jでは`tau_j=lambda*T*a_j/(q*r_j)`、反復数はq*r_j。各pieceのsigned tau、cutoff、law、label意味・必要な数値規約の対応が必要で、総Rだけでは不十分。ALLOCATIONは最低1、floor、remainderの整数処理を持つ。C0.5の人工係数(2,−3,2)では総R=12のq1で(4,5,3)、q2の一outer当たり総6で(2,2,2)となり比例しない。この既存人工例だけを再利用し、実Yoshida/H2/H4/H6の新配分は評価しない。

pieceの順序を無視したmultisetが等しい場合、mass/normalizer等の可換scalar集計は共有できても、非可換operator pathの一致は従わない。normalizerだけの絶対値鍵と、phase・operator用のsigned鍵は別にする。

**根拠・差分・実用性・残責務。** PR2_SIGNAL_CODE、RTE_CODE、SCHEDULE/ALLOCATION、G6_CORE、P5_CODEの既存関係。省けるのは重複したlaw算術・setupで、候補ごとのsignal/native評価を省く保証はない。cacheは有用な工学的最適化だが、新しい選択原理ではない。cold取得、hit/miss回数、同じ最適化をbaselineにも許す条件、保存容量を課金する。実時間利益は未測定で、実用性・新規性の成立を宣言しない。

## 3候補を通じた判定

P-A/P-Bの条件は定式化できるが、既存G6・IS・濃度保証・上下界比較の範囲にある。P-Cは実装上の情報共有である。安価な情報だけから新しい分子PRの安全除外/適格性を与える非自明な条件は今回特定できない。これはあらゆる将来の選択法の不可能性証明ではない。C0.6では新しい独立選択法の命題・科学検証計画を採択しない。
