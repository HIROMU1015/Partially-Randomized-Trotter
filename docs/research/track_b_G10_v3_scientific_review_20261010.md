# Track B G10 v3 科学的研究レビュー
## 全return集約の追加価値、固定辞書sampling下界、低次数構成の位置付けと研究の着地点

- レビュー日：2026-10-10（JST）
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 原結果commit：`fcd3ea6217bc00b667180cec149a70102d75f07e`
- 取得性改善commit：`e65e3c0680fff4cfc4243ed5d6c87429cd5ce753`
- 科学source S3：`b9ed01455351628c9073748f5ba5751aa794b789`
- Authorization／実行HEAD A3：`53a7bc4ca8051bfd76343e98f3122c0f198e37d0`
- 原結果status：`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`。本レビューは変更しない。
- **研究判断：一般full-return生成器の資源優位を主張して自動拡張することは採択しない。この固定入力の性能探索はG10で区切り、低次数構成・一般構成・native資源上の限界を既存証拠から整理する。Track B全体や一般構成の数理的有効性を否定する判断ではない。**
- **Mandatory STOP維持。新しい本番実行、m9、入力探索、合成、sampling最適化、G11を認可しない。**

## 1. 結論と、G9からの判断の更新

G10 v3は、技術的には正常完了した比較として受け入れられる。重要なのは、その比較が研究上の中心課題にどう答えたかである。

G9では、return集約familyの低次数実装であるclosed P5が、登録したordinaryやliteral matched CTSより低いT予算を与えた。その一方、一般のfull-return生成器を使う追加価値は未確定だった。そこでG10は、同じp・x・providerを保ち、m=7で「closed P5＋ordinary tail」を一般full returnと比較することを優先した。[R1–R3]

**その中心比較に対して、今回の固定条件では一般full returnの追加資源利益は得られていない。**

m=7では、一般full returnはclosed P5＋tailに対してT切片が約2.91155%高く、準備・測定呼出し係数も約0.61484%大きい。共通の非負T準備単価を加えても、登録lawの順位は逆転しない。CX、native 1Q、accepted-tail T上限、hard T上限でも改善はない。[R4–R6]

さらに、固定event辞書・固定native合成列・同じBernstein予算規則を保つ限り、samplingの自由度を戻してもfull returnが対照に届かない範囲がある。この点は、単にcanonical samplingが未最適だった、という説明より強い。§6で下界の適用範囲を明示する。

したがって、G9の「一般次数で追加価値を確認できるなら方法研究を拡張する」という条件付き継続から、今回は**「低次数実装の成果を残し、一般full-returnの性能上の役割を縮小する」**へ進む。これはG9レビュー§14にあらかじめ示された分岐に沿っている。七次で勝たなければあらゆる方法研究が無価値だ、という新しい合否条件を導入したものではない。[R3]

今回得られた研究上の核は、次の区別である。

> 形式returnをより多く集約すること、係数ノルムを小さくすること、event supportを減らすこと、実際のnative量子資源を減らすことは、同じ最適化問題ではない。

ただし、この一般的注意点そのものを新規アルゴリズムの発見と呼ぶべきではない。保存された具体例と条件付き分離として価値を評価する。

## 2. 科学taskと評価量を固定する

### 2.1 同一次数内の有限演算子比較

登録targetは

\[
R=\sum_{i=0}^{2}p_iQ_i,\qquad
p=(1/5,3/10,1/2),\qquad x=5/7,
\]
\[
M_m=P_m(-ixR),\qquad m\in\{3,5,7\}.
\]

比較するのは各m内での**full first operator moment**であり、特定状態の信号だけ、あるいはchannel平均だけの一致ではない。3-system-qubitのsynthetic development providerを使う。[R1,R2]

\[
Q_0=Z_0,\quad V_1=R_{X_0X_1}(\pi/4),\quad
V_2=R_{X_1X_2}(\pi/4)R_{Z_0Z_1}(\pi/4),\quad
Q_i=V_i^\dagger Z_iV_i.
\]

対応するPauli表現は

\[
Q_0=ZII,\quad Q_1=(IZI+XYI)/\sqrt2,\quad
Q_2=IIZ/\sqrt2+(IXY-ZYY)/2.
\]

この例はPauli情報が明示的に取得できるI1の文脈である。分子、DF分解、一般的なPauliアクセス不能条件を検証したものではない。[R8]

**m=3の費用とm=7の費用を、そのまま同じexponential精度の性能として比較しない。** Taylor remainder、step数、PR全体の誤差、位相推定／エネルギー推定の誤差は、この同一有限target比較の外側である。これは結果を見た後の限定ではなく、事前契約のscopeである。[R2]

### 2.2 精度と予算

各Re/Im axisの許容誤差は1/200、各axisのfailure配分は49/34000である。34 axes合計の推定failureは0.049、17 rowsの資源tail failure合計は0.001、和が0.05となる。

Rz合成はstrict誤差10^-6、`up_to_phase=false`。root 256 bit、proposal 160 bit、rho=eta=10^-12を固定している。common biasと残余幅は

\[
b=2\{8\rho+8(1+\rho)\,2\epsilon_{Rz}\}
=\frac{1000000500001}{31250000000000000},
\]
\[
s=\frac1{200}-b
=\frac{155249999499999}{31250000000000000}.
\]

sourceの十分予算規則は

\[
N=\left\lceil\ell_+\left(\frac{2M_{2,+}}{s^2}
+\frac{4W_+}{3s}\right)\right\rceil,
\qquad \ell_+\ge\log(2/\alpha_{axis}).
\]

行列residualの1e-10 toleranceは意味論の診断であり、上のbias保証の代わりではない。[R2,R7]

### 2.3 N、K、期待費用の区別

Nはaxis当たりの推定試行数である。full returnでは一部が量子実行前にzero-fillとなるので、実際の量子回路呼出し期待数は

\[
K=2N Z,
\]

Zは受理確率である。Kを2Nと同一視しない。一方、棄却試行を含む古典生成処理を無料とみなしてよいわけでもない。

主評価量は

\[
G_T(h)=T_0+Kh,\qquad h\ge0,
\]

である。T_0は二axes分の期待native T切片、hは共通の一回の準備・readoutのT費用である。hに結果後の都合のよい値を割り当てない。h=0の値は切片であり、「現実の状態準備が無料」という仮定ではない。[R2,R7]

T、CX、native 1Q、workspaceは別々の座標として扱う。native 1QにはTなどの1-qubit gatesも含まれるため、これらを単純に足して独立な資源単位としない。共通のouter 1Qはsource上で5NZ=2.5Kとして別記されている。[R7,R8]

## 3. 実行・証拠の受理と独立確認の深さ

### 3.1 完了を支持する記録

外側process receiptはnormal exit、exit code 0、termination signalなし、stderr 0 bytes、およびCOMPLETE terminal statusを記録している。outer wallは22.2261158517秒、CPUは20.029088秒、whole-runner peak RSSは240.015625 MiBである。result、STOP、completion token、markerと監査記録の整合も報告されている。fileの存在やexit code 0だけで完了認定したのではない。[R1,R9]

全17行・34 axes、10,936 bindings、27新規合成key＋19 G9再利用keyが保存された。m=5の6行はG9のimmutable eventと合成列を使い、共通34-axis policyの予算だけを変更した。これは新しい独立反復実験ではない。G10 v1・v2の失敗結果を救済データとして採用したものでもない。[R1,R10]

### 3.2 監査修正の扱い

post-STOP監査では二つの実装不備が記録された。CTS eventに`provider_calls`が必ずあるという誤仮定と、raw identity変数のshadowingによる初期report fieldの誤りである。修正前の資料を消さず、correctedとfinal監査を正本として区別している。これらはrunner結果の再生成・科学条件変更・本番retryではない。[R1]

本レビューでは`final_saved_output_audit_v3.json`の57,021 PASSを**保存監査の結果**として利用する。この数字をGPTが独立に再実行した検査数として表示しない。[R10]

### 3.3 今回GPT側で実施したこと

- 契約、runner、予算、生成器、native lowering、固定辞書下界、前回研究判断をsourceと照合した。
- 17行の表示CSVの全bytesをローカルに保存し、Git blob identity `705c141b5a927f6b9e326a71cdae470879e6cc5e`を照合した。SHA256もmanifest記録と照合した。
- m=7の主要二行について、取得した正確なT・K、fullの係数ノルム、weighted-root下界などの選択fieldを有理数として転記し、主比較と下界分離を独立に算術確認した。
- 対数下界を標準ライブラリのFractionによる64項atanh級数から別実装で作った。対数enclosure幅は約4.93e-63であり、中心的な符号判定は浮動小数点ではなく有理数で確認した。
- m=7の両方式の先頭3 root eventsについて、保存されたproposal・回路費用を読み、rootの条件付き受理確率を算術確認した。
- 主要二source `g10_saved.py`、`g10_comparison.py`は、S3と取得性改善commitのGit blob SHAが一致することも確認した。
- 先行研究の関連する一次本文・式・アルゴリズムを確認した。文献確認範囲は§10に記す。

### 3.4 実施していないこと

全136 raw fragmentsの独立再結合、全10,936 bindingsの別実装による再認証、全57,021保存監査条件の再実行、全1,352保護pathの再hashは行っていない。原JSON全体の約66.8 MB取得はconnectorのサイズ制約で失敗したが、分割資料・正確な行field・sourceは参照できた。これは未pushや一次データ欠損ではない。[R11]

native synthesis、Hamiltonian／回路行列計算、production generator実行、新しいsampling lawの構成、量子測定、新しい入力、repository変更も行っていない。追加したのは、**保存値の算術と数式の検討**である。

したがって、本レビューは全実装の独立再現報告ではない。一方、登録比較の研究判断に必要な主要な差、下界の意味、科学的claimの限界については、sourceと一次fieldを根拠に結論を出せる。

## 4. 登録資源表の整理

以下は保存display値からの表示であり、単位はT/CX/1Q/Kとも10^6。期待native gate数と準備/readout呼出し係数を区別する。正確な有理数はrepositoryのexact resource tableにある。[R4,R11]

| m | Arm | Events | N/axis | T / 10^6 | CX / 10^6 | 1Q / 10^6 | K / 10^6 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | ordinary | 30 | 1,306,564 | 365.035172 | 21.419239 | 980.769210 | 2.613128 |
| 3 | partial_return_tail | 21 | 993,111 | 270.220208 | 16.572049 | 717.894571 | 1.986222 |
| 3 | closed_P3_tail | 15 | 977,045 | 279.459012 | 16.670105 | 720.920647 | 1.954090 |
| 3 | full_return | 15 | 1,132,235 | 279.999178 | 16.702326 | 722.314113 | 1.957867 |
| 3 | matched_CTS | 21 | 1,714,854 | 400.847851 | 10.470333 | 1036.410493 | 3.429708 |
| 5 | ordinary | 273 | 1,325,813 | 371.085420 | 21.904736 | 996.749433 | 2.651626 |
| 5 | partial_return_tail | 264 | 1,009,902 | 275.437399 | 16.998103 | 731.603229 | 2.019804 |
| 5 | closed_P3_tail | 258 | 993,701 | 284.749630 | 17.095837 | 734.643184 | 1.987402 |
| 5 | full_return | 63 | 1,144,972 | 273.991326 | 16.756376 | 692.820724 | 1.964236 |
| 5 | closed_P5_full | 63 | 975,840 | 272.239831 | 16.649261 | 688.391854 | 1.951680 |
| 5 | matched_CTS | 24 | 1,698,404 | 387.633441 | 10.391732 | 968.944322 | 3.396808 |
| 7 | ordinary | 2460 | 1,326,140 | 371.179562 | 21.914206 | 996.997917 | 2.652280 |
| 7 | partial_return_tail | 2451 | 1,010,188 | 275.518709 | 17.006422 | 731.816636 | 2.020376 |
| 7 | closed_P3_tail | 2445 | 993,984 | 284.832047 | 17.104130 | 734.856468 | 1.987968 |
| 7 | full_return | 255 | 1,145,109 | 280.249068 | 16.756533 | 715.218431 | 1.964243 |
| 7 | closed_P5_tail | 2250 | 976,120 | 272.320319 | 16.657452 | 688.598158 | 1.952240 |
| 7 | matched_CTS | 28 | 1,698,621 | 399.058971 | 10.392764 | 1048.816689 | 3.397242 |

### 4.1 Canonical full returnは全次数でclosed対照に支配される

| 次数 | full returnを支配する登録closed対照 | T切片増加 | K増加 |
|---:|---|---:|---:|
| 3 | closed P3 | 約0.19329% | 約0.19329% |
| 5 | closed P5 full | 約0.64336% | 約0.64336% |
| 7 | closed P5＋tail | 約2.91155% | 約0.61484% |

T差とK差がともに正なので、各比較で任意のh>=0についてfull returnの登録lawは高コストである。m=3のclosed P3が全hで全対照中最良だという意味ではない。

### 4.2 m=3のpartialとclosed P3には準備費用依存の境界がある

m=3のpartial returnはT切片270.220208百万、K=1,986,222。closed P3はT切片279.459012百万、K=1,954,090である。したがって

\[
h_*\simeq
\frac{279459012.3-270220207.8}{1986222-1954090}
=287.5266.
\]

登録T目的では、h<h_*でpartial、h>h_*でclosed P3が有利となる。これは単価を選んでwinnerを作るのではなく、パラメータ依存性の全体を示したものである。full returnはこの両者のtrade-offに新しい最良領域を加えない。[R4]

### 4.3 m=5／7のP5構成

m=5のclosed P5 full、m=7のclosed P5＋tailは、それぞれの次数における登録canonical lawのT切片・Kの両方で最小である。従って共通非負hの全域で、主評価の登録最良構成となる。

m=7の重要な比較を原単位で示す。

| 指標 | closed P5＋tail | general full return | full側の変化 |
|---|---:|---:|---:|
| 期待native T | 272,320,319.314 | 280,249,067.746 | +2.91155% |
| K | 1,952,240 | 1,964,243.128 | +0.61484% |
| 期待native CX | 16,657,451.84 | 16,756,533.06 | +0.59482% |
| 期待native 1Q | 688,598,157.9 | 715,218,430.8 | +3.86586% |
| system外workspace | 1 | 1 | 同じ |
| accepted-tail T上限 | 316,262,880 | 321,276,132 | full側が大きい |
| hard T上限 | 316,262,880 | 371,015,316 | full側が大きい |
| 保存event bindings | 2,250 | 255 | full側が少ない |

上限値は期待値と交換可能ではない。また保存supportの減少は、量子資源の改善ではなく別の性質である。[R4–R6]

## 5. なぜ「より集約したのに安くならない」のか

### 5.1 追加の係数ノルム減少は小さい

full returnの正の係数質量をA_F=Σ_j a_jとする。m=7では正確な保存値から

\[
A_F=1.28845383374058023458\ldots
\]

である。固定辞書下界のK係数が共通因子×A²であることを使い、P5＋tailと比較すると、fullのAは約0.01367%小さく、A²は約0.02734%小さい。後者の相対比較は保存表示値からの近似値である。[R5,R6,R12]

このAと、local fullのrange約1.502278471、reference second moment約1.935616455、acceptance約0.857666444は別の量である。rangeを「集約後normalizer」と呼ぶのは誤りである。

すなわち、P5までの処理後に六・七次returnをすべて取り込んでも、この固定例で残る係数質量上の改善幅は小さい。

### 5.2 量子呼出し数と一回当たり費用の因数分解

\[
T_0=K\,\overline T_{accepted}.
\]

保存値から、m=7の一回の受理回路当たり平均Tは

\[
\overline T_{P5+tail}\simeq139.491210,\qquad
\overline T_{full}\simeq142.675346.
\]

fullは一回当たり約2.28268%高く、Kも約0.61484%大きい。

\[
1.0291155\simeq1.0061484\times1.0228268.
\]

従って、今回の約2.91%差をすべてzero-fillによる追加試行のせいにするのも、すべて一回路の長さのせいにするのも不正確である。[R4–R7]

m=3／5ではfullと対応closedの受理一回当たり平均Tは表示精度の範囲で一致する。G9で示された「同じ理想ensembleの別の生成・normalizer取得方法」という位置付けと整合する。m=7ではP5＋tailとfullのensemble自体が異なり、一回当たり価格差も現れている。[R3,R4]

### 5.3 頻出するroot回転の価格も変わっている

保存されたm=7の空parent eventsを見ると、rootのangle tangentはP5＋tailで`2189372/2921709`、fullで`9011383315/12025404923`である。

| child | P5＋tailのroot T | fullのroot T |
|---:|---:|---:|
| 0 | 136 | 140 |
| 1 | 138 | 142 |
| 2 | 140 | 144 |

各root eventで4 T増えている。保存proposalを用いて算術確認すると、空parentは受理量子回路の約87.7993%（P5＋tail）／87.8094%（full）を占める。[R13,R14]

このため、追加return集約は「めったに発生しない高次数eventだけを軽くする」操作ではない。主に使うroot回転の係数・角度も変え、固定精度で得たnative合成列の費用に影響している。

ただし、このroot比較だけから全T差を一意に原因分解したとはしない。他eventの係数、proposal、native価格、Kも同時に変わっている。上の表は保存された具体的な機構の確認であり、任意のbackendや合成seedで必ず4 T増えるという定理ではない。

### 5.4 local予算の保守性も別途ある

general fullの生成器はordinary envelope Bと、全parentを列挙しないnormalizer上界Uを使用する。Uはroot寄与とreduced raw-word質量から構成され、真の全係数質量Aを無料で参照する方式ではない。[R15]

従って、zero-fillで古典棄却を量子測定前へ移せても、登録予算から直ちに最小のKが得られるわけではない。しかし、§6の下界はさらにsampling自由度を戻した有利なクラスにも適用される。したがって切片付近では、上界を少し精密化するだけで対照を超えられると期待する根拠はない。

## 6. 固定辞書sampling下界：何を排除でき、何を排除できないか

### 6.1 比較クラス

event辞書{V_j}、正の係数a_j、保存されたnative価格T_j、precision、同じ残余誤差s・failure配分を固定する。full-support proposal q_j>0によりweight a_j/q_jを用いる。同じ平均を保つimportance samplingを許すが、event演算子・係数・回路・合成列の変更は許さない。

zero-fillを含む場合、Σq_j<=1として残余を量子費用0のzero eventにしてよい。以下のCauchy–Schwarzはこの場合も成立する。古典生成費用を捨てた有利な下界であり、古典費用を加えることによって下界が破られるわけではない。

### 6.2 下界の導出

\[
M_2(q)=\sum_j\frac{a_j^2}{q_j},\qquad
C(q;h)=\sum_jq_j(T_j+h).
\]

Cauchy–Schwarzから

\[
M_2(q)C(q;h)\ge
\left(\sum_j a_j\sqrt{T_j+h}\right)^2.
\]

またT_i,T_j,h>=0なら

\[
\sqrt{(T_i+h)(T_j+h)}\ge\sqrt{T_iT_j}+h,
\]

なぜなら両辺を二乗した差はh(√T_i−√T_j)²>=0だからである。従って

\[
\left(\sum_j a_j\sqrt{T_j+h}\right)^2
\ge\left(\sum_j a_j\sqrt{T_j}\right)^2
+h\left(\sum_j a_j\right)^2.
\]

登録Bernstein規則の正のrange項とceilingを落とすと、二axes期待費用について

\[
G(q;h)\ge\frac{4\log(2/\alpha)}{s^2}
\left[\left(\sum_j a_j\sqrt{T_j}\right)^2+hA^2\right].
\]

sourceは対数と平方根の下側有理近似を使ってこれを安全側へ評価している。T_j=0のCTS実補正eventも削除せず残る。[R12]

費用と二次momentの積をimportance samplingで扱う原理は既知であり、Cugini–Atif–SubaşıのTheorem 1にも対応する。ここで重要なのは、それをG10の**固定された十分予算規則**に接続したことと、実際の保存辞書でどの差が残るかである。[P1]

### 6.3 m=7における具体的な分離

選択した正確な保存aggregateと独立の有理対数下界から、

\[
G_{full}(q;h)\ge L_F(h),
\]
\[
L_F(h)\simeq277736101.757709
+1946702.79147539\,h.
\]

一方、実際に登録されたP5＋tailの予算は

\[
G_{P5+tail}(h)=272320319.314469
+1952240\,h.
\]

差は

\[
L_F(h)-G_{P5+tail}(h)
\simeq5415782.443240-5537.20852461\,h.
\]

h=0でfullの下界は対照の実現予算より約1.98875%高い。下界同士を比較したのではなく、**full側の全proposal下界と、対照側の具体的な登録予算**を比較している。

この正の分離はh<約978.070885で残る。丸めた安全な例として、**0<=h<=970**の全域は有理数の符号として確認した。h=970でも差は44,690.174365 Tより大きい。

978付近は「実行可能な二方式の勝者が逆転する点」ではない。下界が対照を上回ると証明できる区間の端である。これを超えたhでは、この下界からは分離できない。そこに有利なfull proposalが存在すると判定してはならない。登録canonical fullの不利は、別のT差・K差の両正性から、全h>=0で残る。

### 6.4 最適化をどこまで除外するか

切片について、fullの登録値280.249068百万から下界277.736102百万までの改善可能幅は約0.89669%以下であり、対照に対する約2.91%の差を埋められない。このため、この固定条件のT切片を救済する目的のproposal探索を次に行う情報価値は低い。

一方、次の変更は下界の外側である。

- 別のevent分解・係数再構成、複数eventの統合、回路の大域最適化。
- 別のprecision、合成列、gate set、provider。
- state-dependent variance、別の推定・confidence規則、adaptive measurement、全体taskの別設計。

これらを理論的に不可能だと主張しない。ただし「現在の方法が負けたから制約外を順に探索する」は、次の研究方針の十分な理由ではない。

m=3／5ではfull辞書の表示下界がclosedの登録値より低いので、この同じ下界からsampling改善の可能性を排除してはいない。m=7で得た分離を全次数へ拡張しない。さらに、P3／P5で同じ理想ensembleを共有する場合、既知のsampling変更は一般生成器だけの独自利益にはならない。[R3,R4]

### 6.5 物理的な最低必要shot数の定理ではない

以上は、あらゆる量子推定アルゴリズムに対するquery complexity下界ではない。Nは特定のBernstein型保証から決めた十分予算であり、その**予算規則による期待費用の下限**を述べている。

したがって「最適proposalを実装した」「物理的最小shot数を証明した」「どんな量子アルゴリズムでもP5を超えられない」というclaimはいずれも不適切である。

## 7. CTS比較は残るが、研究の大きさを取り違えない

### 7.1 Tの固定条件での利益

closed P5のT切片はm=5でcanonical literal CTSより約29.7687%低く、m=7のP5＋tailは約31.7594%低い。ordinaryとの比較では両方およそ26.63%低い。[R4]

さらにCTSの固定辞書下界は、m=5で約324.629030百万T、m=7で約334.175448百万Tであり、各P5構成の登録予算を上回る。CTS側の下界のK係数もP5構成のKより大きいので、同じ固定辞書・同じpolicyでは全共通h>=0について分離が残る。これはCTS側の未最適samplingだけで全T差を説明できないことを支持する。[R4,R12]

ただし、約31.76%は**P5＋tail対canonical CTS**の値である。一般full returnの、最も強い簡単な対照に対する追加利益ではない。P5＋tail対partial returnのT切片差は約1.16086%である。比較対象を省略して「約32%の新手法改善」とまとめると成果の帰属を誤る。

### 7.2 CXはCTSが小さい

m=7のnative CXはCTS約10.392764百万、P5＋tail約16.657452百万である。したがってP5系が全資源でCTSを支配するわけではない。

共通の準備CX単価をg>=0とすると、Kも含めたCX比較の境界は

\[
g_*\simeq
\frac{16657451.84-10392763.94}{3397242-1952240}
=4.33542.
\]

g<g_*ではCTSの総CXが小さく、g>g_*ではP5＋tailが小さい。この値を装置の準備単価と仮定するのではなく、感度として残す。

native T/CX/1Q/Kの登録ベクトルでは、m=3はpartial・P3・CTS、m=5はP5・CTS、m=7はP5＋tail・CTSがPareto候補となる。ただし古典生成費用や全実行時間を含めたPareto frontierではない。[R4]

### 7.3 literal CTSをCTS family全体と同一視しない

今回のCTSは明示Pauli情報から同じ有限P_mへ特殊化し、identity／real correction、zero-T event、相対位相を保った比較である。これは公平性に関する強みである。[R8]

一方、CTSのすべての再構成・layering・compiler・precision allocationを最適化した結果ではない。CTS文献に存在するより広いアルゴリズムfamilyとの一般的な優劣は、この固定例からは確定できない。[P4,R16]

## 8. 古典側で何が残っているか

m=7で保存event bindingsは2,250から255へ減った。削減率は88.6667%、件数比は約8.8235倍である。しかし、これは参照ensembleのeventラベル／bindingの件数であり、そのままproductionの時間／メモリ計算量ではない。同一unitaryや同一回路に対応する重複labelを排除した一意回路数とも限らない。

sourceでは、ordinary、closed P3、closed P5＋tailも全event表を入力せずに生成する。general fullもlocal coefficient queryで生成する。従って「fullは255件だが対照は2250件を毎回列挙する」という比較は実装と一致しない。[R15,R17]

またG10 runnerは、正確な期待native費用・意味論確認のために小support参照全体を走査している。これはproductionが表を参照してsamplingすることとは別である。総22.23秒には共有合成、参照処理、I/O等が含まれ、各方式の大L・大mでの古典生成優位の実測ではない。64固定trialずつのinterface traceもscaling証拠ではない。[R1,R18]

一般非列挙構成の数学的価値は残るが、**その価値は今回まだ古典資源優位として実証されていない**。既存ordinary samplerも非列挙であり、その事実はWan–Berta–CampbellのAlgorithm 2でも確認できる。[P2]

## 9. 今回の結果から支持されるclaim／されないclaim

| Claim | 判定・根拠 |
|---|---|
| 固定synthetic provider、各m内の有限meanでnative資源比較が完了 | 支持。保存結果・source・監査を根拠とする |
| m=5／7では登録P5構成がcanonical T目的で最良 | 支持。T切片とKの両方が最小 |
| 一般full returnが低次数特殊化後にも追加T利益を与える | この固定比較では不支持 |
| m=7のfullの不利は単なるsampling未最適化である | 固定辞書／同じpolicyの切片付近では下界により否定できる |
| より小さい係数ノルム・supportが必ずnative総費用を改善する | 保存結果はその含意を支持しない |
| return familyがliteral CTSより低いT予算を与える例がある | 支持。ただし固定辞書・nativeモデル・finite taskの範囲 |
| CTS全手法・全compilerより優れている | 不支持 |
| 全return集約の恒等式・局所生成法が数学的に無効 | そのような反証は得られていない |
| 一般fullの古典scaling優位が示された | 不支持。supportとinterface traceでは不足 |
| 独立新規アルゴリズムとしてのpriority／非自明性が確定 | 未確定 |
| 実分子DF-native／PR全体／QPEで資源を削減した | 検証scope外 |

「不支持」は全条件での不可能性を意味しない。「未確定」はPASSと同じではない。この区別を研究記録と原稿の両方に残す。

## 10. 一次文献と新規性の評価

### 10.1 構成要素とmethod deltaを分ける

| 要素 | 既存知識との関係 | 本研究に残る差候補 |
|---|---|---|
| return／identity／低次数への吸収 | Zhao–Yuanのmodified Taylorで高次寄与をidentityや低次数unitaryへ戻す考え方は既知 | 同一有限P_mでどのreturn集合をどの費用で保持するか |
| 隣接Taylor次数のEuler pairing | ordinary RTE系で既知 | reduced even parentごとのchild pairingと局所angle取得 |
| ordinaryの非列挙sampling | Wan–Berta–Campbellに具体的samplerあり | 全returnを含む生成をどの情報量・bit費用で達成するか |
| free-product Green関数 | Aomoto–Katoの既知構造の特殊化とrepository監査が対応付け | 有限Taylor samplerへの接続。新しいGreen関数定理とは呼ばない |
| cost-aware IS／zero-fill | Cugini–Atif–Subaşıなどで原理は既知 | representation変更後に実native費用を払う価値があるか |
| Pauli collection／CTS | 既知の比較対象 | 同じfinite first moment・access条件での具体的な差 |

Zhao–Yuan §4.2は、捨てていた高次寄与のうちidentityや低次数unitaryに一致する項を係数へ戻す構成を述べている。これは本研究の全有限samplerとの完全同値性を証明するものではないが、単に「高次を低次に吸収した」を新規性の中心に据えることはできない。[P3]

Wan–Berta–CampbellのLemma 2／Algorithm 2は、orderとIID labelから一回転を含む語を生成する構成を持つ。従って非列挙であることだけを既存RTEとの差としては主張できない。[P2]

Aomoto–Katoの§1ではfree-productのGreen multiplierと他因子へのspectral shiftが扱われる。repository G6監査はこれをZ2・形式級数の本研究表記へ対応付けている。今回、その対応を新しい独立定理として再証明したわけではない。[P5,R16,R19]

費用重み付きproposalの基本原理も既知であり、qを変えるだけでは本研究の独立method noveltyにならない。[P1]

### 10.2 全体の組合せが既知だと断定してはいない

今回の確認では、「固定finite P_mの全形式return、even-parent/child pairing、未知global normalizerを入力しないlocal generation、finite-bit補正」を全く同じlaw・access・保証で備えた単一の先行アルゴリズムを特定した、とはしない。

しかし、同一構成の記載を特定しなかったことは、新規性・非自明性の証明でもない。特に本研究が目標とする「PR内部の有益なアルゴリズム改善」としては、構成可能性だけでなく既存の簡単な実装を超える利点をどこに置くかが必要である。

理論的に非自明な構成だけでも研究貢献は成立し得る。今回不支持とするのは一般fullのnative性能claimであり、理論貢献の可能性自体ではない。投稿十分と判定しない理由は、性能勝利がないことだけではなく、構成全体の既知研究との差と、その必要性が主要claimとしてまだ確定していないことである。

G10により、その利点を「一般full returnを使えばP5＋tailよりnative Tが安くなる」という今回の性能claimへ置くことはできなくなった。残る可能性は、限定された情報accessでの一般構成／保証、または別の明確な実用的利益である。後者はまだ実証されていない。

### 10.3 文献確認の限界

本レビューは引用した一次文献の関連本文、式、samplerを確認したものであり、全関連論文・全版・全CTS実装の網羅的なpriority調査ではない。Aomoto–Katoの全スペクトル理論やPR論文全体を今回再証明・再検証したわけでもない。したがって「既存の全手法より新しい／優れる」「新規性が完全に消えた」のどちらも確定しない。

## 11. 検証の十分性と弱点

**G10で十分答えられたこと**は、G9が優先した同じp・x・providerにおける次数比較と、P5特殊化を超える一般fullの登録native価値である。特にm=7は有効な否定結果であり、技術失敗ではない。

**答えられていないこと**は、一般的な入力依存性、異なるnative合成モデル、古典access優位、PR全体への接続、独立新規性の確定である。

重要な弱点は次の通り。

1. **既知development入力が一つ。** 別構造へのtransferや母集団的な性能傾向ではない。m=5再利用を独立追試に数えない。
2. **角度別の固定合成catalogueに依存。** rootの4 T差は具体的なnative evidenceである一方、全backendで不変とはいえない。新seedの勝つ結果を探して現在の結論を置換してはいけない。
3. **十分予算での比較。** state-dependent最適推定の必要shot数ではない。保守性は結論のscopeとして残す。
4. **古典費用の一般比較は未完成。** 既存baselineも非列挙であることを踏まえないsupport数比較は過大評価になる。
5. **compiler探索の最適性なし。** 同じliteral loweringでの公平性はあるが、全回路最適化の代表ではない。

これらを認めた上でも、直ちにすべての穴を追加実験で埋める必要はない。現在の研究判断に必要なのは「結果が負だったから検証不足と見なして続ける」のではなく、**既に答えが出た問いと、新しく別の問いを立てる必要がある部分を分けること**である。

## 12. 代替方針と採否

| 選択肢 | 本レビューの判断 | 理由 |
|---|---|---|
| 同じ方針でm9へ進む | 採択しない | G10の優先比較が負。次数だけを増やす根拠がない |
| 新p・x・provider・seedを同時に探索 | 採択しない | 勝つ条件の選別になり、追加価値の機構を説明しない |
| m7の固定full辞書をcost-aware samplingで救済 | 今は採択しない | 切片付近は下界で分離。大hの未分離だけでは探索の価値が十分でない |
| 高精度合成やcompiler変更で順位逆転を探索 | 保留 | 別研究条件であり、現在の科学的負結果の修復ではない |
| DF／分子／PR全体へ直ちに移す | 保留 | 本来の追加価値とtaskの接続が未確定 |
| 一般fullの数理構成を無価値として破棄 | 採択しない | 資源の負結果は恒等式・有限bit構成の反証ではない |
| **既存成果を構成・資源trade-off・限界の記録へまとめる** | **採択を推奨** | 現証拠に見合うclaimを確定でき、無期限の性能探索を避けられる |

「native費用を考慮して集約するか選ぶ」という方向は、今回から自然に浮かぶ。ただし、これはそのまま新規テーマ採択ではない。cost-aware ISは既知であり、closed P5＋tailも既に低次数までの集約を選ぶ構成である。新しい方式とするなら、sampling最適化ではないrepresentation変更、演算子平均の保存、取得費用、既存のprefix＋tailでは得られない差を具体化する必要がある。現時点でこれを名称だけ変えたG11として開始することは推奨しない。

## 13. 研究としての着地点

現時点で最も根拠のある着地点は、**有限Taylor演算子に対するreturn集約familyの構成と、そのnative資源上の適用限界を整理した技術・方法ノート**である。

推奨する内容は、一般finite-mean構成と仮定、低次数のclosed実装との関係、情報accessとfinite-bit会計、登録native比較、sampling-only救済の条件付き限界である。

中心messageの候補は次のように置ける。

> 同じ有限演算子の乱択実装において、係数質量やsupportの削減だけではnative総費用の優位は決まらない。一般return集約は低次数実装の選択肢を与えるが、その追加集約を使うべきかは、準備費用・合成価格・十分予算・情報取得費用を含めて判断する必要がある。

これは、いま直ちに独立新規アルゴリズム論文として投稿十分だと判断するものではない。独立論文にする場合は、既知構成の組合せを超えるmethod deltaを主張として説明できるかが残る。説明できなければ、既存研究の補足・構成ノート・研究記録として区切るのが適切である。

一方で、今回の結果だけを理由に「今後この系統で新しいアルゴリズムは生まれない」とする根拠もない。**現在の一般full-return性能主張を縮小することと、PR内部改善という研究領域全体を終了することは別**である。

## 14. 次の担当と作業範囲

本レビューの研究判断を採用する場合、次の実務担当は**Codex**である。対象は新実験ではなく、既存成果の整理と原稿／構成ノートの整備である。

同じ作業単位で、主張・証拠・適用範囲を対応付け、m=3のprep境界、m=7のcanonical支配、固定辞書下界、root価格、CTSのT/CX trade-offを図表／本文へ反映する。G9からG10への判断更新を記録し、一般fullの性能主張を主役として扱わない。

実装方法、文書構成、軽微な表記修正を細かく分割してGPTへ戻す必要はない。必要な算術のローカル検算は、保存データの照合として行ってよいが、異なるproposal・合成列・入力・precisionを生成する作業は含めない。原result・source・contract・authorization・STOP・markerとTrack Aを保護する。

今後GPTへ戻すべきなのは、既存claimの文言修正のたびではなく、**別の科学的利益を持つ具体的な新構成を提案する、比較対象や主評価を変える、一般化検証に踏み出す、または論文主要claimを新規性として確定する**節目である。その場合も資料と開始承認を別に確認する。

本レビューはその将来の科学実行を包括承認しない。現時点のresult statusとmandatory STOPは維持する。

## 15. 最終判定

| 項目 | 判定 |
|---|---|
| G10 v3の登録比較を研究判断に用いること | 可。確認範囲と監査依存部分を明示する |
| G10 v3科学的レビュー | 完了 |
| general full-returnの追加native資源利益 | 今回の固定比較では不支持 |
| canonical fullの同一次数closed対照に対するT優位 | 全登録次数でなし |
| m7 sampling-only救済 | 固定辞書・同一policyで切片付近は下界により除外。大hやクラス変更へは一般化しない |
| P5系の固定native成果 | 維持。literal CTS／ordinaryへの条件付き利益を残す |
| 独立新規性・一般優位・PR全体改善 | 確定しない |
| 研究の次の重点 | 一般full性能探索を区切り、低次数構成と一般構成の関係・適用限界を整理 |
| 次の実務担当 | Codex：既存成果の文書化・claim整理 |
| 新規本番計算・m9・新入力・G11 | 未認可 |

**最終結論：G10は成功した実行から有効な限定否定結果を得た段階である。一般full-return生成器を性能上の主役として拡張するのではなく、低次数構成の成果を残し、一般構成の価値と資源上の限界を分離して記録する。**

## 16. 再現資料の読み方

同梱の`analyze_display.py`は、blob同一性を確認した17行の丸めCSVから比率・感度・登録ベクトルfrontierを計算する。小数の長い出力は高精度の入力を意味しない。

`check_exact_m7.py`は、`exact_m7_inputs.json`に明示した選択有理fieldを入力し、m7のT/K支配と下界分離を標準ライブラリだけで確認する。選択fieldの転記と表示値との対応を確認したもので、row file全体や255eventのchecksumを再構成したものではない。

`check_root_examples.py`は、m7の各3 root eventの保存proposal・費用だけを使用する。六eventの回路行列や合成列を独立再生成したわけではない。

実行環境はPython 3.13.5。研究runnerのPython 3.10.12やlibrary stackを再現したものではない。処理は保存有理数・整数・Decimalの算術に限定される。実行は以下でよい。

```bash
python analyze_display.py
python check_exact_m7.py
python check_root_examples.py
```

外部論文PDF・フォント・全repositoryのコピーは再現ZIPに含めない。正本の確認は以下の固定commitリンクを用いる。

## 参考資料

### Repository（取得性改善commitで固定）

[R1] [G10 v3 results and GPT handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)

[R2] [Contract v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/contract_v3.json)

[R3] [G9 v2 scientific review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/docs/research/track_b_G9_v2_scientific_review_20261010.md)

[R4] [17-row resource display](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_degree_result/2026-10-10/v3/resource_rows_display_v3.csv)

[R5] [Exact m7 full-return row](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/rows/row_14_m7_full_return.json)

[R6] [Exact m7 closed P5 plus tail row](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/rows/row_15_m7_closed_P5_tail.json)

[R7] [G10 comparison and budget source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g10_comparison.py)

[R8] [G9 native provider and common lowering](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g9_native.py)

[R9] [Outer process receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_degree_result/2026-10-10/v3/process_receipt_v3.json)

[R10] [Final saved-output audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_degree_result/2026-10-10/v3/final_saved_output_audit_v3.json)

[R11] [Primary-data access package](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/README.md)

[R12] [Fixed-dictionary lower-bound source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g10_saved.py)

[R13] [m7 full-return events part 000](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/events/row_14_m7_full_return_part_000.json)

[R14] [m7 P5 plus tail events part 000](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/events/row_15_m7_closed_P5_tail_part_000.json)

[R15] [Local and canonical generators](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g7_generator.py)

[R16] [G6 prior-art and method-delta audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)

[R17] [G10 closed P5 plus tail generator](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g10_generator.py)

[R18] [Fixed G10 v3 runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py)

[R19] [G6 independent mathematical audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)

[R20] [Result evidence manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json)

[R21] [Exact full 17-row resource table](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json)

[R22] [Access extraction verification](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/extraction_verification.json)

### 関連一次文献

[P1] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1（2026-03-13）。Theorem 1、§IVのZeroFill／Discard。<https://arxiv.org/html/2603.13495v1>

[P2] K. Wan, M. Berta, E. T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071v2（2022-07-13）。Lemma 2、Appendix C、Algorithm 2。<https://arxiv.org/pdf/2110.12071v2>

[P3] Q. Zhao, X. Yuan, *Exploiting anticommutation in Hamiltonian simulation*, arXiv:2103.07988（取得PDFはv2、Quantum accepted 2021-08-24）。§4.2、Eqs.24–29。<https://arxiv.org/pdf/2103.07988>

[P4] J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12,52（2026）。Theorem 1およびCTS本文。<https://www.nature.com/articles/s41534-025-01168-w>

[P5] K. Aomoto, Y. Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1),59–85（1988）。§1、Eq.1.6、Lemma 1.1。<https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf>
