# F2：全処理bit費用の具体的な条件付き会計

2026-10-18 JST。SOURCEの処理列を、公知の算術評価へ写した通常の静的解析。記号上の入力・精度を用い、数値代入・benchmark・SOURCE/test実行はしていない。**frozen symbolic-event generator**、既存手順によるangle enclosure、物理provider、全Nの保存/測定を別層にする。新しいtight bound、lower bound、Full/Partial比較定理ではない。

## 1. 計算モデルと標準primitive

固定基数のbitモデルを使用する。入力有理数は正規化した整数対、各有理演算では正確な整数演算とGCD正規化を課金する。加減算/shift、schoolbook乗除算、通常GCD/rootを用いた**保守的なモデル上界**を使う。標準算術の一次資料はBrent–Zimmermann, *Modern Computer Arithmetic*, arXiv:1004.4710v1, version 0.5.1の§§1.2–1.6。[MCA] Python公式仕様はFractionのGCD正規化とisqrtの正確な意味を確認するために使い、実装の実秒保証には使わない。[PY-FRACTION] [PY-MATH]

| primitive | 採用する保守的な課金 | 条件・区別 |
|---|---|---|
| b-bit整数加減算/比較/shift/読書き | O(b) | operand/resultのbit長に課金 |
| 整数積・商/余り | O(b²) | schoolbookモデル、非零除数、入力/出力が定数倍b bit以内 |
| GCD・正規化を含む1有理演算/比較 | `Q(b)=O((b+1)^3)` | 入力とreduced結果の各整数長がO(b)、一演算の未約分cross productもO(b)。通常GCDの既知のより鋭い評価を使わない保守的課金 |
| root_interval(value,K_root) | `R(b,K_root)=O((b+K_root+1)^3)` | radicandの整数対はO(b) bit。scaled整数はO(b+2K_root)、division/isqrt/端点正規化。公知の整数root処理をこのcubic envelopeで課金 |
| n要素のcomparison sort | O(n ℓ_n) comparisons/key処理 | `ℓ_n=1+ceil(log2(n+1))`。SOURCE sortedの呼出しを標準comparison-sortモデルで会計。比較のbit費用も別に乗せる |
| fresh random bits | H bitsにO(H) bit-supply費用 | 理想bitモデルの**非零**取得費用。実供給器ではT_bits(H)を代入する別条件。SHA診断をfresh-IIDの実証としない |

Q/RはSOURCEを置き換える新アルゴリズムではない。既存処理を同じ意味の標準算術primitiveで評価した条件付き上界である。CPythonの多倍長backend、allocator、hash、serialization制限、OS時間を実測せず保証したものではない。CPU秒、bit operations、T/CX/depthは別座標。

後述の領域評価では、標準整数primitiveのworking領域をO(b) bits、comparison sortの補助index数をO(n)とする通常モデルを明示的に使用する。Python allocatorがその量の実bytesだけを使うという仮定ではない。

## 2. 入力・中間bit長

H_pをp table整数対の総binary長、H_xをxの整数対長とする。Eはη/ρ、必要ならε_axis/primitive_error/τなど**用いる精度パラメータ**の整数対総長。m、L、H、K_root、Nは各々の**数値**を費用変数とする。`ℓ_m`/`ℓ_L`は上と同じlog記号。

既存[BIT]§2の共通分母Dと `D^m den(x)^m m!` の評価を使い、

`b_c = 1 + m(H_p+H_x+ℓ_m)`

と書けば、Green/return係数、a、s、raw mass、tと局所有理radicand等の整数長はO(b_c)。各畳込み/massの途中和も同じ共通分母の倍数で表せ、未約分の単一演算だけを定数倍長で課金できる。これは既存係数bit lemmaへの直接適用であり、bit長一定という仮定ではない。`_mass`はpow/factorialを繰返すため、単なる1回のFraction加算とは数えない。

| 量 | 使用できるbit長の上界・根拠 |
|---|---|
| Green/係数・正確child law・s/a | O(b_c)。child massesは同じ既存分母、sも同じ分母。各q_i=a_i/sの分母はsの整数分子を共有できるという長さ会計。実装を共通分母型へ変更したわけではない |
| root endpoint | O(b_c+K_root)。小さいB_o/order rootsではO(K_root+1)で足りるが、保守的に前者も可 |
| order normalized intervals | 各root整数の総和をT_lo/T_hiとすれば、端点分母はT_lo又はT_hi、共通分母はその積。長さO(K_root+ℓ_m)。midpoint/和にも定数倍の長さで足りる |
| dyadic law | 分母は2^Hを割り、各成分整数長O(H+1)。部分和も同じdyadic分母なので長さO(H+ℓ_n) |
| 実π | 最大m+2個のH-bit dyadic因子の積。分母は2^((m+2)H)を割り、整数対長O((m+2)H+1) |
| c̃とW̃ | c̃=d_mid a_child/sはO(b_c+K_root)。W̃=c̃/πは `w_e=b_c+K_root+(m+2)H+E+ℓ_L+ℓ_m` のO(w_e) bit。G249–251/_event87の再計算も定数倍の一演算長として課金 |
| B_o/U区間・budget | UのDPはD^m、root和はdyadic分母を使えるためO(b_c+K_root)。budgetの(1−η)^(m+2)等を含む長さは `w_b=b_c+K_root+(m+2)E+ℓ_m` のO(w_b)で課金。小さいremainingは無料の安定性仮定でなく、正確有理の記述長に含む |

bit長の存在と、Pythonオブジェクト/JSON出力の容量・文字列化が任意サイズで成功することは別である。出力形式/runtime制約は別項を保持する。

## 3. distributionとselectorを具体化する

### 3.1 dyadic_distribution

n個のintervalの全端点・途中和に、整数長O(g+ℓ_n)の共通分母/分子表現がある場合、SOURCE K55–82のmidpoint、sum、scaled floor、remainders、relative guardの全ての1演算幅はO(g+H+E+ℓ_n)で押さえられる。余りsortを含め、

`D(n,g)=O(n ℓ_n Q(g+H+E+ℓ_n))`

と課金できる。これは共通分母型の新実装を仮定せず、既存Fractionの途中値のbit長を上から数える通常の導出である。任意の外来intervalへこのgを無根拠に適用しない。

本SOURCEではorderのg=O(K_root+ℓ_m)、labelのg=O(H_p)、Full childのg=O(b_c)が使える。canonicalのconditionals/groupにも、SOURCE式とroot端点から同じ種類の共通分母長を得て課金する。一般のn個の独立b-bit rationalを外部から与える場合は、共通分母の粗い長さO(nb)をgへ使う必要がある。全分布を1演算で無料取得するモデルではない。

### 3.2 dyadic_index

K97–110は毎callでlawの和、全scaled countのdyadic/非負検査、その後の累積走査を実行する。正当なH-bit lawについて

`S(n)=O(n Q(H+ℓ_n)+H)`

を課金する。最後のHはfresh-bit供給分。事前lawが保存済みでも検査・線形走査を省かない。lawをalias tableへ変えた将来最適化は採用していない。実bit供給器を扱うなら+Hを+T_bits(H)に置き換え、保証条件も別に固定する。

## 4. Full setupの上界

`n=n_o=(m+1)/2`。SOURCE G216–225では、order roots n回、B_o roots n回、Uのempty root＋tail roots n回で、root_intervalは合計**3n回**。同じrootを自動memoした評価ではない。Uにはempty-parent queryとraw-already-reduced DPも必要。

```
C_setup,Full = O((Lm²+m²+L) Q(b_c+E+ℓ_L+ℓ_m))
             + 3n R(b_c,K_root)
             + D(n,K_root+ℓ_m) + D(L,H_p).
```

最初の項はinput/normalization、既存O(Lm²) Green、UのO(Lm) DP、empty query、tの繰返しpow/factorial等を含む。`factorial(k)`を逐次積で課金してもk以下の整数積、x^kも通常の整数冪評価で課金でき、次数和にm²項を用いた。既存SOURCEが実際にその逐次実装を採用するというCPU主張ではない。

F1の受入不等式を検証する将来の処理は未実装であり、frozen setupには含まれていない。p_min/t_min/d_minとP0〜P5の固定式検証は、既存pow/factorialと有限input検査を使う標準モデルなら、別項 `C_preflight=O((L+m+1)Q(w_e))` で保守的に課金できる。これは**与えられたH/Kの検査**の費用であり、H/K探索/選択の費用や新selectorの設計ではない。

G258–284の固定confidence budgetを呼ぶ場合、別項 `C_budget=O((1+ℓ_m)Q(w_b))+C_log` を課金する。C_logはSOURCE固定96項による有理log上界の処理で、ここでその値を計算していない。任意α APIの費用ではない。

## 5. Full一trialの処理列と3経路

以下ではvalid parent長l≤m−1、child数c≤L。`w_p=b_c+K_root+H+E+ℓ_L+ℓ_m` と書く。

| SOURCE/段階 | 具体的な課金 | 経路 |
|---|---|---|
| G238 order draw | S(n) | 全trial |
| G239 l label draws | l S(L) | 全trial。各drawのlaw検査/走査も毎回 |
| G240/K11–18 raw判定 | O(l ℓ_L)のstack/tuple比較処理 | 全trial。非reducedならここでNone |
| G242/K194–202 local parent/child係数 | 既存 O(m³+Lm²)有理演算をQ(b_c+E+ℓ_L+ℓ_m)へ接続。word系列共有、各child畳込み、各massのpow/factorial込み | raw-valid trial |
| K121–133/230の再sum | **O(L²)有理加算**を追加。child_lawの各a/self.sでsを再計算し、外側のifにもs評価。d²等にも有限回のs再集計 | raw-valid trial。cached sumを将来導入したことにしない |
| K224–229 roots/accept threshold | R(b_c,K_root)を2回、ρ/relative比較・downward dyadic arithmeticをO(Q(w_p)) | raw-valid trial |
| K230–231 child dyadic law | D(L,b_c) | **coinを引く前にpacketで作るため、coin rejectも支払う** |
| K235 ratio/packet | s/aの正確有理divide、phase/time-word/tuple | raw-valid trial。symbolic angle情報だけで、atan計算ではない |
| G243 coin | H-bit供給とexact Fraction比較 O(H+Q(w_p)) | raw-valid trial。不受理ならここでNone |
| G246 child draw/G248 index | S(c)、children.indexの走査 | acceptedのみ。上界にはS(L)を使える |
| G249–251/K239–254/_event79–90 | 最大m+2因子のπ、c̃/π、再sum、alpha/proposal/weightの再計算、reduce、Counter、record | acceptedのみ。次のC_eventで課金 |

`Parent.child_law`のO(L²)は、W1で所在を記録した再集計を今回実数なしにloop合成した結果である。既存の「parent＋全child係数query O(m³+Lm²)」の反証ではなく、**digital packet全体へ追加する項**である。

次を定義する。

```
C_raw(l) = S(n) + l S(L) + O(l ℓ_L)

C_packet = O((m³+Lm²+L²) Q(b_c+E+ℓ_L+ℓ_m))
           + 2 R(b_c,K_root) + D(L,b_c) + O(Q(w_p))

C_coin = O(H + Q(w_p))

C_event = O((m+L+1) Q(w_e) + m²(ℓ_L+ℓ_m)^2).
```

最後の項はword/Counter等のrecord構築について、hash lookupを無料の一定時間とせず、最悪の線形衝突処理も許す保守的会計である。weightの積は長さに沿うFraction処理として数え、全積を一度の単位演算としない。local係数queryを2回してはいないが、G249–251/_event87の有理再計算は残す。

3経路は `raw zero: C_raw(l)`、`coin zero: C_raw(l)+C_packet+C_coin`、`accepted: C_raw(l)+C_packet+C_coin+S(c)+C_event`。全てにaccepted費用を課すなら、**上界として**

`C_trial,max = C_raw(m−1)+C_packet+C_coin+S(L)+C_event`

が使える。一方、期待費用を求めるには各digital経路に対して実行されたprefixを課金しなければならず、この最大費用を期待実費と呼ばない。

## 6. Angle取得・provider IR・nativeの境界

frozen generatorが返すのはtangentのみで、その費用はpacket/eventに含めた。angle enclosureを実装済みとして全費用へ黙って含めない。[BIT]§6の**既存**half-angle/atan交代級数を将来利用する場合は、要求τ、提供されたroot精度K_φ、有限項数J、正当な角区間の証拠を別に持つ。

その既存手順を正確有理区間で評価し、変換後引数の端点bit長をb_z=O(b_c+K_φ)、級数途中値を `b_θ=O(J(b_z+ℓ_J)+E)` と会計すれば、標準pow/加算/除算による保守的費用は

`C_angle = R(b_c,K_φ) + O(J ℓ_J Q(b_θ))`

で、保持する有限個の端点/途中和の領域はO(b_θ+b_c+K_φ)。冪の繰返し計算を定数費用にせず、共通冪分母と有限個の奇数分母の積のbit長を合成した通常評価である。J/K_φを返す新しいangle selector、angle計算コード、native合成を作っていない。J・precisionをτへ接続できる**存在**は既存[BIT]§6、具体的な実装policyは未実装という分類を保つ。

CP2–20のIRはword分CQと、child回転部分CQ2/H4/Rz2/CX2、必要なouter Z、outer+helper=2を記述する。実際のadjointとscalar/σを維持する。外部cacheを読むCP37–58にはsequence長/読込/逆順tokens/出力費用が別途必要で、取得済み全key表を無料oracleとしない。ratio文字列化等も出力形式/runtime制約のある処理である。

物理providerのT/CX/depth・strict error、状態準備、測定/accumulatorは[B-COST]の固定監査を参照するだけ。`C_angle`やquery回数だけでそれらの費用を認証しない。

## 7. Memory・全N・期待値

標準モデルで、frozen Fullのsetup stateとsetup/localの一時領域を保守的にまとめると、

`M_generator = O(Lm b_c + (L+n)(b_c+K_root+H+E+ℓ_L+ℓ_m) + w_e + mℓ_L)` bits

を使用できる。kernel配列、root/law/sortの同時配列、1parent/child masses、整数一時値、eventを含む。Parent.sの再計算は時間を増やすが、過去全parentのcacheにはならない。旧O(Lm)は係数個数であり、このpeak bit領域とは異なる。Python object/allocatorの実bytesを測定した値ではない。

全出力を保持するなら別にS_saved(N)が必要であり、逐次保存すれば自動的にpeakが下がると**実装済み扱いにはしない**。eventの構造的binary record長はO(w_e+mℓ_L+Lℓ_m)で評価できるが、JSON/十進変換、保存先、backend outcome等の形式は別仕様。C_output(N)/S_saved(N)はその形式の演算・容量費用であり0にはしない。

期待値の会計はA/W1にある既存記法を使う。

```
E[c_cl/trial] = E[C_raw(l)]
             + E[1_raw-valid (C_packet(u)+C_coin)]
             + E[1_accept (S(c)+C_event(u))].

C_cl = C_setup + C_preflight + C_budget
     + N E[c_cl/trial] + C_output(N).
```

ここでC_preflight/angle等の未実装層は、使用する場合だけ**別項**として追加する。最大費用による条件付き上界なら `C_setup+C_preflight+C_budget+N C_trial,max+C_output(N)` と書ける。期待経路確率や正確Zを計算して代入していない。

量子は `C_q=N E[1_accept c_q(event)]` の別vector。ideal Z=B_RA/B_oとdigital受理率は別、valid-parent A_uの下界を全trial Zへ代入しない。Nは全trial、量子event数のhard capはN、NZはideal期待に限る。受理event価格がword/angle依存なら、単一価格×未知Zに置換しない。未完成/失敗で費用を0にしない。

## 8. canonicalを同じモデルで扱う条件

G103–212はGreenを作らない。group数g≤n+L+1のSOURCE上界、degreeごとのt、label law、group roots/lawがsetupで必要。同じ標準モデルで

`C_setup,can = O((L+m²)Q(b_c+E+ℓ_L+ℓ_m)) + g R(b_c,K_root) + D(g,K_root+ℓ_g)+D(L,H_p)`

と保守的に会計できる。fullのGreen費用をcanonicalへ転嫁しない。onlineはgroup draw S(g)、modeに応じたlabel draws、conditional/first/root-child lawのD(L,b_c)、eventでのその再構築、正確なideal/proposal/weight、Counterを課金する。IID modeに存在しないconditional再構築を期待実費へ付け足さない。sample/event内の再構築をcache済みにして費用を減らすこともしない。

この表は比較側にも同じ算術/bit/精度/保存/合法最適化の条件を与えるための会計であり、上の粗いupper bound同士から優劣を結論しない。P_Uや一般normalizer順序、W2b/W2cの新しい比較命題は扱っていない。

## 9. 量化と到達範囲

標準モデルの下で、有限入力、供給されたH/K、既存一様lemmaとF1条件、正確な算術primitiveを使う**1trial symbolic生成**のbit費用/領域まで上式で具体化できた。mそのもの、H/Kそのもの、入力bit長と精度記述に対する条件付き評価であり、binary長log mだけの多項式主張ではない。angleは既存手順のJ/K_φを別パラメータとして明示した。

必要Nは精度/残余幅に依存し得る。1trialが多項式でも、全Nが入力bit長のみの多項式時間とはしない。物理oracle費用も除いた古典処理の評価である。ここでの新しい式は、既存lemma・静的loop count・公知算術の**通常合成**であり、非自明な新計算量定理や実用優位ではない。追加の非自明保証が必須と分かった箇所はない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[MCA]: https://arxiv.org/pdf/1004.4710v1
[PY-FRACTION]: https://docs.python.org/3.11/library/fractions.html
[PY-MATH]: https://docs.python.org/3.11/library/math.html
