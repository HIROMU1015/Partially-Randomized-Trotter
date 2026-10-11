# DF三次補正 whole / local Pauli 比較：結果前契約 v1

2026-10-11 JST。**準備のみ。比較計算・compile・Clifford+T合成は未実行・未認可。**
利用者が指定した条件を保持して、別実行のレビュー対象を固定する。実装sourceと実行authorizationは今後別途必要。

## 入力と比較対象

基準commitは `8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f`。旧A–F監査、G10、RA-D0、原source、結果、marker、authorization、STOPは不変。
新branchは `track-b-df-correction-comparison-preregistration-20261011`。
原本固定commitは `f6a8f022947bb9efe1f2c4fffeb0eeda051b2903`。

- [GPTレビュー原本](../../research/track_b_df_sparse_correction_post_audit_scientific_review_20261011.md)：36,306 bytes、SHA256 `d2b3fbd4de208981a2e04e3356b72bce52e059c428c736f101f4e3ab6cf3ffda`。
- [ZIP原本](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/inputs/track_b_df_sparse_correction_post_audit_review_20261011_reproducibility.zip)：50,156 bytes、SHA256 `bb14c641f9b7158cfdfd9efdc17d9c123f65b7ae1335c28c7847ba0ba4b951a4`。
- CRC9件・内部manifest・外部MD一致を確認。付属codeは監査対象であり、今回実行／importしない。18 self-checksをCodexの独立PASSへ加算しない。
- ZIPの局所NPZはGPTによるsynthetic再構成値。今回payloadはloadしない。将来は固定source recipeと独立に照合する参照であり、新しい分子inputではない。
- [機械可読contract](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/contract_v1.json)、[実行計画](df_correction_comparison_execution_plan_20261011.md)を一組とする。

方式Aはwhole nested commutator、方式Bはレビュー§10の**coin-first identity–Pauli pairing**。BをweightなしのPauli qDRIFTで代用しない。
両者に同じlocal T、Gaussian frame、controlled phase、取得情報を与える。全系Pauli表をBだけへ要求しない。

## 二つのdomainを分離する

### CD：全複素信号の誤差・moment・十分shot budget

固定source `audit_df_sparse_correction.py:phase_c_d` のrecipeを保持する。
NumPy1.26.4、PCG64 seed6121107、3-mode QRを順に3回、primitive rank/sign=(1,+1),(2,−1),(1,−1)。
H0=(9/20)Q0、H1=(7/20)Q1、R=(1/4)Q2、ordered D split L_D=2。単一primitive/blockの3-block syntheticであり、分子・held-outではない。

| 条件 | 固定値 |
|---|---|
| t / δ / n | 1 / {1/8,1/16,1/32} / {8,16,32} |
| γ / κτ | 5/4 / 4（floating normで縮めない） |
| η | min(1,γδ)、今回全てunsaturated |
| finite RTE | s=1、paired cutoff4=P5、ordinary paired normalization |
| complex ε | 1/10000 |
| axis ε | X/Yとも1/20000。両axis成功時complex誤差≤√2/20000<ε |
| α | 1/20。X/Yのfamily budgetを各1/40に固定 |
| simultaneity | A/B×3δ=6 task。同時比較のため各axis/taskは(1/40)/6=1/240 |

初期状態は結果前の技術提案として
ψ=(|001⟩+i|010⟩+|100⟩+|011⟩−i|101⟩+|110⟩)/√6に固定する。
little-endian bit labels、amplitude index1,2,4,3,5,6。vacuum/固有状態の探索はしない。
target z=⟨ψ|exp(−it(H0+H1+R))|ψ⟩。初期状態の実回路費用は未取得なのでhに残す。
この状態とfamilywise配分は今回Codexが具体化した契約案であり、実行前の利用者確認対象である。
exact signal／matrix normは将来のdiagnosticだけに使い、primary shot数やbias budgetを結果に合わせて縮めない。

各primitiveの符号はQへ、係数は正hへ吸収する。固定tuple集合は元のtype-I/type-II式の8 label。
A_mass=Σcτ=4159/192000、B_mass=4A_mass=4159/48000、K3=ΣcτTτ。
以下のA_massは方式名Aと区別する。zero Tを浮動小数thresholdで落とさない。

### E：局所native費用、独立したT合成screening

元 `audit_df_sparse_correction_native.py` のseed7101107、**最初の4×4 random_unitary drawを消費してから**r3→r6の各rank2 framesを3回ずつ生成する順序を保持する。
三つのsignは(+1,−1,+1)。θ=1/32、κ=4、u=θ/κ=1/128。
uncontrolled/controlled各1、ancillaはMSB。対象は共有active frame上のcenterだけで、全系embeddingは費用変数に残す。
レビューのη=0.1は診断値であり、新しいCD率として採用しない。Eは条件付きevent費用とweight/biasを記録し、PR総費用にはしない。
**CDの期待頻度2.5とEの回路単価を掛けない。** r3同士でも異なるseed・rank・係数なら別inputである。

## A/Bの有限step semantics

CDでqτ=cτ/A_mass、uδ=A_mass δ³/(2η)。Aはouter coin失敗時I/weight1、成功時qτでexp(iuδTτ)/weight1。
Bは同じouter coin、同じqτ。Tτ=ΣjβτjPj、ℓτ=Σ|βτj|。
成功時jを|β|/ℓで選び、ατ=atan(uδℓτ)、exp(i signβ ατPj)、weight wτ=√(1+(uδℓτ)²)。
T=0はidentity/weight1。左右・step・RTEのdrawは独立。共通乱数による左右drawの共有はしない。

Bのconditional weighted meanはI+iuδTτ、半補正平均はI+iδ³K3/2。
Aの平均は同じ一次項を持つが、finite-stepでBと同じoperatorではない。
目標は両者の同一targetへのbiasを戻した全信号精度であり、unitary単価だけの勝敗ではない。
この恒等式・D5・rank-one Gaussian閉包の独立検証は将来の最初のgate。今回は式を契約化しただけでPASS認定しない。

## 有限step上界と誤差予算

x=(21/20)δ、y=B_mass δ³とし、元監査の
b_sym=2(eˣ−Σ_{k=0}⁴xᵏ/k!)+y(eˣ−1−x)+eˣ(eʸ−1−y)、
e_R=e^(δ/4)(δ/4)⁶/720を使う。∥K3∥≤B_massを使用する。

| 台帳 | A | B |
|---|---|---|
| half-correction誤差 | e_A=δ⁶(B_mass²/η+B_mass²)/8 | e_B=δ⁶B_mass²/8 |
| one-step bound | b_A=b_sym+2e_A+e_R（convex平均のcontractionを使用） | b_B=b_sym+2e_B+e_B²+e_R(1+e_B)² |
| global algorithmic bias | (1+b_A)ⁿ−1 | (1+b_B)ⁿ−1 |

Bはweighted平均なのでcontractionを仮定しない。∥Cbar_B∥≤1+e_Bを戻してからtelescopingする。
これらは計画式であり、将来独立証明・interval scalar arithmeticで確認する。負時間は対象外。

両方式・全δに同じ最大axis bias予算を置く。

| 台帳 | axis上限 |
|---|---:|
| PF＋corrector平均＋finite RTE | 1/40000=2.5e−5 |
| finite-bit sampling | 1/200000=5e−6 |
| native/synthesis（weighted coherent信号） | 1/200000=5e−6 |
| 入力組立・Pauli係数・算術・angle表現の認証誤差 | 1/1000000=1e−6 |

原DF thresholdingはこのsyntheticにはない。無断で新しいthresholdを加えない。
現在のsourceのQR/float行列をexact involutionとみなさず、ideal frameへの組立誤差を別台帳へ戻す。認証不能ならINCONCLUSIVE。
既存保存CDのA boundは最大1.8051980412979107e−5（δ=1/8）。これは過去のfloat記録で、今回は再計算していない。
新しいcertified運用値は未取得。限界超過ならδ/ε/予算を変更せず、理由と代替案を報告して停止する。

未使用予算の規則を固定する。algorithmic項だけは結果非依存の上記certified bound β_iへ置換する。
digital/native/numeric項は実測errorが小さくても**最大割当額を全て控除**する。
s_stat,i=1/20000−β_i−1/200000−1/200000−1/1000000。
従ってβ_iがcap以内ならs_stat≥1.4e−5。oracle観測誤差をこの式へ戻さない。

outer coinのηは今回dyadicなのでexactに保持する。coin成功後のtuple／Pauliのcomplete conditional lawとRTE order lawを48-bit denominatorへlargest-remainderで丸め、tieはlabel lexicographic。
conditional branchesの丸めも含む最終lawとのL1距離を認証する。受理massをηから変えない。T=0のexact identityは保持、near-zero係数のdropは禁止。
安全な全信号boundはWmax×Σdraw Δ_L1（3n complete draws）＋weight/angle roundoffの別台帳。
floor/順位のinterval認証に失敗したらprecision retryせずSTOP。実際のshot/trajectory samplingは行わない。

## normalization、moment、十分shot数

Γ=B5(δ/4)ⁿ、B5(v)=Σ_{k=0,2,4} vᵏ/k! √(1+(v/(k+1))²)。
元RTEのΓを両方式に戻す。correction weight1をΓ=1と混同しない。

ν_A=Γ²、Wmax_A=Γ。
Bの理想momentはΓ²(1+ηuδ² E_qℓτ²)^(2n)、rangeはΓ(1+uδ²ℓmax²)ⁿ。
primary sufficient budgetには共通情報から証明できる**ℓmax=2³×4=32**を使い、Eℓ²≤1024とする。
finite-bit lawでもこのpointwise boundが有効。係数intervalで得る詳細momentはsecondaryで、primaryを結果後にtightenしない。
ℓ≤32の仮定を入力組立の誤差で認証できなければ、この32を黙って拡大せず報告する。

N_i=ceil[log(2/(1/240))(2ν_i/s_stat,i²+4Wmax_i/(3s_stat,i))] をaxis/taskごとに上向き丸めで計算する。
6task×2axisのunion bound≤0.05。全て共通Bernstein sufficient budgetであり最小shot数とは呼ばない。
N_iは保存する資源予算で、量子shot実行は0。native biasには2E[|W|Σgate ε_gate]等のstrict controlled信号boundを使う。
局所Eの1e−6 unitary精度をCDのglobal 5e−6信号予算へ自動流用しない。

## 未取得費用と損益分岐

CDのprep/readout h_x、全PR base費用C_base,x(δ)、各tupleのC_A,τ,x(δ)、PauliのC_B,τj,x(δ)は未取得。
Δ別、T/CX/1Qごとに記号変数で保持する。補正頻度を合成するのは**同じCD input**の単価が得られた場合だけ。

c_A,x=2nηΣτqτ C_A,τ,x、c_B,x=2nηΣτqτΣjpτj C_B,τj,x。
G_i,x=2N_i[h_x+C_base,x(δ)+c_i,x]。
Aの損益分岐は N_A(h+C_base+c_A)<N_B(h+C_base+c_B)。
必要ならc_A<(N_B/N_A)c_B+(N_B/N_A−1)(h+C_base)と表示する。
depth/ancilla/workspace、classical generation/Pauli取得は別座標。time/T/CXを任意scalarへ潰さない。
未取得を0にしない。T全量未取得ならG_Tも記号的／INCONCLUSIVEとする。

## native/T計画と判定の限界

[実行計画](df_correction_comparison_execution_plan_20261011.md)の2worker/単thread、固定compiler、最大22 representative compile、最大64 RZ target keysを使う案。
全RZを合成しない。合成対象は同文書の位置規則と±Pauli anglesで結果前に固定する。
pygridsynth2.0.0、strict phase、共通gate precision1/40000000000、dps100、retry0。
actual部分Tと全circuit Tは分ける。未coverageの角度が残ればwhole全TはT_INCONCLUSIVE_INCOMPLETE_COVERAGE。
RZ×代表Tは推計に過ぎず、primary actual Tとして採用しない。

way A/Bのmean・controlled semanticsの反例が見つかれば以降を実行せず保存してSTOP。
numeric/resource/precision不成立はtechnical INCONCLUSIVE。prefixを科学的positive/negativeへ昇格しない。
resource intervalが重なる場合は判断保留。任意の5%/10%materialityを結果後に追加しない。
generic whole compileは最小値／下界ではない。勝敗はこのbackendと固定domain内の技術証拠で、algorithmの一般優劣ではない。
契約確認→必要なfuture sourceの実装・freeze→別authorization・明示指示を経るまで実行しない。
最終研究判断はGPT/利用者。G10/RA-D0、分子、grid/fixture/precision変更、次stageは未認可。mandatory STOP。
