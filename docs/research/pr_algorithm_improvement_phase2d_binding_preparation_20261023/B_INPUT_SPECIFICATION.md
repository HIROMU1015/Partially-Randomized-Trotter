# Bの数学的入力仕様：P0から変更なし

本書は契約上の静的specであり、scientific input fileではない。文書/SOURCEのhashは管理identityで、fixture_input_sha256ではない。以下から今回数値を生成していない。

4mode、system qubits0–3、ancilla4。ketは|b3 b2 b1 b0>、index=sum 2^p b_p、制御込みはindex+16 b4。Pauli配列はqubit0から順。JWは `a_p=(prod_{j<p}Z_j)(X_p+iY_p)/2`、n=a†a。stateはX0X1|0000>=|0011>、2粒子。full Fock registerで扱い、中間projection/postselectionなし。geometry/AO basis/molecular DF rankは非該当、二fragmentを分子DF rank2と呼ばない。

```text
H = cI + D + F0 + F1
c = 1/37
D = sum h_p n_p; h=(1/20,-1/30,1/40,-1/50)
F_l = B_l (sum d_lp n_p)^2 B_l†
d0=(1/5,-1/7,1/11,0)
d1=(0,1/6,-1/8,1/10)
G_pq(cg,sg) on rows/columns p,q = [[cg,-sg],[sg,cg]]
V0=G01(3/5,4/5) G12(5/13,12/13)
V1=G23(8/17,15/17) G02(7/25,24/25) G01(9/41,40/41)
B_l=Gamma(V_l)
B|vac>=|vac>; B a_p† B†=sum_j V[j,p] a_j†
```

行列積は右端が先。Bではexact orbital Vを生成するが、Gamma(V)／Fock operatorを数値生成しない。native Gaussian helperのU.T／実回路の位相資格はF1で検査する。規約文字列やJW/registerが変わる入力はBで拒否する。

固定D/R splitはLD1のone-body block一つと残差二fragment。`faithful_identity_in_tail`：F_lのIはRへ残し、外部scalarはcだけ。D内nに含まれるIも保持。SはframeのZ/ZZ/I展開で元fragment区別を維持。Pは同じF_lをexact JW代数へ展開し、Pを選んだfragment間だけ同一physical wordをcollectする。S/P間追加collectやframe同値探索、小係数screeningはしない。

| arm | F0 | F1 |
|---|---|---|
| SS | frame | frame |
| SP | frame | physical Pauli |
| PS | physical Pauli | frame |
| PP | physical Pauli | physical Pauli |

BのcoverageはP0のstable ID順I/u/s/v規則で確認し、motif/eventは生成しない。欠落時はINPUT_COVERAGE_FAILUREで停止。

## 独立catalog

A=+1/8 B0Z0B0†、B=-1/10 B0Z1B0†、C=+1/12 X0Y2Z3、I=+1/20 I。A/Bは同じsource frame、Cはphysical Pauli、Iはphase。外部c_cat=-1/31、D=0、T=h=1/20、q=r=1、K2。第五のphysical armではない。Bでは4label inputだけをbindingし、68eventの生成・列挙はF1へ残す。反可換pair groupingは使わない。

## 時間とlaw

F1はT=1/5,q2,delta1/10,r2,h1/20,K2。B.CONTEXTS四armはこのlawのみ。F2は同fixtureのT1/5,q1,delta1/5,r2,h1/10,K0で、Bは仕様と件数だけを記録し、F2数値lawを作らない。

paired cutoff Kの通常Taylor次数はK+1。productsの時間順を保ち最後にchild、paired phaseを保持する規則はIと同じ。Bはイベントを生成しない。F1/F2 signal allowance .025、alpha .05、bias allowance .025とBのAPI toleranceは別物。

各L、lambda、B、law実数、scientific input SHAは今回未取得。混在辞書が有利になるよう値を変更せず、SS/PP最良も正当な将来結果として残す。
