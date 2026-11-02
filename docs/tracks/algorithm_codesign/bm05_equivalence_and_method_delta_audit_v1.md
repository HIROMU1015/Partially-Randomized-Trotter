# BM-0.5：compact BCH同値性・method-delta監査 v1

2026-10-05 JST。対象source文書はBM-0 commit `3b2d624adde979f7d8f983bc7fdbbec88f90c500`。
利用者review `REVISE_BM0_METHOD_DELTA_BEFORE_BM1`に従う、**scienceではない記号監査**。

**結論：固定nested列の三次係数はcompact再帰とBM分解で同値。現adapterのnew-method gateは不通過。**
同じDF backend・同じ集約規則ならscoreと候補順位も同じ。共通部分のcacheはcompact対照にも与えられる。
BM-1を新手法検証として実行しない。application/engineering studyの価値や別deltaの採否はGPTへ返す。
BM一般のno-go、全誤差評価・全scheduleの同値性を主張しない。

## 1. 一次式と規約の対応

一次本文：[Maxwell et al., arXiv:2606.30738v1](https://arxiv.org/html/2606.30738v1)、
§III.2.1 Eqs.(10)–(12)、§III.2.2（再帰PFとcache）、§III.2.3 Eq.(19)。2026-10-05に関連本文を確認した。
三次のcompact表現と再帰構成の再利用は既知である。論文のasymptotic spectral結果は今回の有限signal保証に使わない。

BMはanti-Hermitian `X_i=-iD_i`と、outer半stepがX、innerがYの

\[
C(X,Y)=-[X,[X,Y]]/24+[Y,[Y,X]]/12
\]

を使う。Maxwellの`BCH(inner,outer)`の三次項をこの規約へ直すと

\[
-[[Y,X],Y]/12-[[Y,X],X]/24=C(X,Y).
\]

名前のinner/outer、generator順序、符号を変換してから比較する。
論文のEq.(19)は**このsame symmetric wrapを積み上げる再帰**。
既知compact式をflatにだけ適用し、nested列への再帰・repeat reuseを禁止する弱い対照は採らない。

## 2. 任意group sizeについての三次導出

BM-0の固定operatorは、Aのnative symmetric sweep S_A、Bのouter half-sweeps、中央exact Rを使う。
以下はideal列の形式log、固定整数m、h→0での導出。

1. Aのinnermost exact指数からouter A_iのsymmetric wrapを逐次適用する。
   各wrapで三次へ加わるのは`C(X_Ai, sum_{j>i}X_Aj)`。既存三次項とのcommutatorは五次以上。
   従って`log S_A(u)=u X_A+u^3 K_A+O(u^5)`、
   `K_A=sum_{i<a} C(X_Ai,sum_{j>i}X_Aj)`となる。これはEq.(19)の逆順indexによる帰納法。
2. 同じS_Aのm乗のlogはm倍。`u=h/(2m)`に対して
   `L_A=log[S_A(u)]^m=h X_A/2+h^3 K_A/(8m^2)+O(h^5)`。
3. 中央`exp(L_A) exp(h X_R) exp(L_A)`にcompact symmetric wrapを適用する。
   linear outer generatorはhX_A、A内部三次項は両側から加算されるので
   `h(X_A+X_R)+h^3[C(X_A,X_R)+K_A/(4m^2)]+O(h^5)`。
4. B_bからB_1へouter wrapを加える。各段で
   `C(X_Bi, X_A+X_R+sum_{j>i}X_Bj)`が加わる。

したがってcompact側も

\[
K^{\rm compact}_m=
\underbrace{C(X_A,X_R)+\sum_{i=1}^b C(X_{B_i},X_A+X_R+\sum_{j>i}X_{B_j})}_{K_{\rm floor}}
 +\frac{K_A}{4m^2}=K^{\rm BM}_m.
\]

これは任意a>=1、b>=0、整数m>=1の同じ固定列に成り立つ形式恒等式。
scalar identityはcommutatorへ寄与しない。exact adjacent fusionでoperatorもBCH係数も変わらない。
flatは別列だが、同じcompact再帰で`K_flat=sum_i C(X_Di,X_R+sum_{j>i}X_Dj)`を両armに与えられる。
q同一macro反復では三次global log係数は共通に`T^3 K_m/q^2`。

finite RTEへ置換した積の誤差を、このideal三次恒等式で保証したわけではない。
finite Taylor bias／normalization／numerical guard／signalは今回未評価。

## 3. 独立な形式展開の限定check

[audit script](../../../scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)はstandard libraryだけを使い、
抽象letters A_i/B_i/Rのfree associative wordsをdegree3で打ち切ってFraction演算する。
Hamiltonian、DF小行列、state、science provider、circuitは持たない。

比較した三経路：literal指数seriesの積→log series、Maxwell型symmetric再帰、BMのC式とfloor/internal分解。
fixtureは`(a,b)={(1,1),(2,1),(2,2)}`、各`m={1,2,4}`の9件。
全部のdegree<=3係数が完全一致、二次項0。a>=2の6件では誤った`K_A/m^2`置換を検出した。
この追加fixtureは形式indexのcheckであり、BM-1のmodel/domain追加ではない。

[保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json)は
`EXACT_EQUALITY_THROUGH_DEGREE_3_IN_ALL_REGISTERED_FIXTURES`。
有限9件のcheckを任意sizeの証明と呼ばない。一般sizeは§2の再帰導出に基づく。
PennyLane実装の検証、runtime benchmark、physical semantic testではない。

## 4. Operatorの同値性とscoreの同値性を区別する

DF substitutionとordered-word結合を同じ規約で行う写像をPhi、leading評価をBとする。
一致した形式Lie多項式は同じDF入力へ代入しても一致する。
同じPhiと同じpostprocessorを使えば
`B(Phi(K_compact_m))=B(Phi(K_BM_m))`で、score、同じcostとの選択、tie ruleも同じ。

ただし、**normを取る位置の違いだけでもscoreは変わり得る**。

| Score | 定義 | 差の意味 |
|---|---|---|
| 共通combined評価 | `B(Phi(K_floor + K_A/(4m^2)))` | 両armが同じcanonicalization／結合後評価なら一致 |
| 共通separated評価 | `B(Phi(K_floor))+B(Phi(K_A))/(4m^2)` | 両armが同じgroup別三角不等式なら一致 |
| combined対separated | 上二つを違うarmに与える | 三角不等式・相殺保持のpolicy差。operator差ではない |

Bがnorm／weighted coefficient l1のようなsubadditive評価ならcombined<=separated。
形式word supportが分離していても、DF上の恒等式・結合を含めた一般評価では無条件の等号を仮定しない。
BM-0のseparated leading modelはcombinedの相殺情報を捨て得る。
この違いで順位が動いても、「compact対照から得られないDF設計」にはならない。
compact側も同じfloor/internal groupingを取得できるので、同じpolicyで両scoreを構成できる。

今回DF入力は代入していない。異なるaggregationが具体instanceの順位を変えたとの観測は0。
aggregationを結果後に動かすこと、片方にだけ相殺を許すことをmethod deltaとは扱わない。

## 5. 情報・計算量・再利用・score差分表

| 比較軸 | BM分解adapter | 同情報compact再帰＋DF backend | 判定 |
|---|---|---|---|
| 使う情報 | G_i、lambda_i、N、登録列、group順序、m/q、cost policy | 同じ情報 | 差なし。I2 accessなし |
| 捨てる情報 | leading打切り、group別triangleの場合は相殺 | 同じpolicyなら同じ。combinedは相殺を多く残し得る | policyの選択を新methodにしない |
| formal三次commutators | K_Aに2(a-1)、floorに2(b+1)、計2(a+b) | 同じ再帰式で同じinventoryを構成可能 | 同じorder、m倍のnaive stage展開を強制しない |
| DF評価cost | C_DF：同じbackendでfloor/internalを評価。素朴全tripleならworst O(L^3 n^3) | 同じbackend・同じinventoryに同じC_DF | 速度優位なし。formal countとmatrix costは別 |
| 候補間reuse | floor、K_A、backend norms/wordsをcache。separated scoreならM ratesをO(M) scalar評価 | 同じcache・m依存係数を使用可能 | 現案のreuseは独占できない |
| combined score cost | cached coefficient mapsを組み合わせるcostが必要 | 同じ組合せ・canonicalizationを使用可能 | separatedより重い場合は情報保持とのtrade-off |
| 候補順位 | 固定backend/aggregation/cost/tieで決定 | 同じ値から同じ決定 | 同一policyでは差なし |

**一文の差分：現BM案はcompact BCHの同じ三次係数をfloor/internalへ整理して再利用するadapterであり、
同じ入力と同じ集約規則を与えた強いcompact対照から取得できないscore・再利用量は残っていない。**
これは現在の式・backend案についての技術判定。実装のconstant factorは未測定だが、
弱い再計算baselineとの速度差を独立科学methodの根拠にはしない。

## 6. 結果前new-method gateと今回の判定

BM-1をnew-method検証として提案するには、次をすべて満たすことを要件とする。

1. 同じG_i/lambda_i/N、列domain、DF backend、aggregation、cost、cache機会の強いcompact対照を定義。
2. その対照からは得られない候補順位または評価cost低減が残る具体式／取得手順を結果前に示す。
3. 差を弱いbaseline、triangleの位置、I2 oracle leakage、人工cost weightに帰属させない。
4. 差を判別するcontrol・metric・threshold・budget・sourceをfreezeし、別authorizationと実行指示を得る。

今回1は同情報対照を具体化、2は不成立、3は差と誤認しない条件を整理、4は未認可。
よって`BM05_EQUIVALENT_NO_ESTABLISHED_METHOD_DELTA_MANDATORY_STOP`。
**新手法検証としてのBM-1は実行しない。** 別deltaを今回勝手に探さず、applicationへ縮小するかはGPTへ返す。

[BM-1案のreview反映amendment](bm1_pilot_scope_amendment_v2.md)は、将来別判断があった場合の
leading heuristic／I2評価、primary count／secondary人工costの区別を保存する。
その修正だけでgateが通ったとはしない。B-F closure・原INCONCLUSIVE／R0 BF-A、過去STOP、A evidenceは不変。
