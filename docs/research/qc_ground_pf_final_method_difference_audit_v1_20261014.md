# 厳密ground PF再分割：1-RDM応答生成の最終方法差・計算量監査

2026-10-14 JST。ユーザー指定の一回の最終技術監査。**固定した先頭一体A／最初のdensity Bのclassでは、同じ恒等式・情報・再配分自由度・cacheを許したcompact BCH対照が、同じ応答表を同等の計算量で生成できる。判定は `REDUCES_TO_KNOWN`（この比較モデル内）である。** 現生成法に固有の情報量・演算量・保存量・候補再利用の改善は残らない。Phase Aの停止条件を満たすため、Phase B・pilot準備・改良候補探索へ進まない。

これは歴史的初出性、閉式の理論記録価値、研究の正式継続・終了を判定するGPT科学レビューではない。前回の数式の正確性を再検査せず、固定された式を入力として生成手順を比較する。旧27形式検査の再実行、新しい形式fixture、数値H/eig/RDM/expectation/optimizer/PF/compile/samplingは0。

## 1. 固定資料・範囲

Base `c0bb4350e599e557f7897e64222854d0fe45d620`。以下の本文をGitHubから実際に取得し、[固定identity](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/reference_identity.json)のcommit/path/bytes/SHAで照合した。

- [前回の技術監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md)、[引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/research/qc_exact_ground_pf_repartition_gpt_handoff_v1_20261014.md)。
- [旧method status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/method_status.json)、[旧pilot案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/minimal_pilot_design.json)。
- [方針再レビュー原文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md)。既存source・旧形式checker/result・文献registryもread-onlyで参照し、実行しない。

厳密**量子**groundを利用可能、E0は未入力。期待値情報は無料としない。fixed Hermitian H、scalar/normal-order補正、spin/number sector、density frame、PF ordering、精度・失敗率を両法で共通にする。block-encoding/qubitization/QSVTの新設計、Track A/B、C1–C4や終了系列、第二候補の再開は対象外。

## 2. 現法の正確な入出力

H=A+B+T、Aはone-body、Bは最初のdensity fragment、Tは残る固定fragmentの和。固定S2はAを外側half-step、Bをその直内側に置く。XはBのframeで `X=Σx_p n_p`、`[X,B]=0`。A'=A+X、B'=B−Xで、H・exact state・B frameは変わらない。TとXの可換性は要求しない。

| 入力 | 扱い |
| --- | --- |
| a、Bのframe U_B、元fragment/ordering | 同じHermitian one-body matrixと固定density representation。frameへ移す演算費用も同じ |
| exact quantum ground access | 同じ取得モデル・供給条件。E0や全励起固有分解を入力しない |
| γまたは必要な1-body収縮 | `γ_rs=〈a_s†a_r〉0`。取得費用、誤差interval、failure allocationを共有可能 |
| d0またはその取得手順・interval | 元分解の主要係数 `d0=〈D(0)〉0`。absolute responseを返すとき必要。未取得ならdelta tableだけを返す |
| 合法方向、bounds、gauge、precision、cache | 両法に同じ自由度と再利用を許す。候補のtruthやfinite-h正解を入力しない |

出力は `g,Q`、そのprecision/interval仕様、再利用可能なcache。d0を接続すれば `d(x)=d0+gᵀx+xᵀQx` の主要係数応答を返す。finite-h energy、overlap、量子費用の最適値は出力に含まない。

前回の固定式をそのまま利用する：

\[
\Delta d(X)=\tfrac1{12}\langle[A,[A,X]]\rangle_0+
\tfrac1{24}\langle[X,[A,X]]\rangle_0,
\]
\[
g_p=\tfrac1{12}\operatorname{tr}(\gamma[a,[a,n_p]]),\qquad
Q_{pq}=\tfrac1{24}\operatorname{tr}(\gamma[n_p,[a,n_q]]).
\]

非縮退groundのPF branchの主要固有値誤差として使う条件と、縮退時の単一期待値の限界は前回のまま。今回、BCH符号・PSD・body rank・縮退の証明を反復しない。

## 3. 最も強いcompact BCH対照を具体化する

対照B*は「一般BCHを全候補ごとに再計算」する実装ではない。既知のgrouped recurrenceを、ユーザーが許可した同じground恒等式とCAR閉包で**期待値取得前に特殊化**する。元提案のblack-box generatorを呼ぶ手順でもない。

1. 既知のgrouped S2 recurrenceを固定orderingのprefix/tail表現で保持する。初回d0は共通の最良合法backendから取得・cacheする。
2. A/Bだけを±Xで変更し、元とのdifferenceを取る。内側の未変更tail nodesを、state contractionの前に差から除く。このclassで変更されるgroupは外側二つだけである。
3. 前回の固定operator differenceを用い、`〈[H,K]〉0=0` を適用する。これにより `1/8[H,[X,A]]` の期待値を除く。残った二重交換子にone-body CAR閉包を適用する。
4. XをB frameのdiagonal formal variablesとして扱い、linear/quadratic係数をまとめる。dense n×n matrixのtrace/element演算としてgとQを生成する。n²個のnested commutatorをそれぞれdense計算する弱い実装にはしない。
5. a、γ、d0、g0、Qをcacheする。以後の候補評価と `g(x)=g0+2Qx` に同じcache規則を使う。

response取得のためにL個の未変更nodesを全てmaterializeする必要はない。尾部は未展開の共通記号/pointerに保ち、変更される外側二つのtemplateだけを使う。元Hのinput/preflightと初回d0のL依存費用は、両法共通の取得会計に残す。

既知grouped BCHはfragment sumsを使う再帰・cachingを備える。本比較で追加してよいground恒等式とone-body簡約を適用すると、上の手順から現法と同じ入力依存・出力が得られる。[Maxwellら III.2 Eqs.(9)–(12), III.2.2, IV.1](https://arxiv.org/html/2606.30738v1)

**ここでの静的帰着**は、両法に同じ恒等式を与えるというユーザー指定に基づく。論文や現在のgeneric PennyLane APIが、このexact閉式を既に印刷・自動認識するというclaimではない。現在のreleaseを実行・benchmarkしていない。generic softwareが未特殊化でも、許可されたB*の情報・計算量が現法へ追い付くことは否定されない。

### 3.1 出力を実際に生成するmatrix演算列

両法は共通のB frameで、例えば次の同一の演算列を選べる。これは静的仕様であり、実装・数値実行ではない。

```text
Input: Hermitian a, gamma in the fixed B frame, common precision policy
S := a times a
R := a times gamma
For each p:
  g[p] := (diag(gamma times S)[p]
           + diag(S times gamma)[p] - 2 diag(R times a)[p]) / 12
For each unordered pair r,s:
  w[r,s] := -Real(a[r,s] gamma[s,r]) / 12
  accumulate Q as the Laplacian with these edge weights
Cache g0,Q,d0 and shared acquisition/error records
```

diag-only productsはrow/column dot productsで取得できる。dense full matrix積はa²、aγなど一定数で、残るdiagとedge集約はO(n²)。同じprecision、演算順序、error propagationを使う権利があるため、有限precisionの扱いにも固有の情報差を置けない。実bit一致や実runtime倍率を測ったという意味ではない。

## 4. 情報・計算量・memoryの共通会計

nはspin-orbital/選んだone-particle representationのサイズ、Lはdensity fragment数、mは指定更新方向数、Kは候補数。`Cγ` は必要な1-RDM/one-body期待値の取得費用、`C0` はd0の取得費用、`S_H` は共通の元H/fragment/input storage。これらを0としない。下表のO表記はfixed-precision arithmeticで、bit長・精度・interval保証の共通費用は別である。

| 指標 | 現1-RDM応答法 | 許可されたB* | 独立差 |
| --- | --- | --- | --- |
| responseのRDM次数 | one-body/1-RDM | 同じone-body/1-RDM | なし |
| 必要な期待値情報 | 同じγのlinear functionals。full γ方式なら最大O(n²) real components | 同じ情報集合・取得精度で足りる | なし |
| d0取得 | C0を初回に払う。一般multi-fragmentの絶対d0をγだけで取得できるとはしない | 同じC0/backend/cacheを使用できる | なし |
| frame preprocessing | a、γを固定U_Bへ変換。dense O(n³) | 同じ変換 | なし |
| full g,Q生成 | constant-count dense products O(n³)、edge/diag O(n²) | 同じ演算列と次数 | なし |
| response追加memory | O(n²)。元input S_Hと量子state accessは別 | 同じO(n²)＋共通S_H | なし |
| K dense候補評価 | 各 `gᵀx+xᵀQx` O(n²)、全O(Kn²) | 同じmatrix-vector/scalar演算 | なし |
| fixed directionの再利用 | ℓ=g·y、q=yᵀQyを一回作れば、以後のtはscalar O(1) | 同じℓ/q cache | なし |
| 固定frame内の再配分反復 | Q一定、γ再取得不要、gradient更新O(n²) dense | 同じQ/γ/gradient cache | なし |
| L依存 | input読み/共通C0には残る。first-pair response取得後のg/Q生成には新L因子不要 | 同じtail除去により新L因子不要 | なし |
| m方向への投影 | Vがn×mなら、Vᵀg O(nm)、VᵀQV O(n²m+nm²)、保存O(nm+m²) | 同じprojectionを使える | なし |

full tableの場合の共通総生成・K候補評価の上限は

\[
C_\gamma+C_0+C_{\rm common\ input/frame}
+O(n^3)+O(Kn^2).
\]

この式はruntime/量子gateを無単位に足したscoreではない。Cγ/C0が同じなら、どの共通換算・重み付けを用いても、現法だけの生成差をそこから作れないという会計上の記述である。実単価・総resourceはUNKNOWN。

Qがsparse、γにspin/block構造がある、方向がNp−Nrefなどの少数成分である場合の節約も、両法に等しく許す。m≪nならfull γ/full Qを必須とせず、必要なone-body収縮だけを取得してよい。その節約も現法固有ではない。O(n³)が最適な下界というclaimもない。

初期d0のgrouped評価はO(L)程度のcommutator**groups**を用い得るが、それはtensor構築・量子測定・operator-state contractionの全費用がO(L)であることを意味しない。一般d0の高body-rank表現、特別classでの簡約、誤差interval取得は同じ条件で両法に残す。E0既知のtruth診断sourceをd0の無料取得器へ流用しない。

## 5. 少数の直接評価・既存再分割との対応

固定方向について `p(t)=d0+ℓt+qt²` が与えられ、合法な非zero sがあるなら、p(0),p(s),p(−s)から

\[
\ell=\frac{p(s)-p(-s)}{2s},\quad
q=\frac{p(s)+p(-s)-2p(0)}{2s^2}
\]

を回復できる。これは既知のquadratic interpolationで、実際の値を算出していない。m方向のdense full tableを汎用scalar oracleだけから復元するなら、混合方向も必要で一般にO(m²)点の情報が必要となる。一方向の「三点」をfull tableの「三点」としない。finite-precision/error intervalはsやs²で増幅され得る。

反復full-D評価だけの対照に比べ、解析tableが有利な場合はある。しかし最強B*は差を簡約してから同じ解析tableを生成できるため、三点対照の余分な測定・丸め増幅は、本法とB*の差にはならない。

| 既知手順 | 現法との対応 | 保持する相違・限界 |
| --- | --- | --- |
| [FFF](https://quantum-journal.org/papers/q-2023-01-03-889/pdf/), Eqs.(9)–(10), (14), 結論 | 一体成分の同じ移動自由度、可解fragment、covariance solve、Trotterへの応用 | measurement varianceとPF leading energyの目的は同一ではない。その目的差だけを独立generator改善としない |
| [Hamiltonian partitioning v2](https://arxiv.org/pdf/2210.10189v2), 2.3 Eqs.(37)–(42) | 同じone-body extraction、固定density frame、一体再対角化・合法fusion | reflection/L1/spectral heuristicは別目的。同じ自由度へstate-BCH評価を接続できる |
| [State-specific PF energy error](https://pubs.rsc.org/ja-jp/content/articlepdf/2025/dd/d5dd00185d?page=search), II Eqs.(5)–(9) | leading BCH期待値をpartition/error目的へ接続 | perturbative値をfinite-h厳密上界としない。exact state利用も両法で共通 |
| [Compact/grouped BCH](https://arxiv.org/html/2606.30738v1), III.2/IV.1 | 再帰、partial sums、expectation前のgrouping/caching | generic backendの未特殊化runtimeと、許可された数学的specializationを区別する |
| Ground交換子・one-body CAR・合法cache | 今回の1-RDM表とreuseをそのまま実現する共通手順 | exact state/sector/frameの条件を外す一般化は未検討。未知条件を差分と数えない |

一次本文はこの方法対応に必要な関連節だけ確認した。[文献identity/取得範囲](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/literature_registry.json)。固定local本文をhash照合して再利用し、公式web本文を確認した。RSC出版PDFはweb本文でありlocal SHAを捏造しない。一般文献探索・前回の数学証明再監査は行わない。

## 6. 同値範囲・異なる範囲と最終技術判定

**同値範囲**は、固定S2 ordering、最外側A一体とその隣接density B、XがB frameに対角、同じexact eigenstate情報・γ/precision・ground恒等式・自由度・cacheを使える場合の**leading response table生成**。同じd0を用いれば、同じ候補のleading係数、同じdirection/root/vertex等の選択情報も取得できる。将来数値最適化を実行したという意味ではない。

**同値と認定していない範囲**は、別PF ordering、非隣接fragment、異なる非可換Xの同時更新、frame自体の変更、有限h全log、高次残差、PF branch/overlap、actual gate/depth/compile、quantum ground-energy総費用。これらへ拡張して差を探さない。元の数学的正確性と27保存検査の支持範囲は保持する。

判定は `REDUCES_TO_KNOWN_WITHIN_FIXED_EQUAL_IDENTITY_ACCESS_COMPARATOR`。1-RDMの低情報rankと応答再利用はSUPPORTEDな能力だが、許可された既知B*も持つためexclusiveな改善ではない。固定precisionのmatrix-level scalingはCONDITIONAL、歴史的初出性・理論記録価値・有限h量子利益・研究正式採否はUNDETERMINEDとしてGPTへ残す。

前回の「compact対照も同じ費用に達するか未決」は、今回指定された強い対照を具体化したことで、この生成classについて解消した。前回の式や保存結果が誤りだったという判断ではない。既存報告・method_status・pilot案を編集せず、新記録を現在の技術所見とする。

## 7. 停止と保全

Phase Aの明示STOP条件に従いPhase Bを実施しない。scalar-corrected S2等の有限h比較・新pilotの情報価値分析を追加で展開しない。既存[未認可pilot案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/minimal_pilot_design.json)はread-onlyで保持し、scope/source/input freezeを進めない。

[GPT引継ぎ](qc_ground_pf_final_method_difference_gpt_handoff_v1_20261014.md)、[status/停止](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/method_status.json)、[404旧files保全](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/preservation_index.json)。旧148manifest entries/non-result fields/global description欠落FAIL、元worktree135paths、旧source/results/認可/STOPを保持。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、Track A/B、C1–C4を変更・再実行しない。

独立branch `qc-ground-pf-final-method-difference-audit-20261014` へcommit/pushし、[GitHub再取得receipt](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/publication/receipt.json)でbytes/SHAを照合後Mandatory STOP。同じ第一候補の追加技術監査を反復せず、GPTの研究継続判断の別明示開始承認を待つ。

## 公開確認（追記、2026-10-14 JST）

独立branchのpayload `df11cbe1f5a7084065efbaf3addff59993666c4e` をGitHub SSH経由で独立cacheへ再取得し、変更21blobs／1,597,973bytesをローカルcommit・worktreeとbyte/SHA照合した。[receipt](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/publication/receipt.json)。404旧files・固定9参照・148旧manifest records/non-result fields・既存global FAIL・元worktree135pathsも一致した。外部科学再現やcurrent CIのPASSではない。

Phase Aの限定技術監査完了、Phase B未実施、Mandatory STOP。新計算・新形式検査・旧27再実行・pilot準備/実行・第二候補・GPTレビュー・正式研究判断を開始していない。同じ第一候補の追加技術監査へ進まず、GPT重要科学判断の別明示開始承認を待つ。
