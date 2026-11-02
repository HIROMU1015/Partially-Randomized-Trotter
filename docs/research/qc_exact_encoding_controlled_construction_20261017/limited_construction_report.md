# Exact Encoding × Controlled Evolution：DF平方blockの限定構成と既知法への還元

2026-10-17 JST。Repository `HIROMU1015/Partially-Randomized-Trotter`。ユーザーが認めた一つの構成問題について、紙上の演算整理と固定SOURCE・一次文献の照合を行った記録。

**今回の分類：`REDUCES_TO_KNOWN_METHOD`。** 固定粒子数sector上のspin-free DF平方blockに対し、基準軌道の係数を粒子数恒等式で消す手順を追った。しかし、その演算は既知のDF symmetry shiftの特殊化であり、PR原論文の式(C14)から直接生成できる。後段も既存Gaussian・対角回転・controlled scalar処理である。独立した新生成規則は確認できなかった。

これはこの一案の限定分類である。fermion encodingやHamiltonian simulation全般のNO-GO、実装費の改善不能、研究の正式終了は認定しない。網羅的な新規性判定・重要科学レビュー・新定理の正式受理でもない。主案を別のclassへ差し替えず、ここでSTOPする。

関連資料：[固定証拠・静的検査](evidence_index.json)、[判断用ハンドオフ](decision_handoff.md)、[今回のユーザー指示原本](inputs/user_instruction_exact_encoding_construction_20261017.txt)、[公開照合記録](publication_receipt.json)。

## 1. 再利用した固定資料と承認境界

- 前回preflight `68a87f99e16cff3e20d256e0432b11e6f3fb9d8b` の[技術報告][preflight]、[SOURCE/API明細][inventory]、[証拠台帳][pre-evidence]、[公開receipt][pre-receipt]。各bytes/SHA256を照合し、API調査をやり直す代わりに確定した範囲を再利用した。
- その中に保存された[GPTレビュー原本][gpt-review]。取得済み原本のhashを維持し、§§7–10の入力条件・方法差・sectorと位相の留保を使った。レビューの方向推奨を、今回の規則の成立証拠と取り違えない。
- `56f1cc55c59ab7bc42c6b24e0953675de9b29b1d` の[④研究範囲][coverage]、[残る問題][landscape]、[GPT引継ぎ][old-handoff]、`d09ba606a9fceef806a9313e3f7c43cd21d19e3f` の[4系列引継ぎ][four]・[重複表][overlap]。
- [2026-10-17共通運用指示][instructions]と現ユーザー指示を適用。§8.4の資源・共有環境保護を維持し、新しいCPU/RAM/worker上限・共有環境変更は設けていない。

新規Hamiltonian生成、固有値・state・RDM等の科学数値計算、保存科学値の再解析、simulation、回路build、native compile、T合成、数値correctness test、新科学SOURCE、実行契約、authorization/grant/marker、研究採択はすべて0件。PDFのtext抽出、file hash、文書リンク・固定SOURCEの静的検査は科学数値評価ではない。

## 2. 一つに限定した対象block

### 2.1 classと選定理由

対象は、spinに同じ実対称一体行列 `A` が作用する、**単一のnumber-preservingなspin-free DF平方block**とする。

\[
 L=\sum_{p,q=1}^{m} A_{pq}\sum_{\sigma=\alpha,\beta}
 a_{p\sigma}^\dagger a_{q\sigma},\qquad
 B=c_B I+\kappa L^2,\qquad A=A^T\in\mathbb R^{m\times m}.
\tag{1}
\]

`m` は空間軌道数、`2m` はspin mode数。`κ` は符号を含む元blockの実prefactorであり、`κ d_p d_q` と `c_B` の単位はenergy、`tB` は無次元とする。分子・basis・rank・数値精度・実数係数の値は選ばない。DF因子の単位・prefactorを別の規約へ黙って変更しない。

このclassを選んだ理由は、前回確認済みのDF/Gaussian/diagonal primitiveに直接対応し、粒子数恒等式による操作省略を一つのblockで追えるためである。任意のinteracting Hamiltonianや非number-preserving Bogoliubov blockへ拡張しない。実対称・spin-freeという限定は今回の紙上対象の指定であり、研究全体のHamiltonianを変更・採択するものではない。

### 2.2 必要な入力

1. 元Hと全scalar（nuclear repulsionを含む）、block (1) の出自・prefactor・H内の位置。DF近似を使う場合は元integral Hと固定DFモデルを区別する。
2. fermion mode順、bit-endian、JW符号、Gaussianの行列・作用方向・位相規約。
3. 対象粒子数 `N0` と、求める元問題のgroundがそのsectorに属するという入力上の正当性。`N0` は古典E0や固有vectorを必要としない。
4. `A=W diag(d) W†` と対応するnumber-preservingなFock Gaussian `G`。係数から通常の対角化・Givens生成をする費用は残し、Gを新しい無料量子oracleにはしない。ここで対角化を実行したわけではない。
5. 時間と符号 `U_B(t)=exp(-itB)`、要求誤差、利用可能な状態access。供給されたstate copiesと準備unitary・逆を同一視しない。

初めからfixed-N stateが供給される場合と、状態準備が必要な場合の価格は異なる。新しい準備oracle、逆準備、RDM、gap等は追加しない。

## 3. 既知baselineが行う処理

\[
 \widehat N=\sum_{p,\sigma}n_{p\sigma},\quad
 D=\sum_{p,\sigma}d_p n_{p\sigma},\quad L=G D G^\dagger.
\tag{2}
\]

既存JW＋DF/Gaussianのbaselineは、G†、`exp(-itκD²)`、Gを作用させ、`c_B` をglobal/controlled phaseへ戻す。D²のoccupation表現は一体項と密度対積なので、通常のRZ/RZZ（controlled時は制御付き）で生成できる。[固定SOURCEのdiagonal primitiveとappend処理][ops]を静的参照した。mode順やGのtranspose等の具体API規約は[前回明細][inventory]の条件をそのまま必要とする。

このbaseline自体が既知構成である。さらに強い対照は「粒子数を利用する既知のsymmetry shiftを行った後、通常のcompilerで同じ対角回路を生成するもの」である。素朴なunshifted baselineにだけ勝つ可能性を、新算法の方法差とは呼ばない。

## 4. 検討した一案：基準軌道係数の除去

### 4.1 紙上の書換え

declared eigenmode順の最後の空間軌道 `p*=m` を基準にし、`s=d_m`、`Δ_p=d_p−s` と置く。この選択は最適化・scanではなく、式を一意にするための固定されたindex規則である。

\[
 D_s=D-s(\widehat N-N_0 I)
     =sN_0 I+\sum_{p\ne m,\sigma}\Delta_p n_{p\sigma}.
\tag{3}
\]

基準軌道のα・β両modeは同じ係数を持つので、ともに和から消える。`q_p=n_(pα)+n_(pβ)` はprojectorではなく、`q_p²=q_p` と置いてはいけない。以下は個々のspin occupation `n_i²=n_i` で展開する。

`P_N` をfixed-N0 projectorとすれば、`D_s P_N=D P_N`。`D` と `N` は可換なので、

\[
 (D_s^2-D^2)
  =-2sD(\widehat N-N_0 I)+s^2(\widehat N-N_0 I)^2,
 \qquad (D_s^2-D^2)P_N=0.
\tag{4}
\]

したがって同じsector上では、`c_B+κ G D_s² G†` と元のBが同じ作用を持つ。これは本検討の局所的な紙上式変形であり、独立した新定理・一般数値保証の受理ではない。

### 4.2 exact encoding/isometryと作用範囲

ここで使うencodingは通常のJWである。`J_N` を抽象的なN0-electron物理spaceから、2m-qubit JWのN0-occupation codeへのisometryとし、

\[
 J_N^\dagger J_N=I_N,\qquad J_NJ_N^\dagger=P_N,\qquad
 B_{s,\mathrm{JW}}J_N=J_N B_N.
\tag{5}
\]

よって対象sector上の時間発展と、control両branchの対応を要求できる。ここでは小qubitレジスタへの新しい符号化・圧縮は構成していない。基準軌道のqubitを物理的に削除したのでもない。Gは依然としてそのmodeを含む2m modeに作用する。

sector外では (4) は一般に0ではなく、全Fock空間のunitary一致は主張しない。これはtarget sector上で正確な別extensionであり、元Hの置換近似ではない。ただし入力stateに未知のsector外成分がある場合、その成分の信号は変わり得る。本案はその誤差を0とみなさず、今回の適用域外とする。元Hのground sectorが未確定なら、ground-energy taskへの適用も未確定である。

### 4.3 既知手順としての回路生成の具体化

新SOURCEを実装せず、検討した書換えから通常の回路がどう出るかを記録する。`S={(p,σ):p≠m}`、`Δ_i=Δ_p` と置くと、

\[
 Q_s=c_B I+\kappa D_s^2
     =a_0I+\sum_{i\in S}a_i n_i+\sum_{i<j\in S}b_{ij}n_in_j,
\tag{6}
\]

\[
 a_0=c_B+\kappa s^2N_0^2,\qquad
 a_i=\kappa(\Delta_i^2+2sN_0\Delta_i),\qquad
 b_{ij}=2\kappa\Delta_i\Delta_j.
\tag{7}
\]

通常のoccupation-to-Z変換を行えば、

\[
 C=a_0+\tfrac12\sum_i a_i+\tfrac14\sum_{i<j}b_{ij},\quad
 z_i=-\tfrac12a_i-\tfrac14\sum_{j\ne i}b_{ij},\quad
 z_{ij}=\tfrac14b_{ij},
\quad Q_s=CI+\sum_i z_iZ_i+\sum_{i<j}z_{ij}Z_iZ_j.
\tag{8}
\]

時間順の擬似手順は次のとおり。**既知規則の特殊化を説明する非実行仕様であり、新しい候補SOURCEや実行契約ではない。**

```text
Given the declared DF factor, N0, mode order, t, and scalar assignment:
  retain the existing Gaussian G and its inverse;
  set s = d[m], and form the symbolic coefficients (7), (8);
  apply G† to the system, without external control;
  apply controlled RZZ(2 t z_ij) for each exactly nonzero pair;
  apply controlled RZ(2 t z_i) for each exactly nonzero mode;
  apply Phase(-t C) to the external control, active on control=1;
  apply G to the system, without external control;
  return the gate description together with the scalar and error ledger.
```

`RZ(θ)=exp(-iθZ/2)`、`RZZ(θ)=exp(-iθZZ/2)` の規約である。全対角項は可換なので、上の固定列挙順に新しいPF誤差はない。control=0ではG†とGが戻り、control=1では `G exp(-itQ_s)G†` になる。Gを無制御で挟む手順も既知のcontrolled conjugationであり、今回の差分に数えない。余分な計算ancilla、encode/decode circuit、逆状態準備は要求しない。

near-zeroを結果に応じて切り捨てない。式により正確に消える係数と、有限精度で小さくなった係数を区別する。既存APIの名前・機能を使って擬似手順を書いたことは、現在のSOURCEがこのN0依存の係数台帳をそのまま受理するという動作確認ではない。

## 5. 保存量、scalar、誤差の台帳

### 5.1 保存する量とlogical block境界

選んだspin-free G、L、D、N、D_s²は、logical操作としてN、Nα/Nβ、Sz、S²を保存する。特に同じ空間軌道回転を両spinへ施すという仮定を使う。証明に必要なのはfixed-N0であり、Nα/NβやS²の固有値を新たに指定・採択していない。

一般のN保存変換がNα/NβやS²も保存するわけではない。本classの外へ推論しない。個々のnative分解gate、あるいはα側・β側を分けた中間列が常にS²を保存することまで要求しない。Gの完了時と対角blockの完了時に必要な対応が戻ることと、回路全体のcontrolled作用を区別する。

Pauli Z2 taperingはparity sector、GSEは指定code、Paldusはspin適応変換であり、fixed-N0 codeと同じものではない。別encodingへ移すにはそれぞれのisometry、stabilizer、stateと全logical blockの対応が別途必要である。今回はそれらを重ねて新候補を作らない。

### 5.2 絶対energy原点

`a_0` だけでなく、n→Z変換に由来する和を含めたCをcontrolへ戻す。`c_B` は元の割当てを保持する。元Hのnuclear repulsion等の全体scalarは、既存scheduleで一度だけ割り当て、Bへ重複追加しない。今回の `κs²N0²` を理由に既知energy offsetを落としたり、後から位相合わせをしたりしない。

数学上、正しい式 (3)–(8) を使うと、対象sectorのphaseは元Bと一致し、追加の未知offsetはない。実装時はscalar、回転半角、G/G†の位相規約を明示する必要がある。[既存のcontrolled diagonal処理][ops]がphaseを扱う事実を再利用し、新しいphase primitiveとは主張しない。旧grouped pathのidentity skipを、この用途に無条件に流用しない。

### 5.3 有限精度実装で別管理するもの

| 層 | 必要な記録・条件 | 今回の状態 |
| --- | --- | --- |
| 元modelとDF | rank/truncation・保存binary64入力と元integral Hの差 | 入力契約の問題。新しいモデルや精度を採択しない |
| Gの生成 | Aの対角化誤差、mode/phase規約、Givens合成誤差とG†の逆関係 | 既知生成法を使用する設計だけ。未実行 |
| 係数書換え | s、Δ、(7)–(8) の丸め・相殺・scalar誤差 | 大きなN0や相殺で有利とは限らない。予算値は未設定 |
| controlled対角回路 | RZ/RZZ合成、control phaseの絶対誤差、近似zero除去の扱い | 未検証。新しい許容値を固定しない |
| PF | 元のordered factor・時間係数を保持すること | exact sector作用が同じなら、そのsectorでPF biasを改善したことにはならない |
| state・統計 | fixed-N0入力の条件、準備費・overlap・confidence・読出し | copiesから準備unitary/逆が与えられたとしない。shotsを予測しない |

たとえば係数台帳から `||Q̃_s−Q_s||≤B_coef`、Gの同一規約での誤差 `ε_G`、各回転のoperator誤差が別に与えられるなら、局所unitary差を `2ε_G+|t|B_coef+回転/phase実装誤差の和` で会計する通常の設計が考えられる。これは未採択の記号的な十分予算の形であり、coverageや実platformのcertificateを得たことではない。係数の実データ評価、浮動小数点誤差の認証は行っていない。

## 6. 省略され得る演算と追加費用

数値評価をせず、代数的なgate候補の支持集合だけを比較する。すべての未shifted係数が一般に非零の場合、通常のdiagonal展開は2m modeの対を持つ。(3)では二つの基準spin modeの係数が0なので、生成する対は残り `2m−2` modeの中だけになる。

| 項目 | unshifted DF展開 | 検討した書換え | 留保 |
| --- | --- | --- | --- |
| 密度対／ZZ候補 | `binom(2m,2)` | 高々 `binom(2m−2,2)` | 記号的なdense支持の比較。実gate数の測定ではない |
| 一体／Z候補 | 高々2m | 高々2m−2 | 残る角度は変化する。元からzero/sparseなら増減は別 |
| Gaussian G/G† | 元の2m mode回路 | 同じ回路 | 基準modeをGaussianから削除したとしない |
| control scalar | 元blockのC | (8)のC | 両方に必要。合成精度・角度が変わる |
| workspace・qubit | 既存JW系とcontrol | 同じ | qubit削減・code準備費削減はない |
| 古典生成 | 既知対角化＋二次個数の角度出力 | 同じ対角化＋Δとscalar計算＋二次個数の出力 | 漸近的な生成時間の改善を認定しない |

省略の機構は「exact-N0の恒等式で生成係数を変え、正確に0の対角回転を出力しない」ことである。残る係数が大きくなる、元のsparsityを失う、精度やroutingの費用が増す可能性もある。native T/CX数、controlled depth、倍率、break-evenの数値は得ていない。

metric gごとの総費用を下げるには、同じtask・state access・誤差・成功確率で、共通準備等を戻した費用差が負であることが必要である。同一signalが保証され、同一精度で実装される場合にのみshot scheduleを共通にできる。今回の支持集合の縮小だけでshots・全round RPE・総資源改善を認定しない。より強い既知shift baselineが同じ省略を行える点が、次節の決定的な比較である。

## 7. 既知法への直接的な対応

### 7.1 決定的な一次文献：PR原論文の式(C14)

Güntherほかの [arXiv:2503.05647v2 Appendix C.3、式(C14)][pr-paper] は、各DF平方に対し、平方内へ粒子数差を加える自由parameter `f_j` を導入している。その式で `f_j=−s` と選べば (3) と同じ平方を得る。prefactor・orbital変換の向きを入力規約に合わせれば一致し、基準係数の消去はそのparameter選択と通常のexact-zero除去である。

これは「論文で同じnative最適化・pivot選択の性能が測定済み」という主張ではない。**実装する演算とそれを得る生成familyが既報に含まれる**ため、今回の独立した手順差が残らない、という対応である。Appendix C.1のDF、C.3のshiftを必要箇所として取得・読解した。本文全体の結果を再認証せず、論文の数値利益やstate仮定を本案へ移していない。

### 7.2 BLISSとの紙上対応

`x=N̂−N0 I` とすると、元blockから書換えblockを引く量は、Gによる元basisへの復帰後、

\[
 T=B-B_s=2\kappa s Lx-\kappa s^2x^2
 =2\kappa s Lx-\kappa s^2(\widehat N^2-N_0^2I)
   +2\kappa s^2N_0x.
\tag{9}
\]

Loaiza–Izmaylovの [arXiv:2304.13772v2 §II、式(6)–(9)][bliss-paper] は、number差とその二次式、およびnumber差に可換な一体operatorを掛けるshiftを扱う。ここではLがNと可換で、Aも実対称・spin-freeなので、(9)はその範囲に入る。任意の非number-preserving Fにまで適用していない。

BLISS原論文の主な価格指標やLCU実装は、今回D1内で新研究として使うものではない。既知のsector恒等式を対照として参照することと、block-encoding/qubitizationを新たに採択することは別である。

### 7.3 その他の対照・既存研究

| 必須対照 | 今回の具体的な違い／包含 | 判定上の役割 |
| --- | --- | --- |
| 既存JW＋DF/Gaussian | unshifted diagonal係数を(7)へ変更。G/G†とcontrolled scalarの生成は同じ | 素朴baselineに対する省略機構は具体的。ただし新規性の根拠にはならない |
| 既知shift＋通常compiler | (C14)で `f=−d_m`、定数・一体・対の通常展開、exact-zero除去 | **同じ生成手順を得る強い対照**。独立差分なし |
| tapering／ParityMapper | 異なるparity codeへの写像。追加のisometryとblock対応が必要 | N固定と混同しない。この書換えを独立化する機構は今回ない |
| BKSF／GSE | graph/codeとeven algebraの既知表現 | 本案はgraph・Majorana生成則を変更していない。写し替えだけを新法としない |
| Paldus | 既知のspin適応変換・指定演算 | 本案はspin-freeな既知number shiftであり、新しいGT/CSF上の回路規則を出していない |
| SCDF | [arXiv:2403.03502v1 §§2.3–2.4][scdf-paper] にDFとsymmetry shiftの結合がある | 近接性の補助証拠。同論文の二次tensorへのshiftと(3)が全く同じだとは主張しない |
| ② Gaussian積／basis fusion | Gを変更せず、ordered factorも保持 | active-span・反射積・隣接相殺の新規再提案ではない。旧closureを維持 |
| ④ N1–N3、THRIFT、H1–H4等 | 新selector、別active H、frame変更、分割・順序変更はしない | 停止系列を再開しない。既知還元後に候補探索を続けない |

taper/GSE/Paldusと上流APIの適用条件は[preflight][preflight]と[明細][inventory]を再利用した。今回追加で本文を取得したのは、この一案の還元を判定するための上記3論文の必要部分だけである。①の進捗評価・追加検証、③の保存結果再解析は行っていない。

## 8. 限定結論、未検証事項、次の担当

| 分類候補 | 今回 |
| --- | --- |
| `CONCRETE_CANDIDATE_FOR_SCIENTIFIC_REVIEW` | 選ばない。独立した新生成規則として提出できる差は残らない |
| `REDUCES_TO_KNOWN_METHOD` | **選択**。式(C14)への直接対応と通常生成手順で説明できる |
| `NO_CONCRETE_GENERATION_RULE` | 新規則は得ていないが、検討操作の具体式は書け、既知還元まで確認できたので主分類にはしない |
| `UNDETERMINED_MISSING_EVIDENCE` | native性能等は未検証だが、今回の既知familyへの包含を妨げる不足ではない |

独立した新候補がないため、別ファイルの「新生成規則仕様」は作成しない。(3)–(8)と擬似手順は、主報告中の既知還元の追跡記録として残す。

未検証は、実DF入力でのcorrectness・sector品質・finite-precision coverage、native費用、最適parameter選択、総資源、他の未提示生成規則の存在と独立新規性である。出版版や全関連文献の網羅的差分監査、過去の未取得Claude原本の全件回収もしていない。これらの欠測を「一般に無利益」や「新法の余地がある」という認定へ変換しない。

本案だけを理由に新SOURCE・数値実験・重要科学レビューへ進む必然性は得られなかった。次の担当はユーザーであり、この限定結果を確認し、別の具体的な構成課題を依頼するか判断する。追加の候補class・新しい科学条件・研究の終了/継続方針をCodexが採択しない。

文書・provenanceの静的検査と公開同一性を確認して終了する。**新科学計算0、新科学SOURCE0、未承認作業0。mandatory STOP。**

[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a87f99e16cff3e20d256e0432b11e6f3fb9d8b/docs/research/qc_exact_encoding_controlled_preflight_20261017/technical_preflight_report.md
[inventory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a87f99e16cff3e20d256e0432b11e6f3fb9d8b/docs/research/qc_exact_encoding_controlled_preflight_20261017/source_api_inventory.md
[pre-evidence]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a87f99e16cff3e20d256e0432b11e6f3fb9d8b/docs/research/qc_exact_encoding_controlled_preflight_20261017/evidence_and_checks.json
[pre-receipt]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a87f99e16cff3e20d256e0432b11e6f3fb9d8b/docs/research/qc_exact_encoding_controlled_preflight_20261017/publication_receipt.json
[gpt-review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a87f99e16cff3e20d256e0432b11e6f3fb9d8b/docs/research/qc_exact_encoding_controlled_preflight_20261017/inputs/qc_algorithm_strategy_ii_iii_iv_scientific_review_20261017.md
[coverage]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md
[landscape]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/open_problem_landscape.md
[old-handoff]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/gpt_decision_handoff.md
[four]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d09ba606a9fceef806a9313e3f7c43cd21d19e3f/docs/research/qc_four_track_strategy_handoff_20261017/strategy_handoff.md
[overlap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d09ba606a9fceef806a9313e3f7c43cd21d19e3f/docs/research/qc_four_track_strategy_handoff_20261017/overlap_and_progress_map.md
[instructions]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/research_project_instructions_unified_2026-10-17.md
[ops]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/df_trotter/ops.py#L246
[pr-paper]: https://arxiv.org/pdf/2503.05647v2
[bliss-paper]: https://arxiv.org/pdf/2304.13772v2
[scdf-paper]: https://arxiv.org/pdf/2403.03502v1
