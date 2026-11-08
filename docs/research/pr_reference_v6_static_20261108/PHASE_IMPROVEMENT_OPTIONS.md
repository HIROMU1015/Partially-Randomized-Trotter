# V6：位相保証の候補・SOURCE対応・証明義務

[総括](REPORT.md)／[限定検証案](NEXT_VALIDATION_PLAN.md)。基準 **f2638a7845685a119619382b850441f5d8d33713**。本書は静的整理であり、改善した数値上界・新N・新policyの採用記録ではない。

## 1. すべての候補に共通する証明対象

F、Gは[assemble](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_qualification.py)が**同じcanonical finite-bit event列**から作る保存構成のlaneとする。gate literalとglobal_phaseを取得した後のbinary64値を実数として解釈し、

\[
S_F=g_F-\sum_{j\in\mathrm{RZ/RZZ}(F)}\theta_j/2,
\qquad S_G=g_G-\sum_{j\in\mathrm{RZ/RZZ}(G)}\theta_j/2,
\qquad S=S_F+S_G
\]

と置く。これはSOURCEのvacuum作用の位相であり、元Hのground energyから作る位相ではない。小数文字列へ丸め直した値、未正規化の理想scalar、別のevent順序の和へ取り替えない。

旧アルゴリズムは二つの逐次減算と最後の加算からΦ_oldを作る。各候補は少なくとも次を満たす必要がある。

- 同じF/G、実角、supported gate族とSOURCE inverseを維持し、ΦとSの差を囲む。F/Gのcompose順、q=1のcut、D融合、same-basis run、event符号を変えない。
- Φ−Sの誤差と、uncontrolled lane／controlled SOURCEの**上流normalization差**を分ける。Sを正確に求めても後者は自動で0にならない。
- 全finite-bit supportを覆う。rare trajectoryの期待回数・平均誤差や、有限fixtureの観測だけを一様保証にしない。
- 旧binary64最近接丸め・finite値・overflow/underflowなしの算術モデルを維持する。新しい丸め法を実装するなら、そのruntime適合を追加確認する。
- 新Φのrelative PがSOURCEと一致することを示す。P(Φ)→RZ(Φ)の既存解析恒等式は任意の実Φについてcommon joint phaseまで成立するが、actual角を保持するcompiler contextが変わっていないことが必要。
- 新しい保証だけを旧raw diskへ一度だけ投入する。SOURCE／event／finite-bit／native joint／PFの別誤差を消去したり、旧compiler diskを再加算したりしない。

以下で「静的に成立条件を確認できた」は、**明記した条件の下での構成や不等式**についての状態であり、新SOURCEのend-to-end資格を意味しない。

## 2. A群：回路・算術SOURCEを変更しない候補

### A1：時系列prefix別の誤差伝播

**状態：成立に追加の証明・検証が必要。** 現行SOURCEのまま適用できる候補だが、全supportのprefix列上界とSOURCE対応が未完成。

片側の逐次和をx_i=RN(x_{i−1}+a_i)、そのexact prefixをS_i=S_{i−1}+a_iとする。a_iは実際に取得するbinary半角と符号、S_0は取得済みglobal_phaseである。旧算術モデルでe_i=|x_i−S_i|、u=2⁻⁵³、e_0=0なら、

\[
e_i\le(1+u)e_{i-1}+u|S_i|,
\qquad e_L\le u\sum_{i=1}^{L}(1+u)^{L-i}M_i
\quad(|S_i|\le M_i).
\]

これは実数演算での誤差伝播の条件付き不等式である。一律M_i=Mを用いれば旧γ_L M型を超えない安全な証明へ戻せるが、**より小さいM_iがSOURCEから保証されて初めて有用な改善となる**。新しいM_i、総和の値、改善率は今回評価していない。

必要なSOURCE対応は次の通り。

| 構造 | 利用できる情報 | 追加証明が必要な点 |
|---|---|---|
| 決定論的D反復／D0融合 | 既存336n+24、full D0と二つのhalfのexact binary doubling、固定block順 | 各prefixで完了／途中のDを区別。fusionが次のstepにまたがる境界を正しく囲み、最後の24更新も落とさない |
| RTE order0/2／product | event当たり最大3 basis run、run当たりframeとinverseのRZ数、中央実角上界 | SOURCE順のrun途中のprefix、同basis共有、product順序を全event選択に対して囲む |
| basisの正負対 | exact角度和のnet相殺、途中のbasis絶対振幅 | 現行Mにも既使用。累積accumulatorの入出力がexactに戻るとは言えない |
| global／scalar／event | 取得時のgの有界性、powers of twoの同binary scalarのscaling、別normalization項 | gを未正規化の理想和へ置換しない。event依存丸めは独立ゼロ平均としない |

M_iを毎gate列挙して長round回路をbuildする必要はない。将来、SOURCEの有限block grammarとprefix上界から、反復blockの誤差伝播を閉形式／区間合成する実装が可能である。ただし「j番目のstepならMをj/n倍」とするだけでは、global_phaseの初期値、途中basis、融合境界を覆えない。block式とgrammarの証明が未完成なら数値診断へ入れずSTOPする。

F/Gの誤差上界をE_F、E_Gとした場合、最終加算も残る。安全には

\[
|\Phi_{old}-S|\le (1+u)(E_F+E_G)+u(|S_F|+|S_G|)
\]

を別に課金できる。二つの和の丸め符号が取消すという仮定は使わない。normalization項は旧値を維持する。

### A2：exact更新／条件付き可逆性の絞り込み

**状態：成立に追加の証明・検証が必要。** IEEEモデルでx±0がexact、適切な条件下の二数の減算がexactという局所的性質は使える。しかし、それをSOURCEの各到達accumulatorに結び付ける証拠が必要。

- θ/2や符号反転のexact binary scalingは旧SOURCEにも既使用。これだけで減算がexactとはならない。
- 0角を差し引く更新を丸め回数から除くには、全supportでそのliteralが0であることと演算に異常値がないことを示す。gateを回路から除く操作ではない。
- inverseの二つの更新が離れていたり、間に別角が入ったりする場合、逆順の実数操作であることだけではfloat residualは消えない。
- exact減算の条件を使うなら、入力二数の符号・大きさ関係と全到達範囲を証明する。有限例だけのbitwise可逆性を一般化しない。

局所exact性を一律に新採用せず、A1に証明済み更新だけ組み込む案である。実際に除外できる更新数と利益はUNKNOWN。

### A3：basis／inverseの丸め残差を無条件に取消す

**状態：既存条件では適用できない。** SOURCEの角度和が0であるだけで、逐次和の丸め誤差を0にする案は不成立。同様に、期待的な符号取消、√L型の確率誤差モデル、Haar平均を現在の全support保証に移す案も、固定条件を満たさない。これを救済するための新sampling lawや確率契約は提案しない。

## 3. B群：算術SOURCEを変更する候補

### B1：同じbinary入力のexact集約＋合計一回丸め

**状態：静的に成立条件を確認できた（条件付き構成）。** adapter、runtime適合、実際の新上界・RPE資格は未実装・未検証。

#### 最小変更の定義

F/G構築とglobal_phase setterを一切変えず、取得済みg_F、g_Gと実RZ/RZZ角literalをexactな二進有理数として読む。S_F、S_Gをfloatへ戻す前に合算し、**Φ_new=RN(S)** を最後に一度だけ返すproviderを置く。FとGで別々にfloatへ戻し、従来通り足す案とは区別する。supported gateのreject規則は維持する。

有限floatは`as_integer_ratio`等によりその値自身の有理数へ移せる。[CPython v3.12.3 fractions SOURCE](https://github.com/python/cpython/blob/v3.12.3/Lib/fractions.py)を限定確認した。`Fraction(str(x))`や出力小数の再読込を使う方式ではない。各入力はdyadicなのでexactな和は有限の整数算術で終了する。通常のFractionを最小参照実装に再利用でき、新たな科学管理frameworkや合成word policyは不要である。

#### 条件付き上界

旧phase_boundが片側の全exact prefixに与えるMから、終点も|S_F|,|S_G|≤Mである。入力変換と和がexactで、最終変換が旧binary64モデルの最近接丸めを満たせば、

\[
|\Phi_{new}-S|\le u|S|\le 2uM.
\]

従って旧normalization上界をB_normとすると、同じ位相対応に対する候補上界は

\[
B_{\rm phase,new}=B_{\rm norm}+2uM.
\]

**今回この式に保存M等を代入していない。** これはbinary入力の理想的な一回丸め構成の上界であり、現行SOURCEの上界をこの式に交換してよいという結論ではない。0のexact和を含み、underflowを新たに許す拡張はしない。旧のno-underflow前提を外す場合は絶対誤差項を追加する別scopeが必要。

最終有理数→float変換にも根拠が要る。[numbers.Rational.__float__](https://github.com/python/cpython/blob/v3.12.3/Lib/numbers.py)と[int true division](https://github.com/python/cpython/blob/v3.12.3/Objects/longobject.c)のpinned SOURCEを確認した。CPythonには正しい丸めを狙う整数経路があるが、small-integerの浮動小数点経路を含むruntime条件はなお必要である。SOURCE閲覧だけで現wheel／platformを形式検証したとは扱わない。将来の技術検証でexact有理数と返却floatの残差を独立に囲み、既存のround-to-nearest環境との適合を確認する。

#### 保持する誤差・公平性・費用

- g_F/g_Gは上流で既に丸め・正規化されたbinary値。そのcontrolとの対応に旧B_normを残す。event・D・scalarの解析理想値で上書きしない。
- F/GとF.inverse()は変更しない。新Φは同じ数学的vacuum相対位相を近似する。位相の新しい周期折返しも行わない。
- Pの新しい角度へ、固定compilerのsymbolic lowering／common JOINT phaseの解析資格を接続する必要がある。他gateのcorrectnessを変更角へ無条件転用しない。
- Reference固有の和だけを交換するならB′には同じproviderを挿入する場所がない。共通上流を変えないことを比較条件とし、共通変更を後からReferenceだけへ加えない。
- quantum gate数の削減ではない。classicalに全literalを走査し、整数加算と最終変換が必要。二laneの入力streamは連結して扱え、巨大な追加リストは不要だが、長roundの走査・big-integer費用を無料としない。タイミングやmemory量は今回評価していない。

この構成の一回丸め式は、実装前の限定的な「保証が実現できた場合」の科学screenに使える。そのscreenの利益を実装済みの利益と表示しない。

### B2：math.fsum／補償和

**状態：成立に追加の証明・検証が必要。** `vacuum_phase`の逐次loopを入力streamのfsumへ置き換えることは小さな変更候補。既存metadataのfsum利用を実phaseでの適用済み証拠にはしない。

[Python math.fsum公式説明](https://docs.python.org/3.12/library/math.html#math.fsum)は多重partialによる高精度な和を説明する一方、platformによる二重丸めの可能性も記載する。[CPython v3.12.3実装](https://github.com/python/cpython/blob/v3.12.3/Modules/mathmodule.c)にも、拡張精度中間値、compilerによるreassociation、途中overflowに関する前提がある。一般に「常にexact」「全platformで一回丸め」とは採用しない。

- binary入力の順序・supported familyを固定し、`g, -theta/2, ...`を渡す。C実装が必要とする中間演算条件とoverflowなしを全入力に対して確認する。
- F/Gを別fsumにした場合、二つの返却floatの丸めと最後の加算を残す。B1の2uM式をそのまま使わない。両laneのstreamを一つのfsumへ渡す候補は別のcallsite変更である。
- normalization／controlled相対位相の旧項は残す。途中でτへ折り返す変更は混ぜない。
- Kahan／Neumaier等を自前loopで追加する場合も、適合する誤差定理、最終補償加算、overflow、compiler/runtimeの条件を固定する。fsumと同じ保証と見なさない。
- classical走査費、partial保存費は生じる。B1と異なり既存libraryの利用で実装変更は小さいが、runtime条件を閉じる検証量は必ずしも小さくない。

今回はfsum／補償和を実装・実行せず、新SOURCEを採択しない。B1と並行候補として残すが、精度やshotの予想で順位付けしない。

### B3：pairwise／tree、固定高精度

**状態：成立に追加の証明・検証が必要。**

pairwiseの典型的な誤差評価はtree深さとΣ|a_i|に依存する。現行Mはbasis往復の数学的相殺を使うprefix上界なので、log深さという理由だけでpairwiseが旧γ_L Mより小さいとはいえない。tree順・全supportの絶対和・最後のF/G加算を固定する必要がある。並列和なら再現性も必要であり、並列化そのものを精度保証にしない。

固定p-bitの高精度和は、入力変換、各更新、最終binary64変換の上界とpの選択規則が必要。既存192-bit区間算術は**検証側**の資産で、phase providerが既に高精度という意味ではない。finiteなbinary入力を対象とする最小候補では、B1が新precision tuningを避けるため明確である。高精度方式を否定する負結果ではなく、今回の優先検証から外すだけである。

### B4：解析的block集約／periodic reduction

**状態：成立に追加の証明・検証が必要。** D net drift、basis真空作用、scalar、event寄与をblock解析式で集約できれば長いgate走査を減らせる可能性がある。ただしgate literalの実際の二進値、上流gの逐次更新、F/G cut、D0full/half、signed eventを再現する必要がある。元Hの解析vacuum energyやE0を代入する案ではない。

SOURCEのnormalization済みgと未正規化の解析和は、floatτのwindingsと丸めを含めて接続する。真の2πとτの差を無視した「同じmodだからexact」は**既存条件では適用できない**。新たな周期折返しを入れるなら、remainder回数、整数winding、周期誤差、最終Pのrelative phaseを全supportで保証する。Qiskit設定の変更やupstream builderの置換を最小B1へ混ぜない。

unconditional inverse laneのcommon scalarだけをさらに分離できる可能性はあるが、旧normalization項にはcontrolled SOURCEとの比較も含まれる。項の由来を個別に証明せず全B_normを削除することはできない。該当する鋭い全support boundの既存資格はUNKNOWN。

## 4. 選別と最小変更の比較

| 候補 | SOURCE変更 | 主な追加証明／検証 | 総RPE費用への経路 | 静的評価 |
|---|---|---|---|---|
| A1 prefix／block証明 | なし | 実順序の全support M_i、反復伝播、最終加算 | δ減→ceil境界→N減→全role再会計 | まず検討済み。証明未完成、改善数値なし |
| A2 exact更新の絞り込み | なし | 全到達入力に対するexact性、除外回数 | A1の一部として同上 | 条件付き。既使用binary scalingだけでは追加利益なし |
| B1 exact入力和＋一回丸め | providerとF/G合計callsite | 同じ入力の保持、最終RN、normalization維持、P context接続 | 同上。単価不変が前提 | **式と最小比較対象が明確。最初の条件付き数値screenに適する** |
| B2 fsum／補償和 | provider、必要なら合計callsite | library/runtime条件、中間overflow、最終加算 | 同上 | 小変更候補だが保証の移植を要する。B1と無理に同一化しない |
| B3 pairwise／高精度 | 和の順序／precision等 | absolute-sumやp-bit boundと最終変換 | 同上 | 効果UNKNOWN。現段階で検証を増やさない |
| B4 解析集約／周期処理 | provider、場合により上流 | 実literalとの対応、winding、全source phase | 同上＋classical overhead低減の可能性 | 最小案より変更・証明範囲が広い。未採択 |
| A3無条件取消／τを真2πと同一視 | 保証を欠く | 現前提を満たす証明がない | 正当な経路なし | **既存条件では適用できない** |

全候補について、δ改善がNの整数境界を越えない可能性、Nが下がっても総RZでB′を下回らない可能性がある。native保証の改善は新しいCT precision／domain／word保証を自動的に成立させず、総Tの改善も未確定である。

今回の優先案は**変更なしの保証候補を先に精査した上で、B1の閉じた条件式を保存値だけでscreenする次scope**である。A1の追加証明が先に完成した場合は別の明示されたscopeとして比較可能だが、B1診断が不利だった際に自動で他候補へ探索を広げない。
