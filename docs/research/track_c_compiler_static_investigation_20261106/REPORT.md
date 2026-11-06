# Track C：native compiler誤差の静的原因調査と対処案

2026-11-06 JST、SERVER 2 `GPUA100-gleap02`。調査基準は
`a78054dc48737b06b9396e649f53ecb9efd4bd3b`。
専用branch：`codex/track-c-compiler-static-investigation-20261106`。

**最小再現の消失原因は特定した。新compiler構成のcorrectness・価格は未検証であり、今回の成果は採択前の対処案である。**
既存SOURCE・保存結果・QASMの閲覧とcompiler実装の静的確認だけを実施した。
Qiskitのimportによる調査、pass manager生成・実行、回路build/compile、シミュレーション、sampling、価格取得、T合成、テスト起動は行っていない。
旧run01とそのSTOPは変更せず、未完了セルも再開していない。

## 1. 結論

最小再現の直接原因は、**Rustの交換判定が小さい回転をidentity同様に扱い、本来は非可換なgateを越えた相殺を許すこと**である。
`CRZ(1e-5)`を分解した二つの`RZ(±5e-6)`がこの判定に入り、CX二つとRZ二つが同じ交換集合内で相殺される。
単に「小さいCRZを直接削除するpass」ではない。

これとは別に、回転合算後の角度を`1e-5`で省略するRustの判定もある。
片方だけを変更しても、精度を保つcompiler全体にはならない。
`approximation_degree=1.0`やPython側の同名定数の変更では、これらRust側の固定値を変更できない。

推奨は、**既存gate定義による変換と正確な固定gateの逆相殺を使う参照構成を先に検証すること**。
その後、局所的な誤差を明示して受理できる最適化だけを戻す。
旧opt2から一つのpassを除いただけの構成を、精度検証済みとは扱わない。

H4 Full Y4の保存された差は事実として引き継ぐが、その差の全量をこの交換判定だけに帰属させる証拠は未取得である。
全RPE総G・native成功率・候補の優劣は、引き続き未資格。

## 2. 調査した固定資料と実装の対応

旧pilotはLinear H4、隣接1.00 Å、STO-3G、DF12、生成順LD3/Full LD12、元Hの厳密ground供給仮定・状態準備費除外。
`epsilon_E=1.5936e-4 Ha`、失敗確率0.05、全位相予算0.4 rad、delta0=.02、18 roundの条件を変更していない。

| 資料 | 確認した内容 |
|---|---|
| [旧REPORT][old-report] | 三pilotの条件、価格の部分取得、native比較が未資格であること |
| [最小再現の保存trace][old-min] | BasisTranslator後はRZ二つ・CX二つ。続くCommutativeCancellationで空回路へ変化 |
| [NATIVE_QUALIFICATION][old-qualification]、[STOP][old-stop] | 未計上のlowering誤差、run01停止、再開不可の状態 |
| [native検査SOURCE][old-check]、[価格SOURCE][old-price] | 実際のtranspile引数、測定を残す入力、検査が単一入力状態に対するものと分かること |
| [保存QASM bundle][old-qasm] | 2本の出力QASMをテキストとして閲覧。9 qubit・1 classical bit・末尾のancilla測定を確認。再parse・再シミュレーションなし |
| [C-S合成SOURCE][old-synthesis] | strict guardはQ変換後のRZ→Clifford+Tを検査し、Q変換前に失われた作用は復元しない |

保存されたX期待値差はPartial S2のq1/j0で`3.6637359812630166e-14`、Full Y4のq1/j0で`1.4077178827620607e-5`。
Full Y4のprojective状態差は`2.1933549047339093e-4`。
これらは旧値の引用であり、新たな数値再解析ではない。[Partial S2記録][old-s2-check]、[Full Y4記録][old-y4-check]。

QASMは最終出力なので、削除されたgateや相殺前の交換集合は復元できない。
OPENQASM 2.0の記録だけからjoint global phaseまで含む厳密なoperator履歴を再構成することも主張しない。
旧H4検査は512次元joint space内の**指定入力状態の検査**であり、512次元の全operator検査ではない。

SERVER 2の旧専用venvのMETADATA/WHEELはQiskit 1.3.0、cp39-abi3 manylinux x86_64。
Pythonの`commutative_cancellation.py`、`builtin_plugins.py`、`preset_passmanagers/common.py`は公式1.3.0の同ファイルとbyte一致した。
Rust実装は公式tagをcommitまで解決した
`Qiskit/qiskit@896d1c8b72888a52a46081a6d1f4de8c56db8aea`を参照した。
インストール済みバイナリの再ビルドや実行による対応検証は行っていない。

## 3. 消失を起こす具体的な条件

### 3.1 呼出し経路

Pythonの[`CommutativeCancellation.run`][cc-python]は、Rustの`cancel_commutations`へ処理を渡す。
Rustは[`analyze_commutations_inner`][analysis]でwireごとの交換集合を作り、その中の同種gateを相殺・合算する。
交換の可否は[`CommutationChecker.commute_inner`][checker-main]が決める。

`commute_inner`の固定値は`rtol=1e-5`、`atol=1e-8`。
ここで`rtol`は行列比較だけでなく、`map_rotation`へ角度の閾値として渡される。
[`map_rotation`][checker-map]の条件は次のとおり。

```rust
if (angle % TWOPI).abs() < tol {
    return (op, params, true);
}
```

このtrueを受けると、`commute_inner`は直ちに交換可能と返す。
gate種類の制限、通常のprecheck、正確な標準gate交換表、cache、行列比較より前の処理である。
したがって標準交換表やcacheだけを変えても、この分岐を修正できない。

### 3.2 CRZ最小再現との一致

1. [CRZの既存定義][crz]は、時間順に`RZ(theta/2), CX, RZ(-theta/2), CX`を生成する。
2. 保存traceでも、BasisTranslator後のRZ二つ・CX二つまでは作用素差が約`1.1e-16`だった。
3. `theta=1e-5`なのでRZの角度は`±5e-6`。両方が`abs(angle % 2pi) < 1e-5`を満たす。
4. 交換判定は、CXのtarget上にあるこれらRZもCXと交換できるものとして集合を作る。
5. [`cancel_commutations`][cc-groups]は集合内のCX二つを自己逆として削除し、同じwireのRZ二つを合算する。合算角は0で、RZも消える。

この経路は、旧traceの「最初のCommutativeCancellationではCRZが残り、basis変換後のCommutativeCancellationで空になる」という順序も説明する。
元CRZの角度が閾値と等しいことでは防げず、分解された半角が閾値未満になる点が本質である。

理想CRZの固有値には`exp(±i theta/2)`が含まれるため、小さいthetaでのidentityとの差は`2 abs(sin(theta/4))`。
保存値`4.999999999994792e-6`と整合する。これはjoint global phaseを合わせても消えない、controlled回路の物理的な相対位相の差である。
本段落はSOURCEと保存traceに対する解析的説明であり、再現実行の追加ではない。

### 3.3 独立に存在する合算角cutoff

Rustの[`commutation_cancellation.rs`][cc-cutoff]にも`_CUTOFF_PRECISION=1e-5`がある。
長さが2以上の回転集合について、合算角を2piで正規化し、その絶対値が閾値より大きい場合だけ新しい回転を挿入する。
閾値以下では回転を再挿入せず、元のgateを削除する。phase補正もこの処理の一部である。

最小再現では合算角がちょうど0なので、**このcutoffを0へ変更しても、誤った交換で生じた相殺は止まらない**。
逆に交換判定だけを直しても、非ゼロの合算角を省略する誤差は別に残る。
Pythonに残る`_CUTOFF_PRECISION`は、このRust関数の引数ではない。Python定数のmonkeypatchを修正案にはしない。

### 3.4 presetと他の数値処理

| 実装 | 静的に確認した点 | 対処時の意味 |
|---|---|---|
| [preset init][preset-init] | level2/3のinitにもCommutativeCancellationがある | optimization段だけの除去では不十分 |
| [preset optimization][preset-opt] | level2の反復ループにも同passがある | 再帰・反復を含む実pass構成を固定する |
| [RemoveIdentityEquivalent][remove-id] | degree=1でもfidelityと機械epsilonを比べて削除する | fidelityの二次誤差とcoherent作用素誤差を同一視しない |
| [Optimize1qGatesDecomposition][euler-call] | Rust内部はsimplify=true、atol未指定で呼ぶ | `ANGLE_ZERO_EPSILON=1e-12`を使う簡略化も別途残る。[角度処理][euler-angle] |
| [ConsolidateBlocks][consolidate] | 浮動小数点行列とidentityの近接判定でblockを除去する箇所がある | CC除去だけを全変換の精度保証にしない |
| [translation段][translation] | translator指定でもUnitarySynthesis、HLS、BasisTranslatorの列になる | level0/1への変更も実際の経路を確認する |

`approximation_degree=1.0`は既存のpresetで既に使われる既定値であり、交換判定・合算cutoffのRust固定値を制御しない。
他の数値処理を、今回のH4誤差の原因と確定したわけではない。精度保全構成を設計する際に残る確認事項として挙げた。

## 4. 対処方法の比較

全方法で、基底状態準備費の除外、ordinary controlled回路の意味、元のH4入力・精度・成功条件は維持する。
新しい回路価格や優劣は、今回は取得していない。

| 方法 | 物理correctness・相対位相 | 価格・長いRPEへの影響 | baselineとの比較 |
|---|---|---|---|
| **A：gate定義に基づく参照構成。推奨する最初の検証対象** | 既知gateの分解・BasisTranslatorを使い、角度の小ささを理由に交換・省略しない。固定Clifford等の正確な逆相殺に限定する。数値丸めは別に残す | default opt2と価格が変わる。価格増減は未測定。長qも省略閾値に依存せず、残る丸めの伝播を扱える構成を目指す | 全候補と強baselineを同じ構成で価格化。参照構成を安さの勝者と見なさない |
| **B：精度を確認しながら最適化を残す構成** | 正確な交換関係と非ゼロ角の保持を基本に、数値再合成等は局所作用素誤差を予算内と確認できたrewriteだけ受理する。CCを外す場合はinit/optimizationの両方が対象 | D/D・basis逆変換・隣接gate融合の機会を残せる。小行列で評価可能な変更に限定し、全回路の巨大行列監査を必須にしない | 強加法側にも同じ最大D runの融合と受理基準を与える。新profileで再較正が必要 |
| **C：現opt2の近似を残し、誤差を明示会計** | 削除だけでなく、近似的な交換・再合成・phase補正を含む差の上界が必要。旧traceのgate数だけでは足りない | 長qで誤差がcoherentに累積し得る。C-Sはweight/rangeが増幅するため特に厳しい。予算が閉じなければそのarmは不適格 | profileが本当に同一で条件も一致する旧価格は参照可能。ただし新N/Gと適格性は再評価が必要 |
| **D：連続角をsymbolicのまま変換し、最後に代入** | 既存Parameter/ParameterVector、gate定義、assign_parametersを利用。数値小角の分岐を避けられる候補。最終代入後に同じ数値optimizerを通すと問題が戻る | 値依存の相殺や数値再合成を抑えるので価格は別物。式・scalar phaseの丸めは別途必要。未実行であり安全性を確定しない | 全armへ同じsymbolic化・正確な融合機会を与える。新しい価格cohortにする |

level0/1を診断用参照にする案もあるが、preset名だけで精度を保証しない。
privateなRust修正・別Qiskit版への移行も技術的な候補だが、今回の調査は1.3.0に限定し、修正済みの別releaseを特定・検証したとは言わない。
共有site-packagesの書換えは採用しない。

## 5. 推奨compiler構成の具体案

最初に採択を提案するのはAであり、Dの角度保持を必要な箇所で併用する。
**以下は実装前の構成案であり、採択済み・compile済みではない。**

1. Qiskit 1.3.0、`rz/sx/x/cx`、topology/layout/routing指定なし、seed17を引き継ぐ。既存DF/RTE builder、固定入力、scalar相対位相を再利用する。
2. 既知gateの定義・equivalenceに沿ってlowerする明示的なpass列にする。数値Unitaryへの集約を前提にせず、未対応gateはそこで報告する。
3. CommutativeCancellationとRemoveIdentityEquivalentを使わず、未資格の数値Euler/KAK最適化も参照列には入れない。単にpresetのoptimization段を空にするだけではなく、initとtranslationを含めて列を固定する。
4. 既存InverseCancellationを使う場合は、CX/X/H等の固定自己逆gate、固定逆gate対へ限定する。連続角の近似一致でgateを消さない。
5. 連続角を保持するため、必要なら既存Parameter系で変換を行い、最後に登録角を対応表で代入する。scalar/controlled phaseと共通の角度関係も保持する。代入後は数値gate除去を起動しない。
6. 正確に表現できる同角・逆角の融合や、実adjointとの構造的相殺を回復する場合も、丸めを伴う一般の角度加算とは区別する。Bの最適化は、Aとのcorrectnessを確認してから追加する。
7. C-SのT合成は、資格を満たしたQ変換の後だけに接続する。既存strict guard・actual adjoint・whole-interrogation誤差式を再利用する。

symbolic化は数値精度の万能な解決ではない。[assign_parameters][assign]は値の代入APIであり、代入時の式評価や、opaqueな行列gateの分解まで包含保証するものではない。
CCのRust処理はparameterized gateを相殺集合の処理から除外するが、これだけで全passの安全性が証明されたとは扱わない。[相殺対象の条件][cc-parameter]。
候補A/Dが現DFの全gateを扱えるかは、次の最小検証で確認する。

## 6. 誤差予算と相対位相の条件

compiler誤差を予算化する場合、gateを削除した角度だけを数える方法では不十分。
近似的にABをBAへ並べ替える誤差は`norm(AB-BA)`、その他のrewriteはphase補正を含めた新旧小回路の差として扱う。
ユニタリ積のtelescopingで、受理した変更・lowering丸め・合成の作用素誤差をfull interrogationまで伝播する。
小行列で扱うのは変更するrewriteであり、全gateの個別ログや巨大superoperatorを新設する方針ではない。

複素信号のcompiler diskには、旧契約と同じ保守形

`delta_comp <= 2 sqrt(2) E_p[abs(W_x) d_x]`

を使える。`d_x`は実現xについて両軸に有効なfull joint回路の作用素誤差上界。
raw測定とcorrected weighted測定のnormalizationを重ねず、C-Sでglobal rangeを使う場合は
`delta_comp <= 2 sqrt(2) L max_x d_x`が十分な形となる。
正の参照半径rhoに対し、diskがrho未満であることを確認してから位相幅`arcsin(delta_comp/rho)`と半径減少を会計する。
既存の数値diskと同じ誤差を二重に引かず、全位相0.4 radと正半径条件へ接続する。
compilerに新たに割り当てる数値予算は、今回は確定していない。

同じunitary kernelを反復する構成なら誤差は最悪q倍に累積し得る。
全qを一体compileする場合にはq1の差を単純にq倍して保証にはできず、対象の実pass列・全supportに有効な構造上界が必要。
q1で小さい差や高いaverage gate fidelityが得られても、18roundの十分条件にはならない。
旧traceの`1e-10`は最初の問題passを表示する診断閾値で、RPE全体の許容compiler誤差ではない。

controlled-systemの`U`と`exp(i gamma) U`を同一視すると、ancillaの相対位相を失う。
system段での2pi回転の符号、scalar phase、負時間、実adjointを維持する。
既に構築したfull joint回路全体のglobal phaseを同一視することと、control前のsystem operatorをprojectiveに合格させることは別である。
最終測定に対する厳密な等価性でgateを省く場合は、X/Yの測定等価検査を使い、ordinary controlled unitaryと同一だとは主張しない。

C-Sの旧global rangeは極めて大きく、T合成の`1e-181`だけを満たしても、source→Qの丸め・変換誤差が残る。
通常のfloat64誤差をglobal Lで増幅する上界が閉じない場合、symbolic化だけで成功とせず、局所meanの構造を保つ誤差解析等を別途判断する必要がある。
今回はC-Sの新しい誤差計算や救済計算を行っていない。

## 7. 次に必要な最小検証：別途実行許可の対象

推奨する次の承認範囲は、**参照compilerの小差分実装とcorrectnessまで**。価格gridや三pilotの自動再開は含めない。

| 検証 | 再利用するもの・必要な追加 | 判定の対象 |
|---|---|---|
| 最小再現 | 既存compiler_minimal_reproと保存trace。compiler関数を差し替えられる最小差分 | CRZの消失を防ぐこと、RZ/CXの関係とphaseを保持すること |
| 人工の境界・相対位相 | 既存CRZ/controlled RZZ例、正負角、0、閾値の上下、2pi/4pi、scalar・実adjoint | 小さい2/3qubitの全operator。control前のprojective fitを合格条件にしない |
| H4短q | 既存のPartial S2 LD3とFull Y4 LD12のq1/j0、同じseed・入力、負時間D half参照を再利用 | 両X/Y、control0、full joint状態、sourceとの位相・半径・測定期待値。中間で70次元へ射影しない |
| 長roundへの接続 | 新profileの丸め・変換上界を、既存18round recipeの構造へ接続 | 新しいcompiler diskが全位相・正半径予算内に入るか。短q PASSだけで全round PASSにしない |

許容値は「旧差より小さい」ではなく、採択するcompiler誤差予算から実行前に定める。
この段階でもH4入力照合・SCF・DF fit・ground solveは不要。
C-S/C-P追加計算、追加sampling、価格取得、長q一体compile、T key新規取得は、この最小承認案に含めない。

旧`native_wrapper_check`はX専用なので、再利用時は軸とcompilerを入力できる小差分が必要。
旧`native-check`コマンドはC-Pも列挙するため、そのまま起動しない。
旧OUT/run01/STOPを変更して再開するのではなく、新しい出力先・SOURCEで実施する。
これらは今後の実装事項であり、今回はSOURCEを変更していない。

## 8. 未取得価格セルとbaselineの扱い

旧値はQ 63/306、T 24/108セル完了のまま保全する。残りを0補完せず、run01を自動再開しない。

旧`price_task`/`cs_price_task`は出力fileが存在すればskipし、D部分価格のmemoize引数にもcompiler profileはない。
新出力先とcompilerに対応したcacheを使い、full価格、strong D補正、C-SのT合成前Q変換へ同じ構成を渡す必要がある。
旧pathの中身やSTOPを差し替えて継続する方法は採らない。

- **compiler構成を変える場合**：旧完了セルも新profileの価格としては使わない。未完了セルだけを新profileで埋めると比較が混在する。必要な構成全体を、新しい価格cohortとして採択・取得する。
- **旧profileを保持して誤差会計だけを変える場合**：同一入力・同一sampling law/seed・同一gate列・同一precisionの完了価格は再利用候補。ただし全supportの誤差予算が閉じることが先であり、旧STOPをPASSへ書き換えない。
- **SOURCE mean・入力**：物理条件・原始演算の意味が同一と確認できれば再利用候補。native半径・N・alpha配分・Gの資格まで自動継承しない。
- **strict合成sequence**：正確なangle key、必要精度、tool条件、実adjointとphaseの意味が合い、保存上界が新要求を満たすものは再利用できる。compiler変更後のangle inventory・使用回数・全誤差予算・full価格は更新が必要。

本格価格取得を再承認する場合は、C-NのPartial/Full・3formula、C-PのP0/P1、C-SのS0/S1/S2へ同じ精度保持・fusion機会を与える。
RZ・CX・T/Tdagger・depthを分け、phase・radius・normalization・Nとfull価格を接続して比較する。
q1/2較正、q8 holdoutを再利用する設計でも、新profileの長q価格誤差は別の未解決事項である。
価格gridの再取得を、今回の静的調査で許可されたとは扱わない。

## 9. 未解決事項とSTOP

確定したのは最小再現を起こす交換判定・相殺の経路と、それと独立な合算角cutoffである。
次は未確定のまま残す。

- H4 Full Y4の差のpass別・rewrite別寄与と、他armへの影響範囲。
- 提案した参照構成の全DF gate対応、実際の数値誤差・RZ/CX/T/depth・実行時間。
- 数値最適化を戻す場合のcompiler予算、全round上界、C-Sのweightを含む誤差の閉じ方。
- 新profileの公平な全価格gridと長qモデル誤差、同一精度・成功条件での全RPE総G。

**mandatory STOP：本調査は静的調査と文書公開・remote照合までで終了する。**
新規科学計算・compile・sampling・価格取得は0件。旧SOURCE/raw/結果/grant/ledger/marker/STOPは保全。

**次の担当：ユーザー**。推奨する参照構成と、§7の最小実装・correctness範囲の採否を判断し、必要なら別途明示的に実行許可する。
**次の担当：Codex**。その許可があった場合だけ新SOURCE・新出力先で実施し、価格取得は別途採択された範囲に限る。

[old-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/docs/research/track_c_h4_pilots_20261106/REPORT.md
[old-min]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/COMPILER_MINIMAL_REPRO_FINAL.json
[old-qualification]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/NATIVE_QUALIFICATION.json
[old-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/STOP.json
[old-check]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/src/trottertracks/track_c_h4_native_check.py
[old-price]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/src/trottertracks/track_c_h4_price.py
[old-qasm]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/native_compiled_QASM.tar.xz
[old-synthesis]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/src/trottertracks/track_c_h4_synthesis.py
[old-s2-check]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/native_correctness_measured_profile/CN_S2_h0.02_ld3.json
[old-y4-check]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/native_correctness_measured_profile/CN_Y4_h0.02_ld12.json
[cc-python]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/optimization/commutative_cancellation.py#L80-L92
[checker-main]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_checker.rs#L268-L305
[checker-map]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_checker.rs#L626-L649
[analysis]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_analysis.rs#L55-L139
[cc-groups]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_cancellation.rs#L181-L205
[cc-parameter]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_cancellation.rs#L131-L135
[cc-cutoff]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/commutation_cancellation.rs#L247-L268
[crz]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/rz.py#L256-L280
[preset-init]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L144-L194
[preset-opt]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L589-L599
[remove-id]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/remove_identity_equiv.rs#L36-L53
[euler-call]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1204-L1211
[euler-angle]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L299-L339
[consolidate]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/consolidate_blocks.rs#L175-L181
[translation]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/common.py#L462-L485
[assign]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/quantumcircuit.py#L4251-L4302
