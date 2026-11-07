# O2-A2：PRのcoherent信号への測定結合型Multilevelの適用可否

2026-11-08 JST／SERVER1。判定は **`ACCESS_MODEL_INCOMPLETE`**。既知恒等式のPRへの対応と実装接口の技術調査であり、Multilevel一般の不可能性・科学的NO-GOの証明ではない。

## 1. 一次資料と今回の追加点

[MLMC-qDRIFT v2](https://arxiv.org/html/2604.26865v2) §2.3、Eqs.(26)–(59)、Lemma 2を直接確認した。論文はqDRIFTの状態observableを対象に、共有indexのfine/coarse差を拡張状態へ格納する。二次・四次moment条件と小norm observableによる測定noise評価を持つ。一方、§2.3.6は非unitary updateを明記し、dilationを述べる。本PRの実装済み測定アクセスを提供する資料ではない。

[CV4Quantum v1](https://arxiv.org/html/2502.08735v1) §2.3、Eqs.(38)–(39)/Lemma A.2は、未知の最良metadata controlでも測定内分散が残ることを扱う。SCVの既報不改善から、Born noiseが実際に支配したとは断定しない。

前回調査の一般的な「target/S/dilationが未適格」から進め、以下では **PRのどの状態・observableへ写せるか、どの出力を測り、どこで成功率費用が戻るか** を明記した。以下のPRへの対応とflag会計は技術的な適用上の推論。原論文のPR向け保証ではない。

## 2. PR targetをchannelへ置き換えない

保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector／L_D=3、m0,q1,δ=.02,r1,K2、coefficient_atol=0.0、理想gateという既存A1条件を参照する。N1 actual finite-grid Q・selected full policy、N2保証groundを固定参照とし、実payloadの復元・評価はしていない。

位相・D halves・basisを含むfull pathをUω、同じ入力状態をψと書く。既存のcontrolled `diag(I,Uω)` は、readout前に

\[
 |\Psi_\omega\rangle=(|0\rangle|\psi\rangle+|1\rangle U_\omega|\psi\rangle)/\sqrt2
\]

を準備する。`O_X=X_a⊗I`、`O_Y=Y_a⊗I` の平均はそれぞれ `Re⟨ψ|Uω|ψ⟩`、`Im⟨ψ|Uω|ψ⟩`。これはancilla-firstの数式表記であり、実装のwire順を変える指示ではない。既存Qiskitではancilla index=n、system順も既存codecを維持する。

一次targetは `μ_b=E_Q[⟨Ψω|O_b|Ψω⟩]` というattenuated raw mean。必要なら既知normalizationを別に戻す。`U→e^{iφ}U` はこのmeanを変えるため、system channelの一致・phase fittingでは代用できない。Y readoutはSdg→H、bit0→+1、bit1→−1。

従ってobservable形式への **代数的なlift** は既存Hadamard資産で定義できる。ただしこれだけではMultilevelの測定機構は完成しない。

## 3. Levelと差分の意味

fineをN1の固定target `μ_L=μ` とし、各levelの実law `Q_l` と実装full unitary `U_l` を定義すれば、既知のtelescopingは

\[
 \mu_L=\mu_0+\sum_{l=1}^L E_{\Pi_l}[f_l(\omega_f)-f_{l-1}(\omega_c)]
\]

となる。`Π_l` の両marginalはそれぞれ `Q_l,Q_(l−1)` でなければならない。**各levelの平均が等しいという仮定ではない。** coarseの平均を無料で既知とせず、base測定も費用に含める。

PRのr/K変更、event削除、coarse lawへの写像は本作業で採択していない。qDRIFTの奇数index共有をPRの異なるorder/label lawへそのまま移せない。全levelをN1と同じ平均にすれば代数的telescopingは成り立つが、小さいpathwise差・shot利益は導かれない。

一般のsigned/importance weight `w_l` がある場合の差は `w_f f_f−w_c f_c`。共通weightでない限り、単純な無加重状態差のobservableだけではこの量にならない。例えば `w_f(f_f−f_c)+(w_f−w_c)f_c` という追加項も残る。finite RTE mean、実q、normalization、rangeをlevelごとに対応させる必要がある。

## 4. 欲しい測定アクセスと既存アクセスの差

無加重の場合でも、隣接levelのHadamard状態を `Ψ_f,Ψ_c` とすると必要な量は

\[
 e=\Psi_f-\Psi_c,\quad
 \chi=\begin{bmatrix}\zeta e\\\Psi_c\end{bmatrix},\quad
 S=\|\chi\|^2=1+\zeta^2\|e\|^2,
\]

\[
 \widehat O_b=
 \begin{bmatrix}\zeta^{-2}O_b&\zeta^{-1}O_b\\
 \zeta^{-1}O_b&0\end{bmatrix},\qquad
 \langle\chi|\widehat O_b|\chi\rangle=f_f-f_c.
\]

これは原論文の既知block恒等式を `O_b` と `Ψ` に適用したもの。新しい測定法・定理の採択ではない。元のHadamard ancilla aに加えてblock ancilla bが必要。正規化状態 `χ/√S` での測定値を差分へ戻すにはSが要る。Sは `Re⟨Ψ_f|Ψ_c⟩` に依存し、path metadataのみの既知定数ではない。

| 処理 | 現在利用できる資産 | 残る接口 |
|---|---|---|
| 元ground入力、controlled full path、X/Y | N2 state保証、既存Hadamard builder、SavedEngine | 元state準備の仮定は逆準備・反射oracleを含まない |
| fine/coarseの古典path | rte.py、有限bit/weight処理 | 両marginal・phaseを保つ特定の階層/couplingは未指定 |
| scaled差分状態χの準備 | 単純な制御分岐は記述可能 | 非unitaryな増幅・差分写像の物理的実現は未接続 |
| 正規化・flag | 旧engineはunitary列後のBorn比を計算 | Sまたは既知block scaleを使う成功/失敗込みの測定仕様がない |
| 小norm observable | 代数的blockは上式で指定可能 | 測定基底、追加gateと誤差・価格の資格がない |
| 全費用・有限confidence | Parent/G7等のbounded会計 | 新outcomeのrange、全試行・成功率と追加アクセスの費用が未取得 |

原論文のupdate `[[U_f,ζ(U_f−U_c)],[0,U_c]]` は一般に非unitary。既存のunitary列を作用させる `SavedEngine.probability` へ行列を渡して正規化しても、物理的な準備・成功率・S補正を実装したことにはならない。保存state vectorからSを古典計算する方法も無料oracleとして認めない。

## 5. Dilationを挙げるだけでは不足する理由

既知の成功blockを `K/β` と書き、入力から `K|init⟩=χ` を得る場合を考える。βはそのblock実装に必要な既知scaleである。成功確率は `p=S/β²`。成功時の `z` が `Ô_b` の測定値なら、全attemptを分母とするzero-fill値は

\[
 T=\begin{cases}\beta^2 z&\text{success},\\0&\text{failure},\end{cases}
 \quad E[T]=f_f-f_c,
\]

\[
 E[T^2]=\beta^2S\langle\chi/\sqrt S|\widehat O_b^2|\chi/\sqrt S\rangle.
\]

これはflag付き測定の通常の会計で、実装提案ではない。未知Sの推定を避け得る一方、β²のrange/二次moment負担と失敗attempt費が戻る。例えばcoarse/fineを単純にLCUしてχを作る素朴なencodingではscaleがζとともに増え得るため、observableの小normだけから全attemptのvariance減少は言えない。より良いdilationを否定したわけではない。

successだけを採り `S z` を使う案なら、Sの取得精度と全attempt数/成功率を別に会計する。state逆準備・反射・coherent sampler・postselection retryを無料にしない。元groundの準備費除外という研究上の仮定は、新しいdilationやamplificationの費用除外を意味しない。

## 6. Noise・誤差・費用の判定

既存の独立fine/coarse測定は条件付きvarianceの和が残る。coherent二分岐に固定normのobservableを付けるだけでもnoise床は消えない。A2の候補機構は **scaled差分状態と小norm observableによって測定自体を変えること** で、metadata CV/層別/ISと区別できる。ただし現アクセスでは未完成。

PRの選択された階層について、差分stateの二次・四次moment条件、ζの選択、S/βとweight/rangeの制御は未取得。qDRIFTのlevel別評価をPRへ移送しない。追加register・dilation/flag・observable測定・成功/失敗・native誤差・base測定の全費用もUNKNOWN。

N1/N2/旧engine認証の保証は既存targetと旧kernelのまま維持する。新block gateやpostselectionへ旧engine誤差上界を自動移送しない。今回新しいgate、state作用、Born評価、RNG、shot、compile、variance/N/費用評価は0。

## 7. 今回の判定と再検討に必要な最小材料

代数的target対応はできたが、測定アクセス全体は `ACCESS_MODEL_INCOMPLETE`。数式を実行可能adapterと呼ばず、statevectorで不足アクセスを埋める人工fixtureも作らない。非自明な新理論やdilationを独断で採択せず、指示どおり[J1調査](J1_SAMPLING_RETURN_CONNECTION.md)へ移行した。

再検討に必要なのは、特定のPR階層/両marginalと、既存のどの物理操作でK/βを実現し、何を測り、Sまたはβ・失敗費・rangeをどう扱うかという有限仕様。これが得られるまで実分子pilotを提案しない。追加の科学条件やshot数を本作業では採択していない。
