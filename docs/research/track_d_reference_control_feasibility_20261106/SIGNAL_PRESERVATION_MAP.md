# Track D：信号・測定統計の保存条件

2026-11-06 JST。定義・既知恒等式の適用条件。新しいsimulation・sampling・shot数評価はない。gadgetのoperator順序は [REFERENCE_APPLICABILITY](REFERENCE_APPLICABILITY.md) §1–2。

## 1. 一shotの比較対象

元Hの同じ厳密基底状態ψ=|E₀〉、同じround m・axis b・path ωを使う。状態準備を除外しても、controlled準備や逆準備oracleは追加しない。直接制御の信号をzω=〈ψ|Vω|ψ〉とする。

SE型回路が実装するancilla coherenceは

\[
\widetilde z_\omega
=e^{i\theta_\omega}z_\omega
\langle r|V_\omega^\dagger|r\rangle.
\]

`Vω r=e^{iχω}r` と正しいθω=χωがあれば `z̃ω=zω`。A/Bの共通固有基底を要求する原著のmulti-gadget保証とは条件・対象が違う。今回の照合はfresh inputで終端ancillaを読む一shotに限定する。

[既存wrapper](../../../src/trotterlib/rpe_hadamard_interrogation.py)と同じH/V/(Sdg)/H、bit 0→+1、1→−1を使うと

\[
\Pr(Y_b=y\mid\omega)
=\frac{1+y f_b(z_\omega)}2,
\quad y\in\{-1,+1\},\quad
f_X=\operatorname{Re},\ f_Y=\operatorname{Im}.
\]

従って **各pathの複素coherenceの一致＋同じ二値読出し** は、各axisの条件付きBernoulli分布の一致に接続する。XとYを同一shotで同時に測ったjoint分布を主張しているのではない。

## 2. 保存の強さを分ける

| レベル | 必要な条件 | 保存されるもの／残る問題 |
|---|---|---|
| actual path | 同じVωのgate順・scalar・actual adjoint、reference ray、正しい位相補正 | 一shotのX/Y条件付き分布。system出力状態の一致は不要かつ未主張 |
| raw平均 | 同じp(ω)で上記一致 | E[zω]、raw信号半径・位相。random-unitary channelの一致とは別 |
| 通常の±1測定 | pathごとに同分布、同law、各trial/shotの独立性とaxis scheduleを維持 | 全raw測定列のlaw、従来のN・decoder・失敗配分を理想条件下で維持可能 |
| weighted estimator | さらに同じ保存weight wω、normalizer、全trialの採否・分母、return/zero-fill処理 | (ω, Yb, wω)のjoint law。補正推定量の分布・range/momentも維持可能 |
| 複素平均だけ一致 | E[z̃]=E[z]だけを示した場合 | IID・unweighted二値測定の各axis周辺lawは平均で決まる。ただしpathとのjoint law、weighted分布、相関shot列の一致は導けない |
| noisy/synthesized implementation | 両armの合成誤差、reference leakage、CSWAP誤差、routing、readoutも含めた許容差 | ideal恒等式だけでは実装分布や必要Nの不変は確定しない |

weightedで `E[w z̃]=E[w z]` しか示していない場合も、wYの分布全体が同じとはいえない。weightを固定した各pathの一致が今回の十分条件である。既存importance samplingやreturn法へ平均一致だけを横流ししない。

## 3. phase・normalizationの照合表

| 項目 | 直接制御での所在 | Referenceに必要な処置 | 省略した場合 |
|---|---|---|---|
| path global scalar | 制御下ではancilla-relative phase | χωに含めて既知P補正へ戻す。scalarを省いた実装なら省略との対応を明示 | energy offset・X/Y信号が変わる |
| RTE order符号 | `event.phase=(-1)^(order/2)` | 保持しreference phaseへ追跡 | 高次eventの符号が変わる |
| 負product・identity | product sign、identity rotation phase | 同じsigned unitaryとして会計 | 単純な真空energy補正では足りない |
| Gaussian実global phase | inverse/basis sandwichで相殺、またはcompiler metadata | actual adjointと露出したsplit境界を照合 | 見かけのunitary一致をrelative-phase一致と誤認 |
| finite B/attenuation | [rte.py](../../../src/trotterlib/rte.py)のlawとmean、[accounting](../../../src/trotterlib/rpe_resource_accounting.py)の観測半径 | 同じB、同じw/pを保持 | Referenceを使うだけではattenuationやshot負担は消えない |
| Y軸符号 | Sdg→H、同じbit mapping | P補正の符号・register順と一緒に保持 | Im(z)の符号反転・energy推定の変更 |

`P(χω)`は[SE-QPE v2 Eq.2](https://arxiv.org/abs/2604.14921v2)の|1〉分枝に掛ける正符号の補正である。原著cycle位相との2π換算を保つ。global phaseを無視するoperator比較や、結果を合わせるphase fittingでは代用しない。

## 4. referenceが完全に保存されない場合

`|〈r|Vω†|r〉|<1` なら、phase rotationだけでは一般にその減衰を戻せない。path依存の既知位相だけが違う場合は、各pathごとに補正する。一定のEvac補正や平均phase補正で足りると仮定しない。

有限精度の補正rotation、reference leakage、近似adjointは追加のcoherence誤差源。既存PF/finite-RTE予算と区別して扱い、原Hのenergy精度・成功率の余裕内と確認するまでは「Nは同じ」と実装結果として確定しない。新しい誤差上限値やshot数を本調査で設定・計算していない。

error detectionを付けてreference/cat registerをpost-selectすると、採用shotのlawと全trial費用が変わり得る。原著にerror detectionがあることを、現baselineと同分布・同shot数の無償保証として使わない。まずpost-selectionなしのideal信号を比較するという技術的順序である。

## 5. RPEとcoherent QPEの境界

各RPE shotをfresh ψ⊗rで開始し、全ordered pathを一つのgadgetで測り終える場合、上記条件付き分布が対象である。保存schedule、fresh-IID規約、weights、同じdecoderを維持すれば、ideal同値性から同じ全round統計を使う道筋がある。元H exact ground仮定だけでPF信号半径を1とは置かない。

一方、同じsystemを保持して複数gadgetを繰り返すと、A†による共通back-actionが次の非可換random pathの信号を変え得る。[SE原著Eqs.3–6](https://arxiv.org/abs/2604.14921v2)の共通固有基底条件を各段で確認せず、canonical QPE分布を保存したとはしない。途中reference reset、短いgadgetの反復によるfull path置換にも同じ注意が要る。

[PR原著v2](https://arxiv.org/abs/2503.05647v2) §IV/V・App.Aは制御下のcoherent first momentを対象にし、App.Bは倍時刻のX/Y測定と古典復元を接続する。App.E21はroundとshotの合計費用である。Referenceはこのlaw・normalization・RPE構成を維持するための回路置換候補であって、未知E₀やexact biasを与える方法ではない。
