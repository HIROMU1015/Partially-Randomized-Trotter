# 意味論と継承する未解決条件

原Phase 2CのPO-01～09を全て継承し、充足済みへ変更しない。追加案はknown identityを使った概念設計であり、新しい非自明な証明、numeric certificate、回路認証を実施していない。

## 1. Hamiltonianと誤差層

同一H_ref=cI+H_D+R、同一物理geometry／AO basis／active space／rank policy／register／初期状態を保つ。理想のpair置換は同じ `aP+bQ` を `ρA` と記述するだけ。元model/inputのΔH_input、既存DF誤差、新しい係数・角度の有限bit誤差、PF splitting誤差、有限Taylor誤差、sampling丸め、合成誤差、測定統計を別に保持する。

Iはpair対象外で、元のscalar policyを固定する。負のa/bはA内部へ取り込み、ρを正にした後で外側へ同じ負符号を再適用しない。unmatched atomは元のsign conventionを維持する。ρ=0は認証された両係数zeroの場合だけで、今回対象の非零pairには該当しない。閾値で小さい項を落とさない。

## 2. 有限meanとlaw

既知の反可換性・実係数条件でAはHermitian involution。適用したい既存RTE条件は、残差が自己逆Hermitian componentの実線形結合で、canonical確率とpaired even order lawを正しく更新し、eventを独立に生成すること。新辞書のλは元と異なるため、p、τ、θ、次数law、Bを元辞書から流用しない。

その条件の下で一microstepの補正meanを同じ `T_(K+1)(-ihR)` に接続することが目標。Rとh/Kを固定するが、新production adapterについて認証した成果ではない。元pathと新pathのmarginalや個数を一致させる必要はなく、順序付きoperatorのfirst momentが対象である。channel一致、個別trajectory一致、exact exponential一致は主張しない。粗いλ依存tail boundやNが変わることも区別する。

## 3. Controlled circuitと順序

time orderは `products → child`、matrix orderはその逆。新atomのchildは `W → C(exp(-iθP)) → W†`、productは `W → C(P) → W†`。`iPQ`をHermitian Pauliへ変換するときのordered productの±iと全係数を記録し、phaseを無視しない。atan2のquadrantを固定する。

W/W†はactual adjoint。paired event phase、unmatched負product、I rotation、scalarはancilla-relative phaseへ接続する。source accessor `unsigned_rotation_angle` のsign処理を二重に適用しない。非可換なeventをpairやbasisでsortしない。対照にも同じ相殺とcontrolを与える。

## 4. Sectorと準備

合計Hの粒子数保存だけではP、Q、A、Wの個別sector保存は保証されない。full registerで初期物理sectorとtargetを固定し、中間postselectionを使わない。sector-only encoding／simulationを使うなら全操作がその空間で合法かを別途確認し、不適格なら元辞書へ戻す。sector射影したPauliを自己逆unitaryとみなさない。状態準備やframe移動、Wに必要なgateは無料にしない。

## 5. PO-01～09の対応

| 継承ID | R1で必要になる確認 | 状態 |
|---|---|---|
| PO-01 H identity | 元R、Pauli collect、disjointなpair置換、scalar・元誤差の一致 | OPEN_NOT_EXECUTED |
| PO-02 metadata | pair ID、ρ、φ、ordered generator、basis hash、sign、typed dispatch | OPEN_NOT_EXECUTED |
| PO-03 finite mean | known involution条件と既存paired lawの接続、独立性とordered積 | OPEN_NOT_EXECUTED |
| PO-04 control/phase | Wの向き、actual adjoint、負係数、event/scalar相対位相 | OPEN_NOT_EXECUTED |
| PO-05 price | W/W†追加と既知相殺を含む価格定義。旧Φの自動適用不可 | OPEN_NOT_EXECUTED |
| PO-06 finite bit | ρ、atan2、generator phase、p/B/θ、丸めとgate合成の別予算 | OPEN_NOT_EXECUTED |
| PO-07 statistics | fresh IID bounded outcome、B/range、共通信号allowanceとconfidence | OPEN_NOT_EXECUTED |
| PO-08 fairness | 同じpair候補、情報、recipe、cache、compiler・取得費 | OPEN_NOT_EXECUTED |
| PO-09 execution space | full-register作用、sector-only実行の資格、中間投影の禁止 | OPEN_NOT_EXECUTED |

新たなPR固有定理や保証構造は得ていない。known identityを引用できることと、production実装がこのPOを満たすことは別である。証明・実装・検証を自動で開始しない。
