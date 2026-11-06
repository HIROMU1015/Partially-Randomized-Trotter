# O1 M2：Hermitian規約と新operator判定の提案

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**技術整理・静的SOURCE確認のみ。規約は未採択、科学実行は未承認、mandatory STOP。** 起点`fa7dc379cd16975dea317220148c473a0976a9f8`のFAIL/NOT_QUALIFIED、raw、許容値、SOURCE、STOPを変更しない。新しい残差生成・保存数値再解析・operator計算・テスト実行はしていない。

## 1. 確認した資産と旧結果の意味

| SOURCE／既存証拠 | 再利用と確認事項 |
|---|---|
| [o1_m2_residual_adapter](../../../src/trottertracks/o1_m2_residual_adapter.py) | 元g由来κ、全8frame、normal ordering、R_g=g−g_fit、R_h=h−h_core、核scalar一度。実配列のHermiticity/orthogonalityは有限許容値で受理し、厳密な対称化はしていない |
| [c1_h6_initialization.reconstruct](../../../src/trottertracks/c1_h6_initialization.py)、[c1_stage1](../../../src/trottertracks/c1_stage1.py) | J/U einsum、κ/一体収縮、spin lift、`gamma`/`operation_lengths`/`allowance`。既存allowanceはengineering envelopeでinterval保証ではない |
| [局所JW mapper](../../../src/trottertracks/o1_m2_cutoff_free_pauli.py)、[Pauli実係数化](../../../src/trotterlib/pauli_signal_adapter.py) | unit monomial JW＋fsumで微小実係数を保持。ただし`real_coefficient`は許容された虚部を捨てるため、raw complex operatorの完全保存経路ではない |
| [固定operator runner](../../../scripts/validate_o1_m2_retained_residual.py)、[CAR/density参照](../../../scripts/validate_o1_m2_h4_operator.py)、[Hermiticity診断](../../../scripts/diagnose_o1_m2_residual_hermiticity.py) | 無cutoff CAR、raw全density core、独立label bit action、complex JW、Hermitian/anti-Hermitian部分の区別を再利用 |
| `tests/management/test_o1_m2_residual_adapter.py`、`test_o1_m2_small_coefficients.py` | 正負J、符号、一体補正、scalar/電子I、sector、微小残差の人工回帰。今回実行していない |

以下は[固定報告][M2]の**引用**であり、新しい計算ではない。Rの実Pauli360項と電子identityは保持されたが、raw Rとの差8.391724815037804e-17が事前tau_R=2.0490161611727855e-18を超えた。raw Hermiticity defectは1.6783449630075609e-16。real mapperはHermitian部分に4.1359030627651384e-25、虚部を捨てないcomplex JWはrawに8.271806125530277e-25で一致した。この診断は旧FAILを取り消さない。

保存入力は同じupdate005 J/U全8frameと元h/g/scalar。input commitは`aced1647b81e8d28091058cdc7b6e3e1ee54896a`、core SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`、元H SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。旧LD2 tail・新fit・rank探索は対象外。

## 2. 一貫した規約案――ユーザーの判断事項

H_0、C_0を保存binary64入力・固定core構成から得るraw演算子、R_0をその差とする。実際のbinary64差形成には別途丸め誤差がある。P(X)=(X+X†)/2、A(X)=(X−X†)/2と区別する。

| 選択肢 | 意味と制約 |
|---|---|
| A：rawを変更しない | complex JW/CARでrawの一致を診断する。raw非Hermitian部分を保ったままHermitian unitary kernelの資格を与えない。現実Pauli mapperとの完全一致FAILは残る |
| **B：全3演算子へ同じPを適用（推奨する提案、未採択）** | H_H=P(H_0)、C_H=P(C_0)、R_H=H_H−C_H。実数演算ではP(H_0−C_0)と同じ。binary64ではclosureを別検査する。RだけをHermitian化し、元H/coreをrawのまま厳密再構成と主張しない |

Bは元ファイル・J/U・元h/gを上書きせず、新SOURCEの**派生operator規約**として保存する。normal-ordered packetの対応は、実入力ならh↦(h+hT)/2、g[pqrs]↦(g[pqrs]+g[qpsr])/2（SOURCEのchemist→spin/1/2規約を維持）。任意のtensor permutation平均や新DFは行わない。κ・一体収縮・全density coreとの対応も同じoperator定義から再確認する。native coreがP(C_0)を実装できるかは未検証であり、既存J/Uの読込許容値だけで認定しない。

核scalarはH_H/C_Hに一度、R_Hは0。電子identityは別に保持する。registerはspin8 interleaved、full256とNα=Nβ2のsector36を併記し、途中射影で差を隠さない。元H→H_H、core→C_H、raw residual→R_Hの変更量をそれぞれ記録する。

**元Hとの対応**：H_Hは保存raw H_0と厳密同一ではない。A(H_0)のnormと、元Hの意図されたHermitianモデルへの対応を明示する。非Hermitian rawに対してHermitian固有値の摂動保証をそのまま使わず、energy/元stateへの資格は別に判断する。operator-only計画はstateを使わず、将来も元H保存`state`をcandidate/DF groundへ交換する許可にはならない。

高精度参照は同じ保存float値を正確な入力値として扱い、再構成・差・Pを独立に評価する候補である。再fitではなく、保存時点の非対称性を消せる方法でもない。物理入力生成誤差と、保存値を処理する演算誤差を分ける。

## 3. 新しい事前判定案――「大きいfloor一つ」に置き換えない

u=2^-53、γ_k=ku/(1−ku)を既存helperから再利用する。ただし旧L33288・guard8を全形成過程の証明として使わない。新しい演算path・norm・係数集約・参照精度・定数を、H4の新しい差を見る**前**にSOURCEで固定する。数値閾値は今回算出していない。

| 判定層 | 事前式・実装差分の案 |
|---|---|
| 入力精度・model変更 | 保存floatのbit値を再現対象にする場合、保存値自体は厳密入力。実物理精度まで主張するなら生成SOURCEの精度情報を別に要求する。float64というdtypeだけでSCF/積分誤差を保証せず、未知の保存誤差へ勝手にhalf-ULP保証を付けない。Pによる変更量は独立のmodel差として検査 |
| 形成誤差 | κ、g_fit、h_core、R_h/R_g、Pまでのcomponentwise誤差を伝播。例えばg_fitの和ではγ_{k_g} Σ_tij abs(U_tpi U_tqi J_tij U_trj U_tsj)を使い、収縮・差には入力双方の絶対scaleと既存誤差を加える。k_g等は実einsum/積和pathに基づいて先に固定し、残差scaleだけで上流の丸めを見積もらない |
| closure／native core | H_H−(C_H+R_H)、raw density core→P(C_0)を形成・参照誤差の伝播で検査。総Hスケールの許容値はこの層に限定し、Rの係数消失を許す値として転用しない |
| **mapper自身のR忠実度** | 形成後のcanonical Hermitian residual packetを固定し、Pauli/JWと独立CARを比較。tau_map,R=b_JW(R_H)+b_CAR(R_H)+b_ref,R とし、各項はその残差係数・演算path・参照精度由来、`max(1,S_R)`なし。上流の大きいfloorをここに足さない |
| Hermiticity／虚部／微小項 | P後のHermiticity、捨てる虚部の最大値・L1とoperator影響、nonidentity・電子I・核scalar・一体補正・符号を別検査。mapperの旧1e-12だけを新たな科学的cutoffとしない。虚部は原本を残し、規約と誤差を明示して実係数化 |
| 判定の分解能 | ゼロmapper・逆符号・一体補正欠落・電子I欠落を残差自身の参照で区別する。事前のrelative分解能条件を満たさない、または保守的boundが残差を覆い隠すならNOT_QUALIFIED。後から閾値を緩めず、より精密な形成/参照または限定した主張を選ぶ |

工程scaleの経験的allowanceと、正しく丸めを包絡したinterval/高精度参照による保証は別statusにする。max-entry、Frobenius等のどのnormで判定するかも先に固定する。高精度参照を使っても生成時の物理精度保証へ自動昇格しない。

## 4. 最小実装・検証計画とGO/STOP

1. **ユーザー**がA/B、保存bit値に対するoperator定義、Pによるmodel変更の扱いを判断する。energyへの認定を求める場合は元Hとの許容model差も必要。旧FAIL/旧tau/原rawのstatusはそのまま残す。
2. **Codexの次の実装候補（今回は未実施）**：既存adapter/mapper/参照を再利用する小さな明示Hermitian派生処理、虚部診断、新事前判定だけを追加。J/U・fit・rank・原SOURCEを変更しない。入力を読む前にpath/精度/判定式を固定し、人工のanti-Hermitian perturbation、符号・一体補正・scalar・微小項のnegative controlを検査する。
3. **別途科学承認が必要な最小実行**：同じ保存update005/元Hをdecodeし、rawと派生H_H/C_H/R_H、独立高精度または誤差を定めたCAR/complex JW、full256/sector36、Pによる変更量、closure、残差自身のmapper一致を検査する。新旧結果を別保存し、旧FAILをPASSへ変更しない。新input条件・state解析・PF/RTE・回路・sampling・shots/費用は含めない。

**GO**は規約・新SOURCE・形成/参照精度・事前判定が固定され、残差を区別できる場合の新operator-only検証に限る。**STOP**は規約未決、元Hとの対応不明、精度根拠不足、下流への上流floor流用、無断symmetrize、隠れた虚部/微小項切捨て、旧結果の再資格化、保存J/Uの修正。operator-onlyのGOはenergy/PR/RPEのGOではない。

今回の成果物は本Markdownだけ。主agentによる静的な整理であり、新しい詳細科学レビューや数値評価ではない。独立branchへ公開し、remote SHAと全文一致確認後に**mandatory STOP**。次は**ユーザー**が規約と次scopeを決め、**Codex**が許可範囲のみを実装・検証する。

[M2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa7dc379cd16975dea317220148c473a0976a9f8/docs/research/track_a_o1_m2_retained_residual_validation_20261106/REPORT.md
