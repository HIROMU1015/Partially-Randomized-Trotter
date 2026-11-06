# O1 M2：Hermitian規約Bの最小実装候補

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**実装と人工テストのみ完了。Bは未正式採択、実H4は未資格。公開確認後mandatory STOP。**

起点は[固定計画](../track_a_o1_m2_hermitian_plan_20261106/PLAN.md)（`2bcf8f7b57a5b3bf65b62b367f45817eb88facef`）、旧検証は`fa7dc379cd16975dea317220148c473a0976a9f8`。旧SOURCE・保存入力・FAIL/NOT_QUALIFIED・旧許容値・結果・STOPは変更していない。保存H4のdecode、保存数値の再解析、operator再計算、回路、PF/RTE、shots/費用計算は実行していない。

## 実装と再利用

追加したのは[派生adapter](../../../src/trottertracks/o1_m2_hermitian_b.py)、[人工テスト](../../../tests/management/test_o1_m2_hermitian_b.py)、本報告の3ファイル。既存`prepare_residual`、`SpatialIntegrals`、general-K/frame構造、unit-monomial JW、`spin_lift_unthresholded`、`gamma`、独立CAR/density参照を再利用し、既存実装を変更していない。

`derive_hermitian_b(raw)`は次の順序をcanonicalとして固定する。

1. `H_H=P(H_0)`、`C_H=P(C_0)`を別々に形成する。実保存規約のpacketでは`h↦(h+h.T)/2`、`g[pqrs]↦(g[pqrs]+g[qpsr])/2`。
2. binary64で`R_H=H_H-C_H`を形成する。核scalarはH/coreに一度、Rは0。電子identityはmapper内で別途保持する。
3. `P(raw R)`を別packetに残し、canonicalとの差を形成診断へ記録する。両者のbitwise一致を仮定しない。

元のraw object、元g由来κ、J/U、frame列、sectorを保持する。新fit・rank探索・旧LD2 tailの流用・Rだけの対称化はない。保存update005の全8frameを変更する経路はないが、その実数値・native回路が`P(C_0)`を実装するかは未検証である。

`strict_pauli_map`は既存の微小係数保持mapperの前で、同じunit-JW寄与を複素数のまま`fsum`で集約する。虚Pauli係数が**非ゼロなら明示的に拒否**し、既存の`1e-12`虚部許容値に暗黙依存しない。real adjoint-pairのcanonical packetではunit-JWの2冪係数による虚寄与の正確な相殺を要求する。spatial complex入力はfloatへのcast前に拒否する。共有OpenFermion、`EQ_TOLERANCE`、library設定は不変更。

## 事前判定の分離と限界

| 層 | 実装候補の扱い |
|---|---|
| 入力精度・model変更 | 保存binary64のbit値を再現対象とする。生成時のSCF/積分誤差やhalf-ULP保証は推定しない。H/core/`P(raw R)`のprojection変更量を別記し、ユーザー条件に基づく上限が未指定なら`UNASSESSED` |
| 上流・差形成 | J/Uの絶対積和、κ、一体収縮、projection、subtractionのscaleと`γ_k`を用いる。canonicalと`P(raw R)`の差も別記。独立参照のclosure誤差をこの層で判定 |
| 残差mapper | 実際に渡すcanonical RのCAR係数L1だけから`guard*(γ_mapper+γ_reference)*S_R`を形成。`max(1,S_R)`や上流/総Hのfloorを加えない。残差自身のmax-entry、Hermiticity、ゼロ/逆符号誤差を記録 |
| 判定分解能 | mapper allowanceが参照残差の指定relative割合以上なら`NOT_QUALIFIED`。真の零packetは別扱い。mapのPASSは物理modelや生成精度の資格を与えない |

`RoundoffPlan`はtensor/κ/core一体/mapper/参照の演算数を必須とする。人工テストではそれぞれ512/16/16/2048/2048、guard 8、最大relative割合0.01を**人工入力向けに事前指定**した。これらを実H4へ適用していない。実H4では実einsum/集約/参照pathを覆う演算数・精度・norm・model変更上限を新しい差を見る前に固定する必要がある。この実装のallowanceはengineering envelopeであり、厳密interval証明ではない。

canonical差形成そのものが、大きなH/coreの間にある小残差を丸めで失う場合がある。人工例では`P(H)-P(C)=0`でも`P(raw R)`は非ゼロとなる。mapperがcanonicalの零packetに正しく一致したPASSは、この上流の消失を解決しない。`model_change_status=UNASSESSED`、不適格な形成精度、未解決のroute差があれば、内部map PASSを実H4や物理GOへ昇格させない。

## 人工テスト

**36 passed in 13.57 s**（新14件、既存adapter 17件、既存微小係数5件）。実行runtimeは既存専用環境Python 3.12 / NumPy 2.5.4 / SciPy 1.18.1 / OpenFermion 1.8.1 / Qiskit 2.6.2。新テストは`np.load`と回路生成を禁止し、発明した2 spatial orbitals・2 signed framesのみを使う。

```sh
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/o1-m2-hermitian-b-test-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m2-hermitian-b-test-cache-20261106/mpl \
/tmp/track-a-o1-m2-runtime-20261106/bin/python -m pytest -q \
  tests/management/test_o1_m2_hermitian_b.py \
  tests/management/test_o1_m2_small_coefficients.py \
  tests/management/test_o1_m2_residual_adapter.py
```

独立CARに対するH/core/R全3operatorのPとclosure、raw不変、元g由来κ、符号、一体補正、核scalar/電子identity、sector、`1e-10/1e-16/1e-22`の微小係数保持を確認した。残差mapperの全消失、逆符号、一体補正欠落、電子identity欠落はFAIL、過大allowanceはNOT_QUALIFIEDとなった。旧`1e-12`より小さいanti-Hermitian寄与を検出し、参照とmapperが同じ誤った非Hermitian行列でもFAILとする。complex入力の暗黙cast、canonicalと`P(raw R)`の相違も検査した。

初回は人工density-core比較の差`2.22e-15`が新テストの`2e-15`を超えた1件だけ失敗した。同じ既存人工fixtureの`1e-14`を再利用して再確認した。実H4の許容値の変更や科学PASSへの転用はない。人工PASSは科学性能・実H4 correctnessの証拠ではない。

## 次の固定条件・承認範囲とSTOP

- **ユーザー**：Bを正式採択するか、元Hとのmodel差をどう扱うか、次のoperator-only対象と許容model差を判断する。
- **Codex（別途明示承認後のみ）**：固定計画にある同一update005の全8frameと元H入力を用い、新SOURCEでraw/派生H/core/R、元Hからの変更量、独立参照closure、canonical残差自身の一致を確認する。spin8 full256とNα=Nβ=2のsector36を併記し、途中射影で差を隠さない。実pathに基づく演算数・参照精度・事前判定条件を先に固定する。必要なら同じbinary64値を使う高精度形成を選び、生成精度とは区別する。

現時点で実H4のcounts/model変更予算・独立参照精度は未固定、native coreとBの対応も未検証であり**科学実行STOP**。次の承認をoperator-onlyに限定した場合、state再解析、回路build/compile、PF/RTE、sampling、shots/総費用、新fit/rank探索は含まれない。旧FAIL・旧SOURCE・旧許容値は今後も保存する。本branchを公開・remote内容確認したら**mandatory STOP**し、自動で科学計算や詳細科学レビューへ進まない。
