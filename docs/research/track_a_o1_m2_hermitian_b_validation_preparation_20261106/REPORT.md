# O1 M2：規約B operator-onlyの事前条件・実行準備

2026-11-06、SERVER 2（GPUA100-gleap02）。起点 `6d8f4dd09164f4df65a486d7ce0e51c9f04157c5`。**人工検証・実行準備のみ。規約Bは未採択、新科学実行は未承認。公開後mandatory STOP。** 主agentで候補別worktree/branchを使用した。旧FAIL・旧許容値・旧SOURCE・入力・結果を変更していない。

## 必要最小限の接続

新[operator-only runner](../../../scripts/validate_o1_m2_hermitian_b.py)は、既存Hermitian-B adapter／微小係数保持mapper／[独立CAR・density参照](../../../scripts/validate_o1_m2_h4_operator.py)を直接再利用する。H/coreを別々にPへ写し、binary64でcanonical `R_H=H_H-C_H`を作る順序を変更しない。元g由来κ、全frame、J/U、一体補正、核scalar、電子identity、符号、sectorを保持する。

- CAR参照は既存の整数0/±1の疎monomialを再利用し、各行列要素を`math.fsum`で集約する。JWやmapperの係数展開は使わない。独立density coreのHermitian部分とB core、さらにdensity core＋RとB Hも比較する。
- **形成誤差**は元binary64 H/core係数を`Fraction.from_float`で正確な有理数へ写し、厳密な`P(H_bits)-P(C_bits)`に対するcanonical packetと`P(raw R)`の差を別々に記録する。この参照のprecisionは無限精度有理数であり、SCF生成精度やJ/U縮約を正確化したという意味ではない。
- `P(raw R)`とcanonical Rのoperator差、raw H/core/Rのanti-Hermitian成分、核scalar／電子identity／nonidentity項数を記録する。小残差が形成でゼロ化しても、正しい零mapperのPASSから全体PASSへ進めない。
- **mapper誤差**はcanonical残差自身のCAR参照とのmax-entryとHermiticityで判定する。上流／総Hのscaleを残差mapperの許容値に加算しない。既存strict mapperは非ゼロ虚Pauli係数を拒否する。

I/Oは明示した2入力pathと新outputのみ。固定`UPDATE005` bindingで科学input hashを確認し、output既存時は拒否する。SOURCE commit・関係SOURCEのhashを出力する。承認証明書・grant発行・新管理基盤は追加していない。import/helpではNPZ読込・operator計算は起動しない。

## 実H4結果を見る前に固定した判定候補

以下は今回SOURCEに固定した**新候補の条件**であり、旧FAILの許容値置換ではない。実H4に適用していない。

| 条件 | 固定内容と理由 |
|---|---|
| 入力 | 保存update005の全8frame・4 spatial orbitals、元H。保存binary64 bit値を対象にする。生成時SCF/積分誤差・half-ULP保証は推定しない |
| 参照precision | operator CARはbinary64、整数monomial＋2冪scaling＋`fsum`。P/subtraction形成差は正確なFraction。subnormal underflowを含む厳密interval保証ではない |
| count定義 | `M=1+2n²+4n⁴`。tensorは`max(8 F n^6+8 n^4, 8 F n(2^(2n)+n²))`、κ／core一体は各`2n+2`、mapperは`32 M`、CAR参照は`4 M` |
| H4 count | tensor `264192`、κ `10`、core一体 `10`、mapper `33824`、参照 `4228`。入力shape/frame数から事前固定し、観測差からfitしない |
| 上流allowance | 既存B adapterの絶対J/U積和・κ・core一体scaleと`gamma_k`。tensor global countは直接5-factor全縮約とdensity経路を覆う保守的engineering envelopeで、NumPy BLASのinterval証明ではない |
| P／差形成 | 既存projection/subtraction allowanceを別記する。正確な有理係数差に基づくcanonical形成誤差CAR-L1 boundが、canonical Rの参照max-entryの1%以下でなければ形成分解能は不適格。canonical零でも正確な係数残差が非零なら不適格 |
| mapper | `tau_R=8*(gamma_33824+gamma_4228)*S_R`、`S_R=|scalar_R|+2 sum|h_R|+2 sum|g_R|`。floorなし、rtol=0。max-entry差≤tau_R、Hermiticity defect≤2 tau_R、tau_Rが参照残差max-entryの1%未満を要求。真の零packetは別扱い |
| model変更 | H/core/`P(raw R)`のprojection変更を別判定。H/C/Rそれぞれの許容model差は**ユーザー未決定**。未指定は`UNASSESSED`。旧値や人工値を無条件に転用しない |
| space | full Fock 256と`Nalpha=Nbeta=2`のsector36を併記。途中sector投影で不一致を隠さない |

上流・形成・mapperは別の検査である。正確な有理形成診断は、既に形成されたbinary64 core packetを入力とするため、J/U縮約誤差そのものの厳密高精度測定ではない。独立density比較と上流allowanceを併記する。追加の分解能条件として、観測された独立density/core差が残差参照max-entryの1%以内でなければ全体PASSにしない（canonical零の場合はdensity/core差も零を要求）。この観測条件は参照誤差の厳密保証ではないが、総H allowanceの中に未解像のcore差を隠すことを防ぐ。必要な残差精度を解像できなければ不適格のまま止め、必要なら別承認・別SOURCEで同じ入力bit値の高精度縮約を検討する。

`operator_checks_passed`は実装上の分解されたoperator検査結果に限る。`formal_adoption=False`を保持し、model変更判定と生成精度を別fieldへ出す。CLI終了コード0もoperator＋指定model予算を通過した意味に限り、Bの採択・物理性能・旧FAILの変更を意味しない。

## 人工テスト

**46 passed in 10.49 s**（新10件＋既存36件）。新`tests/management/test_o1_m2_b_validation.py`と既存B/微小係数/残差adapterの計4ファイルを対象にした。発明した2 spatial orbitals・2 signed framesを使い、fsum CARを既存独立CARに照合した。scale `1/1e-10/1e-22`、電子identity、full/sector、残差零化・逆符号mapper、anti-Hermitian寄与、真の零残差を確認した。全H allowanceを通る小さな独立core差でも、零残差に対して未解像なら全体PASSを拒否する人工negative controlも通過した。

大きなH/coreの間の小差ではcanonicalが零、`P(raw R)`は非零になる人工例を含めた。mapper PASSでも形成分解能を不適格にし、operator全体をPASSにしない。初回人工suiteでnegative controlの識別を存在しない属性から読んだ2件は、既存`metadata['model_id']`を読むよう修正した。判定式・許容値は変更していない。新testは保存`np.load`と回路生成を禁止する。runtimeは既存専用Python 3.12 / NumPy 2.5.4 / SciPy 1.18.1 / OpenFermion 1.8.1 / Qiskit 2.6.2、cacheは専用`/tmp/o1_m2_b_validation_prep_cache`、bytecode/pytest cacheは無効。

## 固定入力・次の承認範囲・STOP

元H入力：commit `aced1647b81e8d28091058cdc7b6e3e1ee54896a`の`artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。coreは同commitの`artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。core生成SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。旧LD2 tailは使わない。

- **ユーザー**：Bを採択するか、元Hへのprojectionをどう扱うか、H/core/Rのmodel差上限を決定する。そのうえで、この新SOURCE・固定2入力・上記事前条件によるoperator-only実行を明示承認する。
- **Codex（承認後のみ）**：新runnerを新しい結果pathで1回実行し、operator自身・形成差・mapper誤差を分けて報告しmandatory STOPする。stateに関しては共通の元H厳密基底状態仮定・prep費用除外を維持するが、このoperator-only検証ではstateを読み込まない。

新規科学実行・B採択は未承認なので、今回は実H4 decode、保存値再解析、operator再計算を一切していない。新fit/rank探索、別Hamiltonian、状態solve、native回路、PF/RTE、sampling、shot/価格/総費用は次のoperator-only承認にも含めない。専用branchをcommit/pushしremote内容を確認後**mandatory STOP**する。
