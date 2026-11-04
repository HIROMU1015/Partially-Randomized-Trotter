# Controlled IID測定→数学的decoder・保存DF sector energy：条件付き95%十分保証

2026-11-05 JST / LOCAL。公開SOURCE
[`ffa1e22b3001ad1c4eb275ff1121ed17111452ec`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/ffa1e22b3001ad1c4eb275ff1121ed17111452ec)
を変更せず、ユーザーの別途明示実行許可後に**Arb160 bit・1 pass**実行した。
36軸の直接Hoeffding上界とexact rational集約1回を完了し、native exit0、
`CONDITIONAL_IDEAL_CONTROLLED_IID_MATHEMATICAL_ENERGY_95_SUFFICIENT`。

**指定ideal controlled IID測定・数学的decoder・外部alias条件の下で、
同じ保存DF sector ground energy目標に対する成功確率下界は
0.950307011365666053。** これは実測した成功率ではなく、決定論的包含に基づく条件付き十分下界である。
正式実装全体の認定は行わない。

## 対象・固定入力・再利用

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、単一δ=binary64 `.02`、
K=2、8 system qubit・4電子sector70、保存physical stateのnormalized ray。
m0..17／q=2^m／保存r∈{1,2,4,8,16,32}、N/α/β/ε/μ₀/E_refを維持した。
`extract_identity_phase`、`coefficient_atol=1e-12`、保存component/basis/sign、
`support_for_source_basis_run_le_1`、保存uniform53 component/order lawを変えていない。
cosine=Re、sine=+Im、bit0→+1・bit1→−1、U(t)=exp(−itH)の規約。

以下を再利用し、新しいHamiltonian/state/回路/平均包含/幾何/固有値を計算していない。

- categorical全36軸：`d2e7a7db683e116243eae061c0e4c87865054e9e`、固定round入力`f4905fff592351d33ba07b927892fb9532c99711`。
- controlled平均専用Δ_mean：`1789a8ca731a1de631e80a34d2b8c90657d24458`、固定SOURCE `7a39a78eccdbe457786978e14d36b2f0d7abc290`。
- DのG→branch→alias/lift→literal energy：SOURCE `6942e5176d2dec115ac3c271e48f84377abda915`、結果`89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f`。
- A/Cの旧alpha/aggregateとreference込みenergy：SOURCE `d29a86963f91044a95217d7d2e2ee9703d2f1dc1`、結果`61f569c32a94c2b698df10a62a51e515b12160ed`。

[固定入力commit/path/blob/SHA256](../rpe_controlled_iid_success_energy_preparation_20261105/FIXED_BINDINGS.json)と
[固定実行計画](../rpe_controlled_iid_success_energy_preparation_20261105/SCIENTIFIC_EXECUTION_PLAN.md)を照合済み。
36軸の入力literalと保存上端の欠測・重複なし。SOURCEは実行前後とも無変更。

## 数値結果

各軸に対し元U_catとΔ_meanをexact rationalで一度だけ加え、
t=ε−(U_cat+Δ_mean)>0とp_upper≥2exp(−Nt²/2)を既存Arb160経路で評価した。
既に拡張したinterval、shot半幅、旧uniform上界をU_catの代わりに使っていない。
π64等の計上済み項を追加せず、保存α/N/εを再設計しなかった。

| 項目 | 結果 |
|---|---|
| 全36軸のt strict正値とtail上端 | 36/36 `BOUND_AVAILABLE` |
| 36上端のexact rational総和 | ≤0.0496929886343339467（上向き表示） |
| 目標≤1/20 | `DIRECT_HOEFFDING_95_SUFFICIENT` |
| Pr(G)の下界 | ≥0.950307011365666053（下向き表示） |
| G→数学的nearest-branch | 既存全17隣接strict条件を再利用 |
| G上のreference込みenergy上界 | ≤0.000138689200360119854 Ha（既存Cの上端表示を引用） |
| 保存energy目標 | binary64 `0.00015936001019904 Ha`。旧Cのstrict比較を再利用 |

保存されたexact値：

```text
failure_upper = 1218468535910657108111929276348813106077652184592843003
                /24519928653854221733733552434404946937899825954937634816
success_lower = 23301460117943564625621623158056133831822173770344791813
                /24519928653854221733733552434404946937899825954937634816
```

表は原endpointを18有効桁で指定方向へ表示しただけで、expや科学集約の再実行ではない。
各軸の保存U_cat/Δ_mean/U_ctrl/t/p上端は[AXIS_RESULTS.tsv](AXIS_RESULTS.tsv)、
正式な包含・exact総和・判定は[原RESULT.json](../../../artifacts/rpe_controlled_iid_success_energy/2026-11-05/run01/RESULT.json)。
旧Aの成功下界0.950307030063252182を新controlled値へ転記していない。
旧保存ΣαのUNKNOWN判定・旧uniform18/36判定とrawもそのまま残す。

## 成功事象・energy接続の意味

G=∩_(m,a){|hatμ_m,a−μ₀_m,a|<ε_m,a}は旧Dと同じ。
各軸内のshotは、fresh独立uniform53による全trajectoryと同じ保存rayからのideal ±1測定を仮定する。
この条件で各軸のHoeffdingとunion boundを接続した。round間・axis間の独立性は不要。
固定PCG64列が数学的IIDであることや、任意trajectoryの条件付き平均を保証した結果ではない。

G上の全座標に対して旧Dの矩形幾何、17 branch条件、共通φ_ref=−E_ref δへのalias/liftが成立済み。
保存E_ref=−2.1663874486347616 Haとδのexact binary64値を維持する。
数学的E_hat=−φ_hat/δに対し、旧Cが包含した保存DF exact polynomialのN4 sector最小固有値E_*へ
|E_hat−E_*|≤e_phi/δ+η_refを移送する。Δ_meanをG上の位相・energyへ再加算しない。
旧Cのenergy上端・alias判定は引用だけで、新しいenergy加算・目標比較・固有値評価は0。

詳細は[接続証明](../rpe_controlled_iid_success_energy_preparation_20261105/CONNECTION_PROOF.md)、
[旧D結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f/docs/research/rpe_branch_energy_execution_20261104/REPORT.md)、
[旧A/C結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61f569c32a94c2b698df10a62a51e515b12160ed/docs/research/rpe_three_gaps_execution_20261104/REPORT.md)。

## 実行・技術確認・保全

新規科学callは36 tailセルと1 exact集約だけ、Arb160・1 pass。追加precision/retry/resume0。
H/DF/state生成・数値decode、law/mean/support/geometry/branch/alias/reference再評価、
Float64 decoder、trajectory/sampling/shot、回路build/compile、noise/cost/state-preparation評価、詳細科学レビューは全て0。
準備時の人工148件PASSを再利用し、同じ無変更の数値テストは再実行しなかった。
事後は保存判定・件数・36入力literal・原hash・SOURCE byte・表の転記だけを確認した。
事後照合の初回はsystem PythonのQiskit不在で停止したため、固定venvで技術照合のみ実施し直した。科学実行は再実施していない。

Python3.11.0rc1／NumPy1.26.4／Qiskit1.3.0／python-flint0.9.0／FLINT3.6.0。
affinity32 CPU、実行時MemAvailable54,591,704 kB。GPU deviceなし。
O(36) scalar経路とexact rational集約を用い、root列挙や行列構築を避けた。
人工CPU/RAM/worker/wall-time上限や共有環境の変更はない。
runner elapsed0.021214263513684273秒、process wall0.64秒、maximum RSS107,780 KiB。
比較実測によるspeedupの主張ではない。

原RESULTは64,822 bytes、SHA256 `bc6f5d2213aaa18ddf03d30257f70f5dd61387f35ce0c1760257bb9278e6b53e`。
stdout/stderr/resourcesと合わせて4原本を無編集で保存した。[VALIDATION.json](VALIDATION.json)にhashと技術確認を記録。
LOCALの固定SOURCE実行であり、immutable CI・外部再現・独立科学レビューではない。
専用branch/worktreeだけを使用し、他job・他ユーザー・SERVER2 H8・旧SOURCE/raw/grant/ledger/marker/STOPを変更しなかった。

## 残る保証と次の担当

今回確認したのは、指定ideal logical controlled IID測定と**数学的decoder**に対する保存DF sector energyの条件付き十分保証。
Float64 decoderの全座標域での一様誤差、実PRNGのIID性、compiled/noisy実回路、実state preparation、
外部alias priorの物理的正当性、上流積分/DF誤差、他sectorや真の分子energy、総費用は未保証。

次の担当は**ユーザー**：この条件付き結果の研究上の扱いと、残る保証対象の優先順位を判断する。
**Codex**は別指示後だけ、指定対象の静的gap整理・次SOURCE準備を行う。
新しい科学評価・詳細科学レビュー・実装family変更は自動開始しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。
**結果公開・remote SHA/公開成果物照合後mandatory STOP。**
