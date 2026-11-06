# O1 M2：core-only適格性・部分予算照合の準備

2026-11-06、SERVER 2（GPUA100-gleap02）。baseは `c75d3cfd2115051814c80e14b0af6e55bc51b779`。**既存結果の引用、最小pure helper、架空scalarの人工テストまで。実保存結果へhelperを適用しておらず、新科学計算・再集計／再ランキング・H4入力decode・simulation・回路・shot／費用計算は行っていない。** 新科学実行は未承認で、公開照合後mandatory STOPする。

## 確認した根拠と現在の結論

[既存限定結果][R]は、保存直鎖H4・1 Å・STO-3G、update005 number-CDF rank8／全8 frame、元h/g・一体補正・核scalar、`Nα=Nβ=2`を対象とする。数学的B参照はraw保存binary64 packetにexact Pを作用させた `H_H^ref` と `C_H^ref`、完全残差は `R_H^ref=H_H^ref−C_H^ref` である。同じHermitian保存sector内の最低energy差に対する保守CAR係数上界 **`6.930386640756262e-08 Ha`** を、元報告からそのまま引用する。今回この値や形成済みpacket差を再計算していない。

これは精密なspectral norm／実energy差／元物理Hへのaccuracy判定ではない。`H_phys`の定義、保存／生成精度と参照誤差、model差許容上限、M2の要求energy精度と省略配分は未固定であり、**物理modelはUNASSESSED**を維持する。C0の0.1 HaをM2へ自動採用しない。旧[B operator PASS][B]、旧FAIL／許容値、[研究C NO-GOを保持する調査][S]を変更しない。

## 再利用・最小差分

既存 `o1_m2_core_only_bound` の結果schema（`reference`、`sector`、`energy_bound_reference_H_H_vs_C_H.upper_bound`）を再利用した。既存norm入口はraw packetから上界を再計算するため、今回それを呼ばず、[小さなpure helper](../../../src/trottertracks/o1_m2_core_only_qualification.py) `assess_model_omission_budget` を追加した。実行runner・native PF・価格接続は追加していない。変更はこのhelper・[人工test](../../../tests/management/test_o1_m2_core_only_qualification.py)・本書だけで、既存SOURCEは不変更である。

helperは、既存omission resultに加えて、物理参照ラベル、model boundが対象とするB参照ラベル／sector、model上界、明示的な**model＋omission用配分**を受け取る。全体energy精度をこの二成分に丸ごと割り当てるdefaultはない。提供boundの妥当性・参照精度・正式採択は検証せず、入力ラベルを承認証明として扱わない。

| 条件 | helperが行う限定的な扱い |
|---|---|
| 参照・sector・model上界・omission上界・配分のいずれか未指定 | `UNASSESSED`。不明boundを0やTrueで補わず、合計も算出しない |
| model boundのB参照がomission resultと異なる／sector不一致 | `UNASSESSED`。形成済みH_H用model上界を数学的Bのomission上界へ無補正で加えない |
| 全条件が明示され、参照とsectorが一致 | 提供上界をFractionへ写して正確に加算し、明示配分と照合する。binary64はbit値どおり、integer／rationalはfloatへ丸めず保持する |
| 合計上界が配分内 | `CONDITIONALLY_WITHIN_DECLARED_MODEL_OMISSION_ALLOCATION`。この二成分の条件付き算術だけ |
| 合計上界が配分超過 | `BOUND_INCONCLUSIVE`。実誤差の超過／core-only不成立／残差random化必須を証明しない |

全caseで `physical_model_status`、`native_core_status`、`full_rpe_status` はUNASSESSED、`residual_randomization_needed`は未判定のままにする。参照ラベル一致だけで科学的に正当なboundと認定する機能はない。

## 人工確認

既存専用 `/tmp/track-a-o1-m2-runtime-20261106/bin/python` をread-onlyで使用し、`tests/management/test_o1_m2_core_only_qualification.py`のみ **21 passed in 0.13s**。新moduleは標準libraryのみを使い、分子入力・既存result・NumPy／CAR／mapperを読み込まない。

発明したscalar／参照ラベル／sectorで、未指定と明示0の区別、数学的Bと形成済みpacketの参照不一致、sector不一致、配分内／上界超過、非有限・負boundの拒否、binary64合計やexact integer／rational入力を丸め下げて配分内にしない境界、入力dict不変更とfile I/Oなしを確認した。H4の `6.930386640756262e-08` をテスト値に使わず、実保存結果の新合計・比率・差や順位は求めていない。人工PASSは科学的適格性の証拠ではない。

## 適格性に不足している条件

1. **元物理Hへのmodel接続**：Hermitian `H_phys`の明示的定義と精度、保存h/g・scalarとの対応、生成／参照誤差を含めた `H_phys→H_H^ref` の正当なnorm上界が必要。同じ保存sectorを使うなら保存則・漏れの扱いも必要で、raw非Hermitian H0の固有値を元物理energyへ読み替えない。
2. **energy配分**：M2の要求精度、model・omission・native対応／PF・実装／信号・統計等の配分を固定する必要がある。本helperが扱うのはmodel＋omission成分のみで、未取得成分を0とはしない。
3. **実装対象とstate**：既存formed B packetと数学的B参照、正確な保存J/U core、native coreの対応は別に確認が必要。元Hの厳密基底状態・no-prep仮定を維持しても、そのstateがC_Hや有限PFの固有状態とは限らない。
4. **全RPEの資格**：core内部PF、controlled相対位相／scalar、sector、元H stateでの全必要roundの複素信号・半径・branch・成功確率の接続が未検証。time-domain norm／Duhamel等で信号へ接続する案も、現状は計画条件であり、新時間での値を計算していない。

## core-only／補正法を比べる条件と次の最小範囲

model・omission成分が十分小さくても、全PF／信号／統計精度まで満たせるとは限らない。core-onlyが全条件を満たす場合には、完全残差PRが追加価格に見合う精度やshot上の利点を持つかを反証対照として調べる。省略補正が必要でも、random化だけが選択肢ではない。

比較対象はcore-only、同J/U強決定論PF、元Hに対応する強DF-PF、core＋**完全残差**PR。同じ元H state、状態準備費用除外、energy精度・成功確率・誤差配分、compiler・主資源指標、全必要round／初終端境界を合わせる。旧LD2 tail、own-DF state、prep込み価格、旧serial proxyを新条件へ転用しない。今回その費用接続の実装・計算はしていない。

次は**ユーザー**が物理参照・precision・model／omission配分と比較条件を固定する。**Codex**が行う次の最小科学範囲は、採用された参照に対するmodel上界確認と、同じB参照への省略上界の接続に限定して明示承認を受ける。native／controlled／core PF／元state信号を扱う場合は、そのSOURCE・入力・時間／round・判定を別途具体化し、必要なbuild／simulationを実行範囲として明示する。価格・shot・全RPE費用はその自動的な続きにしない。

**GO条件**は、同じ物理target／B参照／保存sectorの正当な上界と配分が固定され、未検証成分を残したまま部分充足を全体PASSへ昇格しないこと。**STOP条件**は、現時点の科学実行未承認、参照／予算不明や不一致、native／PF／signal資格の不足。保守上界でinconclusiveなら、条件・許容値を事後変更して救済しない。適格性を満たすcore-only対照に対して、完全残差PRの追加利益を同条件費用で示せなければ、そのrandom化候補を継続する根拠は得られない。これはPR一般の不可能性や研究C NO-GOの解除ではない。

準備3ファイルを専用branchへ公開し、remote SHA／本文・SOURCEの一致とlocal cleanを確認後**mandatory STOP**する。既存SOURCE・raw・result・grant・marker・STOPを保全し、新model正式採択・詳細科学レビュー・追加実装／科学実行へ自動進行しない。

[R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c75d3cfd2115051814c80e14b0af6e55bc51b779/docs/research/track_a_o1_m2_core_only_science_20261106/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f891f6e3265e4f0124c598fddfb7775bc1eddd1b/docs/research/track_a_o1_m2_improvement_survey_20261106/REPORT.md
