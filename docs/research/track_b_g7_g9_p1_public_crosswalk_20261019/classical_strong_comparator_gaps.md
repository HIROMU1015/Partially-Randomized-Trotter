# Closed P5・matched CTSの古典費用：あるSOURCEと不足する接続

[入口](README.md) / [identity照合](b3_c1_g9_identity_crosswalk.md)。現在の[BC][BC]は4方式に限り、`execution_scope_excludes`に`G9/P5Closed comparison`を明記している。今回の監査がClosed P5/CTSの実装・契約追加を承認するものではない。

## 1. 既存機能と不足の対応

| 処理 | 既存SOURCE・証拠 | 不足／公平に比較する条件 |
|---|---|---|
| Closed P5 setup | P5Closed constructor：group/root/group_law、completion DP、pair/parent係数 | 新B3でのsetup/preflight/load・cold/warm課金、audit state seal、実driver接続と計測なし |
| Closed P5 online sample | `P5Closed.sample`、`law`、`word_law`、`event` | production sampler機能は既存。全N sampling/angle/requery/出力・X/Y共有・cache費用は未取得。reference_eventsをsamplerの代用にしない |
| Closed P5 audit | 旧`audit_p5`/数学証拠/25 saved checks | 既存formal検証は再実行不要。BS `canonical_state/snapshot`はARMSとGroup属性が異なりそのまま非対応。等価な証拠基準の接続が必要で、旧FullのGreen費用を加えない |
| CTS Pauli収集/setup | G9N `provider_polynomials/exact_target/cts_events`、CTS_operator_certificate | このQではI1収集可能。取得・normal ordering等の別研究課題を再調査せず、既存収集の実setup費・辞書保持/serializationを課金する必要 |
| CTS digital event law | `cts_events`がcoefficient/proposal/weight/pauli/ratio/rotation_signを構成 | 登録有限lawはある。G9 runnerは参照リストを生成して会計し、専用production draw/bitstream/全N accumulatorは提供しない。全列挙を毎trialする弱い対照を作らない |
| CTS audit/phase | 旧full operator/identity-even/phase/native診断 | BS `event_witness`はraw_word/child・positive ratioを要求し、CTS real correction/identityへそのまま適用不可。phase/πW/適正zero等の方式別witnessが必要 |
| Full/canonical4方式 | B3 Layer A/B、F3＋補助canonical adapter、StageLedger | 公開準備版で実G7接続未実行。将来driverを完成済みと扱わず、P1-finalで差分確認 |
| SOURCE/hash/input/law seal | BS `check_source_files/Seal/AuditParity` | 共通の仕組みは参考にできる。悪意あるprocess全般のattestationではなく、実import場所・pins・state callbackを固定する必要 |
| Budget/N load | G7保存N、G9保存budget | 同じpolicyを選ぶ必要。保存Nをloadする費用は残し、再計算しない。G7の全N観測をG9 Nへ比例外挿しない |
| angle | 旧RZ資産、F3 angle enclosure | τ/K_phi/Jとerror allocation、cache hit/miss/取得/再query・phaseの同等scopeが必要。CTSだけ角度oracle無料／Fullだけ毎trial再合成にしない |
| Output/failure/cache | 既存保存record、StageLedgerのsink、terminal failure | 実disk保存・flush/retention/hash、observer負荷、OS中断prefix保全、cache cold/warm・共有poolのscheduleは実driver/契約に依存 |

SOURCEの場所は[P5][P5]、[G9N][G9N]、[G9RUN][G9RUN]、新補助は[BS][BS]。これは実装の静的な存在・境界の確認で、実入力correctnessやruntimeの評価ではない。

## 2. Layerを合わせることと、同じ手順を強制することは違う

Layer Aは方式固有generatorの費用、Layer Bは各方式に必要な監査・angle・保存込み費用である。Fullには旧F3、canonicalには有限law/root witnessという異なる正当な手順がある。Closed P5/CTSへ同じFullのpacket/Green計算を強制しない一方、対照側の監査費用を0にもしない。

同一情報・最適化機会の下で必要なaudit証拠基準を揃え、方式固有の実費を残す。FullのF3と共通overlayを二重に課金しない。CTSには既存の安いPauli辞書・streaming/cacheを許し、Fullにも合法な共有機会を与える。今回新しいsampler/adapter/schemaは実装していない。

## 3. 全trial費用と期待費用

W2c-Aの`C_cl=S+N*mu+Omega`、`Delta_a=S_F-S_a+Omega_F-Omega_a`の定義を継承する。[ABE][ABE]

- 固定全N workloadを選ぶなら、同じ環境・schedule・保存scopeの**完了観測費用**を比較する。短prefixのN倍や旧128 SHA traceから母期待μを認定しない。
- 期待費用を要求するならfresh-IID cost estimand、測定の統計policy、confidence/停止規則が必要。G7/G9の量子推定αはCPU CIではない。
- X/Y間で何を共有するか、arm単独coldか共有key資産かを固定する。既に取得したsequenceがあることと、新実行での取得/load/lookupが無料であることは違う。
- setup・outputがtrial内に入るか外に入るかを明示し、StageLedger exclusive項と非instrumented totalを重ねて足さない。bit上界・CPU秒・peak memory・native T等を未採択重みで合成しない。

G9MANのguard CPUはsmall matrix/reference/native会計を含む既存記録であり、Closed P5/CTSのproduction全N費用ではない。B3の4方式が後日完了しても、R18の不足は自動的には埋まらない。

## 4. 最小限の次の判断

**既存native比較の引用だけで研究判断できるなら、追加の古典実装・測定は不要。** 古典費用を含めた強い対照比較が判断を変えるとユーザーが認める場合のみ、P3で最小のFull/Closed P5/CTSを同じcost estimand・policy・layer/保存条件へ接続する。

その場合、既存sample/CTS law/sequenceを再利用し、足りないproduction draw・audit・永続保存の差分だけを特定して別途実装承認へ戻す。新数値計測にはさらに実行承認が必要。native compileはこの古典費用不足を埋める必須作業ではない。今はP3の契約・実装・科学実行へ進まず、**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[ABE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/break_even_conditions.md
[BC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/execution_contract_candidate.json
[BS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py
[G9N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[G9RUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/scripts/tracks/algorithm_codesign/g9_p5_matched_native_v2.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
