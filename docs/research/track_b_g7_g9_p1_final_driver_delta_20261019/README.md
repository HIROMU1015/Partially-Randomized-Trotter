# G7/G9 P1-final：公開B3 driverの最終静的差分照合

2026-10-19 JST。状態：**`P1_PUBLIC_DRIVER_DELTA_RECONCILED_TECHNICAL_FIX_AND_ADOPTION_PENDING`**。

**D01〜D10の公開固定資料に基づく照合は完了した。科学実行準備の無条件PASSではない。** 保存N・4方式・旧科学SOURCEのidentityは保持される。新driverの失敗記録に技術修正F01が必要であり、τ/Kφ/J等の正式採択と外部atomic admissionのbindingも残る。実G7 correctness・実費順位は未確定。

## 固定資料と差分

- 直接基点：[P1先行照合][P1] `568bf486805dfbd2fd4a13ef5a25771351981ce2`。前回の[D01〜D10][CHECK]を今回の基準とした。
- 新対象：[公開B3 driver準備][DREAD] `e4af6c064db805cbec611bcc8f0ff75a4754d38b`。
- 旧照合対象：[B3/C1初期準備][BREAD] `c8cbeb18f76aef2cb1329d8c810036372fa07cc0`。
- 新SOURCE：[g6_w2c_b3_driver.py][DSRC]、40,477 bytes、SHA256 `d7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee`。
- 旧G7・ReturnKernel・F3の6 SOURCE、補助SOURCEとpackage initのbytesを照合。新対象の10 source pinsと一致し、新driver以外の9件は旧準備と同じbytesだった。既存provider価格schema・RZ証拠は変更なし。

対象はG7 `P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、σ=+1、Ordinary／Partial Return+tail／Closed P3+tail／General Full、X/Y、Layer A/B。known developmentの有限P5 first operator momentであり、分子・geometry・basis・DF rank・split・δ窓は非該当。Closed P5/CTSはこのB3の方式集合に含まれない。

## 成果物

| 資料 | 内容 |
|---|---|
| [D01〜D10結果](d01_d10_final_crosswalk.md) | 分類、一致・差分、証拠、F01と未採択条件 |
| [B3/G9契約差分](b3_driver_g9_contract_delta.md) | schedule/angle/cache/precision/provider/native費用の接続境界 |
| [強い対照の残る不足](strong_comparator_remaining_gaps.md) | 既存G9再利用と、全trial古典費用・C2等の不足 |
| [実行前条件・承認境界](execution_readiness_and_approval_boundary.md) | 技術修正、正式採択、admission、別途科学承認の順序 |
| [Decision Handoff](handoff.md) | Q1〜Q6と次の担当 |
| [差分判定metadata](delta_inventory.json) / [技術指摘](technical_findings.json) | 各分類と実行前の解消要件 |
| [証拠索引](evidence_index.json) / [SOURCE索引](source_inventory.json) | 固定commit、field/関数、bytes/SHA256、読解範囲 |
| [管理検査](verification.json) / [公開照合](publication_receipt.json) | 非数値の整合性・保全・公開同一性 |

## 証拠の強さ

公開driverの人工84件PASS（新36＋既存48）、mock800 trialは**既存記録の引用**。[DTEST][DTEST] 今回そのテスト・preflight CLI・driver・G7/F3をimport/実行していない。SOURCE/テスト/schemaは静的に読んだ。F01は失敗分岐の変数寿命を読んで特定したもので、実科学runで観測した失敗ではない。

G7/G9の基本監査・数学的証明・既存native計算・ランキング・先行研究調査は前回の固定証拠を再利用した。G9の既存matched結果を引用する場合に新compileは不要で、B3の完成からC1 providerの採択や価格bindingが成立するわけではない。

独立branch `track-b-g7-g9-p1-final-driver-delta-20261019`、独立checkoutで新文書・管理JSONだけを追加。共通索引・研究概要・既存SOURCE/raw/契約/claim/STOPは変更しない。Codex Aのprivate worktree/index/lock/processへアクセスしていない。公開固定Git objectだけを取得し、他branchをmerge/rebaseしていない。

**0件**：新科学計算、保存科学値再解析・再ランキング、科学SOURCE import/実行/変更、新adapter/driver実装、sampling/profiling、回路build/compile/synthesis、B2/P2/P3、新保証・重要科学レビュー、科学条件/研究方針/provider採択、authorization/grant/marker/claim発行・消費・再利用。

正式共通指示2026-10-17の同一bytesと§8.4を再確認。観測時点CPU32、available RAM約49 GiB、/tmp空き約374 GiBで、独立Git/hash/read-backを効率化した。共有設定・venv/cache・他jobに手を加えず、旧固定resource契約を維持した。

公開・保全確認後は**mandatory STOP**。次はユーザーがF01の限定技術修正をCodexへ依頼するか判断する。修正・再固定後も、条件採択と新one-shot実行承認は別に必要である。

<!-- FIXED_REFERENCES -->

[BREAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/README.md
[CHECK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/p1_final_delta_checklist.md
[DREAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/README.md
[DSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py
[DTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/artificial_test_results.json
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/README.md
