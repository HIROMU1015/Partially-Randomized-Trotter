# 決定論PF 6候補：全roundの価格coverageと次の検証点

**計画のみ。156軸中、同条件の保存直接価格を再利用できる範囲は48軸（m3～m6）。残る108軸は、別actual幅の短round36軸と、未検証の長q72軸である。** m6の12/12完全一致は既存結論として引き継ぎ、再判定・再取得していない。

最初の新規科学検証には **m9・6候補・X/Y各別、最大12 compile** を推奨する。m7→m8→m9の逐次取得は不要。これはm6の8倍のqにおける凍結モデルの検証案であり、実行も2%規則の拡張採択もしていない。[有限計画・判定案・STOP](NEXT_VALIDATION_PLAN.md)。先に行う保存m3のconsumerへのmetadata bindingはcompile 0件で済む。

新しいH4 preparation、sampling、回路build/compile、statevector、RZ再計数、科学値の再集計・fit・未観測価格予測、N/G計算は0件。C凍結JSON、D Stage 1/2、route別実行SOURCEとSTOPは不変。**mandatory STOP。**

## 既存C表への差分

正本は `0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json`。固定CELL_PLANの6 configuration IDで選択する（結果は `/rows[0:156]`）。旧B0I S2/幅0.1の別候補は含めない。行を複製せず、[COVERAGE_DELTA.json](COVERAGE_DELTA.json) のround規則とm3の元receipt locatorをoverlayする。X/Yはそれぞれ独立のreceiptに対応する。

| 分類 | 意味 | 軸数 | 対象 |
|---|---|---:|---|
| A | 同条件の直接価格を受入可能 | 48 | m3保存12、m4/m5較正24、m6 holdout12 |
| B | 保存候補あり、証拠不足または条件差で保留 | 0 | 今回選択した同物理点の追跡では残らなかった |
| C | 調査scopeに直接価格なし、凍結モデルの適用範囲の検証が必要 | 108 | m0～m2の36、m7～m12の72 |
| D | 調査範囲で所在・対応を確定できない | 0 | 既知carrierの該当候補は解決。未列挙archiveの存在は別にUNKNOWN |

Aは**価格の物理的同等性と再利用可否**の分類であり、全156軸のC guard通過や正式G入力資格を新たに認定したものではない。m3の12軸は旧canonical IDのまま保存し、Cの現在のconfiguration/axisへ元SOURCE付きでmetadata bindingする作業が残る。Stage 1/2の36軸は既存bound receiptをそのまま使う。consumerの追加条件で未完了gateが判明した軸はBへ戻して保留し、再compileで穴埋めしない。

| 候補（各セルはX/Y） | m0～m2 | m3 | m4・m5 | m6 | m7～m12 |
|---|---|---|---|---|---|
| Full / S2 16h/7 | C×6軸 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |
| p7 B0I / S2 16h/7 | C×6 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |
| Full / Yoshida4 16h/3 | C×6 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |
| p7 B0I / Yoshida4 16h/3 | C×6 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |
| Full / M5 16h | C×6 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |
| p7 B0I / M5 16h | C×6 | A×2 | A×4 較正 | A×2 独立holdout | C×12 |

h=0.1、M12、linear H4/1.00 Å/STO-3G、保存tol-only DF10。Fullはone-body＋fragment0..9、p7はone-body＋fragment0..6、生成順の往復S2 sweepをsigned PF piece順に連結する。比較量は測定付きfull controlled Hadamard wrapperのcompiled RZ数、P0除外。決定論PFなのでtrajectory平均の標本誤差はない。価格モデルのsystematicは別問題である。

## 短roundは同じmodel_groupではない

以下のT/q/幅は正本C表のmetadataを転記したもの。価格の予測計算は行っていない。両route共通。

| 公式 | m0：T/q/actual幅 | m1 | m2 | m3 | 凍結較正・m6の幅 |
|---|---|---|---|---|---|
| S2 | .1 / 1 / .1 | .2 / 1 / .2 | .4 / 2 / .2 | .8 / 4 / .2 | .2285714285714286 |
| Yoshida4 | .1 / 1 / .1 | .2 / 1 / .2 | .4 / 1 / .4 | .8 / 2 / .4 | .5333333333333333 |
| M5 | .1 / 1 / .1 | .2 / 1 / .2 | .4 / 1 / .4 | .8 / 1 / .8 | 1.6 |

`43c15bfa34a22f313e55b824223b200332b7f23f:docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json` はm4/m5/m6の36行のみを12 model_groupへ割り当てる。m0～m3にその割当はない。requested幅、公式名、qだけからshort roundを同じgroupに入れない。m3が直接価格で埋まっても、それは別幅へのモデル移送の合格を意味しない。

S2ではm1/m2/m3のactual幅.2が同じでも、qとtotal scalar phaseが異なる。Y4ではm2/m3だけ幅.4が同じ、M5ではm0～m3の幅はすべて異なる。保存m3を別のTへ付け替えたり、m3とm4のp7 M5価格が同数であることを同回路の証拠にしたりしない。短roundは有限の直接価格取得が最も単純で、別幅モデルを作る必然性はない。

## m3保存12軸の元証拠

下表のX/Y価格は既存値の転記であり、再compile・再計数・再fitではない。元axisはX=cosine、Y=sine。各fingerprint、scalar、prep、SOURCE、元selectorはdeltaに保存した。

| route / 公式 | physical ID | q / actual幅 | 保存RZ X / Y | 元証拠 |
|---|---|---|---|---|
| Full S2 | e2b139ad6211e2c63ae6 | 4 / .2 | 31439 / 31439 | PFC `/native_results[129]` |
| p7 S2 | 318d7ec57316942b93d0 | 4 / .2 | 22359 / 22359 | PFC `/native_results[42]` |
| Full Y4 | 2594f35f92e76d58874c | 2 / .4 | 47053 / 47053 | PILOT `H4_TOL_FULL_68d5c67e330e9c5f__0/result.json` |
| p7 Y4 | f144b792e8889063951f | 2 / .4 | 33385 / 33385 | PFC `/native_results[40]` |
| Full M5 | 1b3aa083ff69842d8f31 | 1 / .8 | 85928 / 85928 | M5 direct receipt、旧ID M5_MIN_e847f177d7169dd3 |
| p7 M5 | 4bf5a2525d402abaad45 | 1 / .8 | 60950 / 60950 | PFC `/native_results[38]` |

固定locator：

- **PFC**：`0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c:artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/` の `native_results.json.gz`、`native_result_inventory.json`、`PF_endpoint_costs.json.gz`、`source_input_identity.json`。生成SOURCE `e2179f68f6979f3ad26392bff877fb422ddfb744`。
- **PILOT**：`3eca46148070ba4d4b0f5815ef45943407cc959e:artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/H4_TOL_PILOT_DET/H4_TOL_FULL_68d5c67e330e9c5f__0/result.json`。生成SOURCE `c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。
- **M5**：`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb:artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/` の `direct_measured_axes_92.csv`（candidate_id＋axis）、`stage_completion_receipts.json`、`private_raw_inventory.json`。生成SOURCE `635ba563766875d688ad6912499b7f90397d0cc5`。既存private原trialを**JSONだけ**読み、result SHA `d258be425371f876e83c9bb0b91a10c01e708b726016c695d0071e883a7ad5da` と公開completionを照合した。raw/QPY/vectorを新規に開封・再生していない。
- carrier：`155529e20a8fac39e00786c40b57830b71203c4f:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/reused_compiled_prices.json.gz` のphysical ID key。上記一次記録へのproof・result SHAを持つ。
- Full PFの既存receipt監査：`e7dca5f7301c8633b17a59b6a7161331303aef2d:docs/research/d_full_pf_full_h_v2_final_receipts_20261113/{REPORT.md,COVERAGE_V2.json,RECEIPTS.json}`。Full S2/Y4/M5の該当行はEXACT。136候補全体の監査をやり直してはいない。

確認内容：

1. H hash `c1a6c428…`、state `9bcc0b0e…`、科学snapshot binding `12219b7d…`、tol-only DF10、T=.8、prefix、q、公式を元recordへ照合。PFCの復元containerのbyte hashは旧snapshotと別であるが、保存済み科学array同一性記録・H/state hash・prep同一性を根拠とする。新たな入力復元はしていない。
2. prep hashはFull `bb736dc3158aa52f205ae0c23e4be374ae4e5edfa614d3cbceda04dcd5a28cfd`、p7 `fc83f26f1612a41d1d34caae79e78d3b5ce8e961d0e442ca4a9e92168006cf31`。Stage 1と一致。保存total scalarはFull 2.29310124732、p7 2.293146549384713。Fullはc_H、B0Iはc_H＋discarded-tail identityをtotal Tで一回適用する。同値性は保存scalarとの完全一致とSOURCEで確認し、新しいscalar計算はしない。
3. `h46_m5_plan.py` の全bytesは旧3生成SOURCEと基準 `af284d853890155bacd19e61850c196bd1d5c660` で同一（SHA `2ad8cfa8e5ae71a8100fe16c71995e86236d2a2eee750af5e52034ab31e3e832`）。weightsとpiece順、S2 sweepのfragment順を保つ。旧 `h46_m5_native.compose` のASTも3 SOURCEで同一。旧の固定T=.8と現在の可変Tの差は、今回m3では同じTを与える。
4. 元SOURCEの決定論呼出しは `build_deterministic_native(... formula='2nd', scalar=0., control_policy='symmetric_directional')`、signed pieceを順にcompose、最後に `p(-scalar*T)`。RTEを使わず、DF-tail旧/新controlled-ZZの価格を混ぜる問題はこの6候補にはない。識別は `H46_LONG_TIME_NATIVE_COMPOSE / NOT_APPLICABLE_NO_RTE` と決定論lowererのSOURCEで行う。
5. old callerと現callerは、測定付きnative Hadamard wrapper、X/Yを別構築、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を使用。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。hashだけでlowering同一とせず、既存source_checksの本文証拠を再利用した。Full M5の元configも公開private inventoryのSHAに一致し、そこに固定されたnative v5（wrapperを含む）/価格compilerのprovider bytesは `c31ef424175c02379eafebe8fc6c7548c1d9aabe` および現基準af284と一致した。元生成SOURCEがcatalogでnullだった箇所をこの一次記録で解決した。
6. 元trialの保存PASS、low/high-control、独立occupation作用、Hadamard/compiled-vector両軸の既存gateを参照した。誤差normを計算し直したり、保存状態の作用検査を全入力状態の保証へ拡張したりしていない。P0・wrapper範囲と元STOPを保持する。

旧Full PF direct/旧qモデル、他rank、別prefix・幅・formula、PR/RTE・Pauli価格は、qや表示RZが同じでも今回の価格として受け入れない。

## Stage 1/2は既存索引を再利用

- 較正：`b190bd815c886446291fac8b281c3c7df7343372:docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json` の `routes/groups`。各routeのbound `PRICES/C_INPUTS/SIDECARS/GUARD_CHECK` を参照する。m4/m5の24価格（保存4を含む）。
- 凍結：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`、SHA-256 `845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a`。
- 独立holdout：`a55813373b84b9128cbd10c6812c0d26143a4280:docs/research/d_pf_stage2_holdout_20261115/{HANDOFF.md,run01/RESULT.json}`。12個別PASS、凍結予測とRZ完全一致を引用するだけで、再判定しない。
- Full/p7の実行SOURCEはStage 1 indexとStage 2 HANDOFFのroute別値を保全する。較正・holdout・旧m3を混合して再fitしない。凍結モデルの `qualified_for_candidate=false`、長q未資格も変更しない。

## §10.5の調査範囲と限界

既存Stage 2の `DUPLICATE_AND_INPUT_PREFLIGHT.json` を引き継ぎ、対象T/q/actual幅を展開して差分照合した。長時間carrierのexploration 481、extension_parallel 22、confirmation 39、fixed_delta 1、reused 203 recordをmetadataで検索した。対象物理点への一致はm3の6 cellと既存M5/m4の2 cellだけ。これはcarrier重複であり、独立標本数ではない。Full catalog 136候補も既存調査を再利用した。

関連旧branch `codex/d-full-pf-full-h-rte-v2-paths-20261113` はe7dca5f73、当D branchはStage 2公開a55813373。固定C/A対応表とreceiptの生成SOURCEを照合し、他branchは変更していない。既知private `cost/{EXPLORATION,CONFIRMATION}/<physical_id>/0/result.json` の対象m0～m2/m7～m12の正確なpathに追加trialはない。自ユーザーの関連既存runner名のprocess snapshotにも該当jobは観測しなかった。

未列挙private cohort、別サーバー・別名runner・scheduler job、未fetchの他担当更新は網羅していない。この部分はUNKNOWNであり、「全履歴で未実施」ではない。科学実行前に変更分・関連running jobを再確認し、出所不明の重複候補があればその軸を保留する。

再現：`python3 -B docs/research/d_pf_allround_price_coverage_plan_20261115/check_metadata.py`。stdlibと固定Git JSON/SOURCE text、限定した既存private JSONだけを読む。[delta](COVERAGE_DELTA.json) に66件のmetadata整合check、元source SHA、axis別fingerprint、検索scopeを記録した。科学的精度の新検証ではない。初回のchecker作成時に旧recordのcompiler field名/PASS field形式の違いを修正し、元recordや科学コードは変更していない。

## ロードマップへの影響

統合担当への差分は「決定論PF shortlistはm3～m6の48軸を保存証拠で埋められる。m6再実行不要。短round別幅36軸・長q72軸は未資格。次はm9の12軸を提案、m12未資格を維持」である。参照したロードマップの固定locatorは `fa84c0d6bce0c8e8b8accbe6009069074b466d32:docs/research/RESEARCH_ROADMAP.md`。その較正/holdout未取得という旧記述はStage 1/2一次証拠に追い越されている。

ロードマップ本体は編集しない。Cはm3のconsumer bindingと次の独立検証予測の固定、Dは承認された有限取得、Bは別途N/riskを担当する。今回Nとの積算、G比較、候補採択・winner、全round価格認定への進行はない。
