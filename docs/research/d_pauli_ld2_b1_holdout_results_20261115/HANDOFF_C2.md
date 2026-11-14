# C2向けHANDOFF：LD2 m5/m6独立holdoutの記述的比較

campaign **`DESCRIPTIVE_COMPLETE`**、model_verdict **`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`**。fresh3 path・新規実compile6、保存holdout再利用0、個別receipt有効6/6、4比較単位。モデルのPASS/FAILは付けていない。[REPORT](REPORT.md)に全6価格、gate、event、資源、個別固定locatorを収録。

## 1. 固定入力と元結果

- D2実装SOURCE `13ab1acfaebe5f33958e1db67862f5c45725ab91`、実際の実行HEAD `5dac74959ed9ea49465d393fc792ef5b4727edba`（実装bytes一致、変更0）。
- raw公開commit **`0e1ce866e8f8d4e042801260fb8d543d0b1bee53`**。`run01/M5_ROUND/`と`run01/M6_ROUND/`のRESULT/COMPILE_RECEIPTS/C_INPUTS/PRICES/IMPLEMENTATION_EVIDENCE/SIDECARS_UNBOUND。実行code SHAと結果公開SHAは別々に保持。
- 規則 `f4480ee3f831a7b106d287c401ee78fbd62f8615` の `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/HOLDOUT_RULE_PROPOSAL.md`。今回ユーザーが記述案を採択した。Pauli数値許容差は設けていない。
- 予測正本 `b74e4e536392c2bbe94cd371436615fc56c0564f` の `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`。保存exact predicted_RZを(m,axis)で対応付けた。C2 branch merge/再fit/予測再生成0。
- cell/physical ID/cohort/seedsは元plan043cのLD2 M5_ROUND/M6_ROUND、全詳細と開封前固定記録は[PREOPEN](PREOPEN.json)。B0/W/CAL標本を含めない。

## 2. 4件のexact比較

| 単位 | P（凍結exact） | O（receipt観測exact） | d=P−O | abs(d) | d/O | abs(d)/O |
|---|---|---|---|---|---|---|
| m5-X | 6169957875465/90343469 | 68264 | 2751307649/90343469 | 2751307649/90343469 | 2751307649/6167206567816 | 2751307649/6167206567816 |
| m5-Y | 6169957875465/90343469 | 68264 | 2751307649/90343469 | 2751307649/90343469 | 2751307649/6167206567816 | 2751307649/6167206567816 |
| m6-X | 13521761923033/90343469 | 149435 | 21285633018/90343469 | 21285633018/90343469 | 1637356386/1038498176155 | 1637356386/1038498176155 |
| m6-Y | 13521761923033/90343469 | 149435 | 21285633018/90343469 | 21285633018/90343469 | 1637356386/1038498176155 | 1637356386/1038498176155 |

符号付き相対誤差の表示：m5-X 0.044612%、m5-Y 0.044612%、m6-X 0.157666%、m6-Y 0.157666%。正のdは過大予測、負は過小予測。exact値が正本で、%の丸めを判定に使わない。

[COMPARISON.json](COMPARISON.json)の各rowにP/O/d/絶対誤差/符号付き相対/絶対相対を有理数で保存。m5は同軸2 pathの平均、m6は同軸1 path。欠測は片path平均で補わずO/誤差null、O=0の相対はnull/理由ZERO_OBSERVED_PRICE。許容差必須validate APIは呼んでいない。

## 3. 個別receiptと確認入口

[RECEIPT_ADMISSION.json](RECEIPT_ADMISSION.json)の`units`4項目、各`paths`に計6 receiptを収録。各pathに以下を保持する。

- 元RESULTとCOMPILE_RECEIPTSの`{commit,path,selector}`、実行SOURCE、canonical physical ID、cohort/index/seed。
- 軸別fingerprint、元receipt由来のcompiled_RZ、作用gate、実装guard確認状態。
- 同じraw公開SHAのIMPLEMENTATION_EVIDENCE locator。

[ADMIT.py](ADMIT.py)は固定13abの `compile_receipts` / `check_path` / `check_round_records` を再利用する。各cellの公開後 `SIDECARS_PUBLISHED` と `C_GUARD_CHECK` を別保存した。個別receipt・作用/実装gateの照合完了後に[COMPARE.py](COMPARE.py)で4差を算出した。D2 reader/guardの技術PASSはモデル精度のPASSではない。ADMITは取得時の一回用writerで、C2は保存済みJSONと固定locatorを読めばよい。

rawのQPY原本は専用private cacheに保持し、公開receiptに所在/hashがある。C2別worktreeやモデル正本は変更していない。C2が受領する際は本HANDOFFの最終公開commitと、上記raw公開commitを区別する。

## 4. 証拠の範囲とSTOP

実LD2 order2 productは未観測UNKNOWNを維持した。 非自明event/D2位相も未観測UNKNOWNを維持した。 実eventと出所は[EVENT_COUNTS](EVENT_COUNTS.json)。追加sampling0、較正・親子・X/Yの独立性を水増ししない。各pathの固定作用/compiled-vector gateと、4件の記述差は別の確認である。

凍結modelsの未検証・未資格状態を維持する。今回の結論は指定有限cohortの4予測差まで。全law E[C]・CI、m7以後の長round価格、N/G・winnerは未認定。事後閾値、再fit、追加path、LD1へ進まない。

ロードマップ影響は「C2凍結4予測→独立holdoutの有限記述差取得・公開まで完了、数値許容差によるモデル判定なし」。本体更新は統合担当へ委ねる。D2 branchへのcommit/pushとremote一致確認、引渡し後に**mandatory STOP**。
