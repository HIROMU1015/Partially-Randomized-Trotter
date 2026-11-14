# B/D向け：LD1 Q3候補から元価格receiptへの索引

本ディレクトリの[REPORT](REPORT.md)／[ADMISSION.json](ADMISSION.json)／[COVERAGE.csv](COVERAGE.csv)／[SOURCE_FACTS](SOURCE_FACTS.json)／[追加検証1案](NEXT_VALIDATION_PLAN.md)。E[S]26/26接続、P1/P2 freshのm0/m6各X/Yを有限pathとして受入、保留0。新科学計算・compile・平均・残差fit・Q3再評価・Gは0。

**受入索引・reader・原receipt copyの固定commit：`c99d511e2422b8f250490b11054f754a81e536c5`。** 入口は同commitの`docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json`。各`archived_path`はこのcommitで解決できる。元価格の生成SOURCEやcohortをこのconsumer commitへ付け替えない。

## Bの正本からたどる

1. `a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json`の`candidates[label=PR_DF_LD1].rows`から元configuration/physical ID/m/axis、元candidate、total decoder用candidate、評価済みallocationを読む。
2. 本`ADMISSION.json#rows`の`Q3` locatorがB行を一意に指す。`alignment`は旧C `0b4b2d941…`の元行を保持。`original_law/total_law`はB自身の固定RESULT内のselector。新total IDを元receiptのID欄へ代入しない。
3. `structural.source`から旧C `9d95d3ea…:…/COVERAGE.json#values[i]`へ進む。数値は`mean_selector`、区間は`enclosure_selector`、lawは`law_source`。旧configuration `587d20f7…`と元/total configurationの物理対応は今回のmetadata比較で接続したが、旧記録は変更していない。
4. m0/m6の`finite_paths[*].source`からD `39ddd59e…`のP1/P2 RESULTの元path行へ進む。`scientific_source`は元実行SOURCEであり今回のcommitへ置換しない。`original_price_configuration_id/original_price_candidate_id`、cohort/index、seed/law selector、event digestを保持する。
5. `original_receipt`は元private resultの元bytes、`axis_receipt`は元cosine/sine receiptの元bytesを指す。locatorは`original_path/sha256/archived_path`。今回の公開`receipts/<hash>.json`を読めば元private環境がなくても検査可能。各軸のCと同pathのS、fingerprintを`pair`へそのまま転記した。新しい差分や平均値はない。

P1は元各2path、P2 fresh各4path。P2 replay各1pathは`ACCEPTED_REPLAY_REFERENCE_ONLY`、`counts_as_nonreplay_path=false`。全pathは元DEVELOPMENT、PRNG IID保証はnull。path数nとBの量子shot Nは別単位である。

## consumerで維持する境界

- 索引は平均価格provider recordではない。`compiled_expectation.E_C=null`、全26軸UNQUALIFIED、十分N/G/winner=null。
- C guardのreader/新ZZ emitter SOURCE確認を再利用し、guardは無変更。平均価格を作っていないので`G.check`/`cost`は呼ばない。将来外部で平均/モデルrecordを作る際は、元guardの必要なsidecar/implementation証拠を保持し、今回の受入を全law資格とみなさない。
- scalarは元contextのc＋s_R・総T一回。元receiptに数値直記はなく、実行SOURCE＋入力/preparationの同一性から結び付けた。保存closureのtotal scalar値/hexはSOURCE_FACTSに出所付きで保持。現在のD requestへscalarを補って過去recordを書換えることはしない。
- full-wrapper statevector validation=false、law distance/rare/CI/covariance/長q/R移送UNKNOWNを保持。旧C1は異なるlowering/domainで代用しない。
- Bの固定Q3結果は、この固定USER_SUPPLIED配分で条件付きモデル閾値適合済み。十分Nや全物理RMSEの資格ではなく、価格との積算は未実施。LD2へ移送しない。

## 最初の次段階

推奨1案はNEXT_VALIDATION_PLANのm3追加較正→凍結後m4 holdout。保存P2 fresh m0/m6を較正に再利用し、P1/replayはdevelopment参照のまま。最初は仮4path×X/Y＝8新wrapper、後続holdout仮8wrapper。n4・PR用許容誤差・law目標・科学callの採択/実行許可が残る。PF2%/旧pilot2%・95%を採択済み規則として使わない。

ロードマップ更新担当へ：LD1のE[S]/P1/P2 consumer binding待ちは本scopeで解消。正式E[C]/価格精度/law/rare/独立検証/長q-R移送とN/G資格は未解消。ロードマップ本体はCでは変更していない。

**mandatory STOP：本索引と計画の公開で終了。追加科学計算はユーザーの明示許可後。**
