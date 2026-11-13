# B：decoder統計の入出力・担当

2026-11-13。branch `codex/rpe-decoder-statistics-20261113`、基点B3216d27a6。[REPORT](REPORT.md)／[次の有限科学実行案](SCIENTIFIC_PLAN.md)。本HANDOFFを含む公開commitをSOURCEに固定する。A/C/Dの科学実行や過去STOPは変更しない。

## A → B：13round physical mean

既存V2 configurationとN packetをそのまま利用する。mean recordは各roundで以下を保持する。値はAが供給し、Bは物理mean/normalization/局所誤差/Nを再計算しない。

```python
mean = {
  'm': m, 'identity': canonical_round['identity'],
  'law_binding': canonical_round['law_binding'], 'source': numerical_source,
  'raw': {'real': raw_X, 'imag': raw_Y},
  'corrected': {'real': corrected_X, 'imag': corrected_Y},
  'normalization': {'kind': 'FINITE_LOG_B_SUPPLIED_BY_A', 'log_B': log_B},
  'target': target_descriptor, 'qualification': 'UNQUALIFIED',
  'bias': bias_evidence, 'uncertainty': error_and_precision_evidence
}
shots = {'m': m, 'N_X': exact_integer_or_None, 'N_Y': exact_integer_or_None}
```

target_descriptorは元H/state/snapshotのbinding、`quantity='ORIGINAL_DF_H_TARGET_ENERGY'`、energy/value/reference誤差のSOURCE、`energy_shift_ha=0`を持つ。未取得energy/誤差はnull。uncertaintyはfinite-K/PF bias、state/target、law距離、実装/数値誤差とprecision/資格/SOURCEを区別して供給する。未知はnull。mean.sourceは数値carrier、実装SOURCEとは必要に応じて別欄で明示する。

既存A model recordの物理identityからは`rpe_decoder_statistics.mean_from_a(config,m,record,source_request_id=...,law_binding=...,source=...,target=...,qualification=...,uncertainty=...)`でcopyする。source_request_idは既存N packetの同round行を正本とし、law_bindingは元A requestのSOURCE宣言。canonical IDへ旧recordを単純に付け替えず、物理candidate/binding/structure/profile/time/qとrequest IDを照合する。

`prepare(config,means,shots,target=...,shot_policy=...)`はmetadata検査だけ。shot_policyは出典とmeaning（NOMINAL/USER_SUPPLIED/SUFFICIENT_EXTERNALLY_QUALIFIED）を明示する。meaning名だけで十分Nを認定しない。mean/shotがそろわないroundも残る。今回の保存準備ではnominal 64/60/38だけを転記し、実meanは統計へ渡していない。

## B：count → decoder → loss

- `count_mean(plus,shots)`：任意精度整数の2k−Nを作ってratioをbinary64へ一回変換。exact rationalも保持する。旧count-grid callerとは別の明示conversion ID。Nをfloatへ丸めて保存しない。
- `outcomes_count([+1,-1,...])`：同じ集計経路。bool/欠測/0/非±1を拒否する。
- `decode_counts(rounds,h=...,M=...,prior=...)`：全round X/Yのplus/shotを既存A decoderへ直接渡す。priorは診断のみで選択規則を変更しない。未解決energyはNone、missingはNOT_EVALUATED。
- `simulate`：明示IID policyのもと、供給raw meanを条件とした独立binomial countを生成する。seed/反復数は入力。NumPy int64上限を超えるNは丸めず拒否する。実RTE trajectoryや実H4 meanの生成はしない。
- `simulate_prepared`：prepareの値を受ける薄いadapter。欠測ではsimulationしない。物理未資格ではdefault停止し、明示`allow_conditional_unqualified=True`でも条件付き診断としてのみ実行。実H4の呼出しは今回未許可。
- `exact_counts`：小人工分布を有限列挙。callerのmaximum_statesは有限実験scope。SciPy binomial PMFのbinary64値であり、厳密な確率enclosureではない。
- `summarize`：targetとの差の二乗。UNDEFINEDの未解決lossでは全RMSEを作らない。BOUNDED_INTERVALならenergyを代入せずloss区間を返す。条件付きRMSEと全試行への解決分寄与を区別する。CIは明示IID反復/confidence/loss supportが必要。実mean誤差・十分N/global boundはnullのまま。

独立性policyは`{kind:'IID_SHOTS_INDEPENDENT_AXES_ROUNDS_AND_REPETITIONS',source:...,target_law:...}`。相関を含む観測countsはdecodeできるが、独立binomial simulationやIID CIへ黙って移さない。

## D更新

`rpe_d_evidence_intake.accept_receipt_coverage(previous,current,receipts)`を追加。previous=92ff17deのCOVERAGE、current/receipts=e7dca5f7のCOVERAGE_V2/RECEIPTS。旧分類56/17/63と価格値/physical/sourceを保全し、17件のSOURCE付きaxis receiptをmetadataとして検査して73/0/63へ更新する。receipt追跡・RZ計数を再実行しない。136候補をmenuへ登録せず実Gも計算しない。

## 再現と次の担当

metadataのみのCLI：

```sh
python -B scripts/prepare_rpe_decoder_statistics.py --out /tmp/rpe-statistics-fresh
python -B tests/check_rpe_decoder_statistics.py
```

CLIに科学実行optionはない。[metadata01](metadata01/PREPARATION.json)は実mean未輸入・H4統計未実行、D更新は[D_RECEIPTS_ACCEPTANCE](metadata01/D_RECEIPTS_ACCEPTANCE.json)。原decoder・AのRMSE/shot式・B controller/integration/cost/select・C providerは無変更。

**A**：13roundのphysical mean/normalization/source/error、元target/stateを補完する。**ユーザー**：count caller、shot配分、joint law、未解決loss・loss support、精度/confidence、有限反復scopeを判断・許可する。**B**：許可された全loss統計を実行してconditional/empirical/CI/十分Nの資格を分ける。**C/D**：compiled平均/残差・law/domainを担当。科学計画の採択や実行、実Gへ自動進行しない。今回公開照合後**mandatory STOP**。
