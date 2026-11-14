# Full-H κ2：次回1配分の有限Q3計画（未実行）

本計画の公開は科学実行許可ではない。次回は以下の**新vectorと新しい有限範囲**を明示した実行指示が必要。同等結果が受入可能なら再利用し、新規callから除外する。重複が判断不能なら保留する。

## 1. 対象と不変条件

- 対象は保存H4/1.00 Å/STO-3G/DF10、sector(2,2)、**Full-H pure RTE κ2・K2だけ**。one-bodyを含む全H、元grouping/q/R/event schedule、scalar、H/state/snapshot、h=0.1/M12・全13roundを保持。
- 元configuration：`54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`。
- total configuration：`265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`。
- binding：`579c5de23a2336d402a1d5b195061b34b3c7a1b49e99ea2e1a674c1a086189a5`。
- 13physical ID・round canonical ID・raw SOURCEは[PROPOSAL.json](PROPOSAL.json)の`physical_rows`。shot配分変更のために物理IDを作り替えない。
- 13round packet：**ce6aacfaca8249c510baa2d958849cffc7649cdc**、`docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/{PLAN,B_SIGNAL_INPUT}.json`、元`round_<m>.json`。producer5ccc1dfc3、数値5cc7c07e1、sampler準備9b83b96b8。normalization/scalarは適用済みで、再適用しない。
- decoder：`consecutive_dyadic_nearest_branch_principal_energy_total_v1`、kernel **710aa69b663ef3313439762d25c80a652a03948f**。Z1＋T1のzero HOLD、tie argmin、binary64順序、nearest branch、principal guardを維持。
- target：保存full DF10 solver energy **−2.1663874486347634 Ha**の条件付きproxy。元packetは変更しない。target SOURCE bba150834 `docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json:target`、truth proxy参照89c91c692はPROPOSALに保持。
- law：保存ideal finite-K raw、shots/X/Y/round間のIID二項モデル、`EXACT_AFFINE_OF_BINARY64_RAW`。元paired finite-Taylor component/order law・preparation・law bindingを保全。
- loss：全事象`(Ehat−Eproxy)^2`。直接モデルMSEが主判定、bias²・Varは診断。元Minkowskiは別参照で、δEを加算しない。
- τ：十進εE=`0.00159360010199040 Ha`のexact二乗、`15136964828156312001 / 5960464477539062500000000 Ha²`。判定に表示floatを使わない。

## 2. 固定提案vectorと最大call数

```text
allocation_id = FULL_H_KAPPA2_TOTAL_Q3_CONTROL_SCALE2_PROPOSAL_01
m   = [0,1,2,3,4,5,6,7,8,9,10,11,12]
N_X = [128,120,112,104,94,86,76,68,60,50,42,34,24]
N_Y = [128,120,112,104,94,86,76,68,60,50,42,34,24]
最大新規Q3 science call = 1
```

これは旧USER_SUPPLIED controlの全round2倍という**新しい未評価whole-vector**。同一候補・両軸整数列・decoder・target・law・lossの受入可能結果があれば新規0件。旧control、PR-DF LD1/LD2、Pauli、PF6、R448、κ変更、追加配分を対象にしない。適合を予測せず、最小N・単調性を仮定しない。

## 3. count-only包含と提案scope

新vectorだけを使う整数計数で、`C_m=(N_X[m]+1)(N_Y[m]+1)`は以下。

```text
C = [16641,14641,12769,11025,9025,7569,5929,4761,3721,2601,1849,1225,625]
```

| 範囲 | 前回 | 今回の提案（次回指示で採択が必要） |
|---|---:|---:|
| 全roundのρ | 1.0 rad | 1.0 rad |
| maximum_round_outcomes | 4,225 | **16,641** |
| maximum_verification_pairs | 55,043,173 | **847,519,356** |
| maximum_frontier_states（非終端） | 8,192 | **8,192** |
| maximum_transition_pairs | 141,559,036 | **620,478,721** |

verificationは`Σ_{m=1..12} C_(m−1) C_m`。実際のgood-phase集合は全countの部分集合なので、全隣接組を含む保守的上限になる。good集合・phaseの新計算はしていない。

DPはm0で高々C0遷移、その後は非終端frontier≤8192かつ各roundのgrouped count≤C_mなので、`C0 + 8192*Σ_{m=1..12} C_m = 620,478,721`。既存`finite_bounds`（a82b0f0の`scripts/run_pauli_q3_three_quarter.py`）と同じ純整数漸化式。m0は新supportがfrontierを超え得るため、pruning分を元kernelの残余会計へ必ず残す。終端状態は8192に制限しない。

前回使用した**m0全4225 count→2593初期状態・残余0**という保存supportは旧N0=64に固有で、新N0=128の状態数上限には使えない。旧実測遷移94,048,712、旧閉包PASSも新vectorの証拠にはならない。再利用できるのはSOURCE、入力、集約アルゴリズム、上限証明の方法。新probability/phase/閉包/状態分布/riskは未評価。

この新scopeなら、countだけから全round count・隣接検査・frontier後の遷移の包含上限を示せる。旧scopeのままでは収容を示せない。**旧scopeを黙って使って開始せず**、上表の変更を次回承認範囲へ含める。承認時に旧scopeが指定された場合は、保存済みの新N支持・集約数等で包含を立証できるまで開始保留。包含を調べるためのphase/確率/DPを今回や先行callとして起動しない。

## 4. 次回実行前・実行内gate

1. §10.5の変更分だけ再照合する。候補・vector・decoder/target/law/lossが一致する保存成果・実行中jobがあれば再利用又は保留。同じcontrolや他方式の結果を今回の結果に付け替えない。
2. SOURCE/packet・元/total/binding ID・13physical行・law・normalization/scalar不変を照合。整数vectorとcount-only包含を確認する。新しい出力先は`docs/research/full_h_kappa2_scale2_q3_results_20261115/run01/`を提案（現時点で実行出力は作成していない）。既存runを上書きしない。
3. **caller・INPUTS・科学SOURCEを新commitで固定してから1call**。旧`scripts/run_pr_df_fullh_total_q3_fixed.py`は旧3候補/旧vector/旧scopeを固定しているため、そのまま呼ばない。既存のFull-H metadata route→`rpe_total_binding.create/bind_signal/evaluate`を再利用し、この新1件を明示入力する薄いcaller差分だけを作る。kernel、decoder、DP backendをコピー・変更しない。
4. 原API return・partials・CPU/peak RSSを**worker自身**で保存する既存修正と、親monitor例外fixtureを再利用する。必要な差分だけ人工確認し、実科学callをtestに使わない。新SOURCE・新allocation ID・同物理IDを固定し、原return・保存状態・数値判定を別々に記録する。
5. 固定`total_state_refinement`と高速backend9e294778を使用。旧partial `state_refinement`やMCを呼ばない。既存8worker/batch8は実績のある初期運用設定で恒常的資源上限ではない。共有jobへの干渉を避け、自分のjobの並列化・保存だけを調整する。時間/メモリの科学的予測はしない。
6. call内で候補固有のρ=1 binary64閉包、round・DPの質量保存、nonfinite、残余loss、principal guard、既存数値allowance、MSE上下界・momentsの整合を確認。count包含だけで閉包をPASSとしない。frontierによる残余やguard-failed質量を0として捨てない。bias²・Var、zero/tieの追跡mass、全mass/残余loss、coarse/lemma/DPの有効上界とmin、CPU/RSSを保存する。

## 5. 判定規則・STOP

- 必要gateが有効で、MSE上界≤τ：固定配分の**条件付きモデル適合**。
- 有効MSE下界>τ：固定配分の**モデル上の不適合**。
- 有効上下界がτをまたぐ：**INCONCLUSIVE**。保守的上界の超過だけを不適合としない。
- SOURCE/ID/law不一致、ρ閉包不成立、非有限、質量・残余loss・上下界不整合、宣言scope/allowance超過等：**UNKNOWN**、理由・部分結果を保存してSTOP。
- 実際の資源不足・保存不能等の重大異常もSTOP。監視だけの一時異常では正常workerの保存を妨げず、科学状態と運用記録状態を分ける。

**不足分最大1callの終了又は重大異常でmandatory STOP。** 無断retry、追跡上限拡大、frontier/ρ/target/decoder/shot変更、2本目のvector、価格/G/shot探索へ進まない。

新vectorが適合しても、有限候補1本に条件付くモデル評価に留める。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`、ground未認証・実sampler移送未認証・数値certificateなしを保持する。D1較正は待たず、価格の無い段階でGや方式間winnerは出さない。

## 次回指示への転記要旨

> 保存Full-H pure RTE κ2のみ、PROPOSAL.jsonの同一physical/total binding・target/law/lossで、上記X/Yの2倍whole-vectorを各軸固定し、最大1回の新規Q3とexact τ判定を承認対象とする。新scopeはρ1、round16,641、verification847,519,356、非終端frontier8,192、transition620,478,721。実行前に重複・SOURCE・count包含・worker保存を確認して新caller/INPUTSをcommit固定する。gate失敗はUNKNOWN、無断retryや範囲変更なし。終了後commit/push・remote照合してmandatory STOP。旧control・PR-DFは再実行しない。十分N・global RMSE・G・winnerの認定は含めない。
