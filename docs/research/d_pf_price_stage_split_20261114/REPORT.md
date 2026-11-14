# D：Full PF A1・Truncated p7 T3価格callerのCALIBRATION／HOLDOUT段階分離（Stage 0）

2026-11-14 JST、Codex D。**Stage 0（段階分離の技術実装と人工検証）だけを行った。**

比較policyとholdout許容差は、ユーザーによる採択をまだ確認できていない。そのため、Stage 1（CALIBRATIONの新規22 wrapper）は開始していない。実H4のbuild、compile、価格取得、fit、予測、N/Gはいずれも0件である。

- 固定SOURCE：**`1fd47847b3300395f1b0c0de1868ed7b83905497`**（branch `codex/d-price-records-c-bridge-20261114`）
- 関連文書：[HANDOFF](HANDOFF.md)／[実行手順と条件](NEXT_EXECUTION.md)／[test結果](TESTS.txt)

## 実行前条件の確認：未採択

- **指示書**：§5に「現時点では、まだユーザーによる正式採択はありません」とある。
- **Cの提案**：`2fa88065` の `POLICY_PROPOSAL.md` は「提案のみ。採択していない」。`USER_ADOPTED_POLICY_SOURCE` も未確定である。
- **remote**：作業開始時に `git fetch origin` で確認した。C branch `codex/m5-m4-receipt-admission-20261114` の先頭は `2fa88065` のままで、採択を記録したcommitはなかった。

したがって §5 の規定どおり、Stage 0で止めた。2%案は自動採択していない。

## branchとSOURCEの扱い

- **branch**：D価格callerの系列であるbranch `codex/d-price-records-c-bridge-20261114` を再利用した。
  - D②（Pauli B0）は、別branch `codex/d-pauli-ld1-m0-b0-results-20261114` と別worktreeで進んでいる。同じbranchへの競合pushは起きない。
- **旧SOURCEの保全**：固定SOURCE `62ae1bd5`（全セル一括のcaller）は、このbranchの履歴にそのまま残る。
- **Cの成果**：Cの受入bundle `3368805575`／`2fa88065`、およびC guard `c628e9e8` は、変更もmergeもしていない。CのCELL_PLAN、保存receiptの入力、sidecar、guard結果は、git blobとして読むだけである。

## 段階分離の実装

再利用を優先し、価格計算の実装は作り直していない。

### 新規module：`scripts/pf_price_stages.py`

段階の選択、gate、出力の整理だけを担う。

1. **`select`**：CのCELL_PLAN（`3368805575`、`2fa88065`とbyte一致）を役割の正本とする。
   - 再生成した各cellが、CELL_PLANのX/Y 2行と全fieldで一致することを要求する。対象fieldはroute、formula、actual幅、configuration、m、axis、physical candidate全体、action、features、model groupである。
   - roleとroundの対応も照合する（CALIBRATION＝m4/m5、HOLDOUT＝m6）。
   - CELL_PLANの全36行が過不足なく対応していることを確認し、そのうえで指定stageのcellだけを返す。
   - 新しい候補生成器は作っていない。既存の `plans()` の結果をroleで絞るだけである。
2. **`execute_stage`**：処理は次の順に進む。
   - SOURCE byte検査（実行するcaller、T3 caller、stage module、固定native/wrapper）。
   - stage選択と、stage別wrapper数の検査（Full 10/6、Truncated 12/6）。
   - HOLDOUTの場合はgate。
   - stage別namespaceを新規作成する（既存dirがあれば停止）。
   - provider-rootの子processを起動し、そのstageの `NEW_COMPILE` cellだけを渡す。
   - 価格record、sidecar（placeholder付き）、implementation evidence、Cへの入力（model group別）を書き出す。
3. **`run_cells`**（子process側）：
   - 既存の `checked_trial`／`full_trial` を、fork worker上でcellごとに並列実行する。worker数は、既存long-time runnerと同じidle CPU／available RAM規則で決める。
   - 失敗したcellは部分保存し、それ以降は新しいcellを開始しない。retryはしない。
   - 各wrapperのfingerprintとqubit数を、`compile_cached` の呼出し記録から付ける。compileの経路・設定は変えていない。
4. **`check_group`**：Cの1 modelは「1つの (route, formula, actual幅, axis)」かつ「1つのstage」とする。異なるgroup・軸・stage・m6の混入、round不足、重複を拒否する。最後にCの `price_model_validation._domain` を適用する。
5. **`saved_calibration`**：保存M5/m4のX/Yを、Cの受入bundleのrecordとして参照するだけにする（再compileしない）。
   - 参照先は、bundleの `SAVED_CALIBRATION_INPUTS.json` と `PRICES.json`、`2fa88065` の束縛済み `SIDECARS.json` と `GUARD_CHECK.json`。
   - cohortは原 `EXPLORATION` のまま、SOURCEも原carrier `155529e20` のままである。
6. **HOLDOUTのgate**：次のすべてを、compile前に要求する。
   - CALIBRATIONの7ファイル（RESULT、PRICES、SIDECARS_UNBOUND、IMPLEMENTATION_EVIDENCE、C_INPUTS、SIDECARS、GUARD_CHECK）が、HEADにcommit済みで、byte不変で、dirがcleanであること。
   - CALIBRATIONが完了状態であること。
   - placeholderが残っていないこと。
   - HOLDOUTのcellが、CALIBRATIONで取得済みのcellと重複しないこと。
   - Cの凍結model（下記）が揃っていること。

   実行後には、CALIBRATIONの7ファイルを再照合する。変わっていれば `STOP_HOLDOUT_PRICE_CHECK` とする。
7. **`frozen_models`**：`--frozen-models COMMIT:PATH` で指定したC文書の中から、Cの `fit` 出力（`PRICE_LINEAR_FEATURE_MODEL_V1`）を探す。このrouteの6 groupそれぞれについて、次をすべて満たすmodelがちょうど1つあることを要求する。
   - 状態は `CALIBRATED_UNVALIDATED`、featuresは `['one','q']`、`qualified_for_candidate=false`。
   - 較正点（cell_id＋SOURCE）が、公開済みのCALIBRATION入力と完全に一致する（m6を含まない）。
   - domainが一致する。

   groupを混ぜたmodelがあれば拒否する。D自身は実データでfitしない（係数を照合・記録するだけである）。
8. **`bind`**：stageの出力を公開した後、実際の公開commitをsidecarに埋める。そのうえでCの `c628e9e8` guardを、明示的な技術互換性probe policy（`TECHNICAL_COMPATIBILITY_PROBE_ONLY_NOT_ADOPTED`、Cの受入時と同じ形）で実行し、全recordの検査と `DETERMINISTIC_DIRECT` tierを確認する。結果は `SIDECARS.json`／`GUARD_CHECK.json` として書く。正式な比較policyではない。

### 2本のcallerの変更（`run_full_pf_shortlist_prices.py`、`run_truncated_p7_prices.py`）

- **CLIの置換**：`62ae1bd5` の全セル一括実行（`--execute-approved-full-pf`／`--execute-approved-t3`）を削除し、次の3つに置き換えた。旧flagは引数エラーになる。
  - `--execute-approved-stage CALIBRATION|HOLDOUT`
  - HOLDOUT専用の `--frozen-models`
  - `--bind-stage-evidence STAGE --evidence-commit SHA`
- **出力先**：stage別に分けた。
  - 公開側：`docs/research/d_full_pf_shortlist_prices_20261114/{calibration,holdout}/run01`、`docs/research/d_truncated_p7_t3_prices_20261114/{calibration,holdout}/run01`
  - private側：`/home/AbeHiromu/projects/d-{full-pf-shortlist,truncated-p7-t3}-20261114-private/{calibration,holdout}/run01`
- **子process**：cellの逐次loopを `S.run_cells` に置き換えた。各cellで呼ぶtrial関数は従来のまま。
- **共通化**：`fit_inputs`／`c_model` をT3へ移し、両routeで共用する（Full PFでは同名のaliasを残した）。roleは明示必須とし、既定値をなくした。
- **preflight**：stage別の分割と、保存M5/m4の参照を表示する。
- **変えていないもの**：`plans()`、`native_row`、`checked_trial`／`full_trial`、`price_bindings`、`provider_checks`、cohort名、分類、scalar quantity、保持block、tail identity、index0の作用gate、compiler、native、wrapper。

### recordに追加した情報

`record_schema` は `D_PRICE_RECORD_B_KINDS_V2` のままである。追加したのは次の情報だけである。

- `source.stage`、`source.stage_module`、`source.cell_plan`
- `diagnostic.stage`
- `diagnostic.compiled_wrapper`（fingerprint、wrapper／compiledのqubit数、cache再利用）

RESULT.jsonには、cellごとのtrial（全compiled metric、gate、wall秒、peak RSS）、worker数とresource観測、経過時間を残す。C guardは `diagnostic` を参照しないため、既存の検査はそのまま通る。

## CALIBRATION 22／HOLDOUT 12の分離（CELL_PLANと照合、metadataのみ）

| 方式 | stage | cell（formula m・q） | 新規wrapper |
|---|---|---|---:|
| Full PF p10 | CALIBRATION | S2 m4 q7・m5 q14、Y4 m4 q3・m5 q6、M5 m5 q2。M5 m4 q1は保存receiptの参照で、compileしない | **10**（＋保存2） |
| Truncated p7 B0I | CALIBRATION | S2 m4 q7・m5 q14、Y4 m4 q3・m5 q6、M5 m4 q1・m5 q2 | **12** |
| Full PF p10 | HOLDOUT | S2 m6 q28、Y4 m6 q12、M5 m6 q4 | **6** |
| Truncated p7 B0I | HOLDOUT | S2 m6 q28、Y4 m6 q12、M5 m6 q4 | **6** |

- 合計：CALIBRATION 22＋HOLDOUT 12＝**34**。CALIBRATIONで使う価格は、新規22と保存2の計24件になる。
- 実幅：S2が16h/7、Y4が16h/3、M5が16h。
- model group：方式2×公式3×軸2＝12。Cへの入力（`C_INPUTS.json`）はgroup単位で、新規recordと保存recordを分けて持つ。
- 保存M5/m4（`0cd1527c80f7b9841a83`）：Cが受け入れたX/Y各85,960 RZ。Cのrecordをそのまま参照しており、Dは数値を変えたり再計算したりしていない。

## テスト

[TESTS.txt](TESTS.txt) を参照。すべて作者自身の技術確認であり、独立レビューではない。

- **D**：54件PASS。
  - 新規：stage parent 13件、stage child（artificial circuits）3件。
  - 回帰：Full PF parent 6件／circuit 2件、Truncated parent 3件／circuit 5件、Pauli pilot parent 8件／circuit 5件、Pauli接続 8件。
- **C**：回帰48件（`check_paper_g_price_routes` 34、`check_price_implementation_guard` 14）PASS。

新規テストで確認した内容は次のとおり。

- CELL_PLANが正本であること、受入公開版とbyte一致すること。
- CALIBRATION 22／HOLDOUT 12／合計34、保存2件はjobに入らないこと。
- 公式・幅・q・round・X/Yの対応、Full PFとTruncatedの同一q/T/幅。
- CELL_PLANとの不一致8種で停止すること（m6をCALIBRATIONに、m5をHOLDOUTに、physical ID、保存の再compile化、X/Yの役割分裂、行の欠落、行の追加、group）。
- 架空の子processによるstage実行：
  - 名前空間がstage別になること。
  - recordの分類とstageが正しいこと。
  - placeholderはUNBOUNDにだけ残ること。
  - 同じstageを再実行すると、子processを呼ぶ前に停止すること。
- bind：未公開の状態では拒否されること。公開後はC guardが全件を検査し `DETERMINISTIC_DIRECT` になること。二重のbindは拒否されること。
- HOLDOUT：
  - 凍結modelなし、CALIBRATION前、未bindの場合は拒否され、子processも出力dirも作られないこと。
  - 凍結modelの欠落・混合・m6混入・別SOURCE・検証済み・feature追加・重複・軸違いを拒否すること。
  - 正常な場合はm6のみを取得し、CALIBRATIONのファイルはbyte不変であること。
  - 実行中にCALIBRATIONが変更された場合はSTOPになること。
- model groupの混合を拒否すること（公式・軸・stage・m6・route・round不足・重複）。
- 実gitでの `committed`／`source_gate` の動作。
- CLIに旧一括flagがないこと。
- 子process：並列cellでtrial関数・scalarが不変であること、X/Yのfingerprintとqubit数、失敗cellの部分保存、失敗後に新しいcellを開始しないこと（retryなし）。

## 実行前の未確定条件

- **ユーザーの判断**：比較policyとholdout許容差の正式採択（Cの推奨はH-b、固定2%、観測値分母）。採択されるまで、Stage 1は開始しない。
- **Cの凍結形式**：Dのgateは、Cの `price_model_validation.fit` の出力を、そのままcommitに固定する形を前提にしている（[HANDOFF](HANDOFF.md)）。別の形式で凍結された場合、HOLDOUTはcompile前に停止する。その場合は技術修正が必要になる。
- **未確認の実行条件**：実H4での回路規模、compile時間、メモリ。保存M5 m4 q1の原trialは、205秒・1.2 GBだった。
- **主張しないこと**：m7〜m12の外挿、全qでのaffine性、方式間の順位、G、十分Nは主張しない（指示§10）。

commit/push・remote照合後 **mandatory STOP**。
