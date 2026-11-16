# B2向け：PR-Pauli候補を必要時に準備する入口

固定技術SOURCE：`6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb`。branch `codex/a-pauli-on-demand-preparation-20261116`。
[REPORT](REPORT.md)／[SOURCE一覧](SOURCE_INDEX.json)／[受入結果](ACCEPTANCE.json)。

## B2から渡すrequest

既存`run_rpe_optimizer_config_generator.py`と同じ`kind/items/base/changes`を受ける。保存baseがなければ、B2のNEEDS候補行の`configuration.L_D`を使う。NEEDS行そのものも入力できる。設計変数の再入力を利用者へ要求する手順はない。

```json
{
  "kind": "pauli_native",
  "items": [{
    "item_id": "B2-selected-LD3-kappa-x2",
    "base": {
      "commit": "421258d400a86fffd7c4955d5cf1bd24ac5729cb",
      "path": "docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json"
    },
    "changes": {"formula": "2nd", "delta_over_h": "1", "kappa_multiplier": "2", "K": 2}
  }],
  "saved_packets": [{
    "commit": "1537dfdf9d7fbf52c24609ed3a188381cc24eb8c",
    "path": "docs/research/b2_pauli_kappa_signals_20261116/packets/LD3_S2_delta_h_kappa_x2/MEAN_HANDOFF.json"
  }]
}
```

`base`の代わりに`"configuration":{"L_D":4}`と同じ`changes`を渡せる。B2 NEEDSのbaseline要求でchangesがなければ、保存S0のexactκ/S2/δ=h/K2を基準として解決する。base指定時のκ倍率は、そのbaseのκに掛かる（B2既存と同じ）。別の基準を要求する場合は、そのbaseまたはB2のchangesを渡す。

`changes.delta_pf`は絶対幅、`delta_over_h`は相対幅。同時指定は厳密に一致する場合だけ許可する。`"1/5"`または有理数dictと、JSON数値`0.2`は異なる値として扱う。floatを文字列へ変えてからFractionにしない。式はB2と同じ`2nd`・`4th(m5_best)`。KはBのspecへ渡し、signal providerの対応は別に返す。Yoshida4はこのB2 native generatorの対応外。

## 実際の呼出し

独立A worktreeの入口を、B2からsubprocessで呼ぶ。B2のcwd・branch・runを変更する必要はない。

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/a-pauli-on-demand-preparation-20261116/scripts/prepare_pauli_candidate.py \
  --request /path/to/B2_REQUEST.json --output /path/to/A/PREPARATION.json \
  --cache /path/to/A/SPLITS.json
```

通常呼出しはmetadataのみ。公開構成と一致すれば固定locatorを返し、未公開metadata構成は本体と`.CONFIGURATION.json`を返す。cacheのlocal構成を公開commitとして扱わない。

`saved_packets`は旧LD1/2・LD3・LD5の既定一覧に追加される。新packetのSOURCEと物理keyが一致すれば再利用する。`saved_configurations`にも追加の固定locatorを渡せる。新しい辞書を含む保存構成からは、そのP0 descriptor・辞書SOURCEを解決する。

## 保存構成と未登録構成の応答例

以下は実CLI応答の主要欄。完全な圧縮JSONと再現用requestは[examples](examples/)にある。

```json
{
  "item_id": "LD1", "status": "GENERATED",
  "preparation_status": "REUSED_CONFIGURATION",
  "configuration_id": "d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27",
  "readiness": {"configuration_complete":true,"signal_13round_complete":true,"price_26axis_complete":false}
}
```

`status=GENERATED`は既存B2 generatorとの互換欄であり、新しい科学計算を実行したという意味ではない。保存/新規metadataの区別は`preparation_status`。

```json
{
  "item_id":"LD4", "status":"NEEDS_SPLIT_PREPARATION", "L_D":4,
  "configuration":null,
  "next_actions":["DERIVE_DESCRIPTOR","P0_DICTIONARY_AND_OPERATOR_GATES","P1_CONFIGURATION"],
  "readiness":{"configuration_complete":false,"signal_13round_complete":false,"price_26axis_complete":null}
}
```

LD7/9では最初のactionが`REUSE_DESCRIPTOR`となる。入力そのものが不足・不整合なら`missing_inputs`に項目・SOURCEを返す。別LDの係数を代入しない。

完成応答の`configuration`・`configuration_id`・`configuration_source`と`schedule`をB2へ登録する。scheduleは13roundのphysical ID、T、q_RPE、q_PF、actual Δ、R_outer、signed weights/allocation/event数、law bindingを持つ。`lambda_P`と`lambda_DF`は別欄である。

## 辞書がない場合の次段階

今回の検収はここを人工系で確認した。実H4の新辞書はまだ作っていない。

対象1候補のP0科学実行を別途承認された場合だけ、同じrequestへ次を付ける。

```text
--prepare-missing --source <固定A HEAD> --p0-directory <A worktree内の新規directory> --cache <AのSPLITS.json>
```

- 既存辞書が見つかれば再利用する。cache hitでは新科学SOURCEの実行条件を要求しない。
- 未保存descriptorのみ、保存snapshotと既存係数helperで導出する。新SCF/DF/state生成はしない。
- 既存stage1の要求prefix＋LD2 controlを実行する。operator/scalar/sector/元gateを維持し、失敗時は部分結果でSTOP。retry/fallbackはない。
- 成果をそのA worktreeで限定commitし、辞書とP0 gateの固定locatorを作る。これは科学来歴の固定であり、承認の証明ではない。
- 同じ入力で再要求された場合は保存dict/descriptorを使う。κ・幅だけの変更は辞書を作り直さない。

stage1の一時metadata bindingはprocess単位。並列化するなら分割ごとの別processと別出力/cacheを用い、同じcacheの同時writerや同じsplitの重複P0を作らない。

## 信号担当への引渡し

`signal_plan`は既存`A_PAULI_SIGNAL_ENTRY_PLAN_V1`。各roundのREUSE/REQUEST/UNSUPPORTEDと理由・SOURCEを使う。`signal_plan_command`は実行可能なreadonly plan呼出しである。

- 保存packetが揃う場合：13roundすべてREUSE。公開LD3 κ×2は追加packetを渡すと13REUSE、渡さないとm0/m1のみREUSE・残11REQUEST。
- 厳密有理数のactual stepを現binary64 providerが表現できない場合：未対応理由を返す。丸めて取得しない。
- 旧contextにないdescriptor：出力したdescriptor envelopeと`--tail-descriptor`をplan/execute/assembleすべてへ渡す。committed値・H binding・prefixが照合される。
- 新規mean実行は別途承認後に既存信号入口の`--execute --approve-new-means --source … --out …`。本prepare入口はmeanを起動しない。

configuration完成を信号取得済み・新LDの数値検証済みと扱わない。`split_signal_validation`も確認する。

## D2への引渡し

`price_handoff.request`をD2 `c1f95e080` の`PriceRules.quote`へ渡す。policyはMODEL。渡されたconfigurationの各roundを使用し、κから別scheduleを作り直さない。

- 保存契約がある場合：当該LDのimplementationと固定SOURCEが返る。quoteの適用判定はD2の仕事。
- implementation=null：`needs`にある候補固有の実装契約・価格根拠をD2が解決する。他LDのregistry名・係数を埋めない。
- 新LDをB2の旧`registry[item.registry]` wrapperへ無理に通さない。契約が解決した後、configuration＋明示implementationを直接D2入口へ渡す。
- `price_26axis_complete`は保存carrierの技術coverageである。quote未実行ならモデル予測を取得済みにしない。E[C]・CI・十分N・Gの認定ではない。

## B2側の接続点

公開`98b6f8e1a`の`rpe_method_optimizer.enumerate_candidates()`は、`saved_configurations_only=false`なら未登録LDを生成するが、`generate_variants()`はconfiguration_idのある候補へ絞る。このため、flag変更だけでは未登録分割を準備できない。

**B2が選んだ候補を、`generate_variants()/run_generator()`のsaved-only filterへ入れる前に本入口へ渡す。** baseがある既存itemはそのまま、baseがない候補はNEEDS行または`configuration.L_D`＋B2が選んだchangesを渡す。`GENERATED`ならID/本体/13round scheduleを登録し、信号plan・D2 handoffを各担当へ渡す。価格不足をconfiguration登録の条件にしない。準備要求ならその分割のP0をpendingとして保持する。

AはB2の探索順、L_D/κ選択、shot探索、実行中run、optimizerコードを変更していない。採用する探索規則・実科学実行範囲はB2/利用者が決める。

技術検収・公開後mandatory STOP。実H4の新辞書・信号・較正・compile・Q3へ自動進行しない。
