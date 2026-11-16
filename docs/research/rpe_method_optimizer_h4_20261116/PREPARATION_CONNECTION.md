# AのPR-Pauli候補準備入口の接続

2026-11-16 JST、B2（Claude Code）。確認は作成者自身によるもので、独立レビューではない。指示はcomment(B1114)の追記。

- **Aの入口**：SOURCE `6e4b6bd6a`、HANDOFF `7cf4e1f05`（`docs/research/pauli_candidate_preparation_20261116/HANDOFF.md`）。
- **B2の実装**：`05c7022fa`。entryへ渡すpathを絶対pathにする修正が `76bb92184`。
- **計算の範囲**：metadataだけを扱った（`--prepare-missing` は使っていない）。新しい辞書・平均値・価格fit・compile・Q3は0件。

## 接続したもの

1. **未登録L_Dの準備要求**
   - 接続箇所：`generate_variants()` がconfiguration_idのない候補を除外する前。
   - 探索範囲（roundのL_D範囲）にある、configurationのない候補を、Aの入口へ送る。
   - 送る内容：B2の要求形式（`configuration: {L_D}`）と、基準の変更（S2・Δ_PF=h・κ×1・K=2）。
   - 呼び出し方：入口は固定SOURCEの読取専用view（`b2-route-view-a-6e4b6bd6a`）からsubprocessで呼ぶ。HEADとsrc/scriptsが変更されていないことを確認してから呼ぶ。
2. **完成したconfigurationの登録**
   - 価格不足は、登録の条件にしない。
   - `GENERATED` で構成が完成していれば、ID・本体・固定locator・13round schedule・信号plan・D2への価格引渡しを持つ候補として登録する。
   - 信号packet・価格・Q3経路がそろうまでは、`ready=False` のままで、Q3探索の対象にはならない。
   - 価格は、既存のD2 PriceRules（MODEL）の引用を、そのまま適用する。
3. **準備が必要なL_D**
   - `NEEDS_SPLIT_PREPARATION` の候補は、configurationなしのまま残す。
   - その候補には、Aのnext_actions（例：`SPLIT_PREPARATION:DERIVE_DESCRIPTOR,P0_…`）と `scientific_request` を付ける。
   - V2の入力要求（`NEW_L_D`）として、最安のL_Dに近い順に出す。
4. **V2のκ案**
   - `search.prepare_kappa_proposals: true` のとき、κ案をAの入口へ送る。
     - baseは、そのL_Dの保存構成。
     - 完成したconfigurationは同じ規則で登録し、価格を引用する。
   - 雛形ではfalseにしてある。新しいκのmetadata構成を作ることになるため、使う範囲は利用者が決める。

## 接続確認（保存結果と人工系）

[preparation_connection_76bb92184/PREPARATION_CONNECTION.json](preparation_connection_76bb92184/PREPARATION_CONNECTION.json) で8/8 PASS。

| 検査 | 結果 |
|---|---|
| 保存LD1 | `d905d219…` を登録した。固定locatorは `714247619`。13roundすべて信号REUSE。D2 MODELの引用（PAULI_LD1）では26軸すべてが `NEEDS_PRICE_EVIDENCE` で、価格は不完全だが登録した。残る必要項目：packetの組立て・価格・Q3経路 |
| LD4・6・8 | 準備要求として保持した（DERIVE_DESCRIPTOR→P0→P1） |
| LD7・9 | 準備要求として保持した（REUSE_DESCRIPTOR→P0→P1） |
| 保存LD2/3/5とκ変種6件 | 価格・IDとも変化なし |
| κ案の経路 | 公開済みのLD3 κ×2は `23ec1791`、LD5 κ×1は `4d6e2c9c` を返した。ともに13round REUSEで、既存候補への `ALREADY_REGISTERED`（二重登録なし） |
| 探索 | 登録しただけで信号・経路のない構成（LD1）は、Q3の問い合わせを受けない |
| 人工の入口 | 新規の完成構成（価格なし）は登録された／準備要求は保持された／要求round（m11・m12）は必要項目になった／既存IDは二重登録しない／呼出し失敗では候補を残し、失敗を必要項目にした |
| 固定SOURCE以外のview | 入口を拒否した |

## 次roundでの出力

[load_check](priority_v2_next_round/load_check/ROUND_RESULT.json) を `76bb92184` で出力した（budget 0、起動0件）。

- `generation.PR_PAULI.preparation` の内訳
  - LD1：`REGISTERED_WITHOUT_PRICE_REQUIREMENT`
  - LD4・6・7・8・9：`PREPARATION_REQUEST`
- V2の入力要求：κ案3件、新L_D 5件、LD1（構成の入力不足）。
- 応答の実体：`load_check/preparation/PR_PAULI/unregistered_L_D/`（REQUEST・PREPARATION・CONFIGURATION）。

## 失敗例と修正

- **起きたこと**：`05c7022fa` の接続確認で、相対pathの出力directoryを渡したところ、入口が要求fileを見つけられずに失敗した。入口は、viewをcwdとして動くためである。
- **round modeへの影響**：round modeはrun directoryを絶対化しているので、影響はなかった。
- **修正**：`76bb92184` で、渡すpathをすべて絶対pathにした。相対の出力directoryで再確認し、8/8 PASSだった。

## 回帰（`76bb92184`、作成者自身による）

| 受入 | 結果 | 出力 |
|---|---|---|
| 新規則 | 6/6 PASS | [priority_acceptance_76bb92184](priority_acceptance_76bb92184/PRIORITY_ACCEPTANCE.json) |
| 実行器 | 18/18 PASS | [executor_acceptance_76bb92184](executor_acceptance_76bb92184/EXECUTOR_ACCEPTANCE.json) |
| 生成 | 11/11 PASS | |
| Aの固定受入 | PASS 20（PENDING 1） | |
| 単体test | 18 OK | [TESTS_PREPARATION_CONNECTION.txt](TESTS_PREPARATION_CONNECTION.txt) |
| C test | 12 OK | 同上 |
