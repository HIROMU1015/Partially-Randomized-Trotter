# Native証拠・RZ cacheの再利用可能性

[入口](README.md) / [identity照合](b3_c1_g9_identity_crosswalk.md)。分類は**指定scopeでの証拠再利用**の判断であり、科学的新規性・一般費用優位・新provider採択ではない。異なるscopeには同じartifactでも別分類がある。

## 1. 再利用表

| ID・対象 | 分類 | 固定field/関数 | 再利用条件・不足 |
|---|---|---|---|
| R01 G9登録matched比較 | `DIRECTLY_REUSABLE` | G9R、G9RAW `/rows`、G9AUD、前回ABASE | 同じP5・G9 policy・3-qubit I1・direct primary6の既存T/CX・affine判定を引用。再縮約/compile不要 |
| R02 G7の保存B1/B2 | `DIRECTLY_REUSABLE` | BREUSE→G7RAW `/rows/4`〜`/rows/7` | BCの旧4方式・旧budget・helper/query scopeの引用。新粒度・G9 Nへの換算を含めない |
| R03 G7 N×G9 native総量の直接比較 | `NOT_COMPARABLE` | BC `/fixed_science_policy`対G9MC confidence/bias/resource | 16/22 axes・3ρ/8ρ等が不一致。provider一致でも解消しない |
| R04 G9の19 RZ資産 | `REUSABLE_IF_CONTRACT_MATCHES` | G9KEY `/keys`、G9RAW `/synthesis_cache/<ratio>` | full angle-key、scale、ε、strict phase、tool/options、sequence/hash、正負規約が一致する既存key。新合成不要 |
| R05 G7由来18 keyの来歴 | `DIRECTLY_REUSABLE` | G9KEY acquisition=`reuse_G7`、G7RAW cache、G9MAN `/reused_sequence_keys` | 旧保存record/sequenceの同一性は来歴として利用可。旧wrapper費用・独立再現を意味しない |
| R06 新cache guardとの結合 | `ADDITIONAL_BINDING_REQUIRED` | BS `rz_cache_witness`、G9 cache `key/angle_key/epsilon/...` | row形は対応。引数keyはratioではなくfull `atan:…:scale:1:epsilon:…`。新precision_id/tool manifestとのbindingが未固定 |
| R07 negative RZ/actual adjoint/W | `ADDITIONAL_BINDING_REQUIRED` | G7 CP `bind_synthesized_provider_ir`、G9N `inverse/adjoint`、BS `price_row` | 新price rowが求めるforward artifact SHA/検証flagと、旧SOURCE・正列hash・W保持規約を結ぶ。負列再合成は不要 |
| R08 具体CQ model | `REUSABLE_IF_CONTRACT_MATCHES` | G9MC `/input/provider`、`exact_provider_delta`、G9N `V/CQ` | 同じQ/basis/control・exact Clifford+T contextを別途採択する場合。encoding/sectorを分子へ外挿しない |
| R09 将来C1の物理provider | `UNKNOWN` | BC `/C1_proposal/physical_Q=null`、`/decisions_required` | 公開準備版では未選定。将来のA実装/採択を予測しない |
| R10 既存event価格→新価格schema | `ADDITIONAL_BINDING_REQUIRED` | G9RAW `/rows/*/events/*/{event,native_ir,cost}`、PRICE、BS `price_key/price_row` | event context/implementation digest、9-field ID registry、artifact/certificate SHA、整数区間への記録対応が必要。primitive価格の完成poolと誤認しない |
| R11 compiler/cancellation | `REUSABLE_IF_CONTRACT_MATCHES` | G9MC `/compiler`、G9N `native_ir/simplify/cost` | 同じlowering・CZ基底・adjacent cancellation・cost conventionなら引用可。whole-eventをliteral primitive加算へ置換不可 |
| R12 T/CX/1Q座標 | `ADDITIONAL_BINDING_REQUIRED` | G9 event `cost.{T,CX,1Q}`、row per_trial/two_axis、PRICE `coordinates` | 1Q名、event/outer項、per-trial/2-axis・期待/上限を結合。新N/価格で縮約し直す場合は別承認 |
| R13 serial/whole-circuit depth | `ADDITIONAL_EVIDENCE_REQUIRED` | PRICE `/coordinates/serial_depth`、G9N `cost`にdepth fieldなし | 旧gate数からdepthを推測しない。depthが研究判断に不可欠なときだけ追加scopeを検討 |
| R14 ancilla/workspace | `ADDITIONAL_BINDING_REQUIRED` | G9RAW row `workspace_beyond_system`、G9MC system3、PRICE ancilla_peak | direct1/helper2は旧model scopeで引用可。system・state・compiler workspaceを含む新peak定義と合わせる。N倍しない |
| R15 CQ errorとevent error | `ADDITIONAL_BINDING_REQUIRED` | BC CQ allowance0、G9MC delta0、G9RAW `event_strict_error_analytic_bound` | CQ=0とRZを含むevent=2εを分けて適用。eventをCQ allowance0へ押し込む／誤差をρへ吸収しない |
| R16 provider/precision/compilerの不一致 | `NOT_COMPARABLE` | BC未決fieldとG9MC固定値、PRICE request | 不一致の価格を直接転用しない。既存証拠で接続できるか先に確認し、未証明差を0にしない |
| R17 実state/prep/readout/backend | `ADDITIONAL_EVIDENCE_REQUIRED` | BC C2 null、G9RAW affine K/outer1Q | 共通非負parameterとして旧結論を引用可能。実QPE/RPE/総資源の取得済み価格ではない |
| R18 強い対照の全trial古典実費 | `ADDITIONAL_EVIDENCE_REQUIRED` | B3、BS ARMS、G9 `P5Closed.sample`/`cts_events`/runner | sample/構成機能はあるが、同等audit/production/全N計測・保存費用が不足。native compileではこの不足を埋められない |

各判定のcommit/path/fieldは[証拠索引](evidence_index.json)と[再利用metadata](reuse_inventory.json)に固定した。`DIRECTLY_REUSABLE`は旧scopeの引用に限る。R10等のbindingは今回**仕様上の不足を特定した段階**で、新しいprice artifactやadapterを作成していない。

## 2. Cacheの同一性をどこまで確認したか

G9は結果前inventoryで19 keysを固定し、18をG7保存列から再利用、CTS用1 keyを取得した。[G9KEY][G9KEY] [G9MAN][G9MAN] 今回はinventoryと保存cacheのkey/angle/ε/sequence SHA、G7由来recordの同一性を管理検査した。保存`error_pass`とstrict boundは既存証拠として扱い、行列や三角関数・誤差上界を再評価していない。全eventの再列挙や費用再集計もしていない。

`rz_cache_witness`の要求fieldは旧recordに揃っている。ただしそれだけではtool/package/compiler version・seed・`up_to_phase=false`との結合を検証しない。G9MCの`tool_identity`、`runtime_executable_sha256`、`synthesizer_options`、取得SOURCE/保存result/既存監査を一緒に固定する必要がある。[BS][BS] [G9MC][G9MC] 既存strict certificateの引用と、新環境での独立数値再検証は別である。

## 3. Compileが不要な範囲と、必要性を再判断する条件

**不要**：旧G9結果をそのまま引用する、同じ19 RZ列を再利用する、同じevent実装・compiler・価格座標の旧証拠を新manifestへ結び付ける場合。metadata bindingのために旧回路build/compile/synthesisをやり直す必要はない。数値的な新縮約はcompileとは別にP2/B2の承認が必要。

**必要性を判断する条件**：新Q/encoding・control/ancilla・angle/precision・gate set/最適化・価格座標が変わり、そのcontextを支える保存価格/証拠が存在しない場合。その場合も、必要な差分key/価格だけを対象化する。新RZ角があることだけで旧19 keys全体の再合成を要求しない。

depthは現資料にないが、depthが今回の主張に不要なら取得しない。gate列から新depthを算定する処理も追加費用評価であり、今回は実施しない。新native取得が不可避かは、採択した費用問いと旧証拠coverageによって決まり、現段階で一律のcompile計画を作らない。

## 4. 公平性と証拠限界

全方式に同じdirect/helper選択・合法compiler最適化・基底共有・cache/情報アクセス機会を与える。CTS real correctionやidentityのT=0 eventを削除しない。G9のexplicit I1 Pauli取得を一般Q/I0で無料とせず、query-only比較へCTSを無償追加しない。

既存のT-primaryとCXトレードオフは[前回の保存結果引用][ABASE]を参照。今回新しい表・順位・費用重みは作っていない。価格schema/guardの静的対応はplatform certificateでも回路正しさの独立証明でもない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[ABASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff0bed24ba8fdb361a2890646ef84347ae37d457/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/existing_strong_baselines.md
[BS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py
[G9KEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_p5_native_preparation/2026-10-10/synthesis_inventory_v1.json
[G9MAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/evidence_manifest_v2.json
[G9MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
