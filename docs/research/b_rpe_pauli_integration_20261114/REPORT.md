# B RPE基盤統合・Pauli全round signal受入

2026-11-14 JST。独立branch `codex/b-rpe-pauli-integration-20261114`。
技術実装・保存metadata照合・人工/回帰テストのみ。新しい科学risk/N/G評価は実施していない。

## 結果

- 固定B DPの2結果は保存完了、元worker 2 processは終了。元branchは `e7ce23803b5644bd9df0e83c877b390aeb9405e0` のままclean。kernel、runner、元結果はbyte一致で保全し、保存MSEの再計算はしていない。[保全確認](PRESERVATION.json)。
- 既存I1/I2/E1とB価格guard patchを独立worktreeへ競合なく適用。元Bへのmerge/cherry-pick、他担当worktreeの変更はない。
- A Pauli LD1/LD2の各13round、計26 recordを受入。P1再利用4件と新規22件の元carrier・numerical SOURCEを区別したsource-bound viewを追加。[受入metadata](PAULI_INTAKE.json)。raw/normalization/allowance/UNKNOWNはコピーし、確率・N・MSEを計算していない。
- 原論文型nominal入力とMSE資格付き入力を分離する小さいreadiness APIを追加。十分N資格を発行せず、G/winnerは常にnull。
- **人工/回帰137件PASS**（新6、既存131）。実H4/H6の新signal・risk・N・価格・G評価なし。既存の人工MC回帰1件を含むが、実MCや主経路の自動MCは実行していない。[テスト記録](TESTS.txt)。
- 技術実装SOURCE `d86a38eb061754d84bf8bbf7af08a79206e66f2a` に対する既存I1のB wiring preflightもPASS（provider呼出し0）。[WIRING_PREFLIGHT.json](WIRING_PREFLIGHT.json)。patchのreverse適用チェックPASS、manifest JSONと差分の整形確認PASS。

## 再利用と変更範囲

| 成果 | 固定SOURCE・取込み方法 | 今回の変更 |
|---|---|---|
| I1/I2/E1 | `3a27f99feef2073341dfd0b980ae6b6a23906e9f`、基点`ce0517de6e7b58cf4bf4c69f99f998d86099c406`との差分11 file | 同じ修正を適用。definition wiring、parsed parser cache、旧fixtureのSOURCE修正。candidate結果をcacheせず毎回検証 |
| B価格guard | `67876f2287a3bc004d06c8d84d92e9aee2774cff` の `B_guard_connection.patch` | 既存3 fileをそのまま適用。先行3 commitとHANDOFFを参照、guard規則は変更しない |
| C guard | `3a8b37f1a282c0c06ac1a363b070f4376e9ff41a` | B wrapperが固定C moduleをbyteそのままprivate namespaceで実行。C本線を更新しない |
| Pauli metadata | `714247619afb4714fd663eb15c0470ef32bdc7af` | Aの純粋な`dictionary_identity/job/numerical_recipe`をSOURCE固定で再利用。新intakeは辞書・carrier照合とview作成のみ |
| 統計caller | 既存`rpe_signal_statistics_caller.prepare` | 明示された全round implementation SOURCE mapを任意で受け取る。旧単一SOURCE経路は不変 |
| 二水準 | 既存`B.assess`、価格guard、exact整数の検査 | readinessと契約保持のみ。新controller、N producer、risk kernel、価格会計器は作らない |

適用対象と差分は [INTEGRATION.patch](INTEGRATION.patch)。このpatchは固定B上で適用する技術fileのみで、元DP結果や本報告を含めない。[PATCH_PROVENANCE.json](PATCH_PROVENANCE.json)に元3 commit、無競合、SOURCE照合を記録する。

## Pauli packetで一致を確認した内容

- canonical V2全schedule、各roundのcandidate/identity/law、T/qPF/R_outer/event数、signed piece/整数allocation、共通κ/K2、元H/state/snapshot、profile/control/full wrapper/P0。
- P0辞書のkind、canonical JSON hash、prefix、Pauli λ、保存scalar、bit順、coefficient/scalar/event-space policy。辞書payloadの同一性を確認し、新しいλ集計やoperator検証はしていない。
- 4再利用recordはnumerical `05447d1d4f1e3c0e9e335a7085aa58a877c139f2` / carrier `e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7`。22新recordはnumerical `7084fbc9a43cf7b7e24681b4005d37cb59c1e465` / carrier `fce4e4f7eae620e4440ac4103312d7afed9e9608`。assembly commitで置換しない。
- 元full DF-H targetはA旧bundleの保存targetから同じbindingを照合して参照。`truth_proxy_adopted_for_new_simulation=false`、ground未認証、`energy_ha`なしを保持。新proxyの採択はしない。
- `X=Re(raw), Y=Im(raw)`、既存`EXACT_AFFINE_OF_BINARY64_RAW`による確率変換規則を**人工座標のみ**で検査。保存rawには適用していない。
- 全26 meansは依然UNQUALIFIED。nominal Nも今回packetでは未取得、十分N/global bound/C/G/winnerはnull。旧DF signal・fixed-R診断のIDへ合わせない。

## 価格guardとC接続の到達範囲

`B.cost/B.select`は従来の直接算術primitiveのまま。正式比較は`rpe_price_implementation_guard.cost/select`、またはguard付き`rpe_shot_allocation.compare`を使用する。lowering/domain/scalar/wrapper/axis/candidate/law/SOURCEの不一致はBの価格算術より前に拒否する。有限cohortと期待価格、MODEL false・区間未資格は別の不足として残す。

Cの`signal_input(packet)`への受渡しは成功し、`5cc7c07e1`と固定C guard commitで関数本体が同一であることも確認した。ただし**Cの正式plan/price bridgeはPauli未対応**。旧A physical checkerはlegacy candidate IDを要求し、Pauliの`RMSE:...` physical IDを与えると `Physical recipe differs from B round` を返す（metadata test）。この拒否を迂回せず `UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED` を返す。受入viewの成功は正式C価格/G経路の完成を意味しない。

## 科学的に残るUNKNOWN

1. Pauli同一configuration/law/dictionaryに束縛された価格とimplementation sidecar。旧DF・fixed-R・異lowering価格は代用不可。
2. 原論文型nominal packetの供給。既存`nominal_axis_shots`を保存logBへ新適用していない。
3. MSE目標を満たすwhole-shot評価と、既存B N規則に適合する外部資格packet。今回のmeans取得はその証拠ではない。
4. target/state/sampler law、数値allowanceの移送資格、bias/Var、元Minkowskiと直接riskの接続。元H targetに対するraw直接riskへAのδEを足さない。
5. Z1＋T1、別version decoder/identityとのbridgeは独立Codexの担当。今回は取込み・重複実装・採択をしていない。W0は条件整理の参照であり新decoderやrisk契約の承認ではない。

次の担当・入出力は [HANDOFF](HANDOFF.md)。独立branchの公開・remote照合後は**mandatory STOP**。新しい科学実行・正式比較へ進まない。
