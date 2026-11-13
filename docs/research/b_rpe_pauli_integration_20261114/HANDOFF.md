# A/B/C/Dへの技術HANDOFF

固定B `e7ce23803b5644bd9df0e83c877b390aeb9405e0` からの独立統合。元branchへのmerge権限を意味しない。保全とテストは[REPORT](REPORT.md)。

## Pauli入力から統計/C inputへ

```python
from trottertracks import rpe_pauli_signal_intake as P
accepted = P.accept_saved(root, integration=C_pipeline)
# metadata only: 2 configurations, each with 13 saved round means
item = accepted['results'][0]       # LD1; index 1 is LD2
config = item['configuration']     # A's complete canonical V2, not a new ID
packet = item['signal_packet']     # source-bound copy; original A unchanged
c_input = item['signal_input']    # existing C.signal_input(packet)
```

`C_pipeline`は既存`rpe_optimization_pipeline`の`signal_input`を持つmodule。testsは固定Cの同じ純粋関数を科学provider抜きで使用する。全C graphをBへ移植していない。

| Aの保存field | B受入view / Cの入力 | 意味 |
|---|---|---|
| packet.configuration | configuration全体、packet.configuration_id | V2 hashと全round生成規則を既存Bで検証 |
| row.identity/law_binding | mean.identity/law_binding | 辞書・分割・全scheduleを含む正本。旧DFへ変換しない |
| mean_source commit/path/selector | mean.source、provider_binding.numerical_source | **raw carrier**（既存C field名を維持） |
| mean_record.source | mean.implementation_source、provider_binding.implementation_source、packet.implementation_sources[str(m)] | 各roundの実numerical実装。P1とcompletionを区別 |
| A完成HANDOFF | packet.source | assembly来歴。各round実装の代用ではない |
| physical_context | provider_binding.physical_context | H/state/snapshot、physical ID、辞書、law、compiler/wrapper、numerical recipe |
| input_source/reference_source/state_evidence | 同名field / provider_binding | 元operator/reference/stateの来歴。matrix再読込なし |
| signal.raw/corrected/reference/exact_tail | 同名コピー | rawだけが将来の測定モデル入力。補正や減衰を二重乗算しない |
| signal.log_B/B/attenuation | normalization（carrier SOURCE付き） | supplied normalization。今回再計算しない |
| allowance/numerical/qualification | uncertaintyとcarrier_qualification | 経験的・未認証・nullのまま保持。mean.qualification=UNQUALIFIED |
| 旧DF bundle.target | packet.target、target_view_source | 同じH targetの参照。proxy採択やenergy補完なし |

canonical ID：

- LD1 `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27`
- LD2 `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`

全roundのphysical ID/時間/q/R/allocation/辞書/SOURCEは [PAULI_INTAKE.json](PAULI_INTAKE.json)。上の26は**round record数**であり、各record内にX/Y raw座標を含む。

`K.prepare(config, packet, allocations, evaluation_target=..., statistical_contract=..., integration=C_pipeline)`も既存のwhole-shot準備を再利用できる。任意追加の`implementation_sources`は全round mapとauthoritative=trueを必須とし、未指定なら従来の単一SOURCE検査を行う。今回、実packetに新shot配分を与えて準備・評価することはしていない。明示evaluation_targetを要求する既存規則は不変。主評価のMC自動起動はない。

## 二水準のAPI対応

新`rpe_evaluation_levels.prepare`は**readinessのみ**。NもMSEもGも生成しない。

| | 原論文型 | MSE資格付き |
|---|---|---|
| level | PAPER_EMPIRICAL_NOMINAL_ONLY | MSE_QUALIFIED（入力区分名。資格発行ではない） |
| 入力 | paper_packet | mse_packet（従来B N packet） |
| N意味 | nominal_N_X/Y、rigorous_sufficient_N=false | 明示した全round shot配分と外部RMSE証拠 |
| 状態 | 存在・全round/identity/law/source/整数を検査 | 既存B.assessがpolicy/error contract・十分性・finite-K等を検査 |
| SOURCE/契約 | paper packetと各roundのSOURCE、元policy | 別packetのSOURCEとrmse_evidence、元policy |
| C | 各levelにprice_inputを保持。共通configuration/law/profile/guardが必要 | 同左。MODEL/cohort/期待値の資格を既存Bで判定 |
| 不足時 | UNKNOWN。nominalと価格がそろっても科学的十分Nにはしない | UNKNOWN/UNQUALIFIED。旧HP型N・診断boundだけでは適格化しない |
| G・winner | この準備APIではnull | この準備APIではnull |

paper packet形：`{configuration_id, policy, source, status: 'PAPER_EMPIRICAL_NOMINAL_ONLY', rigorous_sufficient_N: false, sufficient_N: null, global_RMSE_bound: null, rounds: [{m, identity, law_binding, source, nominal_N_X, nominal_N_Y}]}`。巨大整数はintのまま。既存`pr_rpe_rmse.nominal_axis_shots`の数式を再実装せず、将来の明示されたproducer結果をこの形で受け取る。今回実Pauliのnominal算出なし。

```python
from trottertracks import rpe_evaluation_levels as L
views = L.prepare(config, paper_packet=paper_or_none, mse_packet=mse_or_none,
    prices=prices, cohorts=cohorts, root=root, price_bindings=bindings,
    comparison_policy=explicit_implementation_policy)
```

INPUTS_READYは指定入力の技術的readinessだけである。価格guardはB.assessより前に実行され、policy欠測や価格欠測・未資格なら両水準の該当出力はUNKNOWN。nominal packetをmse_packetとして渡すと拒否する。

原論文型の将来のnominal資源指標は別名・別claimとして会計する。既存Bの十分N条件を緩めたり、nominalをQUALIFIEDに変えて`B.cost`へ通す経路は作っていない。MSE水準の後続会計は次の正式入口を使う。今回のtechnical readinessは実nominal費用や正式Gの算出ではない。

## guard付き正式入口と不足bridge

- `R.cost(root, config, n_packet, prices, cohorts=..., price_bindings=..., comparison_policy=...)`、`R.select(...)`：R=`rpe_price_implementation_guard`。固定C guardが先に検査し、その後で従来B primitiveを呼ぶ。
- 複数whole-shot配分は既存`W.compare(..., root=root, comparison_policy=...)`。既存`D.compare`もcost_optionsをこの入口へ渡す。正式scienceでpolicyがないと算術に進まない。
- `B.cost/B.select`は直接primitive。これだけを正式比較入口として使わない。guardはlaw期待値やN/CIの科学資格を新たに発行しない。
- C `signal_input`への受渡しは検証済み。一方`C.plan`と固定C guardのlegacy physical adapterにはPauli IDの正規bridgeがない。実Cplanを作ったかのように表示しない。P0辞書・Pauli lawを保つ新bridgeは**次の技術課題**であり、legacy IDへの置換で回避しない。
- 価格未取得、finite cohortのみ、C1 MODEL false、未検証外挿、未知CI/covarianceを推定値や0で補完しない。Dの新ZZ loweringと旧RZZは別SOURCE/domain。

## 次の担当とSTOP

1. **Codex B/C（次の指示後）**：Pauli canonical physical contextと既存C plan/guardを照合する最小bridgeを整備。AのID、辞書、lawを保ち、人工入力で拒否/一致を確認する。今回の独立成果を元Bへ無断mergeしない。
2. **A / 価格担当C・D**：必要になったとき、同一configuration/law/dictionary/profile/loweringに束縛されたnominal入力・価格証拠を供給。今回は新しい計算を依頼・起動していない。
3. **独立Codex**：Z1＋T1の別version実装を担当。今回はその成果の待機・無断取込み・重複開発なし。
4. **ユーザー**：次に進める評価水準と、target/loss/decoder/誤差契約、有限science範囲を指示する。W0の提案は自動採択しない。rawの元targetへの直接riskにAのδEを再加算しない。元Minkowskiは不変。

新しいPauli/DFのrisk/N/G科学実行は別途明示許可が必要。公開・remote照合後は**mandatory STOP**。
