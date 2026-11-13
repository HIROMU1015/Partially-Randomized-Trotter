# B → A/C/D：新証拠と正式V2の接続

2026-11-13。[REPORT](REPORT.md)、最終成果は[run02](run02/PUBLISHED_EVIDENCE.json)。基点Bf0c4bb559、正本B030f3f548、canonical ID=`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。本ファイルを含む公開commitを固定する。

## A：新m1 scope

A実結果`3e035c0d61386010f59ad7ea792bc7a5cefd18ac`の`docs/research/h4_pr_m1_rmse_results_20261113/run01/{result,N_HANDOFF}.json`を受入済み。数値SOURCEはac873e7fc。m1 nominal60/60は転記だけで、元S0 m0=64/64・m6=38/38を保全。他10round未評価、全十分N/global bound/nullとUNQUALIFIEDは不変。

新APIは`rpe_diagnostic_intake.validate_m1_packet(config, packet, previous, numerical_source=..., adapter_source=...)`。previousはA75b5c89の元packet、adapter_sourceはAac873の保存PREFLIGHT.source。元S0-only validatorは変更していない。m1を元S0へ偽装せず、新数値SOURCE、previous locator、`diagnostic_scope`、各rowの`diagnostic_source`を検査する。全13identity/law/policy/profile/physicalは同じ正本へ結ぶ。

受入を再現するmetadata CLI（新規directory専用）：

```sh
python -B scripts/intake_rpe_new_evidence.py --out /tmp/rpe-new-evidence-fresh \
  --a-release 3e035c0d61386010f59ad7ea792bc7a5cefd18ac \
  --a-result-path docs/research/h4_pr_m1_rmse_results_20261113/run01/result.json \
  --a-packet-path docs/research/h4_pr_m1_rmse_results_20261113/run01/N_HANDOFF.json
```

## C：K0/K1の受入契約

C a637ed1bの公開run02/RESULT/CHECKS/REDUCTIONだけを受入済み。局所状態UNKNOWN、physical transition/reward parity=null、K0/K1新実結果は未受入。E[S]/E[C−S]/E[C]は作成しない。既承認のC科学実行やそのSTOPをBは変更しない。

将来の小さな技術envelope：

```python
packet = {
  'configuration_id': canonical_id,
  'source': source_locator, 'kernel_source': kernel_source_locator,
  'stage': 'K0',  # またはK1。Taylor cutoffのK=2とは別
  'K0_status': 'UNKNOWN',  # K1には保存K0 PASSが必要
  'quantity': 'MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C',
  'qualified_for_candidate': False,
  'mean_compiled_C': None, 'mean_C_minus_S': None,
  'rows': []  # m0/m6×X/Y。部分取得なら欠測を残す
}
```

各rowに`m/axis/identity/law_binding/physical_candidate_id/profile/source/law_context/checks/uncertainty/mean_S`を持たせる。row.sourceはpacket.sourceと一致。K0ではmean_S=null、K1でもpartial/nullは維持。未知のsystematic/covariance/CIを0へ変えない。`accept_structural_packet`はcell値をcopyし、compiled_price_records=[]を返す。Cの実formatが異なれば固定SOURCEの最小adapterで対応する。`--c-release/--c-packet-path`はこのenvelopeをGitから読む入口で、科学実行optionではない。

MeanPriceLookupやcostへ構造Sを渡さない。適格E[C]には別途C−S/law/domain/全wrapper/不確かさ証拠が必要。C1 false・有限cohortと全law・rare/長q資格を維持する。

## D：B正式interfaceの利用

```python
from trottertracks import rpe_d_evidence_intake as intake

# spec/layoutはD0536646bのSOURCE付き入力。条件をBが選ばない。
bound = intake.bind_physical(spec, layout_by_round=layouts)  # PFはlayouts=None
assert intake.validate_binding(bound)
config = bound['v2_configuration']
requests = bound['requests']       # 既存B側D.requests/validate_requests
identity_map = bound['identity_map']
```

Dの`physical_schedule`を固定Git0536646bから再利用し、B既存generatorへ接続する。D branchを変更していない。標準4次の元IDは`YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64`、B正本は`PF_STANDARD_YOSHIDA4_BINARY64_V1`。元af284d85係数SOURCE・binary64 signed順序を検査してから結ぶ。m5は別公式。Full-HのD元kindとB`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`はsource/canonicalの別欄で保持し、κ=None、q/R/K/全H λ/one-body/scalar/law/profileを保全する。新native hookはない。

`accept_example(entry)`は公開D人工例のpending/null欄を過去の完成結果へ書き換えず、B生成identityとの外部対応を返す。人工例のMOCK_COMPILERでは保存科学価格に結合できない。この不一致はそのまま保全している。

`accept_saved_coverage(cov)`はD92ff17deの既存136件分類をmetadataとして受入れる。`join_saved_coverage(bound,cov)`は明示されたwhole scheduleだけを既存`for_packet`へ結び、H/state/snapshot・SOURCE/formula・compiler/profileを検査する。136件へh/M/RMSEを自動付与しない。新receipt/V2 mapを公開する際は固定commit/pathを示すこと。分類のEXACTは現RMSE資格ではない。

## 不足・次の担当

[METHOD_GAPS](run02/METHOD_GAPS.json)／[既存planner結果](run02/EVIDENCE_PLAN.json)。4方式の正式menuや実Gは未評価。

- **ユーザー**：Full/truncated PF・Full-H RTEのformula/幅/layout/K/M/RMSE比較条件と、追加科学実行の有限scopeを判断する。
- **Codex A**：別途許可された範囲だけ、残10roundの診断・全round十分N/global RMSE/finite-K/κ証拠を取得・公開する。旧HP型を再解釈しない。
- **Codex C**：既承認K0/K1の結果をSOURCE付きで公開する。新C−S/compile/fit等は既存承認の範囲を確認し、不足scopeには別許可が必要。
- **Codex D**：このB interfaceへ自branchを追従させ、receipt/physical V2 mapの更新を固定commitで公開する。保存136件は比較menuにしない。
- **Codex B**：後続指示で公開証拠を受入れる。実N×C/G/最適化は別の明示許可まで行わない。

人工13＋前回93＝106 test PASSと公開A fixtureのmetadata受入を確認。新科学計算0。公開SHA/tree/主要成果物照合後**mandatory STOP**。
