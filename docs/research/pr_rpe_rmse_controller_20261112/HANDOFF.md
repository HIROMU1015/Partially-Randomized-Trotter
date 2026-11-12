# A/Cへの接続・B controllerの使い方

[共通仕様](INTERFACE.md)／[JSON](INTERFACE.json)／[人工I/O](SYNTHETIC_IO_EXAMPLE.json)／[REPORT](REPORT.md)。先行共有commitは`d562bc38d3b63df51f775aa4eb0998afbf1873e2`。科学的RMSE・κ定義は未採択の技術案である。

## A/Cが利用するAPI

```python
from trottertracks import pr_rpe_rmse_controller as b

config = b.generate(spec)  # またはevent_mapper=Aの明示SOURCE付きcallback
# config['rounds']のidentity/law_bindingをそのままA/Cのrequestにする。
qualification = b.assess(config, n_packet, prices,
                         cohorts=cohorts, price_policy='qualified_expectation')
# ここまでmetadataと供給資格の照合のみ。Gはまだnull。

# 人工データ、または別途明示許可された実評価だけで呼ぶ。
evaluation = b.cost(config, n_packet, prices,
                    cohorts=cohorts, price_policy='qualified_expectation')
selected = b.select(configs, evaluations,
                    unexplored_scope={'outside_explicit_menu': True})
```

`configs`はユーザー/Aが明示した外側(method,L_D,ΔPF)と内側(M,κ)の有限集合をgenerateしたリスト。Bがh/formula/K・追加rangeを自動探索するAPIはない。selectはwhole-configuration単位で選び、欠測や未探索・方式別coverageを残す。単なるassess結果を渡してもG評価済みにしない。価格policyや科学evidenceと人工evidenceを混ぜない。

### A → N_packet

必須field：

- `configuration_id=config['id']`, `policy=config['spec']['rmse_policy']`, `source`。
- `status='RMSE_SCHEDULE_EVALUATED'`, `evidence_kind='SCIENTIFIC'`（人工testは`SYNTHETIC_ONLY`）。
- `rounds`：m=0..Mすべて、各rowの`identity`/`law_binding`、正の整数`N_X`/`N_Y`。単位は量子Hadamard shots/axis。
- `rmse_evidence.source/status/error_contract/rmse_bound_ha`。
- `covers_state_target_alias_prior_decoder_all_rounds=true`。finite-Kを用いる方式は`finite_K_status='QUALIFIED'`、`signed_piece_kappa_status='QUALIFIED'`。

科学値はpolicy.scientific_status=`EXTERNALLY_QUALIFIED`が必要で、誤差契約の採用SOURCEも明示する。Bはこれを自動付与しない。AのRMSE導出に必要なbias/finite-K/参照状態/round間誤差・丸め等の証拠をsourceへ残す。Bは供給boundと目標を比較するだけで、biasの二乗和やNを計算しない。未評価・精度未資格・INFEASIBLEを元の状態で渡す。

Aは`b.paper_event_mapper`を共用できるが、これは候補整数化のdraftにすぎない。別κ定義の場合は、event_mapping.kind/sourceとkappa.kind/value/definition_sourceを揃え、callbackが同一κ/SOURCEとR_outer/event_countを返す。名前なし・callbackなしでは未解決のまま。有限K・負時間・control/実PF幅の数学的対応はAの科学契約で資格する。

A-P1の12件はalpha/betaと`CONDITIONAL_ROUND_PASS`を持つ旧高確率型として保存照合済み。新schemaに変換せず、RMSE packetの代用品にしない。A0のlog B/attenuation/shot係数もNに変換しない。

### C → 価格record

既存C recordと同じfield名を用い、config rowと同じ`identity`/`law_binding`を返す。`axis=X/Y`、measurement/observable/符号・単位は既存Bと同じ。`mean.rz_count`は1 full-wrapper interrogationの平均価格。

DIRECT、MODEL、cohort/n_paths、expectation、SOURCE、model domain/binding/compiler/validation_source/coverage/q_variableを明示する。MODELの`qualified_for_candidate=false`は不適格のまま扱う。PF/PRのq_variableはqPF、全H純RTEはevent_count。有限cohortは`finite_cohort_scenario`を明示した条件付き点比較だけで利用でき、全law期待価格にはならない。X/Yのcohort/kind/expectation/標本数はpairedにする。rare forced cohortを無条件平均へ混ぜない。

旧C1や固定R価格は、新identityの適用証拠なしに付け替えない。未検証q/R・未知CI/共分散/系統差はnull/未資格のまま。normalizationや古典標本数を価格へ乗算しない。`cohorts[(candidate_id,axis)]`が選ぶ1 scenarioだけを供給する。

## テスト・保存成果の確認

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  tests/check_pr_rpe_rmse_controller.py
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  docs/research/pr_rpe_rmse_controller_20261112/CHECK_SAVED_COMPATIBILITY.py
```

前者は人工条件のみ、後者は固定Git metadataの照合のみ。[TESTS](TESTS.txt)と[LEGACY_COMPATIBILITY](LEGACY_COMPATIBILITY.json)を参照する。後者はN/Cの数値演算・Gを一切呼ばない。

## 未完了事項と次に必要な承認

**A/ユーザー**は数学的RMSE/κ・finite-K/signed配分・h/ΔPF/時間半減・state/error契約を確定する。**C/ユーザー**は価格domainと校正/独立検証・全law/不確かさ・採用基準を決める。そのうえで**ユーザー**が固定tuple・有限range・SOURCE/input/precisionに対するN/C科学実行を明示許可し、**A/C**が成果を取得する。**B**はその公開recordを接続し、実G/M/κ/分割比較は別の明示許可された有限scopeで行う。

解析κの最適式は未実装で、明示seed入力としてのみ将来利用できる。全law・共分散やPF baseline最適性、全RPE科学資格は未取得。今回の人工PASSを性能改善・科学的採択へ読み替えない。公開照合後mandatory STOPする。
