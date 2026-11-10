# C1価格recordの利用方法と今後の受入条件

[REPORT](REPORT.md)。読み手は、B pipelineでC1価格を扱う担当（Codex B／Claude Code）と、A/Cの将来の結果を接続する担当。今回の取込みは公開JSONのcopyで、実N×C/GやR比較の許可ではない。

## 1. 受け入れた価格recordのSOURCE・path・cohort

- 公開commit：`2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`
- run directory：`docs/research/r_price_phase1_results_20261111/run01`
- 凍結commit：`7eeabf0ae482ee610d667e28470692d209f2bb21`
- C実行SOURCE：`1a87e533…`
- PLAN：`c596e88d…`

| file | record | 区分 | cohort（Bでの名前） |
|---|---:|---|---|
| CALIBRATION_EVALUATION.json `prices` | 24 | 保存校正direct 8、新校正direct 4、校正点model 12 | `C1:DIRECT:EXPLORATION`、`C1:DIRECT:H4_DF10_V2_INDEPENDENT_Q_20261110`、`C1:DIRECT:H6_PRICE_V2_20261110_Q2_CALIBRATION`、`C1:DIRECT:R_PRICE_P1_20261111_{H4_R8,H6_R32}_CALIBRATION`、および各`C1:MODEL:…` |
| VALIDATION.json `prices` | 8 | 独立検証direct 4、検証点model 4 | `C1:DIRECT:R_PRICE_P1_20261111_{H4_R8,H6_R32}_VALIDATION`、および各`C1:MODEL:…` |

元のcohort名は`source_cohort`、元のlaw宣言は`c1_original_law_binding`に残している。`c1`欄には次を記録した：cell・役割・stage・`price_kind`・publication・凍結commit・join key（cacheの一致証拠ではない）・不確かさ。

## 2. 既存N/C pipelineへ渡す方法

```python
from trottertracks import rpe_r_price_intake as ci
from trottertracks import rpe_provider_integration as b

intake = ci.C1PriceIntake(root)
accepted = intake.read_published('2e6a0cdf5a647058cc7afeeb5741bfafa24c974b',
                                 'docs/research/r_price_phase1_results_20261111/run01')
packet = b.prepare_menu(root, 'H6', {4: [16, 32], 5: [16, 32]})   # C1行を含むmenu
prices = packet['prices'] + ci.records_for(accepted, packet['menu'])
cov = ci.coverage(packet['menu'], packet['shots'], prices, cohort_selection={},
                  intake=intake, accepted=accepted)
```

CLIで再現する場合（Aの全menu＋C1）：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B scripts/plan_rpe_r_price_intake.py \
  --output-dir <fresh-dir> --publication-commit 2e6a0cdf5a647058cc7afeeb5741bfafa24c974b \
  --run-dir docs/research/r_price_phase1_results_20261111/run01 --full-menu
```

C1行を持つmenuでは、`run_missing`はC1行のC providerを呼ばない（direct/model両軸のrecordが揃っているため）。C1外の行は、従来どおり`RScopedFixedPrices`／`FixedSavedPrices`がnullか固定R catalogを返す。

## 3. finite cohort scenarioの扱い

- 既定の`qualified_expectation`では、C1のPR価格はすべて`UNQUALIFIED_C`になる。
- direct cohortを使うには、次の3つがすべて必要：
  - `price_policy='finite_cohort_scenario'`
  - 候補・軸ごとの`cohort_selection`で、X/Yとも同じ`C1:DIRECT:…`を選ぶ
  - 同じcandidateのN
- それでも、得られるのは「その有限cohortの価格での条件付き値」であり、全law期待価格ではない。
- 保存校正セルでは、固定catalogの`DIRECT:<cohort>`と`C1:DIRECT:<cohort>`が同じ保存pathを表す（`same_saved_cohort_as_fixed_catalog`）。どちらか一方だけを選ぶ。
- model scenarioは`qualified_for_candidate=false`のため選べない。採用には、ユーザー／GPTが採用基準を決め、Cがその資格をrecordで示す必要がある。

## 4. Nとの同条件結合に必要な資格

- N recordは同じ`identity`を持ち、`law_binding=r.request_context(menu,row)['law_binding']`、`N_comparison_policy=R_AWARE_EMPIRICAL_ACTION_COUNT_V1`である必要がある。
- C1新R 4セル（`6e1efff1ba1f7ac95a3b`、`201d498a3fa5eaa3b5c5`、`4269eafc8c0ef8922c23`、`24fec6bdd44d6a728f1e`）のNは現在欠測。
- 保存R4/R16のNは旧allowanceなので、R比較には同policyによるscalar再解析（別途の科学実行）が必要。
- Aが詳細Nを公開した場合は、そのcandidate IDとallowance policyが一致するrecordだけを結合する。A0 normalizationはN・Cの代わりにならない。

## 5. 長q・大Rの価格不足

- C1は小qの8点だけで、次の価格は供給しない（`NOT_ACQUIRED`／`NEW_R_PRICE_NOT_ACQUIRED`）：
  - H4：C1の4点以外のすべて。m3/m4のR16以上、m5以降の新R、m12（q4096）のR8192等
  - H6：C1の4点以外のすべて。m4のR11、m5のR64、m6以降の新R、m12（q256）のR2048/8192等
- Cのmodel式でこれらを予測するには、その点の構造価格Sと、別途の検証・採用資格が必要。B側では予測しない。

## 6. Cが追加検証した場合の受入条件

- 新しい検証が同じPLANの延長ではない（新PLAN・新run）場合は、新PLAN用のpinned設定（`C_PLAN`、`C1_R_MENU`、セル役割）をadapterに追加する。既存のC1受入を変更せず、別の受入として扱う。
- 受入に必要なものは次のとおり。
  - 固定PLANとSOURCE
  - 校正→Git凍結→独立検証の順序と、bytes一致
  - 校正cellと検証cellの分離
  - full identityとlaw対応（rte.pyの同一blob等）
  - `FINITE_COHORT_ONLY`などの区分
  - 未知の不確かさはnullのまま
  - model資格の根拠（採用閾値と、それを満たした検証）
- `qualified_for_candidate=true`のmodelは、PLANに採用閾値がある場合にだけ扱い得る。現在のadapterでは拒否する。
- 長q・大Rの価格は、そのq/R点のdirectまたは資格済みmodel recordとしてだけ受け入れる。C1点から補間・外挿しない。
