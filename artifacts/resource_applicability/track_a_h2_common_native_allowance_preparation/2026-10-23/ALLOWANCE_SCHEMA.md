# 共通適応allowance：schemaと未採択規則

提案policy：`H2_COMMON_ADAPTIVE_A_RULE_V1_PROPOSED`。旧Aの経験的allowance規則を第一候補とし、正式採択はユーザー判断に残す。共通規則から候補ごとのu/marginを作る提案であり、共通の固定値を強制しない。

## 元規則と人工実装

固定旧providerの `h6_matched_accounting_v1.empirical_allowance` に基づく。

```text
W = primaryとindependentの実action count合計
M = 両path・必要な全stageの最大norm
rounding = 64 * machine_epsilon * (1 + W) * max(1, M)
agreement = reference_error + max(path_signal_errors, path_vector_errors)
            + corrected_signal_raw_closure
u_real = u_imag = 10 * max(1e-12, rounding, agreement)
log_B_margin = 10 * max(1e-12, rounding, independent_log_B_discrepancy)
```

演算順を元規則に合わせる。入力欠測をmaxのdefault=0で通さずSTOPする。係数64・safety10・floor1e-12を提案として明示する。人工入力だけでこの算術を実装した。保存された実科学値はdomain guardで算術より前に拒否する。shot数N・費用Gを生成するコードは追加していない。

旧PR保存u=1e-9、margin=1e-12もfloorを含む生成器の出力であり、単なる説明なし定数とは扱わない。ただし、その生成器と旧Aの診断定義が同一であることにはならない。native1e-9/compiled1e-7のcorrectness閾値をuへ転用しない。biasは別項で、agreementへ再加算しない。

## 入力と単位

| schema項目 | 単位・範囲 | 必要な証跡 |
|---|---|---|
| reference_id / reference_error | 共通reference、CORRECTED_SIGNAL | state/H/T/scalarの対応、旧combined診断の意味 |
| path_signal_errors | CORRECTED_SIGNAL | primary対independent、cos/sin等の保存定義 |
| path_vector_errors | CORRECTED_STATE_VECTOR | 同一state/phase/wire/normalizationの両path差 |
| corrected_signal_closure | CORRECTED_SIGNAL | B×raw signalとcorrected signalのclosure |
| log_B_discrepancy | LOG_NORMALIZATION | 両pathのlog_B差 |
| work_primary/work_independent | 整数deterministic/tail実action数 | instrumentation定義、両path区別 |
| maximum_norm / norm_coverage | CORRECTED_STATE_VECTOR、両path全必要stage | 最大値に加えtrace_evidence |
| native_policy / input_binding_sha256 | 共通系列・入力結合 | operation/schedule/referenceとのidentity |

全値に有限・非負を要求し、boolや非整数countを拒否する。異なる単位・scope・referenceや部分stage coverageを拒否する。各Observationはevidence locatorを持つ。新系列への結合がなければ、旧Fullの診断値が揃っていても共通入力として受理しない。

## 保存値の接続状況

`map_old_diagnostics` は旧Fullのreference誤差、signal/vector差、corrected-signal closure、normalization差、両path action数、保存最大normを転記する。元correctness SOURCEが表すnorm範囲を注記するが、新共通nativeへの結合は `UNJOINED_SAVED_NATIVE` として不足扱いにする。

`map_pr_diagnostics` は受領PRのindependent forward signal差とnormalization errorだけを対応項目へ転記する。path-vector差、両path実work、corrected-signal closure、全stage norm、reference combined診断はNone/空のまま。不足一覧は[diagnostic_coverage_4.json](diagnostic_coverage_4.json)。

PRの `raw_corrected_vector_closure` はraw vector単位の量であり、Aのcorrected-signal closureへ直接代入しない。`maximum_intermediate_norm` は一方の経路のpiece endpoint観測で、両pathの全stage coverageにはならない。schedule上のstage proxyを実action数の代わりにしない。PRのBを1と置かない。

## reference・closureの最小追加案

旧FP referenceとその保存されたcombined empirical diagnosticを共通referenceとして再利用する案を優先する。保存PRのreference scalarは丸め水準でも同一とは限らないため、採択後に保存corrected meanを同referenceへ再結合する算術を別途承認する。今回は差分を再計算していない。経験的reference診断を厳密誤差上界に格上げしない。再利用が科学的に不適切な場合だけ、新reference診断を提案する。

closure変換が必要なら、state/normalizationを固定し、内積を取る同一state、Bの定義、raw-vectorからcorrected-signalへの変換、log-domainのoverflow条件と検証を仕様化してから承認する。保存log_Bを使ってexpを計算し、新closureを作る処理は今回行わない。

次の科学runnerには、独立pathを保ったinstrumentation、必要stageの定義・trace、input/native/reference結合の実receiptが必要。今回の人工規則は経験的候補であり、rigorous bound・strict bias保証・母集団単価上界を与えない。
