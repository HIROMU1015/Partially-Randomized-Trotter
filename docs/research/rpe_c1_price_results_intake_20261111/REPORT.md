# C1実compiled価格32 recordのB取込みとN/C coverage更新（2026-11-11）

Codex CのPhase 1（C1）公開結果を、既存のB受入adapter（`724d313b9c8445763f0fe9c41feb573df28ba5d5`）で取り込んだ。

- 32の価格record（校正24、独立検証8）をすべて受け入れた。受入状態は`PHASE1_COMPLETE_RECORDED`。
- SOURCE、stage、凍結、候補、law、compiler、cohort、価格値の照合はすべて成立した。
- adapterとtestは変更していない。CLIにだけ、Aの全R menuでcoverageを作るoption `--full-menu`を追加した。
- 新しい科学計算は0件：sampling、build/compile、S計数、fit、価格予測、残差評価、signal/N、実N×C/G、R最適化。

成果：[INTAKE.json](INTAKE.json)（受入結果）／[COVERAGE.json](COVERAGE.json)（全menuのcoverage）／[READINESS.json](READINESS.json)／[VERIFY.json](VERIFY.json)（実データ照合、[VERIFY_INTAKE.py](VERIFY_INTAKE.py)）／[TESTS.txt](TESTS.txt)／[HANDOFF](HANDOFF.md)。

## 1. 使用したC1成果

| 項目 | 値 |
|---|---|
| C1公開commit | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`（`docs/research/r_price_phase1_results_20261111/run01`） |
| 係数凍結commit | `7eeabf0ae482ee610d667e28470692d209f2bb21`（公開commitの祖先） |
| C1準備PLAN | `c596e88d…`。run01/PLAN.jsonと同一bytes |
| C科学実行SOURCE | `1a87e533…`。全recordの`execution_source`。公開commitの科学4ファイルも同一bytes |
| STOP | `PHASE1_COMPLETE_MANDATORY_STOP` |

凍結commitにはVALIDATION.jsonとRARE_DIAGNOSTICS.jsonがない。FROZEN_MODELS、PLAN、CALIBRATION、CALIBRATION_EVALUATIONは、凍結時と公開時で同一bytes。validationは`frozen_model_commit=7eeabf0…`、`validation_refit=false`。つまり、検証価格を取得する前に係数を凍結し、その後に再fitしていない。

## 2. 32 recordの受入状況と区分

| 区分（`c1.price_kind`） | record数 | セル |
|---|---:|---|
| 保存済みRの校正direct | 8 | H4 q8/q16 R4、H6 q1/q2 R16 |
| 新Rの校正direct | 4 | H4 q8 R8、H6 q1 R32 |
| 凍結後の独立検証direct | 4 | H4 q16 R8、H6 q2 R32 |
| 校正点のmodel予測 | 12 | 校正6セル |
| 独立検証点のmodel予測 | 4 | 検証2セル |

保存値の転記（X/Yは同値。単位はfull Hadamard wrapper 1回当たりのcompiled RZ、cohort平均）：

| セル | 役割 | n | direct | model予測 |
|---|---|---:|---:|---:|
| H4 q8 R4 | 保存校正 | 9 | 16857.111 | 16857.111 |
| H4 q16 R4 | 保存校正 | 33 | 33527.242 | 33527.242 |
| H4 q8 R8 | 新校正 | 8 | 23752.375 | 23752.375 |
| H4 q16 R8 | 独立検証 | 8 | 49638.125 | 49808.823 |
| H6 q1 R16 | 保存校正 | 9 | 179583.333 | 179583.333 |
| H6 q2 R16 | 保存校正 | 8 | 358243.0 | 358243.0 |
| H6 q1 R32 | 新校正 | 8 | 188903.75 | 188903.75 |
| H6 q2 R32 | 独立検証 | 8 | 377494.75 | 377604.736 |

- 校正点では、model予測がdirect平均と一致している。3セル・3係数の厳密fit（lack-of-fit DOF 0）なので当然で、モデル精度の証拠にはならない。精度について情報を持つのは独立検証の2点だけ。
- Cが保存したpath別相対RMSEは、H4 1.039841%（`0.01039841107204328`）、H6 0.229959%（`0.002299590195904343`）。値を転記してVALIDATION.jsonと照合しただけで、再計算はしていない。B側で合格閾値は設定していない。
- directとmodelは別scenario（`C1:DIRECT:<cohort>`／`C1:MODEL:<cohort>`）。X/Yは同じpathのpair。Bのexact会計はcandidate・軸ごとに1つのscenarioしか選べないため、同じ候補のdirectとmodelを同時に加算することはない。
- rare診断の4 pairは`rare_diagnostics`に別unitとして保持した。価格record・cohort・fitには入れていない。

## 3. candidate・law・compilerの照合

既存adapterの受入検査と[VERIFY.json](VERIFY.json)で確認した。

- **8セルのfull identity**：B `build_menu/request_context`と一致した。対象はH/state/snapshot、configuration、T・qPF・qRPE・δ・formula・prefix・K・R、signed weights・allocation・event数・scalar。
  - H6のallocationは、R16で`[1,2,1,2,1,2,1,2,1,2,1]`、R32で`[3,3,3,3,2,5,2,3,2,3,3]`（11 signed piece）。
  - 全セルで event数 = qPF × R。
- **時刻**：実際の発展時間は`identity.candidate.T`で、次のとおり。
  - H4：q8が0.8、q16が1.6（δ=0.1）
  - H6：q1が1.6、q2が3.2（δ=1.6）
  - いずれも`T = base_time·2^m`かつ`δ = T/qPF`を満たす。
  - `identity.binding.T=0.8`は、B base taskが持つ旧参照値。B menu側のidentityにも同じ値があるので、一致比較は成立する。C1の固定結果は書き換えていない。
- **compiler・測定**：
  - compiler hash `1239304c…`、Qiskit 1.3.0 opt1 seed17、symmetric_directional、測定付きfull Hadamard wrapper、P=0。
  - X=cosine/Re、Y=sine/Im、`P0_minus_P1`、RZ gate_count。
- **値**：32 recordの値は、それぞれのevaluationの`observed_cohort_mean`／`predicted_cohort_mean`と一致した。B recordはC recordのcopyで、変更したのはB `law_binding`と`C1:`付きcohort名だけ。
- **law**：
  - B（N側）のlawは`IDEAL_FINITE_RTE_MEAN`、C1のsamplerは`PCG64_numpy_float64_CDF_saved_cohort`。両方の宣言とSOURCEを保持した。
  - 対応付けの根拠は、両者の`rte.py`が同一blob（sha256 `620a7007…`）であること。
  - `ideal_sampler_distance_qualified=false`。
- **不一致の検出**：公開fixtureのcopyに次の10種の改変を加え、すべて拒否されることを確認した：law定義、compiler、candidate R、cohort、n_paths、価格値、validationの校正への混入、model資格、実行SOURCE、凍結commit。
- **数値処理を呼んでいないこと**：数値計算の入口（C fit・評価・価格関数、B evaluate・会計、H4/H6予測）をすべて無効にした状態で再取込みし、同一のINTAKEが得られた。

## 4. 既存providerとの接続

- C1 recordは`records_for`により、B menuの中でexact identityが一致する8行にだけ付く。
- 固定catalog（H4 R4/H6 R16）の保存scenario、たとえば`DIRECT:EXPLORATION`とは別名で並存し、重複も同一視もしない。
- C1外のq/Rにはrecordを付けない。固定RのMq/Affine係数を新Rへ移すことはない（`RScopedFixedPrices`のguard）。B側で新しいmodel予測も行っていない。

## 5. 資格の状態

- **finite cohort**：32 recordすべて`FINITE_COHORT_ONLY`。
  - 既定の`qualified_expectation`では価格として使えない（`UNQUALIFIED_C`）。
  - 明示的な`finite_cohort_scenario`では、direct cohortを限定的なscenarioとして扱える。今回はそのscenarioでGを計算していない。
- **model**：全16件が`qualified_for_candidate=false`。独立検証の観測値は、モデル採用の承認ではない。
- **取得済みとして保持した不確かさ**：paired X/Y共分散、path間価格分散、nominal mean共分散、係数のnominal共分散、shared calibration group、validation summaryとmetrics、元SOURCEとcohort。
  - 係数を共有するのは、system別にmodel予測8件と校正direct 6件。`dependence_groups`に記録した。
- **未取得（null）のまま**：長q・大R外挿のsystematic、cross-R・cross-round共分散、joint共分散、全law期待価格、total CI、全RPE成功。

## 6. coverageの更新

[COVERAGE.json](COVERAGE.json)は、Aの元のR menu（PR 86候補）とC1 8セルの和集合でcoverageを作った。
- 行数はH4 66、H6 52、計118。
- 削除した行はない。`R_upper_bound=null`、`range_complete=false`も保持している。
- 行の内訳はPF、早期H6、C1追加2点を含む。

| 項目 | 状態 |
|---|---|
| A0 normalization | 既存の24候補のstatusのまま。N・Cへの換算はしていない |
| 詳細signal/phase/radius/N | 保存Nは各系26件（H4 m10 PRの補正済み整数`1070910265913026585642`も不変）。C1新R 4セル（H4 m3/m4 R8、H6 m4/m5 R32）はN欠測（`MISSING_N`） |
| C1 direct価格 | 8セルだけに、direct/model × X/Yの4 scenarioが付く |
| C1 model予測 | 同じ8セルの別scenario。未採用 |
| 長q・大R価格 | 例：H4 m12 R8192、H6 m12 R2048/R8192はA0 normalizationが記録済みだが、価格は`NOT_ACQUIRED`。小qのC1価格は代入しない |
| 実総費用G | 全行`NOT_EVALUATED`、costs=null |
| 最適R | 未採択 |

## 7. N/C/Gで残る不足

- C1新R 4セルのN（R-aware allowance、phase/radius）。同じ点で実N×Cを見るには、Aの別の科学実行が必要。
- 長q（H4 m12 q4096、H6 m12 q256等）と大Rの価格。C1の小q 8セルからは外挿できない。
- modelの採用基準と必要精度。全law期待価格とrare寄与。samplerとideal lawの距離。共分散と同時区間。全RPE成功。

## 8. 技術テスト・保全・STOP

- **[TESTS.txt](TESTS.txt)**：130 test method（129 PASS、既存の環境変数依存skip 1）。adapterとtestのSOURCEは`724d313b`から変更していない。
- **実データの照合**：[VERIFY.json](VERIFY.json)が`status=PASS`。内容は、凍結・SOURCE、32 recordのcopy忠実性、時刻・allocation、数値処理なしでの再取込み一致、公開fixtureの10種の改変の拒否、coverage。
- 作者自身の技術確認であり、独立した科学レビューではない。
- **保全**：
  - Cの公開結果、private raw、worktreeは読取専用で、Git objectだけを参照した。
  - Aは実行中の詳細N作業に干渉していない。
  - SOURCE、input、raw、result、grant、ledger、marker、KEEP、STOPを保全した。共有環境は変更していない。
- **STOP**：commit/pushとremote照合の後、**mandatory STOP**。実G計算やR最適化へは自動で進まない。
