# 研究C：Stage P／Compiler適用性の統合と期待bias校正の実行可能性

2026-11-04。技術実装・人工検証のみ。新規実H4科学call、sampling、実回路build/compile、保存rawの科学再解析、Stage C、重要科学レビューはすべて0。
正式共通指示の原本 `research_project_instructions_unified_2026-10-25.md` §§8.1・8.4・10を適用した。

本作業は、固定1-wordの包囲を全lawへ拡張するものではない。現在の期待bias校正は **NOT_READY**。
次の最小候補として、保存native recordに対する高速certificate評価器の **7-call限定pilot** を準備した。未承認・未実行であり、校正法やabort方式の正式採択も行っていない。

## 1. 固定条件と証拠の接続

Linear H4・1.00 Å/STO-3G、number CDF8/LD2、partial S2/s1/r32/K0、h=δ=.1、m0〜12、controlled full512。
PR endpoint SHA256は `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
Qiskit1.3.0/P0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、既存Gaussian/relative fusion、own-DF ground-state oracle・状態準備費用除外、IID 53-bit grid、既存CA/RPE政策を維持する。

以下は既存結果の引用であり、本作業で再計算していない。

| 証拠 | 固定result commit | 資格・限界 |
|---|---|---|
| Stage P後続完了 | `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e` | [報告](research_c_stage_p_completion_execution_20261104.md)。独立15/15 COMPLETE、SOURCE `2c454ed089fd6995f22c87d0a34886e1f680f3a1`。指定F6/F7交互32-event wordのみ |
| Stage P旧部分実行 | `5e1450b1071359b81819fe5de481cdeccc9d3f35` | [報告](research_c_stage_p_literal_execution_20261104.md)。30 attempts、29 COMPLETE、1未完了。旧41-call runを正常完了へ変更しない |
| Codex-A compiler | `60dcaa94fd44675bc3184e409aa4ebdef7a07dbf` | [固定技術報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60dcaa94fd44675bc3184e409aa4ebdef7a07dbf/docs/research/research_c_p0_compiler_totality_readiness_20261104.md)。全word成功は未資格。失敗反例を得たという意味ではない |
| Stage R | `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039` | [報告](research_c_population_range_schema_execution_20261104.md)。256/256 COMPLETE、literal Dの全正q-support216カテゴリのpopulation norm |
| Head bridge | `369e2dfad9a4251ba5eb2fda28f183d559a791cc` | [報告](research_c_head_bridge_execution_20261104.md)。26/26 COMPLETE。sector36のhead参照接続。全word compiler保証ではない |
| B2 | `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0` | [報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7a3db09e395d36965e5b9dc28ae4a299db3b2f0/docs/research/research_c_reference_bound_execution_20261103.md)。27/27 COMPLETE。atom→数学的CAR→binary64 referenceの参照誤差 |
| IID-grid Stage A | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` | [報告](research_c_iid_grid_stage_a_execution_20261103.md)。38/38 COMPLETE。理想IID-grid law、実PCG64 streamの保証とは区別 |
| 期待bias準備 | `bd560a45410409573ae849ca5aaf4050343910bf` | [報告](research_c_iid_grid_expected_bias_readiness_20261104.md)。非実行設計・楽観的計算量例 |

各result/SOURCE tree・契約・STOP・報告の固定対応は [fixed_evidence_refs.json](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/fixed_evidence_refs.json)。契約がSOURCE版とresult版で同一であることも確認した。

Stage Pの引用上界は C→R `1.585733058571e-11`、R→D `3.957579638642e-14`、C→D `1.589690638209e-11`、compiled actual inverse `5.542118759907e-15`、単一occurrence raw complex signal `4.496362061560e-11`。
約1667秒は後続15-call処理のwall実績であり、一般的per-word価格ではない。旧部分実行のC/R約1000.44秒、R/D約166.94秒も、この固定word・当時の実装の記録に限る。

機械可読な [evidence_graph.json](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/evidence_graph.json) は次を別々に保持する。

| 資格 | 現状 |
|---|---|
| 指定1-wordのcompile成功・operator/inverse/shell包囲 | 既存固定証拠あり |
| literal population norm/range | 既存Stage R固定証拠あり |
| 全合法wordのcompile成功 | false：未証明 |
| 全合法wordのsynthesis保証 | false |
| IID-grid期待bias校正 | false |
| IID-grid条件付きRPE十分条件 | 未接続 |
| actual CA achieved | null |

Stage Rのexact値は表示小数へ置換しない。

```
a = 1569275433846670467439326996090883863351856440850483005231
    /1569275433846670190958947355801916604025588861116008628224
M = 3138550867693340658398274351892800467377445301966491633455
    /1569275433846670190958947355801916604025588861116008628224
M = 1+a > 2
```

`||C(w)-D(w)||≤1+a` は、各対象wordでCが定義されたunitaryであることを要する。
未成功wordのCを未定義のまま母集団の誤差変数を作らない。別scopeを正式採択しない限り、compiled-native populationへのrange接続は未資格とする。Head+B2は参照legであり、Pのnative誤差と混同・重複計上しない。

## 2. Arb full-basis certificateの最小高速化

既存 `c_pr_aware_df_fixed_certificate` のgate定義、入力検査、resource plan、Arb192、列norm包囲とrecenterを再利用した。
新しいopt-in [c_arb_column_batch.py](../../src/trottertracks/c_arb_column_batch.py) は旧コードを変更せず、次だけを追加する。

- 全入力列を小batchの`acb_mat`へまとめ、局所FLINT行列積を共有。各gate・global phase後の丸め半径を**列ごと**に外向き加算する。
- qubit scatter/gatherと同一hexパラメータgate行列をprocess-localにcacheする。候補間の近似的同一視や共有cache変更は行わない。
- 2個のnative unitary recordについて、厳密に同じ先頭・末尾gate列だけを除く。両global phaseは保持する。

最後の変換は `A=e^(iφ_A)L A_res R`、`B=e^(iφ_B)L B_res R` の共通unitary L/Rによる左右不変性を使う。
operator 2-normとFrobenius normは保存され、残差recordの全512入力列・全出力から元operator差の上界が得られる。
因子を除いた後の個別列は、**元の同名計算基底列そのものではない**。出力schemaにこのbasis変更を明記した。
Frobenius包囲からoperator上界を取り、下界は最大列norm。入力gate・register・phaseをbindし、phase fitはしない。
native-unitary同士だけは既存の上界2を使用できる。**非unitary literal Dにこの削減やclipを適用しない**。

| 選択肢 | 今回の扱い |
|---|---|
| Common Gaussian/headの重複gate作用 | 厳密な共通unitary因子除去とlocal gate cacheを実装 |
| CAR/head expのword間cache | 入力bits・candidate・h・block・precisionにbindする必要あり。新しい実H4 cache評価は行わない |
| Spin/block構造 | literal CAR referenceの構築共有には適合。native Gaussian gateを個別sector射影してfull512検証を代用しない |
| `c1_h6_state_action`、ColumnStream | Aer/complex128または浮動小数点全列の数値評価。Arbの丸め包囲を代用できない |
| State/signal限定 | 将来のdirect biasには候補。operator保証と資格が変わり、oracle/stateの誤差包囲も必要。現契約へ適用しない |
| Round間共有 | 同じkernel母平均の資格を13roundへ伝播可能。独立occurrenceの標本数・compiler要求を省略できる意味ではない |

人工benchmarkは [script](../../scripts/benchmark_research_c_arb_column_batch.py) と [全記録](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/artificial_benchmark.json)。分子入力・保存H4 record・QPYを読まない、決定的な人工gate列で3反復、old/newの実行順を交互にした。

| 人工問題 | old中央値s | new中央値s | old/new |
|---|---:|---:|---:|
| 4qubit、batchのみ、因子除去なし | 0.02684 | 0.02515 | 1.07 |
| 9qubit/full512、共通因子あり | 2.68294 | 0.42069 | 6.38 |

batch単独の小問題では明瞭な速度差を示さない。6.38は共通因子削減を含む人工結果であり、H4や未知wordのspeedup・wall予測ではない。
観測CPU affinity32、当時available RAM約56GB。重い人工比較は32 process、残差が小さい場合はspawn費用を避ける既存work判定を使う。利用可能資源と独立batch数からworkerを決め、CPU/RAM/wallの人工capは追加しない。共有venv・他job・設定は変更していない。

## 3. Compiler失敗を含むscope

[c_p0_compiler_totality.py](../../src/trottertracks/c_p0_compiler_totality.py) はCodex-Aからbyte一致で取り込んだadapter（原blob `67d86d45ba807021202a4bfd3ce43ab7d5921d17`）。その純粋検査と1回限りP0呼出しを再利用し、Aの非実行契約を実行可能へ変更していない。
全word成功の障害は、固定1000反復以内のfixed point到達、全到達数値入力・phase/definition処理の有限性・例外排除である。

| 案 | 得られるもの | 残る条件 |
|---|---|---|
| 元P0の全word成功証明 | 元の無条件lawでCが定義される | 1000反復・全例外経路等の静的証明が未完 |
| Fresh固定Nの失敗率評価 | 固定compiler map下のp_failのconfidence上界 | 全word成功証明ではない。未観測・非停止・途中STOPを除外しない |
| 明示abort scope | 失敗を運用リスクとして扱える | 成功条件付きlaw/estimand、RPE接続、operational abortの正式な定義が必要 |

技術的第一候補は **明示abort scopeと固定N失敗率評価の組合せ**。未採択であり、fallback回路を実装・採択しない。
全成功より弱い「有限時間で成功または明示abortへ到達する」証明も有用だが、非停止が未解決ならこの資格も得られない。

新adapter [c_totality_calibration.py](../../src/trottertracks/c_totality_calibration.py) は人工outcomeの評価機能を持つ。
raw build failureと、raw成功後のcompile exception/controller guard failure/invalid native outputを分離する。
pipeline失敗率は全N件、compile条件付き失敗率はraw成功件数を分母とし、raw失敗をcompile成功へ数えない。
UNKNOWN・非停止未解決・実行中断は固定N資格を返さない。恣意的timeoutは導入していない。締切をfailure定義に採用するなら別のアルゴリズムscopeの判断を要する。

固定N全件を評価する統計実験と、初回failureでSTOPするcorrectness実験は異なる。
draw IDは固定prefixを維持し、失敗wordの削除・再抽出・retry/resumeを拒否する。同じwordが再び抽出された場合は**重複標本として保持**する。
成功0失敗の信頼上界は既存Arb adapterによる `p_U=1−δ_compile^(1/N)>0`。
非零失敗は既存Hoeffding半径を使う保守的上界であり、非零kのexact Clopper–Pearsonとは主張しない。
固定N・IID・既知のBernoulli outcomeを必要とすることは [NISTの二項信頼限界](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) と整合する。

### 成功条件付けと確率政策

失敗wordを捨てて得るq|Sは元のqと異なる。
例えば `E_q[z_D]=(1−p)E[z_D|S]+pE[z_D|F]` であり、別途 `|z_D|≤Z_D` を証明すれば選択による平均差は `2 Z_D p` 以下になる。
条件付き平均を元の無条件biasとして渡してはいけない。
分析上のzero-padding/killed-signal変数を定義する案も、実行可能fallback回路を採用することとは異なり、元native平均を自動的に定義しない。
いずれもestimandとRPEへのtransportが未採択なので、accuracy engineは未接続のままにする。

独立kernel occurrenceごとにfresh wordを用いる既存政策では

`T_draw = Σ[m=0..12] 2^m (shots_X[m] + shots_Y[m])`。

これはsymbolicな要求数であり、新しい実shotsを計算していない。Actual inverseは同じrealized compiled recordから生成し、別wordを再抽出しない。
exact word cacheはcompile呼出しを減らし得るが、fresh draw数・標本重複数は減らさない。
source/candidate/angle/head/basis/P0/environment/phaseを含むkeyと決定的なword→outcome map、または明示的に正当化したjoint lawが必要。seed17だけでその性質を証明しない。
共有shellのcompile件数は別に数え、既存の資格付きshell再利用が成立するときだけ追加0にできる。

同じ前提で全運用abortはunion bound `min(1,T p_U)`、独立性があれば `1−(1−p_U)^T` を使用できる。履歴依存・cache・適応的compilerではその前提を再確認する。
δ_cal、δ_compile、p_fail、運用abort、RPE測定α=.01を分離する。共通kernelの校正confidence eventを13roundへ再利用するときは同一eventを13回加算しない。
仮に各資格が接続できても総failureは概念上 `α+Σδ_cal+Σδ_compile+P(abort)` となり、測定α=.01だけから全体.01は得られない。
成功条件付けの下で従来のRPE α政策が有効かも別のpremiseである。adapterはそのpremiseなしに無条件failure上界を返さずnullを維持する。新しいconfidence配分は正式採択していない。

## 4. 期待bias校正法の比較

既存 `c_per_word_population_range` の `threshold_tail`、`variance_sensitive`、`calibration_inputs`、`propagate_calibrated_mean` と、`c_iid_grid_expected_bias` の `mean_upper`、`paired_round_upper`、`assess_if_adopted` を再利用する。accuracy engineは再実装していない。

| 項目 | Threshold tail | Variance-sensitive | Direct round raw complex bias |
|---|---|---|---|
| 標本 | 各wordのoperator差dの包囲・threshold決定 | 各wordのdの区間 | 各roundで対応するC/Dの符号付きX/Y信号差の区間 |
| Range | 採択scopeでd∈[0,M] | 同左 | round literal normとshellから別途信号rangeが必要 |
| N・confidence | thresholdを跨ぐ区間をgoodにしない。zero badでもtail非零 | 既存empirical Bernstein、区間不確かさを保持 | 既存paired X/Y平均のconfidence。round間配分も必要 |
| Native費用 | 単純cold policyでN build/compile、setupは別 | 同左 | cold policyでΣN_m 2^m occurrences、shell/setupは別 |
| Arb費用 | wordごとのfull-basis enclosureが支配 | 同左。分散が小さいことを別途観測する必要 | state/signal evaluatorが資格付けされれば軽量化候補だが未実装のrigorous oracle/state接続あり |
| m12 | 共通μをcoherent反復へ伝播するため小さいkernel許容誤差を要求 | 同左、range依存項は残る | kernel単位の2^m bias増幅を避け得るが標本の構築費用が増大 |
| Compiler失敗 | 全成功または採択した代替estimandが必要 | 同左 | 同左。長roundは多数のcompile要求を伴う |
| Stage C接続 | 校正・参照transport・confidence全項が必要 | 同左 | 対応するround signalと全誤差項の資格が必要 |

Threshold法は `μ≤t+(M−t)p_bad`。variance法は [Maurer–Pontil Theorem 4](https://arxiv.org/pdf/0907.3740) の標本分散項に加え、正規化前には `7 M log(2/δ)/(3(N−1))` の項を持つ。小分散でもこのrange項は消えない。これらの方法を比較する実H4の分散・tail・期待誤差は新しく計算していない。

既存[楽観的設計例](research_c_iid_grid_expected_bias_readiness_20261104.md)の引用では、δ_cal=.05、a=1、shell誤差0、raw room .25という**未採択の仮定**で、kernel Hoeffding N=12,866,572,143、zero-bad tail N=555,301、direct N/round=1779、13round合計14,571,789 kernel occurrencesだった。
これは現H4の必要Nの確定値でも一般的下限でもない。Stage Pの小さい1点から小分散・低tailを推定しない。Stage RのM>2も保持する。

現段階では3法のどれにも、m12まで有用で現実的なフル校正実行を保証する根拠がない。直接法は将来候補として残すが、既存AerやColumnStreamのdouble値をrigorous信号boundへ昇格させない。

失敗率評価のcall規模は、候補setup `S_bind+G_basis+G_relative`、固定Nについてword抽出・raw build・compile・outcomeの最大4N、集計1という設計式までとした。setup/cache/N未定なので数値上限はnull。bias側はoperator/signal方式とoccurrence政策が未定でcall式もnull。未確定を0や任意の小さいNで埋めない。

## 5. 次の最小検証と未承認契約

第一に実行する価値があるのは、**保存済み同一wordのnative pair評価器を2組だけ確認する7-call pilot**。
期待bias校正のNを先に増やすより、支配費用を維持精度のまま減らせるかを小さく確認する。
このpilotで分かるのは固定2組の数学的target保存・包囲・wallであり、未知word性能・母集団bias・compiler全word成功は分からない。

| 契約 | 状態 | calls |
|---|---|---:|
| [固定pair pilot](research_c_fixed_pair_batch_pilot_contract_unapproved_20261104.json) | 実行経路・SOURCE/input固定、passive preflight PASS、**別途許可まで未実行** | 最大7 |
| [期待bias/失敗率設計](research_c_stage_p_totality_calibration_contract_unapproved_20261104.json) | UNAPPROVED_DESIGN_NOT_EXECUTABLE、方法/N/δ/abort/cache/非停止政策未採択 | null |

Pilotは旧保存compiled/raw/compiled actual inverseのnative JSON recordを用いる。
比較はC/Rと、保存actual inverse対compiled recordの数学的adjointの2組。旧literal reference・旧certificate・旧QPYは生成し直さない。新sampling/build/compileは0。
Arb192、full512全列・全出力、batch16、元hex phase/word/digestを固定し、既存resource planからprocess数を決める。
新旧包囲の非矛盾も検査するが、区間の重なり自体を正しさの証明とせず、target binding・unitary因子削減の導出・Arb丸め包囲を根拠とする。tightnessが変わることは許容し、未知wordの速度を認定しない。

| Pilot logical unit | 上限 |
|---|---:|
| 保存証拠metadata | 1 |
| 保存native record load | 1 |
| input/word/phase identity gate | 1 |
| 高速full-basis certificate | 2 |
| 旧包囲との整合記録 | 1 |
| summary | 1 |
| 合計 | 7 |

Fresh出力は `artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01`。
準備済みentryは `scripts/prepare_research_c_totality_calibration.py`。引数なしは**passiveのみ**。
将来、固定7-callへの別途明示許可を得たときだけ `--execute-fixed-pair-pilot` を使用する。今回は実行していない。
SOURCE/input/runtime/fresh gate不一致は科学開始前STOP。実行中の不整合・例外はpartial evidence/journal/STOPを保存し、retry/resume0・追加pair/word/fallbackなしでmandatory STOPする。旧枠は使用しない。

研究C・ユーザーの後続判断は、まず7-call pilotを実施するか。その後、compiler代替scope、failure/非停止定義、cache政策、N、δ_cal/δ_compile、運用abort許容、RPE transportを採択するかである。今回Codexはこれらを正式決定していない。

## 6. 技術検証・保全・公開

新規 [テスト](../../tests/management/test_research_c_totality_calibration.py) 40 PASS。関連head/Stage P・fixed CARを含め108 PASS、2 deselected。
除外2件は旧実行済みH/P出力先に対するcold passive preflightであり、過去のfresh namespaceを再利用しないため除外した。新契約のcold preflightは数値library import禁止条件でPASSした。
検証範囲は、解析的位相差の包含、full512・batch幅・共通因子削減、非unitary入力拒否、旧SOURCE保全、compiler成功/失敗・例外時の非retry、非停止未資格、標本除外禁止、confidence event二重加算防止、未資格Stage C拒否、契約の型/word/model/precision/fresh検査を含む。実分子の新しい性能証拠ではない。

証拠： [tests.txt](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/tests.txt)、[regression_tests.txt](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/regression_tests.txt)、[passive preflight](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/passive_preflight.json)、[SOURCE](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/SOURCE.json)、[保全](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/preservation.json)、[技術STOP](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/technical/STOP.json)。
技術SOURCEは172ファイル、保存inputは74ファイル。旧SOURCE/input234 unique files、直前Stage P後続出力16ファイル、Codex-A原moduleの同一性を確認した。旧41-call部分失敗と15-call完了はそれぞれ原状のまま保全し、旧SOURCE/raw/契約/grant/ledger/marker/STOPを変更しない。

Branch：`codex/research-c-stage-p-totality-calibration-feasibility-20261104`。親はStage P result `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e`。本報告・SOURCE・人工証拠・未承認契約の公開commitは集約用であり、既存実験の科学SOURCEを付け替えない。
独立worktreeで技術作業を実施し、公開後remote SHA/tree・報告本文・主要blobを照合する。科学実行は0、fresh科学出力先は未作成。**Mandatory STOP**。
