# H8長時間・高次PR：Stage C独立標本比較の計画

今回の成果は設計・技術準備。Stage Cは未実行。提案の採択と科学実行は別で、公開SOURCEと下記条件に対するユーザーの別途明示許可後にだけ実行する。

## 固定した比較案と根拠

対象はLinear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit、spin sector4900。T=1.6、ε=.0005、P=0、元state・fragment順序0..27を維持する。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/topologyなし。各軸の費用はmeasurement付きfull Hadamard wrapperのcompiled RZ数とする。

| 役割 | 保存candidate ID | 条件 | 保存N_cos / N_sin |
|---|---|---|---|
| PR | `7eb0a3d691b94089969b` | 4th(m5_best)、prefix8、q1、R_outer32、K4 | 84916951 / 88339169 |
| PF固定対照 | `1425ae6ed83fd9098b39` | B0I、4th(m5_best)、prefix15、q1 | 71893270 / 73757762 |

IDは[公開結果](../h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)から取得した。推測・新規採番ではない。方向は `G_PR < G_PF` に固定する。
PFは保存済みの両軸価格613049 RZ、G=89291219516568を固定量として使い、Stage CでPFのsampling/build/compileを行わない。保存PF観測のcohort=EXPLORATIONは書き換えない。
PRの探索標本は候補選択・計画根拠だけに使い、確認平均・分散には混ぜない。Stage Aのbias、数値allowance、headroom、N、適格性は保存endpointのままとし、新reference・signal・shot再計算は行わない。

根拠は公開commit `7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84`。Stage Aは69候補（適格40、不適格29）、Bは19候補・51標本・19代表133検査PASS。
Signal SOURCE `d1272016592ff06c0567862f770ed796fd61b366`、native SOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`、位相補助SOURCE `18ce41942051065fa097644a8a937bd11adba708`。
[scope案](../../../configs/h8_long_time/stage_c_proposal_20261104.json)にbinding、Stage A/Bの圧縮結果hash、正式ID、endpoint、元aggregateを保存した。

## 推奨する規模と精度

ユーザーの方針は「独立cohortの標本比較を先に設計する。母集団保証は別に扱う」。推奨案は**探索と同じ規模・報告精度を目標にする独立再現試験**とする。

- `cohort=CONFIRMATION`
- **n=9** の新しいPR trajectory、両軸計18価格。
- **absolute_SE_G_target=121655623062.28098 RZ**（約1.216556e11）。探索結果に保存されているsample SEそのものを引用する。
- 固定PF対照との確認標本平均の向き、およびこの記述的SE目標の達否を別々に報告する。
- 95%等の信頼水準・有意水準は正式採択しない。population guaranteeは常に別判定とする。

保存されたPRの探索aggregateは、n_B=9、平均G=67323362591920、sample variance `133200815604046129447200`、sample SE `121655623062.28098`。これらは引用であり、今回価格や分散を再集計していない。
推奨nは新しく最適化した値ではなく、実際に取得済みのcohortサイズを引き継ぐ。h_Gも任意の便宜的な数値を置かず、到達済みの報告精度をそのまま目標とする。公開報告の観測差約24.6%と、同報告のsample SE/平均約0.181%を踏まえ、最初の独立標本比較で同じ精度を目指す計画である。
**最小n・最適n・母集団精度を保証するnを求めたものではない。** 希少次数を調べる規模として十分だとも主張しない。n・h_Gは科学実行前にユーザーが採択する提案値である。

各標本は `G_i=N_cos C_cos,i+N_sin C_sin,i` とする。

\[
s_G^2=N_c^2s_c^2+N_s^2s_s^2+2N_cN_s s_{cs},\qquad
SE_G=s_G/\sqrt n.
\]

既存の `paired_G` は同一trajectoryの両軸を重み付けした後で分散を求め、共分散を含む。
一般の計画式は `n_plan≈ceil(s_G,B²/h_G²)`。h_Gを未丸めの探索SEと同じに置けば、式の内部は記号上n_Bに対応する。今回、この式を実H8の保存値で新しく数値評価していない。保存されたbinary64のSEからceilを再計算してnを決め直す処理も行っていない。
将来、別の精度・rare coverage・母集団保証でnを設計し直す場合は、その入力・目標・式を明示した科学的数値評価が必要となる。今回の9標本をその設計の代用にしない。

## 希少orderと母集団について分かること

PR候補の保存sampling lawは `UNCHANGED_PCG64_NUMPY_CHOICE_FINITE_PAIRED_RTE`。公開aggregateから引用すると、1 trajectoryでorder2以上を少なくとも1回含む確率は `0.017503368117490408`、order4以上は `1.7365913367114017e-6`。
探索9標本での未観測確率はそれぞれ `0.8530601056102244` と `0.9999843707865361`。実際の288 eventはorder0のみで、order2/4は0件だった。
同じlaw・同じnを使う今回の推奨案にも、この既存lawに基づく未観測の限界がある。PCG64の数学的独立性が証明されたという意味ではなく、保存lawのモデルに関する引用である。新しい希少確率の数値評価は実施していない。

通常群を0、希少群をr、希少trajectory確率をpと書けば、

\[
\mathbb E G=(1-p)\mu_0+p\mu_r,
\qquad
\operatorname{Var}G=(1-p)\sigma_0^2+p\sigma_r^2+p(1-p)(\mu_r-\mu_0)^2.
\]

希少群を観測しないsample varianceから、未知のmu_rやsigma_rを0とは置けない。ゼロ観測は費用寄与が小さい証明ではない。PFとの差に対してどの程度寄与し得るかも、この未知費用を扱わずには保証できない。
`population_tail_cost_bound` と未観測次数の条件付き費用は引き続きunknown/nullとする。

### 既存コードからの費用上界の可能性（静的調査）

`h46_long_time_native.compose` はsigned pieceごとに、保存blockのinstruction boundと実際のtail命令長を使ってnative回路を構成する。
`df_rte_qiskit.QiskitDFRTEEventCircuitBuilder.build_sequence` はevent applicationを展開し、同じbasisの隣接取消を処理する。有限K・有限component集合から、未compile構造の上界を検討する入口はある。
`rte_compiled_cost.estimate_df_rte_structural_size_upper_bound` の既存経路も未compileの構造上界である。一方、価格は `transpile_and_measure_cost` が固定compilerを適用した**後**のRZ数である。
前者を、そのまま後者の一様上界とみなす根拠は確認できていない。opt1でgate数が単調減少すると仮定せず、全supported gateの分解とcompiler passまで含む上界の妥当性が必要になる。
今回は実gate条件の新compile、希少費用の数値評価、全event列の列挙を行っていない。数値上界は未確定であり、観測最大値を上界として流用しない。

仮に全希少列に対して `C_c≤M_c, C_s≤M_s` を示せれば、`G_max=N_c M_c+N_s M_s` として希少群寄与を `E[G 1_rare]≤p G_max` で抑える候補になる。
このM・G_maxは今回求めていない。独立性・有界性が妥当で、ユーザーが保証基準を事前採択した場合には、bounded concentration等を別に検討できる。保証水準やその手法を今回正式採択していない。
希少群の観測を計画する式 `n≥ceil(log(eta)/log(1-p))` も、未観測許容確率etaを事前に決める必要がある。etaや新しい数値nは今回設定しない。

## 再利用と必要な変更

| 再利用先 | 今回の扱い |
|---|---|
| `h8_long_time_adapter.preflight` | 科学入力・旧native source mapのmetadata照合 |
| `h8_long_time_runtime.Context / price_job / map_jobs` | 将来のH8 restore、元と同じ7検査、並列trajectory処理 |
| `h46_long_time_native.sample_requests / compose / compile_cached / rarity` | 元sampling law、signed native、compiler価格、希少診断 |
| `h8_compiler_phase_bookkeeping.certify / correct_phase` | SOURCE18ce4194のまま。実際の局所operatorからglobal phaseのみ会計 |
| `h8_long_time_adapter.paired_G` | 将来取得するCONFIRMATION標本だけのexact paired会計 |
| `confirmation_n` | 既存計画式を調査。今回実H8数値で呼び出していない |

追加は[薄いStage C入口](../../../scripts/h8_stage_c.py)と人工テスト。元SOURCEのファイルは編集していない。
既存汎用C経路はPFも新規にbuild/compileし、比較関数も両群のcohortをCONFIRMATIONとして扱う。今回の「保存PFを固定量として使う」仕様に合わないため、PFの探索originを保持する小さな比較関数を追加した。
また元runtimeの直接compileには位相補助が入らないため、compile callbackで既存helperを適用してから元の7検査へ戻す。代表cache hitでQCがない場合も再compile価格照合後に位相補助を適用する。

## 将来の実行順序と科学call範囲

1. 公開されたStage C SOURCE、scope案、n=9、h_G、比較ID/方向をユーザーが採択し、**科学実行を別途明示許可**する。CLI `--execute` は許可の証明ではなく単なるdispatch。
2. metadata preflightでStage A/B bytes、元SOURCE/input、N/endpoint、phase helper、現在の実行コードと公開SOURCEの一致を確認する。privateのStage B seedファイル2本もhash照合する。
3. fresh outputのrun_tagを固定する。新namespace `H8_STAGE_C_CONFIRMATION_V1` とCONFIRMATION、実行SOURCE、binding、run_tag、candidate/index/outer/pieceから新seedを生成する。元Bの全seed・旧入力で除外指定されたseed・今回cohort内との実値衝突を拒否する。
4. 既存H8 loaderで保存入力をrestoreし、PR prefix8だけの新9 trajectoryを元sampling lawで生成する。新SCF/DF/state生成、新T reference/finite mean、PFの科学再計算は含めない。
5. signed native build、full measured Hadamard wrapper、固定compilerによる両軸価格取得。完全fingerprint・compilerが一致する元のraw compiled cacheだけを読取再利用できる。別cohortのtrajectory/ベクトルを確認標本へ流用しない。
6. 代表index0で元と同じ7検査。low/high control、algebraic event、cos/sin期待値は1e-8、compiled vectorは1e-7。位相整列なし。実際の非basis gate/parameterは局所operatorで確認し、helper由来のglobal phaseだけを補正する。RZ数を変更しない。
7. 新CF標本だけでpaired G、sample variance/SE、rare診断を計算し、固定PFとの差・点方向・記述的SE目標の達否を分けて保存する。母集団保証は自動認定しない。

n・h_G・候補・方向を確認結果から変更しない。精度未達、方向不成立、希少orderの未観測はそのまま報告し、top-up・候補差替えを行わない。
SOURCE/input/seed/cohort/fingerprint/compiler/normalization不一致、未対応gate、非有限値、代表検査失敗はHOLDを保存し、自動retryしない。単なる共有負荷や他worktree更新では中断しない。

### 入出力と実行SOURCE

公開入力はscope案にあるStage A/Bの圧縮metadata。物理入力はStage A/Bと同じprivate configの既存SOURCE・bindingを使う。
専用cloneのprivate configはmetadataだけ準備済み。seed実値・event列・raw/QPY・内部pathは公開しない。出力はこのclone専用private root内のfresh run_tagへ閉じる。
実行SOURCEは**本準備branchの公開最終commit**を将来CLI `--source` に指定する。scopeにはそのcommitとコード/計画hash、元3 SOURCEを別々に記す。
defaultはmetadata inspectで科学呼び出しを行わない。今回実施したのもinspectと人工テストだけである。

```bash
python -B scripts/h8_stage_c.py --proposal configs/h8_long_time/stage_c_proposal_20261104.json
```

科学実行の将来入口は `--config <private-config> --source <公開Stage-C-SOURCE> --run-tag H8_M5_P8_Q1_CONFIRMATION_01 --execute`。現在は起動しない。
結果は `scope.json`、`seeds_PRIVATE.json`、`preparation_binding.json`、`prices/<PR-id>/<index>`、`compiled_cache`、`result.json`。失敗時はHOLD.json。PFには新価格ファイルを作らず、結果内のfixed_PFとして元のorigin付きで引用する。

CPU/RAMの恣意的上限は置かない。実行時の空き資源・共有負荷・回路サイズを見てworkersを選び、独立trajectoryと局所operator検査を並列化する。共有設定・他利用者のprocess/環境は変更しない。

## 三段階の結論と完了条件

1. **標本比較**：新CF平均が固定PFより低いか。
2. **記述的不確実性**：paired sample SEがh_G以内か、どのorderを観測したか、何がunknownか。
3. **母集団優位**：一様費用上界・rare寄与・独立性・事前の保証基準が揃うか。1と2から自動的に成立としない。

今回の技術準備後はmandatory STOP。新科学実行、保存価格の数値再解析、詳細科学レビューを開始しない。採択判断は[次段判断](NEXT_STAGE_DECISION.md)、人工検証と証拠は[VALIDATION](VALIDATION.json)を参照する。
