# 研究C：固定U tail LP・微小係数保持profileの限定科学実行

## 1. 結果と検証範囲

固定SOURCE `c40f8128197177fd5ec2388cd306d89d36c3bd67`を変更せず、[新しい50-call契約](research_c_fixed_u_tail_lp_numerics_contract_unapproved_20261101.json)に基づきLPを1回実行した。**50/50 logical units COMPLETE、correctness FAIL 0、retry/resume 0**。登録されたLP数値診断と全保持gateがPASSし、結果は[`NUMERICAL_LP_IMPROVEMENT`](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/result.json)。

対象は保存直鎖H4・1.00Å・STO-3G、通常number CDF8、Nα=Nβ=2/sector36、元frame順LD2、head0/1・tail2..7、partial ordered symmetric S2 h=.1/s1/r128/K0。元U全byte・head J・元g補正κ・scalar・frame順を保持し、tail対称Jだけを変数とするexact Pauli L1問題である。

| 主結果 | 値／状態 |
| --- | --- |
| 元tail λ [Ha] | 0.9545432846581517 |
| 新tail λ [Ha] | 0.9544320281272091 |
| λ削減 [Ha] | 0.00011125653094268273（約0.0116555%） |
| 登録した改善判定閾値 | 2.0454696431443494e-8 |
| Numerical optimality／strict improvement | true／true |
| Tensor／sector36・full256 CAR／head・phase／DF予算／1 h-step PF・補正K0保持 | 全PASS |
| 新CDF fit・CDF optimizer iteration | 0／0 |
| 元CDF optimizerの資格 | **UNQUALIFIEDのまま** |
| 厳密LP certificate・actual CA・shots・native価格・総資源・独立新規性 | 未資格／null |

これは固定した一つの保存終点・固定Uの係数再分解に関する数値pilotである。小さいが登録閾値を超えるλ削減を確認した。λ差はtail Pauli係数質量の差であり、energy誤差の改善量やRZ削減率ではない。厳密な大域最適性や一般のDF性能優位は認定しない。

## 2. SOURCE・入力・実行前確認

独立branch `codex/research-c-fixed-u-lp-numerics-execution-20261101`、fresh output `artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01`で実行した。[技術準備](research_c_fixed_u_tail_lp_numerics_preparation_20261101.md)の実装・未承認という歴史的契約metadataは変更せず、チャットでの明示許可に基づく。承認原文保存・承認記録・新grantは作成していない。

| Identity | 固定値 |
| --- | --- |
| Science SOURCE | `c40f8128197177fd5ec2388cd306d89d36c3bd67` |
| Implementation commit | `45e0dc292343a5cb38d69a94563d4e5a6dc757fa` |
| 契約SHA256 | `a02b6422cf87a34de647b153da65dbf6250d3c2411937d7fbc08babdf002c66a` |
| SOURCE manifest SHA256 | `0c27ca5e99f8b3d72f983f2b9ddfc13fd5939b40305a7e0788901dfa16b90fb7` |
| 元PR export SHA256 | `3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21` |
| 元physical input SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 保存reference SOURCE | `d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6` |
| 新candidate SHA256 | `eb064fcae6dae771fed3e07cc447530023cd53646726dbe30ddb56a4bc8279ca` |

[実行preflight](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/execution_preflight.json)でSOURCE109件、保存入力12件、依存lockとNumPy1.26.4／SciPy1.14.1等の環境、旧raw945件、fresh output未作成を確認した。準備時の104人工・回帰PASSは同一固定SOURCEの証拠として再利用し、今回はSOURCE変更0のため不要な再テストを繰り返していない。

実行entryは`run_research_c_fixed_u_tail_lp_numerics.py`で、旧LP/execute/CAR/PF-RTEの適合経路を再利用した。旧run01の再起動や残41 unitsの流用ではなく、新しい固定scopeでのLP1回である。

## 3. LP原結果と独立数値資格

[Solver原結果](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/LP_solver.json)はsuccess=true/status0、HiGHS Optimal、nit389、fun=0.9544320281272076。`highs-ds`、presolve=true、primal/dual solver tolerance各1e-10、small_matrix_value=1e-12で、retry・別option探索・座標変更・行削除・dual修復を行っていない。

[警告原本](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/solver_option_handling.json)には固定SciPy1.14.1の既知2警告だけが保存され、登録した設定処理gateがPASSした。「Using default」という警告は技術準備で切り分けた整数用・実数用ループへの重複登録に由来する既知のもの。新しいnativeエラーを黙って抑制・fallbackした結果ではない。

[独立診断](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/LP_diagnosis.json)は、solver内部の処理後の行列ではなく、元の全A/T/bで実施した。

| 登録診断 | 実測値 | 基準 | 判定 |
| --- | ---: | ---: | --- |
| 元j0のfull-row equality residual | 2.42861286636753e-17 | 1.0767046484579418e-10 | PASS |
| 返却点のfull-row equality residual | 4.864164626638967e-15 | 同上 | PASS |
| Epigraph violation | 6.13170538649177e-16 | 1e-9 | PASS |
| u非負違反 | 0 | 1e-9 | PASS |
| Dual equality residual | 9.886980123496869e-12 | 1e-9 | PASS |
| Dual box／inequality sign／u-dual違反 | 各0 | 1e-9 | PASS |
| Complementarity | 6.13170538649177e-16 | Objective基準、約1.95454e-9 | PASS |
| Fun consistency／epigraph surplus | 1.88738e-15／3.33067e-15 | 同上 | PASS |
| Primal-dual gap | 2.55351295663786e-15 | 絶対値が同基準以内 | PASS |
| Objective arithmetic guard | 9.092380497324095e-11 | 同上 | PASS |
| Retained condition | 99.84975897081932 | 1e12以下 | PASS |

Epigraph objective=0.9544320281272057、dual objective pilot=0.9544320281272032、dual dot absolute scale=58.45145643263055。全256等式行・60 signed J変数・216 epigraph変数・432不等式を維持した。SVD effective rank46／near-null columns14は診断値であり、exact nullspaceとして制約を捨てていない。

[前回の旧STOP](research_c_fixed_u_tail_preserving_lp_execution_20261101.md)ではdual残差7.8707、gap−6.54246e-8、相殺guard.001053が未資格だった。**今回の固定設定ではこの3項目が登録基準内となった。** 旧返却点λ1.1204135568を新DFとして採用したり、旧STOPを成功へ書き換えたりしていない。内部presolveや微小係数処理の各寄与を個別に証明したものではなく、前回FAILの唯一の原因を確定したとは言わない。

Nativeには1e-12以下の係数のゼロ扱いや内部presolveが残り、全実数LPの厳密certificateは得ていない。既存の独立数値残差・guardを通るnumerical qualificationと数学的保証を区別する。

## 4. Tensor・物理sector・位相・PF/RTE保持

LP数値資格後にのみ、新しい[candidate](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/candidate.json)をexportし、以下の登録gateを実行した。

| Gate | 保存結果 | 判定 |
| --- | --- | --- |
| U／head J | 元U全byte・head0/1 Jが同一、polar/codec補正なし | PASS |
| 独立full tail tensor contraction | max entry4.86416e-15／Frobenius1.58592e-14、基準1e-10 | PASS |
| 新6 frame・sector36一般K対raw CAR | 6/6 PASS、最大entry残差2.77556e-16 | PASS |
| 新旧各6 frame・full256 CARのtail合計 | max entry3.28071e-14／Frobenius2.57462e-13、基準1e-10 | PASS |
| Full CARからsector36への新旧写像 | どちらも最大entry1.11022e-16、基準1e-10 | PASS |
| κ／scalar／head／frame順 | 登録gateを維持。抽出identityは旧.2557642181923188／新.25576421819232853で基準1e-10内 | PASS |
| 1 h-step partial PF対保存旧kernel | max entry2.64372e-15、基準1e-10 | PASS |
| 1 h-step corrected K0対保存旧kernel | max entry2.64258e-15、基準1e-10 | PASS |

原本：[tensor](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/tensor.json)、[head/phase](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/head_phase.json)、[tail CAR](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/tail_CAR.json)、[PF/RTE](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/PF_RTE.json)。

[DF保持](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/DF_preservation.json)は、保存guard込みDF bound7.216837988604903e-9 Haに、新旧sector Hamiltonian drift norm4.84263296609596e-14 Haとdrift guard1.171284763290944e-8 Haを加えた登録数値評価である。新numerical triangle bound **1.8929734047844004e-8 Ha < CA/8=.00019875 Ha**で予算PASS。元physical HamiltonianとのDF誤差normを再計算した値ではなく、guardはengineering pilotで、厳密enclosureではない。

[Ideal K0 IID attenuation](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/attenuation.json)は13 roundすべてで登録単調性gateを通った。m12は旧.8643440178248293から新.8643733902370624となった。ShotsやCA達成は評価していない。全frameを決定論的に並べる別S2の不変性、13-round PF/RTE誤差、ground/alias、native価格を新候補へ自動継承しない。

## 5. Call・資源・証拠保全

[Call journal](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/science_calls.jsonl)は50 attempt／50 COMPLETE、FAIL0。[STOP](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/STOP.json)のcall_countsは契約各上限と完全一致し、LP_solve=1、retry/resume=0、mandatory_stop=true。全50 unitsを使い切った後の追加数値評価・追加LPは行っていない。

| Call群 | 実績 |
| --- | ---: |
| Evidence／input／export／reference読込＋固定input gate | 5 |
| LP assemble／solve／diagnose | 3 |
| Candidate export／tensor gate | 2 |
| 新旧arm／law／係数gate | 6 |
| Sector／full CAR context | 2 |
| 新sector6 frame／一般K6 gate | 12 |
| 新旧full256 CAR各6 frame | 12 |
| Full tail／head-phase／DF保持 | 3 |
| Kernel／PF-RTE gate／attenuation13round | 3 |
| Qualification／comparison | 2 |
| **合計** | **50/50** |

[資源原本](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/resources.json)：wall2.062393148秒、CPU24.565441645秒、peak RSS774,780 KiB。[動的resource plan](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/resource_plan.json)はaffinity32 CPU・空きRAM約56 GBから12 workersを選択し、worker並列中だけBLAS1 threadで過剰並列を避けた。LPは逐次依存の1 solve、重いfull CAR12因子は既存の並列経路を使用した。CPU/RAM/worker/wallやsolver iteration/timeに人工上限を追加していない。共有環境・他ユーザーjob・共有venv/cacheは変更していない。

Fresh runのraw31件にA/T/b/j0、solver生x・slack・con・marginals/residual、全警告、新J/U、SOURCE、契約、各gate、新reference、journal、resources、STOPを保存した。[Postflight](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/execution_postflight.json)でSOURCE109／入力12／旧raw945／新raw31、旧SOURCE・契約・STOPの保全を照合する。Scientific SOURCE・数値allowanceの修正0。追加fit・rank/U探索・RC2 LP・native build/compile・shots/総費用は0。

## 6. 結論の限界と次担当

今回確認できたのは、固定U・head・tail合計tensorを保持する限定LPで、元の数値基準を満たしながらtail λを約0.0117%下げられたこと、登録したtensor/operator/phase/DF/PF-K0保持がPASSしたことである。元CDFの未収束資格は解消しておらず、RC2対照一般や別分子・別rankへ一般化しない。

Native価格は係数support・basis切替・合成に依存するため、このλ削減率を回路価格の削減率と同一視しない。全law synthesis/implemented-law距離・CA・shots・総資源も未資格のままで、旧candidateの価格・shotsを新candidateへ転記しない。

次担当はユーザー／研究C：この小さいλ削減と保存attenuationの変化を踏まえ、新candidateに対する追加PF/RTE・native価格検証の情報価値を判断する。Codexは別途明示許可された範囲だけを新たに固定して実施する。今回のLP枠は消費済みで、追加LP・候補B・新fit・13-round再評価へ自動進行しない。

結果・SOURCE・raw・STOPをcommit/pushし、remote SHA/tree・公開本文・SOURCE・成果物を照合後、**mandatory STOP**する。
