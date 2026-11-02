# Track A AX-2A：実装・検証・AX-2B pilot草案 v1

公開時のリンク補修（2026-10-10）：未公開sourceへの参照は[公開依存関係](track_a_ax2b_gpt_review_index_v1.md)。原文と補修一覧を保存し、科学的主張・数値は変更していない。

2026-10-09 JST。準備のみ。科学pilot未実行・未認可。
研究方針と比較条件は[追補案](track_a_ax2a_research_amendment_v1.md)を参照する。

## 1. 保護と作業場所

AX-1b結果commit `b2e1bf65e21893b6c617223b42313623d3186f12`から
branch `track-a-ax2a-preparation-20261009`の隔離worktreeを作った。
作業場所は `.worktrees/track-a-ax2a-preparation-20261009/`。
元PR2 worktreeの未commit変更は移植・編集しない。
既存source、科学出力、AX-0/1a、AX-1bモデル、原稿、Track B、旧validation manifestは不変。
本準備は未commitのlocal証拠。元レビューはbyte一致の写しを別文書として収録する。

## 2. 再利用できる実装と不足

| 機能 | 既存source / 保存資産 | 再利用範囲・不足 |
|---|---|---|
| M1/M2信号 | `pr2_matched_accuracy_m1_execution.py` | H4 dense block・spectral cacheとforward/tail/reverse、phaseの意味論をoracleにできる。H6/H8のdense拡張は採用しない |
| sector/matrix-free | `df_hamiltonian.py` | number/spin sector、CPU Python/Numba/chunk operator、ground solverが既存。primitive保存性とbasis対応を別途検証 |
| PF state-action | `pf_c_system_size_validation.py` | 回路half作用・sector lift/reference経路がある。finite-RTE平均のTrack A接続は今回追加 |
| GPU statevector | `df_gpu_statevector.py` | parameterized template/cache、Aer GPU、phase補正が既存。今回はimport/実行しない。finite mean接続のGPU検証も未実施 |
| finite RTE | `rte.py` | exact finite distribution、paired Taylor、normalization、samplerが既存。distributionだけ再利用し、samplerは呼ばない |
| 高次PF | `product_formula.py` / `pf_decomposition.py` | 既存Yoshida四次係数とglobal stage iteratorを再利用。P-Dのexact二block/inner高次試験をfull-wrapper結果へ読み替えない |
| DF回路 | `df_partial_s2.py` / `df_partial_s2_repeated.py` | Gaussian basis、diagonal control、repeated S2を再利用候補にする。新global四次・directional native loweringは未接続 |
| wrapper cost | `df_rpe_hadamard_compiled_cost.py` | cosine/sine measured wrapper、compiler context、trajectory別cost統計が既存。今回compile/samplingしない |
| analytic/proxy | `df_partial_randomized_pf.py` / `rpe_hadamard_compiled_cost_proxy.py` | deterministic RZと局所校正proxyが既存。旧domainを超えた適用と新controlへの変更は未検証 |
| AX-1b FEW | `artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/model_fits.json` | full210モデルをbyte hashで凍結。fitも保存値の解析再実行も行わない |

source関数の所在・hashは[再利用inventory](../../artifacts/resource_applicability/track_a_ax2a_preparation/2026-10-09/reuse_inventory_v1.json)に記録する。
過去のP-D S2 STOP、R3 no-method-delta STOP、H12未決定を変更しない。
高次PFの存在とTrack Aの強いbaselineの実検証完了は異なる。

## 3. 今回追加したmodule

- [ax2a_state_action.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：
  callback経由のfinite paired Taylor平均、partial S2 complex signal、global二次/四次PF、
  primitive projection検査、既存DF tail operator接続、参照位相と残差allowance。
- [ax2a_control_plan.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：
  対称PF ordinary controlを表すdirectional意味論IR。scalar/event phaseを別にcontrolする契約。
- [ax2a_preparation.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：
  stdlibのみのsource hash、固定モデル参照、prefix候補、三状態headroom、実行認可を持たないpilot草案。
- [prepare_track_a_ax2a.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：
  metadata writer。`--review`と新規`--output`だけを受け、exclusive createする。science起動optionはない。

library callbackは同じbasisのHermitian term作用をcallerが保証する。
DF tailはone-bodyをゼロにし、molecular constantを除き、抽出identityを一度だけ引く。
`DFHamiltonian.select_blocks`はone-body/constantを保持するので、tail生成としてそのまま使わない。
matrix-free tailはOpenFermion sector順。Qiskit順との変換は既存H4 snapshotで確認する必要がある。
`primitive_sector_certified=True`はcallerの明示宣言であり、flagだけで数学的証明にならない。
Hamiltonianのsector保存だけから中間primitive保存を推定しない。

finite平均は `P_(K+1)(τ)`。Hornerにより1短stepでK+1 matvec、O(d) vector storage。
K=0/2/4/6、正負時間、q outer step、r tail step、`B=b^(qr)`を扱う。
corrected/rawを別経路で評価し、中間・最終normを勝手に1へ戻さない。
log Bを保存し、binary64のnormalization overflow/raw underflowではNoneと明示statusを返す。
codeのmatvec budgetは両経路の合計。通常最大 `2qr(K+1)`。
deterministic action budgetも両経路を合算し、wall/RSS上限とは別に数える。

scalarとextracted identityはphaseとして残る。controlled signalではglobal phaseも相対位相となる。
有限平均の非unitary numeratorをsampled unitary trajectoryと取り違えない。
単一termの四次はexactに処理し、legacy merged multi-term iteratorの退化caseは使わない。
古いiteratorは変更していない。

`eigenphase_reference`はcallerがHermitianと保証したHのRayleigh Eと残差ρから
`exp(-iET)`とsignal allowance `|T|ρ`を返す。
ground-state証明、eigensolver、roundoff保証は提供しない。
H6の参照signalをこのsurrogateで代用できるかはAX-2Bで決める。

## 4. synthetic検証

[専用tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)：57 passed、local。
固定した非可換2×2 operatorと2 orbital synthetic DFだけを使い、分子snapshot/科学artifactのopenをguardした。

確認内容は、既存dense finite Taylor oracleとのcorrected/raw一致、K+1次数、非unitary norm、
q/r・正負時間・空tail・scalar、global四次と負時間composition、control両branch全basisの作用一致、
central event phase、各primitiveのleakage拒否、DF tailのone-body/constant/identity除去、
残差allowance、B0のsigned discard/PF分解、三状態headroom、操作上限と無効入力、
normalization underflowの明示、prefixとpilotのmodel非依存性。

Python 3.11.1、NumPy 1.26.4、SciPy 1.14.1、pytest 9.0.3を既存環境から使い、
BLAS/OpenMP/MKL各1 thread、pytest plugin autoload/cacheを無効化した。
JUnitとsource hashを[準備artifact](../../artifacts/resource_applicability/track_a_ax2a_preparation/2026-10-09/)に保存した。
分子科学計算、実trajectory、科学回路compile、GPU、immutable CIは0件。

## 5. 古典メモリと量子資源

STO-3G水素鎖・全spin orbital・complex128の単一bufferの理論サイズ：

| 系 | qubits / full d | full vector | full dense matrix | number sector d | Nα=Nβ sector d |
|---|---|---|---|---|---|
| H4 | 8 / 256 | 4 KiB | 1 MiB | 70 | 36 |
| H6 | 12 / 4096 | 64 KiB | 256 MiB | 924 | 400 |
| H8 | 16 / 65536 | 1 MiB | 64 GiB | 12,870 | 4,900 |

これは分子生成・compiler・複数operator/eigenvector cache・Krylov workspace・Python overheadを含まない。
H6でもrank×dense blockと固有分解cacheにより8 GiBを超え得る。H8 full denseは計画から排除する。
matrix-freeはsector vectorが小さくても、1 matvecのDF rank・primitive transition/cacheの費用がある。
CPU Python/Numba backendの比較はprofile後に判断する。GPU存在だけから必要性・速度を推定しない。
量子qubit、RZ/CX/depth、trajectory費用標本、量子shotと、評価側CPU/wall/RSS/diskを別欄に保存する。

## 6. AX-2B task草案と実行上限案

[pilot草案JSON](../../artifacts/resource_applicability/track_a_ax2a_preparation/2026-10-09/pilot_plan_draft_v1.json)は
`DRAFT_NOT_AUTHORIZATION`、`assigned_resources=null`。結果に依存しない構造的cellを選ぶ。

H4は3 bundle：legacy dense/action・basis/phase/primitive、finite K2/4/6・raw/corrected・B0分解、
global四次とordinary/directional controlledのnative一致。
まずH4 correctnessが通ることを必要条件とする。詳細snapshot・prefix/q・seedはlaunch前に列挙する。
H6はB1二次/四次×q=1/8の4 cell、B2 fraction1/2・q=R=1・K2、
B3 fraction0・q8/R64・K6の計6 cell。H8は0 cell。
B2/B3のrはR/qで、各trajectory費用標本は2本を提案する。
これは負荷と実装のpilotであり、2標本で費用優劣の精密判断はしない。

| 上限 | 提案値 | 確定に必要なこと |
|---|---|---|
| CPU/process/BLAS | 1 core / 1 process / 1 thread、GPUなし | 実hostとaffinity、使用可能CPU |
| memory | address-space 8 GiB | RSS測定とRLIMITASの区別、実host割当 |
| wall | H4 bundle 900s、H6 cell 1800s、全体14,400s | 独立watchdogと子process停止方法 |
| output | 全体512 MiB | partial/terminalも含む上限とexclusive output |
| full wrapper | 全体64 compile call | axis×trajectory×control variantの全taskを数える |
| random cost sampling | random cellごと2 trajectory | seed identityとdraw数を列挙 |
| tail matvec | 一経路448、corrected+raw計896 | reference/solverのmatvecは別に上限を固定 |
| circuit gate count | 未固定 | rank・diagonal term・stageの静的上界を先に算出 |
| reference matvec | 未固定 | solver iteration capとexpm action capを先に固定 |

旧AX-1bの300s/8 GiBを許可として転用しない。上記は今回の新しい提案。
per-circuit gate/reference cap、state・DF政策、callback/native lowering、H4 task展開、実割当が空欄なので
**まだ実行できるsealed manifestではない**。科学runnerやlaunch authorizationは作っていない。
実装準備→H4 correctness→H6 bounded profile→GPT研究reviewの順で進む。

失敗時はfail reason、予定/実行済みtask、wall/RSS/counter、partial出力hashを保存してSTOP。
best-effort値を科学的成功へ昇格しない。自動retry、resume、rank・sector・quota変更はしない。

## 7. 次に必要な技術作業

native DF global四次とdirectional diagonal/scalar lowering、同一wrapperの接続を追加し、
synthetic circuitでbranch/phaseを確認する。その後、旧H4 snapshotのbasis変換、primitive証明、
reference accuracy、DF tol、seed、実資源、gate/matvec上限を確定する。
新しい科学実行は別の対象・予算・STOPを明示した利用者認可後だけ開始する。
pilot結果はH6 developmentであり、H8の独立性は保持する。
