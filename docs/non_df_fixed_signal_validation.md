# 非DF Pauli PR/PF：固定時間信号の技術実装

2026-10-22 JST。実装・人工技術検証の報告であり、実分子検証・優位性評価・GPT詳細科学レビューではない。
分岐元は `b81d4096e806efb6523304764d3a0680b7663320`、専用branchは
`codex/non-df-pauli-fixed-signal-20261022`。原本のDFコード・科学SOURCE・結果は変更しない。

## 調査結果と実装の境界

| 機能 | 既存経路と今回の接続 |
|---|---|
| Hamiltonian構築 | `chemistry_hamiltonian.jw_hamiltonian_maker` は既存のPySCF→OpenFermion→JW経路だが、今回は呼ばない。供給済みJW和と実数MO積分からの変換を新adapterに実装 |
| Pauli分割 | `partial_randomized_pf` の非identity項の絶対係数順・prefix分割を踏襲。旧APIはH-chain metadataを要求するため、汎用モデル用adapterを追加 |
| Pauli PF | `pf_decomposition` と既存Pauli S2参照を確認。今回の実装は各Pauliの厳密指数を対称順に並べるS2 |
| Pauli PR | `rte.py` の汎用Hermitian involution・paired finite-RTEを再利用。既存 `test_rte_h2.py` にPauli接続はあるが、実分子生成を伴うため今回は除外 |
| 理想partial-S2 | `pauli_partial_cgs_validation` の中央exact-tail演算はdense参照であり、ランダム回路やshot会計の実装とは区別 |
| DF専用部分 | fragment分割、one-body、basis registry、identity抽出、native circuit builderは既存のまま再利用 |
| 共通部分 | 固定時間信号・bias・Hoeffding会計・完全Hadamard wrapperのcompiled RZ・paired MC費用・結果形式・provenanceを新interfaceで接続 |
| 旧Pauli解析費用 | 旧 `partial_randomized_pf` のgamma・連続最適化・簡略費用モデルは今回のC/N/Gには使用しない。旧qは誤差配分比であり、今回の整数q（時間分割数）と別物 |

新規libraryは `fixed_signal_accounting.py`、`pauli_signal_adapter.py`、
`fixed_signal_evaluation.py`、`fixed_signal_io.py`。runnerは
`scripts/run_fixed_signal_evaluation.py`、人工テストは
`tests/test_fixed_signal_evaluation.py`。回帰選別・分子処理ガードはopt-inの
`tests/no_molecular_plugin.py`。証拠は
[`artifacts/non_df_fixed_signal_validation/2026-10-22/`](../artifacts/non_df_fixed_signal_validation/2026-10-22/)。

## Hamiltonianと状態の規約

Pauli入力は `H=cI+sum_j h_j P_j`。Qiskitラベルの右端がqubit 0、状態indexは
`sum_q occupation[q] 2**q`。OpenFermionのmode indexはqubit indexへ直接写し、
行列のendiannessは人工Y項を含む2量子ビット参照で照合する。
同一ラベルを統合し、係数は有限実数（Hermiticityの虚部許容差1e-12）。
係数cutoffは0で、非zero項を勝手に捨てない。identityを分割対象から外し、常に一度だけ保持する。
非identity項は厳密な `(-abs(coefficient), label)` 順。旧Pauli経路の同率項の挿入順とは異なり、
新APIでは入力順に依存しないcanonical hashと分割を採用し、その規則をmetadataに記録する。

`PauliHamiltonian.from_spatial_integrals(one_mo, eri_chemist, nuclear_repulsion=...)` は
**供給済みの実数・直交空間MO**を受け取る。AO→MO変換やSCFは行わない。
入力ERIは `(pq|rs)`、OpenFermionの `h[p,q,r,s]=(ps|qr)` へ
`transpose(0,2,3,1)`、spin orbitalは `2*p+alpha(0)/beta(1)`、二体係数は1/2。
この規約はインストール済みopenfermionpyscfの `compute_integrals` と照合した。
spin liftingは明示的に行い、`spinorb_from_spatial` の暗黙cutoffを持ち込まない。
核間反発はscalarに含め、JW後のidentity係数と分けて生成metadataに保存する。
積分・Hamiltonian・状態のhashを保存する。geometry、basis、orbital生成法は呼出側が供給する。

DF入力は既存 `DFHamiltonian`。新adapterは既存公開APIの
`weight_ranked_prefix` を使用し、実際のfragment indexとblock順を記録する。
固定科学契約でgeneration順等を採用した保存実験を、このadapterへ暗黙に読み替えない。
Pauli方式のDF rankはnull。DF rank・L_Dの異なる単位を同じHamiltonian／分割とみなさない。
参照は常に各入力自身のfull-H、状態は供給済みvectorで、ground stateを新たに求めない。

## PF・PR・B0/B0I

PFは全項を保持するS2。`q`回、時間幅 `delta=T/q` を反復する。
PRでは先頭L_D項を決定論側に置き、前half sweep→中央finite-RTE→逆half sweepを反復する。
DFではone-bodyも決定論側に残す。高次PF・qDRIFTを指定すると明示的な未対応例外を返す。

Pauli Tailは `lambda_R=sum_j |h_j|`、`p_j=|h_j|/lambda_R`、
符号を吸収したHermitian involution `sign(h_j)P_j` とする。
`rte.finite_rte_distribution(lambda_R*delta/r,K)` の偶数次数0,2,...,Kを使用。
order nのeventは既存定義どおり
`(-1)^(n/2) exp(-i atan(tau/(n+1))*P_rotation) P_n ... P_1`。
application順、負係数、product符号、eventのglobal phaseを保持し、control下でも保持する。
正規化はこのPauli法則自身から計算し、DF値を流用しない。

1 eventの平均はdegree K+1までのpaired Taylor多項式を有限正規化sで割ったもの。
全q*r eventの `B=s**(q*r)`、raw測定信号は補正前の平均、corrected信号はB倍。
評価にはサンプリング平均でなく、独立な疎行列の多項式状態作用を使用する。
量子shotの推定器は `B * mean(independent ±1 outcomes)`、**shotごとのfresh IID trajectory**を仮定する。
古典cost MC標本を量子shot数と混同しない。空Tailは決定論的経路でB=1。

B0はTailを省略。Pauli方式ではTailが非identity項だけなのでB0IはB0と同一で、補正phaseは0。
同値性と理由をmetadataに明記する。DF方式のB0Iは既存のTail identity抽出係数を一度加算する。
科学的にB0とB0Iが異なる場合と同一の場合を一律に扱わない。

## 固定時間の精度・shot・費用

参照信号は `z(T)=<psi|exp(-iHT)|psi>`、biasはcorrected法則平均と参照の差。
両軸の既定精度は `epsilon_sig/sqrt(2)`、既定失敗予算は各0.025（union 0.05）。
APIでは非対称配分も受け取り、`hypot(epsilon_cos,epsilon_sin)<=epsilon_sig` を検査する。
各軸について

```text
h_a = epsilon_a - abs(bias_a) - u_a
N_a = ceil(2 * B_upper**2 * log(2/alpha_a) / h_a**2)
G_RZ = N_cos*C_RZ,cos + N_sin*C_RZ,sin
```

を用いる。これは現行DF比較の `ax2b_numerical_accounting.u_aware_shots` と同じ目的・算術順・適格性。
保存された旧関数との人工差分テストで、ELIGIBLE/INELIGIBLE/UNDETERMINEDと整数Nを照合する。
`u_a` はcandidateとreferenceを合わせた数値allowance。normalizationのlog上側marginも呼出側の明示入力。
本実装はそれらをcertifyしない。未知のallowance、normalization上側値、または安全な整数範囲外では
N/Gをnullにし、成功値を代入しない。大きなNは既存同様log-domain診断まで。
添付例の1e-10/1e-12は人工テストの**宣言した経験的許容差**であり、一般のrigorous certificateではない。

回路は状態準備を除いた進化・ordinary control `diag(I,U)`・scalar/event位相・
cosine(H)／sine(Sdg-H)・ancilla測定を含めて一体compileする。
DFは既存controlled native builderを直接使用。Pauliは厳密単Pauli指数からcontrolを生成する。
既定compilerはQiskit 1.3.0、`rz/sx/x/cx`、opt1、seed17、topologyなし（versionは実環境から記録）。
既存 `TranspiledCircuitCostCache` を再利用し、実際の回路・compiler・backendで重複compileを抑える。
PRのCは有限MC平均。両軸を同一trajectoryでpairにし、GのSEには共分散を含める。
名目Student-t 95%区間は希少eventや母平均への厳密保証ではない。NとC/Gの不確かさは別物。

## API・CLI

```python
from trotterlib.pauli_signal_adapter import PauliHamiltonian
from trotterlib.fixed_signal_evaluation import FixedSignalConfig, evaluate_fixed_signal

h = PauliHamiltonian.from_terms([("I", .1), ("Z", .7), ("X", -.2)],
                                 model_id="artificial_1q")
config = FixedSignalConfig("pauli_jw", method="PR", T=.2, q=2, L_D=1,
                           r=2, K=2, seed=17, cost_samples=4)
result = evaluate_fixed_signal(h, [1., 0.], config, epsilon_sig=.1,
    numerical_allowance={"cosine": 1e-10, "sine": 1e-10},
    log_B_margin=1e-12, evidence_kind="EMPIRICAL",
    evidence_ref="artificial declared tolerance, not certified", verify_compiled=True)
# DFHamiltonianとFixedSignalConfig("df_native",...)も同じinterface。
```

```bash
PYTHONPATH=src /home/AbeHiromu/venvs/trotter-common/bin/python \
  scripts/run_fixed_signal_evaluation.py \
  --input artifacts/non_df_fixed_signal_validation/2026-10-22/inputs/pauli_1q.json \
  --output /tmp/non_df_pauli_PR_example.json \
  --hamiltonian-type pauli_jw --method PR --T .2 --q 2 --L-D 1 --r 2 --K 2 \
  --cost-samples 4 --epsilon-sig .1 \
  --numerical-allowance 1e-10 --log-B-margin 1e-12 \
  --evidence-ref 'artificial declared tolerance, not certified' --verify-compiled
```

CLIの入力は `input_kind=artificial` と明示したJSONだけ。既存出力を上書きしない。
直接Pauli和、人工MO積分、供給済み人工DF配列を選択できる。
`--numerical-allowance` 等を省略した場合もCとsignalは評価でき、N/Gは未定義のまま保存する。
結果schemaは `fixed_signal_resource_evaluation_v1`。
`validate_fixed_signal_result` でreference hash、Pauliのrank null、axis/sample形式、N/G欠測、
有理数で保存したN×C=Gを検査する。compiler・seed・MC標本・source/input/state/environmentのprovenanceを含む。

## 技術検証と保全

人工テストでは1/2量子ビット非可換系、Y項・qubit順、identityとcontrolled phase、
S2時間反転と局所三次誤差、Pauli event全列挙、人工DFのexact法則平均と回路平均、
seed、raw/corrected平均、wrapperとcompiled signal、C/N/G、欠測と異常入力、CLIを検査する。
24量子ビットの人工Pauli circuit生成が行列を作らないことも検査する。
signal評価は疎行列・状態vectorを使用し、dense全演算子を作らない。
指数的な状態空間そのものを解消したわけではなく、大規模state-actionの性能検証は未実施。

実行コマンド・件数・除外ID・初期技術失敗・最終結果はartifactの
`validation_summary.json` とJUnitに記録する。予定回帰集合は既存17moduleに限定し、全repository suiteではない。
実分子処理5件は除外。初期点検で1件の間接SCF呼出しをガードがSCF前に拒否し、除外へ追加した。
固定runtime `Python 3.11.0rc1` を要求する回帰1件は、現環境 `3.12.3` と不一致のためskip。
このテストの初回試行では16ケースの数値比較を通過後、runtime assertが失敗した。
元のfixture・基準値・環境条件を緩和したPASSとは扱わない。
初期人工DF列挙比較の失敗はQiskit Statevectorのglobal-phase適用による入力vector共有を切り分け、
テスト側でvectorをcopyして修正した。実分子の失敗runやSTOPを再実行したものではない。

source固定前の人工テストはdirty local evidenceであり、immutable CI／外部再現ではない。
専用source commitとbyte hash、人工CLI出力、実行環境、保全検査を新namespaceへ保存する。
旧科学結果、SOURCE、grant、ledger、marker、STOP、図・スライド、固定科学契約は変更しない。
共有venv・共有設定・他jobは変更しない。小規模人工回帰はprocess内のmonkeypatch/cache検査を含むため
1 pytest processで実行し、独立したCLI例は利用可能資源の範囲で並列実行する。
CPU/RAM/workerへの一律の低い資源上限は新設しない。

## 未対応・将来の実分子検証条件

- 実分子PySCF/SCF・DF生成、新規の実分子状態／signal／compile／samplingは今回0件。
- 実分子の資源比較、候補探索、新fit、全RPE最終総費用、PR優位性は未検証。
- S4以降、qDRIFT、新しいrandomization law、新しいshot科学モデルは未実装。
- 複素MO積分、非直交AO積分の変換、active-space/frozen-core規約の新規採択は未対応。
- 独立したcompiled法則平均からのN再評価、厳密数値certificate、希少event母平均保証は未実装。
- `verify_compiled` はwire layout不変の人工検査に限定。routing/layout付き検証は未対応を明示する。
- RPE各roundの位相誤差予算・複数round配分・RPE全体costは今回接続しない。

実分子へ進むには別の明示実行承認に加え、分子・geometry・basis・軌道／積分・spin/JW順・
状態／参照Hamiltonianのhash、係数cutoff、PauliまたはDF分割の順序とL_D、T/q/r/K、
推定器・数値allowance/normalizationの根拠、epsilon/alpha配分、compilerと回路scope、
MCのseed／標本・精度条件、SOURCE/input/environment、出力領域と既存STOPの保全条件を固定する必要がある。
DFと非DFのHamiltonian一致は仮定しない。今回の完成後はmandatory STOP、push・公開・詳細科学レビューへ進まない。
