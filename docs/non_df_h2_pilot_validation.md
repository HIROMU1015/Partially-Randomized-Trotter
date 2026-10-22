# H2非DF固定時間signal pilot

2026-10-22 JST。利用者の一括実行承認に基づく新規H2 pilot。
base commit `87e5f4a5ab0182f11c4b3af58ebf5e1c036c80d1`、専用branch
`codex/non-df-h2-pilot-20261022`。詳細科学レビュー、新しい科学モデルの採択、H4以降への拡大は行わない。

## 固定契約とSOURCE

[`execution_contract.json`](../artifacts/non_df_h2_pilot/2026-10-22/execution_contract.json)と
[`authorization.md`](../artifacts/non_df_h2_pilot/2026-10-22/authorization.md)を先に固定する。
`src/trotterlib/h2_non_df_pilot.py` は契約とH2入力・独立dense/sparse参照、
`h2_non_df_pilot_validation.py` はoperator/event/trajectory検証、
`scripts/run_h2_non_df_pilot.py` はSOURCE・grantの結合とone-shot起動を担当する。
`tests/test_h2_non_df_pilot.py` は人工preflight。前回のDF/Pauli共通interfaceは変更しない。

SOURCE commitを固定した後に `--bind` を実行し、全library・runner・test・契約・承認のhash、
環境version、利用可能資源をSOURCEへ保存する。`--execute` はSOURCE/契約/grantを照合し、
`run01` の排他的mkdirでone-shotを消費する。以後の失敗は `FAILED_NO_RETRY_MANDATORY_STOP`。
契約・数値許容差・候補数・標本数を結果に応じて変更しない。入力生成条件を実行前に固定し、
新規生成した積分・DF・状態snapshotを保存/hash化してから下流caseへ渡す。

## 物理条件・表現・分割

H2、(0,0,0)/(0,0,1.00) Å、STO-3G、charge0、spin0、canonical real RHF、
全2 spatial/4 spin orbital、active space/frozen coreなし。
新規RHF1回、収束1e-12/max_cycle100、同じMO積分からPauli-JWとDFを生成する。
ERIはchemist `(pq|rs)`、spin orbitalはalpha/beta交互、qubit/stateはQiskit little endian。
核間反発とJW identityは省略しない。Pauli係数cutoff0。
DFは既存 `df_hamiltonian_from_integrals` のtol-only1e-8、final_rankなし（H2期待rank3）。

状態は非DF full-HのNalpha=Nbeta=1 sectorの最低固有vector（最大成分を正実に固定）。
同じ状態で各表現自身のfull-H参照信号を評価する。別にDF sector基底状態とPySCF FCIを求め、
エネルギー・残差・重なりを照合する。状態準備費用は評価対象外。
DF全HamiltonianとJW、既存DF sector operator、参照信号の差を検査し、
DF truncationがある表現を無条件に同一とはみなさない。

Pauli PRは非identity項を `(-abs(h),label)` 順に並べた前半
`floor(num_terms/2)` を保持。DF PRは既存の重み順prefix1 fragmentとone-bodyを保持。
同じL_Dが同じ演算子分割を意味するとは扱わず、実際の項ラベル・DF index・block順を保存する。
これは構造で事前指定したpilotで、旧DFのgeneration順・最良候補cohortの再現ではない。

## 固定時間評価

T=.8、q=4、delta=.2の単一点、外側S2。各表現でPF全項保持とPRの2方式、計4固定case。
PRは既存のpaired finite-RTE、r=2、K=4。normalizationは各Tail自身の法則から計算する。
各表現PRのcost64 trajectory、両軸paired、seedは固定SHA規則。標本追加・candidate探索はしない。
epsilon_sig=.05/.01/.005/.001/.0005を同じsignalとCから会計し、精度ごとの回路再選択を行わない。
cos/sin各epsilon/sqrt(2)、alpha各.025、補正付き独立±1のHoeffding十分N。
量子shotごとfresh IID trajectoryを仮定。Nは実機で測った最小shot数ではない。

数値allowanceは既存 `empirical_allowance` の式（roundoff倍率64、安全倍率10、floor1e-12）を踏襲。
新しいdense/sparse計算経路のoperation proxyを契約で定め、参照差・Horner/forward差・
raw/corrected closureを反映する。数値certificateには格上げしない。
既存同様u/headroom>.01は未解決としてN/Gをnullにする。bias不適格もnullで、設定を変更しない。

費用Cは状態準備なしのfull ordinary-controlled Hadamard、scalar/event位相、測定を含むcompiled RZ。
`rz/sx/x/cx`、opt1、seed17、topologyなし。Python3.12.3/Qiskit1.3.0/Nature0.7.2、
NumPy1.26.4/SciPy1.14.1/OpenFermion1.6.1/PySCF2.7.0を固定する。
PRのC/Gは有限標本平均、GのSEはpaired covariance込み、名目Student-t 95%。
希少eventの未観測を費用0とせず、未観測orderと法則確率を記録する。厳密な母平均保証ではない。

旧DF C/N/G図にはsymmetric-directional/nativeの別構成と別cohortがあるため、
今回のordinary-control・weight-ranked固定caseの費用をそのまま旧値へ同一視しない。
整合性は今回の同一積分、既存library operator、native primitive、参照信号で検査する。
DFと非DFのPF/PR近似信号・C・N・Gが同じ値になることは要求しない。

## 事前固定の技術gate

Hermiticity/MO直交1e-10、Hamiltonian演算子差1e-9、DF/FCI energy差1e-9、
state residual1e-10、state norm1e-12、reference signal差1e-10、Horner/forward法則平均差1e-10、
raw/normalization closure1e-10、trajectory/primitive operator差1e-9、compiled probability差1e-9。
operator比較ではglobal phase alignmentを行わない。

DFの各native blockを元の二次量子化演算子指数と±half-stepで照合する。
Pauli/DFとも、抽選されにくいTaylor order0/2/4を強制したeventの代数回路検査を行う
（cost MCへは混ぜない）。各caseの先頭最大3 trajectoryを独立な演算子積で照合し、
全取得標本のcos/sin compiled wrapper信号を確認する。
これは有限の実施済みscopeで、全抽選列の証明や独立compiled法則平均に基づくN比較ではない。

## 実行・保存・停止

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/venvs/trotter-common/bin/python \
  scripts/run_h2_non_df_pilot.py --bind
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/venvs/trotter-common/bin/python \
  scripts/run_h2_non_df_pilot.py --execute
```

上記は消費後に再実行しない。入力log/checkpoint/snapshot、source/input hash、各caseのsignal・
numerical evidence・C/N/G・全MC価格・validation、grant/marker/STOPを新namespaceへ保存する。
入力依存工程の後、4独立caseを並行実行。共有venv・共有設定・他jobは変更せず、
資源利用は実際の空き資源を確認して選ぶ。固定workloadを超える科学処理は行わない。

結果保存後に既存tracked科学ファイルと過去manifestエントリを照合し、
研究現状・本検証文書・日付ノート・新manifestエントリを更新する。
ローカルcommit後mandatory STOP。push、詳細科学レビュー、新規モデル採択、H4以降、retryは承認外。
