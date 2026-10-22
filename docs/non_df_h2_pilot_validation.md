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

## 実行結果：部分取得・FAILED_NO_RETRY（2026-10-22 JST）

**4固定caseのうち2件が完了、2件がQiskit runtime例外で失敗した。pilot全体は未完了。**
科学SOURCEは `b5ed2c6e2399b55a0c24eb8437f842892a6703ef`、runは `run01`、全体STOPは
`FAILED_NO_RETRY_MANDATORY_STOP`、実行process exit1、wall246.816秒。
新RHF1回・同一MO積分からDF1回・FCI照合1回は完了し、その後4caseを同時起動した。
失敗をretryせず、同時起動済みcaseを回収して終了した。SOURCE・契約・許容差は変更していない。

|固定case|分割条件|最終状態|C/N/Gの受理|
|---|---|---|---|
|Pauli PF|非identity全14項、L_D=14、DF rank非適用|完了・実施したgate PASS|受理|
|Pauli PR|予定L_D=7 Pauli項、r2/K4|`Already mutably borrowed`|未取得null|
|DF PF|rank3・全3fragment保持、L_D=3|operator validation中に `Already borrowed`|保存途中値は未受理|
|DF PR|rank3・既存weight順位fragment0を1個＋one-body、L_D=1、r2/K4|完了・実施したgate PASS|有限MCの条件付き値として受理|

DF PFにはnumerical evidenceとprecision_rowsが保存されているが、validation/resultの完成前に失敗した。
これらの値を成功結果や優位性比較へ用いない。Pauli PRのsignal・C/N/Gを仮の値で埋めない。
同じL_Dが同じ分割を示すとは扱っていない。

### 入力と既存DF経路の整合性

H2・1.00 Å・STO-3G、4 spin qubit、同一canonical real RHF MO積分、Pauli14非identity項、
DFはtol-only1e-8/rank3（今回truncation値0）という新snapshotで照合した。
数値はこの条件・新cohortに限定し、旧generation順最良候補cohortの再現結果とはしない。

- 非DF sector ground energy：−1.1011503302326184 Hartree。独立PySCF FCIとの差2.22e-16 Hartree。
- DF/nonDF full-H operator差（2-norm）：8.88e-16（事前許容1e-9）。
- 既存 `df_linear_operator` のsector行列との差：4.62e-16（許容1e-9）。
- state residual：8.92e-16（許容1e-10）。DF ground energyとの差は今回の数値で0、state overlap差2.22e-16。
- T=.8のfull-H参照信号差は今回0（許容1e-10）。参照は約0.6364415911143844+0.771324899831317i。
- 全4 native DF blockの±half-step、計8 primitive operator gateをPASS。

[入力receipt](../artifacts/non_df_h2_pilot/2026-10-22/run01/inputs/receipt.json)と
[入力snapshot](../artifacts/non_df_h2_pilot/2026-10-22/run01/inputs/snapshot.npz)を原値とする。

### 受理した固定時間C/N/G

以下は上記H2/geometry/basis/stateで、T=.8、q4、delta=.2の**単一点**、S2、
各軸epsilon_sig/sqrt(2)、alpha各.025、P=0、full ordinary-control HadamardのRZ。
Pauli PFはL_D14、DF PRはrank3/L_D1/r2/K4・新64 trajectory。
Nは経験的allowance下のHoeffding十分値、PRのC/Gは有限MC平均であり、最終総RPE費用ではない。

|方式|C_cos [RZ/shot]|C_sin [RZ/shot]|epsilon_sig=.05のN_cos+N_sin|同G [RZ]|
|---|---:|---:|---:|---:|
|Pauli PF|3850|3850|7044+7038=14082|54,215,700|
|DF PR|2030.921875|2030.921875|7152+7146=14298|29,038,120.96875|

DF PRのG名目95%半幅は29,529.2401 RZ（SE14,776.8809）、各軸CのSEは1.03349 RZ。
この2caseの費用値から非DF PR優位、DF優位、方式一般のランキングを結論しない。

|epsilon_sig|Pauli PF N_total|Pauli PF G|DF PR N_total|DF PR G（有限MC平均）|
|---|---:|---:|---:|---:|
|0.05|14082|54215700|14298|929219871/32|
|0.01|357943|1378080550|363455|47241517445/64|
|0.005|1462238|5629616300|1484776|24123712463/8|
|0.001|43644174|168030069900|44322488|720124083469/8|
|0.0005|223521528|860557882800|227040767|29510531853893/64|

小数でないGは正確な有理数表記。各精度で回路を選び直した結果ではない。
各軸N/C・bias・normalization・全64価格は原JSONを参照する。
Pauli PFの複素signal絶対誤差は1.03943e-4、DF PRは1.04092e-4。
DF PRのBは1.0076627782285141、attenuationは0.9923954934189536。
Pauli PFのuは各軸9.37145e-8、DF PRは3.78351e-8で、いずれも経験的でcertifiedではない。

受理caseでは計130軸wrapperを検査し、compiled probability最大差は
Pauli PF4.33e-14、DF PR2.28e-14（許容1e-9）。
独立trajectory operator比較、DF ordinary control、DF PRのTail再構成・order0/2/4強制event検査をPASS。
DF PRの実MCは512 event位置が全てorder0、order2期待数0.24418、order4期待数1.94262e-5。
order2/4のcost母平均寄与は未同定で、0と見なさない。名目区間を厳密母平均保証にしない。

### 失敗の技術記録と未解決事項

Pauli PRのtracebackはQiskit high-level synthesisのequivalence library keys、
DF PFは追加operator検査時のcontrolled RZZ構築→basis translationでborrow例外を示す。
同一processのThreadPoolExecutorでQiskit処理を並行したことと、
インストール済みQiskit1.3.0がprocess-global `SessionEquivalenceLibrary` を持つことを静的確認した。
共有mutable stateの競合が疑われる。原因の独立再現・修正後の科学再実行は行っていない。
作者による技術確認で、独立科学レビューではない。

人工preflightは87件PASS、最終のpilot固有16件もPASSしたが、人工caseは逐次実行だった。
実際の4case同時実行に対する十分なpreflightになっていなかった。
これは実行方式の問題であり、失敗2caseの科学結果を得たことにはしない。

未完了なのは非DF PRの実分子C/N/GとDF PFの最終operator検証・結果受理。
将来の再実行を検討する場合は、Qiskitをprocessごとに分離した実装と人工並列preflightが候補になる。
ただし現契約を変更せず、今回のgrant/marker/STOPは消費済みのまま保全する。
新SOURCE/新規出力・新しい明示承認なしで科学retryを行わない。

[機械可読の部分取得・未取得台帳](../artifacts/non_df_h2_pilot/2026-10-22/result_inventory.json)、
[全体STOP](../artifacts/non_df_h2_pilot/2026-10-22/run01/STOP.json)、
[保全検査](../artifacts/non_df_h2_pilot/2026-10-22/preservation.json)を保存した。

旧tracked 6870ファイルと旧manifest152エントリの内容同一を確認。
凍結科学SOURCEのbyte hashは不変。旧raw・grant・ledger・marker・STOP・図を保全した。
結果と失敗記録を専用branchへlocal commitし、pushなしでmandatory STOP。
