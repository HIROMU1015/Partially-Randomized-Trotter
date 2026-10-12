# N3 OCC v1：固定8入力の一回比較結果

2026-10-12 JST。専用branch `n3-occupation-transfer-v1-execution-20261012`。
判定は **NO_ADDITIONAL_ACCEPTANCE_OBSERVED**。共通強下界の前提は8/8で成立し、既存対照B*、NORM、raw OCCはいずれも8/8を保証した。
OCC固有の追加保証は0入力/0base、事前2-base gate不通過。未取得の有限探索を保証族の不可行性と解釈しない。
結果後Mandatory STOP、追加探索・再試行なし。中心研究テーマの採択は行わない。

## 1. 認可・固定source・入力

利用者の「次の作業に進んで」を[一回実行認可](n3_occupation_transfer_v1_execution_authorization_20261012.md)として記録し、実行前commit `a4311eece09275e4a27259db3fe12e98974dbadc` のclean worktreeから一回だけ起動した。
比較条件は[追補v2](n3_occupation_transfer_amendment_v2_20261012.md)、[固定plan](n3_occupation_transfer_v1_execution_plan_20261012.json)。過去の準備時未認可の記録は保持する。

- frozen source: `83f51cd8a4848572c66a5535c67d0d0a5f3868cd`（8ファイル変更なし）
- source fingerprint: `41b951dae4968a2b679e61c9e61e761dc8253babeb2c610ccc1886fe3646fcb6`
- plan SHA: `a825ece44d9bf8266a9c83d390572139d3799b748419b644515d79230e5c0d0f`
- input SHA: `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`（48,201 bytes、再生成なし）
- primary ledger seal: `8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69`
- attempt1、retry0、old12/T3再探索0、many-body truth0、quantum compile0、GPU0。

実行command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 /tmp/prt-representation-venv/bin/python scripts/run_n3_occupation_transfer_v1.py --run --authorization docs/research/n3_occupation_transfer_v1_execution_authorization_20261012.json
```

## 2. Hamiltonian・精度・検証scope

実対称one-body＋非負unordered density、一外部modeを空に固定するactive-space射影、固定ν-sector。
4base各2つの同norm方向（edge_pair/distributed_three）の事前固定8入力。乱数標本や統計的blind試験ではない。

| base | full modes | active m | ν | 外部mode | scale | δ |
|---|---:|---:|---:|---:|---:|---:|
| C1 chain | 7 | 6 | 2 | 1 | 1 | .004 |
| R1 ring | 8 | 7 | 3 | 4 | 1.5 | .006 |
| L1 ladder | 9 | 8 | 3 | 0 | .75 | .003 |
| S1 star | 10 | 9 | 4 | 8 | 1.25 | .005 |

geometry・化学basis・DF rank/方針・split L_D・PF/time evolution windowは適用なし。
保証対象は基底エネルギーの射影差 `0≤a−E0≤δ`。時間発展のoperator normやQPE/RPE shot費用を評価していない。
Uは固定determinant/equal-pair trialの係数Rayleigh上界、cは係数からの厳密な外部sector下界。
真の基底状態・真のgap・many-body対角化をselector/certificateへ使用していない。

## 3. 同一U,c_star,δでの比較

raw96record（3下界行×4方法×8入力）を保存。B*は三行のNORM/FIVE/OLD_DIRECTのPSD-valid証拠を同じc_starへ継承したOR。
OCC gainはraw_OCC_at_c_star AND NOT B*だけ。OCC自己継承、旧法fallback、運用unionを追加保証数へ含めない。

| input | NORM | FIVE | OLD-DIRECT | raw OCC | 強対照B* | OCC gain |
|---|---|---|---|---|---|---:|
| C1_edge_pair | PASS | PASS | PASS | PASS | PASS | 0 |
| C1_distributed_three | PASS | PASS | PASS | PASS | PASS | 0 |
| R1_edge_pair | PASS | 有限探索未取得 | PASS | PASS | PASS | 0 |
| R1_distributed_three | PASS | PASS | PASS | PASS | PASS | 0 |
| L1_edge_pair | PASS | 有限探索未取得 | PASS | PASS | PASS | 0 |
| L1_distributed_three | PASS | PASS | PASS | PASS | PASS | 0 |
| S1_edge_pair | PASS | 有限探索未取得 | 有限探索未取得 | PASS | PASS | 0 |
| S1_distributed_three | PASS | 有限探索未取得 | 有限探索未取得 | PASS | PASS | 0 |

FIVEはc_starで4/8、OLD-DIRECTは6/8、NORMとOCCは8/8。FIVEやOLD-DIRECTだけとの差をOCC固有の改善とはしない。
全raw行の採択数はNORM22/24、FIVE11/24、OLD-DIRECT16/24、OCC22/24。
whole-case未判定0。raw未採択25件の内訳はNORM条件不成立2、有限探索未取得23。
stage別・case別理由は[機械summary](../../artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/result_summary.json)。
finite不成功23件は数学的不可行性証明ではない。

同norm2方向ではUと下界は同じ。次表の小数は表示用で、採否計算は保存有理数を使用した。

| base | U | c0 | c_occ | c_dh | c_star |
|---|---:|---:|---:|---:|---:|
| C1 | -0.8559094809 | 0.07980743647 | 0.0852171527 | 0.04432346604 | 0.0852171527 |
| R1 | -1.735714286 | -0.9878915787 | -0.8195091627 | -0.9179063441 | -0.8195091627 |
| L1 | -0.8912946429 | -0.4363709688 | -0.3514075431 | -0.4206758675 | -0.3514075431 |
| S1 | -1.833646617 | -1.661575576 | -1.289181916 | -1.274673029 | -1.274673029 |

S1のc0ではNORM/OCCとも2方向で未採択、c_occまたはc_starでは両方が通った。
この変化は共通外部下界の強化で説明され、方向付き条件固有の追加保証は観測されなかった。
全8入力でc_starのNORMが既に通るため、今回の入力集合で強い対照が認めない縮約をOCCが追加する事前目的は達成しなかった。
このscopeの結果を、全HamiltonianにおけるOCCの無価値や全既知算法への優位・劣位に一般化しない。

## 4. 一次seal・secondary・独立監査

8 primary / 96 rawを保存・SHA seal後にsecondaryへ進んだ。8件ともgainがなくNOT_ELIGIBLE。
secondary eig0/PSD0、separator提案0。旧族分離は未実施であり、separator探索の失敗ではない。
一次ledger、primary checkpoint、未取得証拠、raw行、継承・fallback証拠、resource/event DAGを保持する。

[独立保存audit](../../artifacts/n3_occupation_transfer_v1/2026-10-12/run1/independent_saved_audit.json)は **5,008,702 checks PASS**、checker最大監視bit長1167。
checkerはstdlibの有理LDL/元行列再構成を使い、optimizer/eigensolver/ground solver0。
別processで`scripts/verify_n3_occupation_transfer_v1.py`を保存結果へ実行し同じauditを確認。
[別process確認記録](../../artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/local_saved_verifier_confirmation.json)。
local検査とGitHub再取得は同一環境での保存証拠監査であり、immutable CIまたは外部環境の科学batch再実行とは区別する。

## 5. 計算費用・資源・実装上の限界

32 CPUs/affinity32、実行直前available memory 55619198976 bytes。
Python3.12.3/NumPy1.26.4、2 spawn workers、BLAS/OMP/MKL/Numba/Rayon各1thread。
凍結済みのcase内matrix/spectral/PSD cacheを使用した。GPU driver接続不能、CPUのみ。
科学条件・precision・capsを変える最適化や結果後tuningは行わない。

- 全batch（保存checkerを含む）wall 7.660213s、parent CPU 6.870775s、children CPU 2.200394s。
- producer eig総数2272、PSD総数719。case最大eig383/850、PSD177/512。
- OCC spectral pair総数142、bound pair総数96。denied callsなし。
- producer監視bit長最大1156/4096。explicit operand/resultだけを監視し、Fraction内部temporary integerは対象外。
- parent peak RSS 74432512 bytes、最大child peak RSS 74432512 bytes。
- sampled aggregate RSS peak 173293568 bytes（8GiB budget、違反なし）。sample45、目標間隔50ms、実最大gap 0.434613s。間の瞬間peakは未測定。
- run1合計11673420 bytes、最大shard 5831263 bytes、run.log257 bytes。32MiB shard/64MiB output/8MiB log cap内。wall/CPU time capなし。

以下はproducerのevent依存とfirst-observed primitive費用からの単独運用経路再構成wall見積り（秒）。
cold standalone実測や量子総資源の比較ではない。

| input | BASE-OR | OCC operational |
|---|---:|---:|
| C1_edge_pair | 0.007305 | 0.008166 |
| C1_distributed_three | 0.007384 | 0.008158 |
| R1_edge_pair | 0.012135 | 0.013246 |
| R1_distributed_three | 0.012387 | 0.013374 |
| L1_edge_pair | 0.018222 | 0.019608 |
| L1_distributed_three | 0.018776 | 0.020035 |
| S1_edge_pair | 0.096838 | 0.390802 |
| S1_distributed_three | 0.099440 | 0.392357 |

この再構成では追加保証0に対してOCC operationalの見積りが全8入力でBASE-ORを上回る。
whole batchには全factorial行を計算する費用と保存checker費用が含まれ、単独経路見積りと分離する。
speedup倍率・実分子scale・総量子costの結論は出さない。

## 6. 保存・公開・STOP

[一次result](../../artifacts/n3_occupation_transfer_v1/2026-10-12/run1/result.json)、[run audit](../../artifacts/n3_occupation_transfer_v1/2026-10-12/run1/run_audit.json)、
[STOP](../../artifacts/n3_occupation_transfer_v1/2026-10-12/run1/MANDATORY_STOP.json)。source/plan/inputと旧results/STOPを保持して専用HIROMU1015 branchへ公開し、GitHub remoteから再取得する。
公開・再取得のcommit/receiptは後続の[publication文書](n3_occupation_transfer_v1_publication_20261012.md)に追加する。
既存global manifestの`relevant_commits[7].description`欠落FAILは保持し、新entry検査・保全auditと区別する。

次の科学的判断事項は、今回の共通強下界では独立gain0だった結果を受けて、N3の研究上の価値と方向を評価すること。
このbatchを成功するまで反復しない。N1追加探索停止、N2保留、中心テーマnull、実分子・多外部・一般二体への拡張は未認可。
