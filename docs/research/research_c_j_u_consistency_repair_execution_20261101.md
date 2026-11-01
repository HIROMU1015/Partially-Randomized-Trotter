# 研究C：J/U整合gateの評価点修正・限定再実行（2026-11-02公開）

保存パス・契約名の20261101は固定scope IDとして保全し、公開作業の記録日は2026-11-02 JSTとする。

## 1. 結果

ユーザーの修正・再実行指示に基づき、旧runを保全したまま、保存rawの限定15-call診断、評価点の修正、人工回帰検証、同じ元初期値からの新しいpaired実行を一括して実施した。

**診断15/15 COMPLETE、修正版fit328/415 COMPLETE、FAIL0。今回計343/最大430 logical units。** 旧run内のretry/resumeを解除・上書きしたのではなく、新SOURCE・新しいfixed scopeでpaired再実行を1回行った。各新run内retry/resume0。旧210-call部分STOP・旧未使用205 units・さらに旧159 unitsは転用していない。

修正後は両armの終端整合・codec忠実性・独立sector表現・DF予算がPASSした。しかし、**PR/RC2ともJ/U同時一次資格は未達**。PRは35 cycles後のJ基準未達、RC2は64 cycles上限でU未達のまま終了。技術的な誤検出を修正した結果を、最適化の収束改善や強いRC2に対する優位とは扱わない。

[集計](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/execution_summary.json)／[fit result](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/fit_run01/result.json)／[fit STOP](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/fit_run01/STOP.json)／[保存診断result](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/diagnostic01/result.json)／[保全postflight](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/execution_postflight.json)。

## 2. 旧FAILを切り分けた証拠

### PR：Newton-CGのraw jacの評価点

固定SciPy1.14.1の[Newton-CG実装](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_optimize.py)は、最後の更新前に計算したgfkを返す場合がある。更新後のxに対してraw jacが常に等しいというこちらの検証仮定が誤っていた。Library/optimizer自体を変更せず、実際のSciPyを使う人工回帰試験でもpre-step jacを再現した。

[保存PR J0の15-call内診断](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/diagnostic01/PR_comparison.json)：

| 元の複合gateの内訳 | 保存点での差 | 元allowanceによる判定 |
| --- | ---: | --- |
| raw fun対終点chart値 | 0 | PASS |
| chart値対元座標の独立値 | 3.81639e-16 | PASS |
| 元座標勾配を変換した値対chart終点勾配 | 4.55206e-11 | PASS |
| raw solver jac対終点chart勾配 | 7.86716e-7 | FAIL（比較点が異なる） |

保存点の独立physical J gradient∞は1.3643355409e-6だった。修正版fresh runのJ0では、raw jacと記録したlast_step_start勾配が差0で一致し、終点との差7.86716e-7もそのまま保存された。原因は単に許容差が小さすぎたのではなく、raw gradientの点を取り違えたことだった。

### RC2：codec前後の異なるUでの微分比較

元candidate Uとcodec/xから復元されたUはbinary64で微小に異なる。正規化loss_scaleが小さいため、目的差が小さくてもJ-gradient差は拡大し得る。旧terminal gateはffsimのx点の微分と、元U点の独立微分を同一点として比較していた。

[保存RC2 state64の診断](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/diagnostic01/RC2_comparison.json)：

| ffsim J-gradientと比較する独立経路の点 | gradient maxabs差 | 固定atol1e-6/rtol1e-7 |
| --- | ---: | --- |
| 元candidate U | 1.39850465e-6 | FAIL |
| NumPy codec decode後 | 2.29805143e-7 | PASS |
| 実際のADと同じJAX decode後 | 4.95138580e-8 | PASS |

JAX decodeと元Uのmax-entry差は5.77316e-15。Codec忠実性gateは別途維持した。微小U差のある点のgradientを厳密同一点の微分として検証したことが、当該旧FAILの原因だった。旧STOP/rawを成功へ書き換えていない。

## 3. 修正範囲と維持した条件

新core [c_separated_j_u_consistent.py](../../src/trottertracks/c_separated_j_u_consistent.py)は最後のaccepted zとpre-step zを保存する。生jacを対応する記録点または終点で再計算したgradientと照合し、**終点のchart gradientと元座標physical gradientは必ず別に再評価**する。旧jacを上書きせず、一次資格へ流用しない。偽gradient、nonfinite、objective増加は引き続きSTOPする。

新pilot [c_separated_j_u_consistent_pilot.py](../../src/trottertracks/c_separated_j_u_consistent_pilot.py)は、ffsim ADが実際に使用するJAX decode点のJ/Uを用い、独立contraction/adjointと比較する。元candidate J/Uは変更せず、最終sector/DF/一次資格も元candidateで評価する。Decode後のprojector、tensor、NumPy/JAX rotationの忠実性は元codec allowanceで確認する。

**変更しなかった数値条件**：J/U gradient整合atol1e-8/rtol1e-7、native J-gradient整合atol1e-6/rtol1e-7、value atol1e-10/rtol1e-8、一次閾値各1e-5＋従来margin、codec component1e-10とtensor allowance、既存DF guard・CA/8。数値allowanceの緩和、新しいpenalty、rank/強度/初期値探索、solver/ε/cycle政策の変更はない。

新traceの`J_consistency`と`terminal_consistency.json`には、必要な各差分を例外前に保存する。異なる点の参考gradient差はFAILの観測値として残すが、同一点の微分検証に使わない。必要gateのPASSと、参考比較欄のFAILを区別する。

人工・回帰**291 distinct methods＋29 subtests PASS**。[テスト記録](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/technical/test_results.json)。実SciPyのpre-step jac、独立終点判定、codec内の微小U差、元candidateの不変、false success/nonfinite/gradient不一致/非増加FAIL、人工parent-child E2Eを確認。旧SOURCEと共有環境を変更していない。人工成功を実H4収束の証明にはしない。

## 4. 実H4の新しい結果

保存直鎖H4、隣接1.00Å、STO-3G、Nα=Nβ2/sector36、number CDF rank8、元順LD2/head0/1・tail2..7、全signed対称J/実直交U。元g補正κ・nuclear scalar一回、γ=α=.1、ε=1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074、P0=4.410884790420646、DF予算CA/8=.00019875Haを維持。

PRは同じaffine residual/解析Hessian Newton-CG、RC2 Jはaugmented gelsd、Uは同じ固定J residual/Jacobian TRF。最大64 cycles、同じ元qualified J/U/x0から両arm各1新trajectory。旧終点warm-start・旧run resumeなし。

| 測定値 | PR-tail | RC2 |
| --- | ---: | ---: |
| Complete cycles | 35 | 64 |
| 停止status | J_STATIONARITY_UNQUALIFIED | OUTER_LIMIT_UNQUALIFIED |
| J/U分類 | BOTH_UNQUALIFIED | U_UNQUALIFIED |
| 自己F | .012265732322593252 | .029193402354005357 |
| J gradient∞ | 1.807688809e-5、NOT_MET | 6.037097047e-7、MET |
| Post-J物理U gradient₂ | 2.954837920、NOT_MET | .8667167193、NOT_MET |
| 最終固定J U step直後の物理gradient₂ | 1.376116617e-6 | 6.480206478e-8 |
| exact tail λ / Ha | .7737752701910898 | 8.285810103455127 |
| Tensor Frobenius誤差 | 1.119553644e-8 | 3.963520052e-9 |
| Tensor max-entry誤差 | 1.807648170e-9 | 6.418298115e-10 |
| 物理sector operator norm / Ha | 1.106293783e-8 | 5.017996929e-9 |
| 既存numerical guard / Ha | 3.227166764e-9 | 3.175576524e-9 |
| DF予算 | PASS | PASS |
| sector raw-CAR対general-K max-entry | 5.77316e-15 | 2.22045e-15 |
| 同じAD decode点でのJ-gradient差 | 1.61612e-7、PASS | 4.95139e-8、PASS |

両candidateの独立表現・終端必要gateはPASS。JAX decodeの元candidateに対するtensor Frobenius差はPR1.84423e-14/RC2 6.94963e-15で、元codec allowance内。元U点との参考gradient差はPR4.67460e-6/RC2 1.39850e-6であり、これらを隠したり同一点検証PASSと偽ったりしていない。

PRの旧joint λ .9545432846581517や旧分離v1 λ .7738207630625358に対して、今回のλは記述的に低い。ただし両新候補とも一次収束未資格で、`qualified_tail_comparison=null`。異なるPR/RC2 raw Fを直接順位付けしない。RC2の自己Fは旧joint .02232445369551521より高い。修正によって収束上の課題まで解消したとは報告しない。

固定JのUを小勾配まで解いても、J refit後にU勾配が大きくなる状態が今回も残った。追加反復、再最適化、別rank/強度/初期値、PF/RTE/native/shotsへ進まず終了した。DF guardはengineering pilotで厳密enclosure/actual CA証明ではない。

## 5. SOURCE・入力・実行・call

| Phase | 固定科学SOURCE | 最大 / 実績 |
| --- | --- | ---: |
| 保存raw限定診断、optimizer0 | a6a2c55aa8400428329165578d54e2f65bff4636 | 15 / 15 |
| 修正版paired再実行 | 24a7268359506abd171686132f12b4876f9e62bf | 415 / 328 |
| 今回合計 | 別scopeとして固定 | 430 / 343 |

診断[契約](research_c_j_u_consistency_diagnostic_contract_20261101.json)とSOURCE122/input9、再実行[契約](research_c_j_u_point_consistent_reexecution_contract_20261101.json)とSOURCE128/input17を実行前にcommit固定した。診断は15 named operations、修正版runはparent11/PR115/RC2 202、Jsolve101/Usove99/state103、全COMPLETE/FAIL0。

修正版SOURCE manifest SHA256 `fbc00f50f0deae0cbeadbdc4b914dd024504df75a60c18ba292f5de357d8456a`、契約SHA256 `22657253e7478cdf4db5bcc2e5590e42a8574e9185af5e88e3b5b2b6ff2e34f5`、実装commit `b0dc86e61e29469bf9d19f8709e3c68a613f9aeb`。

元input SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、元qualified J/U/x0 SHA256 `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856`。旧失敗raw・input・SOURCEの来歴は独立保持した。新規trajectory2、修正版paired再実行1回、ffsim joint fitter0、各新run内retry/resume0。旧残枠を転用しない。

[実行preflight](../../artifacts/research_c_j_u_consistency_repair/2026-11-01/fit_preflight.json)で両環境・SOURCE/input・fresh output・資源を確認。ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38の既存fit環境、Qiskit1.3.0/OpenFermion1.6.1等の評価環境を分離し、共有venv変更なし。診断は依存順に実行し、新pairedは空きメモリ約56GB/affinity32 CPUsに対し既存plannerの独立2 worker。固定U cacheとJAX shape cacheを使用し、CPU/RAM/wallの恣意的上限は追加しない。

```bash
# 保存診断（消費済み、再起動しない）
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_j_u_consistency_diagnostic.py \
 --contract docs/research/research_c_j_u_consistency_diagnostic_contract_20261101.json --execute
# 修正版paired実行（消費済み、再起動しない）
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_separated_J_U_consistent.py \
 --contract docs/research/research_c_j_u_point_consistent_reexecution_contract_20261101.json \
 --execute \
 --output /tmp/prt-c-j-u-consistency-repair-20261101/artifacts/research_c_j_u_consistency_repair/2026-11-01/fit_run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

診断wall1.073898秒、paired parent wall5.420162秒/CPU1.603068秒/peakRSS288212KiB。PR worker wall3.332217秒/CPU6.093249秒/494004KiB、RC2 wall3.617865秒/CPU6.387784秒/492376KiB。目的・予算・停止状態が異なる旧runとの速度優位を主張しない。

## 6. 保全・次担当・mandatory STOP

診断raw9＋fit raw345＝354ファイルを保存し、旧raw1471（旧曲率run226を含む）・旧SOURCE/contract/grant/ledger/marker/STOPを保全する。各FAIL/PASSの差分を保存し、旧2 FAILを上書きしない。科学manifest・研究概要・dated noteへ今回の診断と再実行を追記し、過去の履歴を保つ。

今回のユーザー指示で許可された修正・再実行は完了した。次担当はユーザー／研究C：技術的な評価点問題と分離型最適化の未収束を分けて、J/U結合・conditioningを扱う次の研究方針を判断する。Codexはこれ以上のfitや科学的条件変更を自動で行わない。重要科学レビューも今回実施していない。

独立branch `codex/research-c-j-u-consistency-repair-20261101`へSOURCE/診断/結果をcommit・pushし、remote SHA/tree・公開本文・入力/旧原本を照合後、**mandatory STOP**する。
