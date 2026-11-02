# 研究C：J/U分離型PR-tail・RC2の限定科学実行（2026-11-01）

## 1. 結果と資格

明示的な科学実行許可後、公開SOURCE `cbbb11e7e4f2a133bd53c1efdf59b1e02c4d5287`を変更せず、独立worktreeのfresh run01へ一度実行した。**258/最大417 logical units COMPLETE、correctness FAIL 0、retry/resume 0。両候補とも物理DF予算はPASSだが、両方の一次収束資格は未達**となった。

最終statusは`ONE_OR_BOTH_FIRST_ORDER_UNQUALIFIED`。PRは11 complete U→J cycles後のJ stationarity未達で登録停止し、RC2はJ stationarityを満たす一方U方向が未達のまま64 cycles上限に達した。残り159 unitsを追加反復・別開始点へ流用していない。

PRのexact tail λは保存joint終点0.9545432846581517から今回0.7738207630625358Haへ約18.93%低下した。ただし**未収束終点同士の記述値**であり、分離型の収束優位・安定性・十分最適化したRC2に対する優位とは認定しない。RC2の同じ自己目的は旧joint終点より高く、全体としてjointを上回る一次資格は得られていない。`qualified_tail_comparison`はnullを維持した。

[run結果](../../artifacts/research_c_separated_J_U/2026-11-01/run01/result.json)／[小さい集計](../../artifacts/research_c_separated_J_U/2026-11-01/execution_summary.json)／[STOP](../../artifacts/research_c_separated_J_U/2026-11-01/run01/STOP.json)／[公開前保全照合](../../artifacts/research_c_separated_J_U/2026-11-01/execution_postflight.json)。STOP.failed=falseは実装例外・correctness FAILなしを表し、一次収束PASSの意味ではない。

## 2. 固定条件・SOURCE・実行

- 保存直鎖H4、隣接1.00Å、STO-3G、Nα=Nβ=2、物理sector36。
- 通常number CDF rank8、全signed一般実対称J、実直交U、元frame順LD2（head0/1、tail2..7）。元g補正κ・nuclear scalar一回。
- PR: γ=.1、ε=1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074。RC2: α=.1、P0=4.410884790420646、full ordered spatial Z=J L2。異なるraw目的同士を順位付けしない。
- 同じ元qualified J/U/x0からPR/RC2各1独立trajectory。旧joint/LP終点からのwarm-startなし。固定U LPのtail tensor保存等式やhead固定は今回適用しない。
- PR J: SciPy BFGS、全方向SVD初期計量、gtol1e-7/maxiter1000。RC2 J: augmented gelsd。U: 局所右実反対称expm、JAX64 gradient、L-BFGS-B maxiter25/maxfun500、最大64 cycles。
- 独立isometric J gradient∞と全物理U tangent gradient₂はそれぞれ1e-5＋固定engineering marginで判定。詳細は[準備報告](research_c_separated_J_U_optimization_preparation_20261101.md)と[固定契約](research_c_separated_J_U_contract_unapproved_20261101.json)を参照。
- 契約SHA256 `fb0e137564e4013225bd12c4812017af882cee4fd5af69c382149a5eb75735df`、SOURCE manifest SHA256 `f2c3618c9e75bd7949a3824f9fed9213f537d3f58cea1f1616dd59198910c4f8`。実装commit `e75c6b66bfd7711ca93d3e3942b0dae6ece08be7`。
- 元入力SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、共通初期J/U/x0 SHA256 `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856`。旧joint SOURCEは`e2f7dfd44357510d54ab9c6c9c5032e0072ab376`。

契約の歴史的なunapproved名/statusは変更せず、チャットでの今回の明示許可に基づいた。承認証明書・原文保存・新grantは作成していない。既存227人工/回帰methods PASSを同じ固定SOURCEの技術証拠として再利用した。

両環境のpassive preflight、SOURCE114/input13、実空き資源・fresh出力を照合した。[実行preflight](../../artifacts/research_c_separated_J_U/2026-11-01/execution_preflight.json)。最初の技術的資源確認は非公開cgroupパスを無条件に読んでFileNotFoundとなったため、存在する資源情報だけを記録し直した。科学SOURCEや環境を変更せず、科学run開始前の確認で科学retryではない。

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_separated_J_U.py \
 --contract docs/research/research_c_separated_J_U_contract_unapproved_20261101.json \
 --execute \
 --output /tmp/prt-c-separated-j-u-execution-20261101/artifacts/research_c_separated_J_U/2026-11-01/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

このrun01は消費済み。再起動・resumeするための手順ではない。ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38の既存fit環境と、既存Qiskit1.3.0/OpenFermion1.6.1の評価環境を分離し、共有venvの変更はない。

## 3. 正しさ・DF物理資格

| Gate | 結果 |
| --- | --- |
| SOURCE・入力・依存環境・fresh output | PASS |
| 保存初期DF資格・同じ元J/U/x0・loss/penalty scale | 両arm PASS |
| J/U objective非増加・finite・直交性 | 全152 blockが登録allowance内PASS |
| 終了codec・独立ffsim objective/J-gradient照合 | 両arm PASS |
| sector36 raw CAR対一般K | 両arm PASS |
| 物理DF norm＋既存guard ≤ CA/8=.00019875Ha | 両arm PASS |
| J/U両方の一次資格 | 両arm UNQUALIFIED |

| 測定値 | PR-tail | RC2 |
| --- | ---: | ---: |
| Tensor Frobenius誤差 | 1.1195919926e-08 | 3.96366795854e-09 |
| Tensor max-entry誤差 | 1.80775100445e-09 | 6.41854763783e-10 |
| 物理sector operator norm / Ha | 1.10639656677e-08 | 5.01816354787e-09 |
| 既存numerical guard / Ha | 3.22717032499e-09 | 3.17557661905e-09 |
| sector raw-CAR対一般K max-entry | 3.99680288865e-15 | 1.99840144433e-15 |
| 独立ffsim J-gradient差 max-entry | 4.17205958832e-07 | 9.15736633594e-07 |

独立gradient照合allowanceは固定atol1e-6/rtol1e-7で、一次資格閾値とは別。数値差を見て変更していない。DF guardは`sector_operation_scale_guard8_proposal_v1`というengineering pilotで、厳密enclosureではない。旧rank_assessmentのstatus文字列CONVERGEDを新optimizerの収束と読み替えない。今回はsector36を評価し、full256や実回路を再評価していない。

## 4. 同じ目的内での旧joint終点との比較

旧joint J/Uは各arm一度だけ読込み、新しい物理metricを一度だけ評価した（契約内4 units）。旧結果や旧停止statusを書き換えず、旧終点を最適化していない。

| Arm・終点 | 同armのF | J gradient∞ | 物理U gradient₂ | exact tail λ / Ha |
| --- | ---: | ---: | ---: | ---: |
| PR 旧joint | 0.0186661040073517 | 0.365777499035 | 0.579861586517 | 0.954543284658152 |
| PR 新分離型 | 0.0122671746519134 | 1.10243877371e-05 | 2.95486445917 | 0.773820763062536 |
| RC2 旧joint | 0.0223244536955152 | 0.0319986797996 | 0.0593639804898 | 8.5456871282301 |
| RC2 新分離型 | 0.0291934506564528 | 3.36198009634e-07 | 0.866763842696 | 8.28580975959188 |

PRのFとλは旧PRより低い一方、U stationarityは旧値より大きい。RC2はJ stationarityが改善した一方、自己目的とU stationarityは旧RC2より大きい。両者のJ/U同時資格は成立せず、十分に最適化したRC2との比較は未成立である。PRとRC2のraw Fを直接比較していない。旧jointと新trajectoryの古典budgetは同一ではなく、速度優位・分離そのものの因果効果や多初期値安定性は認定しない。

PRは初回J0だけでF≈.01226764443、λ≈.77383557984となり、11 cycle後はF≈.01226717465、λ≈.77382076306だった。今回観測された大半のPR目的改善は、元Uを固定した全J更新で既に生じている。旧LPはhead/tail tensorを保存する別の制約問題だったため、その改善率と今回を同じ探索範囲の成績として比較しない。

## 5. 停止理由・計算負担・科学call

PRのJ BFGS全12回はraw status2（precision loss）。初回のpost-J stateは独立J gradient∞≈7.02e-7でMETだったが、11 cycle終了時は1.1024387737e-5でNOT_METとなり、登録規則により停止した。Marginや閾値を緩和して継続していない。最終物理U gradient₂≈2.95486446も未達である。

RC2のaugmented solve全65回はraw status0。最終J gradient∞≈3.36198e-7でMETだが、物理U gradient₂≈.866763843が残り、64-cycle上限で停止した。両armのU block全75回は各25 iterationのstatus1。登録済みの有限inexact更新であって収束成功ではない。

| Arm・block | 呼出し | raw成功 | solver nit | solver nfev / njev | block内VG実回数 | inclusive wall / 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR J | 12 | 0 | 617 | 2069 / 1920 | 2093 | 0.364389 |
| PR U | 11 | 0 | 275 | 297 / 297 | 330 | 0.132556 |
| RC2 J | 65 | 65 | 0 | 0 / 0 | 130 | 0.142823 |
| RC2 U | 64 | 0 | 1600 | 1984 / 1984 | 2176 | 0.669869 |

固定UでのJ問題は今回大幅に改善できたが、PRでは内側の数値stationarityも停止要因になり、RC2では外側U更新が一次資格の主な未達項目となった。現在の固定J U更新とJ再solveの結合を調べる技術的動機が得られた。原因の一意な確定や、別更新政策で収束するとの保証ではない。ここで新しい診断計算・最適化は追加しない。

| 科学call | 実績 / 上限 |
| --- | ---: |
| Parent | 11 / 11 |
| PR worker | 44 / 203 |
| RC2 worker | 203 / 203 |
| **合計** | **258 / 417** |
| J solve | 77 / 130 |
| U solve | 75 / 128 |
| state診断 | 79 / 132 |

新規交互最適化trajectoryは2、ffsim joint fitter呼出しは0。内部のJ/U library solveをそれぞれ計上し、一つのtrajectoryを一つのsolveへ隠していない。Parent/PR/RC2のjournalで258 attempt/COMPLETE、FAIL0。Retry/resume0、追加rank・強度・初期点・fallback0。元scalar/κ・frame順は保持した。

実行前の空きメモリ約56GB・affinity32 CPUs・低いloadに対し、既存plannerが2独立workerを選択。各trajectory内は依存順序J→U→Jを維持し、固定U A/T cache・shape-cached JAX graph・process-local tiny BLAS調整を使用した。Cgroup quotaは今回の実行環境から公開されず、値を捏造していない。CPU/RAM/wallの人工上限や共有設定変更はない。

保存wallはparent3.849162秒、PR1.719404秒、RC2 2.153718秒。Process CPUはparent1.551132秒、PR3.259076秒、RC2 3.716975秒、peak RSSは順に279416/397044/397340KiB。Inclusive block timingと起動・初期診断・exportを含むworker timingを混同せず、他手法との速度比較には使わない。

## 6. 研究上の問いへの今回の到達点

| 問い | 今回得られた証拠と限界 |
| --- | --- |
| Q1 jointより良い一次状態か | J残差は改善。U残差は両方旧jointより大きく、同時一次資格は未達 |
| Q2 PR目的/exact λを安定に減らせるか | 今回の1初期値ではPR自己F・λが低下。多初期値安定性や収束後の改善は未確認 |
| Q3 DF予算内か | 新両候補とも登録engineering DF予算PASS |
| Q4 強いRC2でもPR λ利点が残るか | RC2がU一次資格未達のため判断できず、qualified比較null |
| Q5 J/Uどちらが制約か | PR J precision-loss/最終勾配未達、両arm U非収束。RC2の主反復負担はU block |

数値pilotの候補と未資格終点を保存するが、厳密最適性・最適RC-DF一般に対する優位・actual CA達成・PF/RTE・attenuation・shots・native価格・総費用は未資格/nullを維持する。新候補へ旧PF/RTE/native価格を転記していない。

## 7. 保全・次の担当・mandatory STOP

新SOURCE114/input13と元契約、旧LP/raw976/旧SOURCE7群・旧contract/grant/ledger/marker/STOPを照合し、原本の不変を確認する。新raw269ファイル、block raw x/jac、state J/U/勾配、全trace、sector/codec診断、journal/resources/STOPを保存した。研究概要・本報告・dated note・機械可読manifestに今回の未資格結果を追記し、過去の科学結果を上書きしない。

次の担当はユーザー／研究C：この未収束pilotを受けて、U blockのconditioning・更新政策とPR J precision-lossを切り分ける技術調査を優先するか判断する。Codexによる新しい実H4再解析・条件変更・追加最適化は別の固定範囲と明示許可が必要で、現在のrunの残枠は再利用しない。重要科学レビューも今回実施していない。

独立branch `codex/research-c-separated-j-u-execution-20261101`で結果をcommit/pushし、remote SHA/tree・公開本文・新SOURCE/input/rawの整合を照合する。**公開照合後mandatory STOPし、追加fit・PF/RTE・native・shots・総資源へ自動進行しない。**
