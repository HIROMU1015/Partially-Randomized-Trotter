# 研究C：曲率利用J/U分離pilot・整合gateで部分STOP（2026-11-01）

## 1. 結果

明示科学許可後、固定公開SOURCE `ecb2e7dc0ebfebe15246dfd3c290bcae72eeb246`を変更せず、独立worktree/fresh run01へ一度実行した。**210/最大415 logical unitsを開始し、208 COMPLETE・2 FAIL。登録correctness gateで停止、retry/resume0。** 新しい最終DF予算・表現gate・qualified比較は未実施で、PASSとは扱わない。

- PR：初回J0のNewton-CGはraw success/status0を返したが、直後の`J_CHART_OR_GRADIENT_MISMATCH`でFAIL。Post-J state、U更新、最終candidate exportへ進まなかった。
- RC2：開始済みの独立workerを契約通り回収し、64 cycles後に`OUTER_LIMIT_UNQUALIFIED`、J MET/U NOT_METを記録。その後の`INDEPENDENT_FFSIM_OBJECTIVE_J_GRADIENT_MISMATCH`でFAILし、candidate exportへ進まなかった。
- 両workerとparentのSTOP.failed=true。残枠205 unitsを別試行・条件・rank・診断へ転用しない。SOURCE・閾値・allowanceを変更せず、追加再評価や修正再実行を行っていない。

[小さい集計](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/execution_summary.json)／[parent STOP](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/run01/STOP.json)／[parent failure](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/run01/failure.json)／[保全postflight](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/execution_postflight.json)。最終result.jsonは生成されていない。

## 2. 固定条件・preflight

保存直鎖H4・隣接1.00Å・STO-3G、Nα=Nβ2/sector36、number CDF rank8、全signed対称J/実直交U、元順LD2/head0/1・tail2..7、元g補正κ/nuclear scalar一回。PR γ=.1/ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074、RC2 α=.1/P0=4.410884790420646。DF予算CA/8=.00019875Ha。両armは同じ元qualified J/U/x0から独立開始し、旧終点warm-startや旧159-unit残枠の再利用はない。

[固定契約](research_c_separated_J_U_curvature_contract_unapproved_20261101.json)のSHA256は`4c631022c08b2d2a88f67ffbcc8ce960119bc228035c861ee79ce0933d7a9796`。[SOURCE119](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/SOURCE.json)のSHA256は`6170207a0b86badd26bda1a0c46a5748d32045ddc8cfdd2edbbc6af5fb6d7f8e`、実装commitは`d1b37cd8e1cc9916a5d2948133d769be1d6119da`。

元入力SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、初期J/U/x0 SHA256 `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856`。保存入力17件を照合し、旧joint/分離型は固定JSON scalar履歴だけ再利用した。古いNPZ/旧点の目的・勾配を再評価していない。

[実行preflight](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/execution_preflight.json)でSOURCE/input/fresh・両環境をPASS確認。既存257人工/回帰methods＋29 subtestsを同じSOURCEの技術証拠として再利用した。これを今回の実H4 gate成功の代用にはしない。

既存fit環境はffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38、評価環境はQiskit1.3.0/OpenFermion1.6.1等。共有環境を変更しない。契約の歴史的unapproved名/statusはそのまま維持し、今回のチャット許可に基づき実行。承認証明書・原文保存・新grantは作成していない。

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_separated_J_U_curvature.py \
 --contract docs/research/research_c_separated_J_U_curvature_contract_unapproved_20261101.json \
 --execute \
 --output /tmp/prt-c-separated-j-u-curvature-execution-20261101/artifacts/research_c_separated_J_U_curvature/2026-11-01/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

このrun01は消費済みで、再起動・resume用の手順ではない。

## 3. Gateと未実施範囲

| Gate | PR | RC2 |
| --- | --- | --- |
| SOURCE/input/環境/fresh | PASS | PASS |
| 保存初期DF資格・元x0・scale同一性 | PASS | PASS |
| 旧JSON scalar履歴再利用 | 完了 | 完了 |
| J0 solver直後の独立座標・勾配整合 | **FAIL** | PASS、計65 J solve完了 |
| U更新 | NOT_RUN | 64 block完了 |
| Post-refit J/U同時一次資格 | NOT_EVALUATED | U未資格、64-cycle上限 |
| 終了codec/ffsim独立目的・勾配整合 | NOT_RUN | **FAIL** |
| final.npz export | NOT_RUN | NOT_RUN |
| 新candidate sector DF norm/guard・表現gate | NOT_RUN | NOT_RUN |
| Qualified λ比較・最終DF予算 | null | null |

元入力のphysical sectorはparentで一度構築されたが、新終点に対するDF誤差・budget gateとは別である。保存初期DF PASSを新candidateのDF PASSへ転記しない。

PRのraw J0記録はsuccess=true/status0、fun=.012267644429814947、nit481/nfev785/njev785/nhev481。しかしraw solver成功は同値性を保証せず、復元した物理Jの採用資格・一次残差は未確定。生x/z/jac/座標変換は`pr/J_raw_0.npz`、raw metadataはtraceへ保全した。新しいNPZ decodeや勾配再評価は加えない。

RC2の保存されたpartial stateは以下の通り。ただし終端の独立ffsim gateはFAILしており、これを検証済み最終candidateや方法間の性能比較へ昇格させない。

| RC2 partial・64-cycle後の既存記録 | 値 |
| --- | ---: |
| 自己F | .029193402354005357 |
| J gradient∞ | 6.03709704720401e-7、MET |
| Post-J物理U gradient₂ | .8667167192797767、NOT_MET |
| 最終固定J U step直後の物理U gradient₂ | 6.480206477792986e-8 |
| exact tail λ / Ha | 8.285810103455127 |
| 直交性max-entry残差 | 1.5432100042289676e-14 |

固定JのU勾配が小さくてもJ refit後に大きい勾配が残るという記録は得た。実H4の根本原因・新手法の優位・DF予算適合を確定するものではない。目的の異なるPRとRC2のraw Fを直接比較していない。

## 4. FAILの意味と不足する診断

PRの複合gateは、affine chartとraw solver/元座標での目的、変換した勾配、dtype/shape/finite等の整合を確認する。RC2の複合gateは、既存codecへ再parameterizeしたffsim objective/J-gradientと独立J経路等の整合を確認する。いずれも固定allowanceの下で不一致を検出したという意味である。

**今回のfailureメッセージは複合gate名だけで、どのsubconditionが超過したか、各差分の測定値をすべて保存していない。** そのため、丸め/conditioning、chart復元、勾配実装等の原因をこの記録だけで一意に特定しない。科学STOP後に新しい値を計算して閾値を調整することもしない。

次の技術候補は、複合gateの各比較値を例外前に保存する最小instrumentationと、それを人工stiff問題で検証すること。既存保存rawを新たに目的/勾配評価する限定診断は別の科学範囲・明示許可が必要。初期点へのfallback、元初期値の自動採用、無断の閾値緩和・追加fitは行わない。

## 5. Call・資源・保全

| 実行主体 | attempts | COMPLETE | FAIL |
| --- | ---: | ---: | ---: |
| Parent | 4 | 4 | 0 |
| PR worker | 6 | 5 | 1 |
| RC2 worker | 200 | 199 | 1 |
| **合計** | **210 / 415** | **208** | **2** |

J solveは66（PR1/RC2 65）、U solve64（RC2だけ）、state診断67。新規trajectoryは両arm各1、ffsim joint fitter呼出し0。Parentのfinal読込・DF/表現・比較7 units、両candidateのexport/tensor等は依存処理として停止。未使用205 unitsは転用しない。

実行前affinity32 CPUs、空きメモリ約56GB・低いloadに対し、既存plannerが2独立workerを選択。固定U A/T・shape-cached JAX residual/Jacobianを再利用し、J→U→Jの依存は逐次。PR失敗時にはRC2が開始済みで、既存launcherの範囲で回収した。共有環境/他ユーザーjobを変更せず、恣意的CPU/RAM/wall上限も追加していない。

Parent wall4.349575秒/CPU1.284766秒/peak RSS276428KiB、PR wall1.217646秒/CPU2.572758秒/392012KiB、RC2 wall2.904747秒/CPU5.408893秒/432424KiBを保存。失敗した有限経路の実測であり、収束速度の優位比較には使わない。

新raw226ファイル（生block/state NPZ、JSONL trace、solver結果、全call、failure/STOP/resources/stdout）と新SOURCE/input identityを保全する。旧分離run269を含む既存raw1245・旧SOURCE/契約/grant/ledger/marker/STOPは不変。研究概要・本報告・dated note・機械可読manifestへ部分STOPを追記し、旧結果を変更しない。

## 6. 次担当とmandatory STOP

今回、Q1〜Q7の最終比較に必要な独立終端gateと新DF資格が揃わなかった。J solverの成功、partial stateの一次診断、DF物理資格、最終比較資格を分離し、未測定値はnullを維持する。PF/RTE、ground/alias、回路、shots、actual CA、総資源の追加計算は0。

次担当はユーザー／研究C：最小のgate内訳保存・人工検証と、保存raw限定数値診断のどこまでを次に許可するか判断する。Codexは今回の固定契約を修正・再実行せず、部分証拠のcommit/push、remote SHA/tree/公開本文/SOURCE/raw照合までを完了する。

独立branch `codex/research-c-separated-j-u-curvature-execution-20261101`。**公開照合後mandatory STOPし、追加診断・再最適化・閾値変更・下流科学計算へ自動進行しない。**
