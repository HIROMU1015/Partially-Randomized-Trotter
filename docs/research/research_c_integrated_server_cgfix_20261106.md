# 研究C：サーバー実行のSTOPとmissing-jac修正準備

2026-11-06 JST。H4直鎖1.00 Å/STO-3G、number CDF8/元順LD2、Nα=Nβ=2/sector36。
重要科学レビュー、H6等、新surrogate、論文化GO判断は行っていない。

## 今回の科学実行

ユーザーが許可したSOURCE `200418968aa9e60e5ce842425e0373a4d8fafdb4f78c53a22df37b204989afd6`、
契約 `7615803fd047e19ec052b4ff39f5b1259778aa9371add1ef0db9e11e7fb06c69` で
00:42:08 JSTにfresh実行を一度開始した。約865.22秒後、`FAILED_MANDATORY_STOP`。

- 開発初期化27 unitsが完了。
- 開発18trajectory中17本が64更新まで完了し、各16/64の計34checkpointを保存。
- C0/ENTRY/strength=.3の1本が `PR_INNER_POINT_CONSISTENCY_MISMATCH` で停止。
- 実消費は初期化27＋local task記録17,385＝17,412 logical units。
- global STOPのcompleted_stage_units=27は完了stage分だけであり、途中stageのlocal消費を含まない。
- native価格校正・候補選択・final64・強D評価・1.15 Å入力生成/移送は未実行。
- retry/resume=0。SOURCE全ファイル、旧契約、raw、STOPは保全した。

原本root：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106`。
その `artifacts/research_c_integrated/2026-11-06/INTEGRATED_STOP.json`、
`server_technical/execution_outcome.json` と各local journalが今回の記録である。
native32に未到達のため、登録バッチ実測による下流所要時間の更新はまだできない。
科学的な資源順位・頑健性・移送性の結論は得ていない。

## STOPの技術的原因

保存診断 `k000_4_refine2` はSciPy Newton-CGのstatus3/nit0、初回CG収束失敗だった。
固定SciPy 1.14.1はこの分岐で`gfk`を代入する前に終了するため、返却`jac=None`となる。
旧guardがこれを長さ80のgradient配列として照合し、shape=[]とshape=[80]の不一致で停止した。

同じ保存診断にあるchart/physical gradient、chart/physical value、raw/endpoint value、
binary64同一配列valueはすべてPASS。これはoptimizer成功や実Hessianの性質の保証ではない。
追加の実分子VG/HVP/fit・保存実配列の数値再解析は行わず、診断JSONと固定実装を確認した。
人工的なCG非収束の注入でもstatus3/nit0/jac=Noneを再現した。
その人工Hessianは故障注入用であり、実分子Hessianの不定性を示す証拠ではない。

## 別worktreeで準備した修正

新作業領域：`research-c-integrated-cgfix-20261106`、
branch：`codex/research-c-integrated-cgfix-20261106`。元runのSOURCEは書き換えていない。

solver呼出し中にVG callbackが実際に返したpoint/gradientをcopyして保存する。
返却jacが存在する場合は旧判定を維持する。jac欠落は、status3/nit0、受理stepなし、
初期点から移動なし、記録したsolver入力pointがendpointと完全一致し、そのgradientが
独立endpoint評価と元の許容差で一致した場合だけ、別のqualification routeを使える。

後から再計算したgradientでraw jacを埋めない。raw jacは未取得と明示し、
`success=false/status3/nit0`を保つ。元のraw-jac比較はPASSに書き換えない。
route名は `CAPTURED_SOLVER_INPUT_INITIAL_CG_FAILURE`。
既存のprimal/adjoint/value、binary64、stationarity、physical DF等の検査は維持する。
optimizer、科学条件、許容差、VG/HVP回数、更新/line-search予算は増やさない。

これは欠落raw返却値についての**資格判定方法の明示的な追加**であり、
旧runがこの判定でPASSしていたとは表現しない。新しいSOURCE/契約への明示許可が必要。
偽の返却jac、別point、受理step後の欠落、成功statusでの欠落、実solver callback記録なしは引き続き停止する。

## 成功17本を再計算しない新実行案

保存初期化、元H/CAR context、17本の16/64 checkpointと独立physical referenceを
bytes同一の最小入力として登録した。146ファイル・約22 MB。
旧run全rawの再整理・再計算や成果物移送を目的にした作業ではない。
旧原本rootを記録し、元のSOURCE/STOPとの来歴を維持する。

`Evaluation.from_context`で元H/CARを再構築せず再利用する。
`Run.optimize`は登録済み17結果を保持し、新規dispatchはC0/ENTRY/.3の1本だけにする。
その1本は新SOURCE・fresh outputの新trajectoryであり、停止した旧trajectoryをresumeしない。
17本の証拠は旧SOURCEのまま保持し、全18本を新SOURCEで最適化し直したとは主張しない。

その後の開発selection/native、独立final64、同J/U D、recipe凍結、1.15 Å移送は元の登録通り。
共通r32と公平r16/32/64、selection32/final64の独立性、全512演算子と実逆回路の検査、
`Ctotal=W*mean_kernel+shell`、shots/W/kernel単価/shellの分離を維持する。
新SOURCE digestによる新native cohortを使い、旧標本をpoolしない。
新規科学作業の上限は **118,188 logical units**。
研究gridは18開発trajectoryのままで、17本を保存入力として再利用した分だけ新規計算を減らす。
候補数やnative標本数を削減する案ではない。

## 固定成果物と検証

- [修正SOURCE](../../artifacts/research_c_integrated/2026-11-06-cgfix/technical/SOURCE.json)
- [新契約](../../artifacts/research_c_integrated/2026-11-06-cgfix/technical/contract.json)
- [保存入力と原本来歴](../../artifacts/research_c_integrated/2026-11-06-cgfix/technical/saved_input_handoff.json)
- [passive preflight](../../artifacts/research_c_integrated/2026-11-06-cgfix/technical/preflight.json)

SOURCE SHA256：`c5c48ba38b88d5c914c51f710113e65f806a900a67d7ade03fd76a29290d9608`。
契約SHA256：`670f0faf58781e88919213604d22a803934d135a4397fab0dc981e6e7a601a66`。
fit人工/回帰34件、eval/RPC/保存17+新1接続/下流政策46件がPASS。
実分子で修正routeが成立したという証拠ではない。
固定fit/eval package pins、ffsim指定SOURCE、extended precisionもpassive照合済み。

新出力は `artifacts/research_c_integrated/2026-11-06-cgfix/{development,transfer}/run01`。
どちらも未作成、科学再実行0。旧runのmandatory STOP・retry/resume=0を保持する。

passive確認（科学計算なし）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-cgfix-20261106
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106/.venv-eval/bin/python -B scripts/prepare_research_c_integrated_cgfix.py
```

次の担当：ユーザーは上記資格判定差分・新SOURCE・保存17本＋新1本のfresh科学scopeを実行するか判断する。
明示許可後、Codexが上記コマンドに`--execute`を付けて一度開始し、最初の登録native32バッチ実測で所要時間を更新する。
承認証明書、承認原文保存、消費済みgrantの再利用は要求・実施しない。
