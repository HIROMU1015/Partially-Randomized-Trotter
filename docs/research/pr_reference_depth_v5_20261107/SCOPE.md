# 新A V5：固定scope

2026-11-07 JST、SERVER 2。基準 `224440f740051ed5794e8db87a57908fc02dd02f`。専用branch/worktree `codex/pr-reference-depth-v5-20261107` / `pr-reference-depth-v5-20261107`、専用出力 `artifacts/pr_reference_depth_v5/2026-11-07/`。

Linear H4 1Å/STO-3G/DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical finite-bit law、保存18round q/r、energy 1.5936e-4 Ha、失敗率.05、phase cap .4rad、Qiskit1.3.0/compiler A/RZ-SX-X-CX/全結合を固定する。B′4-CX中央回転、reference F/G・vacuum位相・8-CX CSWAP・basis共有・D境界融合・actual inverseを変えない。9/17qubit、shot8598/8606、旧誤差・alpha・aliasを維持する。

実施はV5-A〜Eの保存SOURCE/局所lowering/QPY/whole価格再利用、frontier状態と構造的depth下界・上界の実装・人工検査・保存cohort照合、finite-bit有理期待値、18round shot加重再会計。新しい回路builder/compilerは作らない。現在の実行予定では新build/compile/sampling/simulationは不要。追加取得が不可欠なら目的・q/r・path/seed・件数を取得前に限定し、scope超過時はSTOPする。

保存局所D/frameの依存鎖に基づくB′下界、reference2 laneの並列構造に基づく上界を検討する。平均event数で作る回路のdepthを期待depthとはみなさない。上下界が重なればUNDETERMINED。局所加法値・cohortを全supportの厳密な期待値へ読み替えない。

新PF/law/schedule/条件/gadget/F-G分割/compiler最適化、V3角度分類変更、Clifford+T合成・precision・T-depth・新長r探索、実機時間・routing/noise、baseline追加ancillaは対象外。旧SOURCE/raw/結果/grant/ledger/marker/STOPは読取専用。V5完了又は指定の不整合・保証不足・scope変更条件でmandatory STOPし、GPT／ユーザーへ返す。
