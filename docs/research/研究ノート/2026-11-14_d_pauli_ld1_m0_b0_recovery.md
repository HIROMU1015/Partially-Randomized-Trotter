# 2026-11-14 D：Pauli LD1/m0 B0復旧完了

旧run01のimport STOP公開後、ユーザーが「復旧再実行してよい」と承認した。
当worktreeだけのtracked referencesを展開し、元blob/input SHAと照合した。
run02出力先と実restore moduleまでのimport-only preflightを最小追加し、科学関数は元62aeとAST同一。
親9人工/metadata test（内部子5件もPASS）後、実行SOURCE9abbcc563をcommit/pushで固定した。

元C canonical d905d219/RMSE:62a9…、H4/1Å/STO-3G/保存tol-only DF10、Pauli LD1/m0、
S2/Δ.1/T.1/q1/R3/K2/P0、184項辞書、normal1 path/index0/元seed14788159701244282917を保持した。
一回の手動復旧で3event（order0:2、order2:1）、X/Y最大2compileを完了。
保存bit順/head/tail、s_R、c+s_R一回、low/high control、独立full-space vector、X/Y、compiled vectorがPASS。
両軸RZ1380/CX578/totaldepth834、全wall13.48秒、peak626888KiB。短いm0の実測に限る。

raw結果commit8502dd33a。raw sidecarは保全し、別のpublished sidecarで同commitのproofへ結び付けた。
C既存guardは実recordのm0X/m0Y metadataをPASS。law平均/modelの資格ではない。
旧run01とSTOPは変更せず、追加retry/seed/path/compile、mean/CI/fit/N/G/B1は0。
今回価格はDIRECT_COMPILED_FINITE_SAMPLE/FINITE_COHORT_ONLY/n_paths1/law_expectation=false。

[REPORT](../d_pauli_ld1_m0_pilot_results_20261114/RECOVERY_REPORT.md)／
[C HANDOFF](../d_pauli_ld1_m0_pilot_results_20261114/RECOVERY_HANDOFF.md)。
ユーザーが次の有限科学scopeを判断するまで、公開照合後mandatory STOP。
