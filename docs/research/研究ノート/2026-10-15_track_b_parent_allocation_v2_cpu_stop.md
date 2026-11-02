# 2026-10-15：Parent allocation v2 one-shotのCPU技術停止

固定S2 `a874f443411e03ad4d9d329834f09073a64a6621` のauthorization-only直接子A2 `2f4886c9f0389311acbaa57724d8c2a428e51809`から一回起動。
known 3-qubit distinct-basis、p=(1/5,3/10,1/2)、x=5/7、finite P5、r1。
総CPU3600秒でouterがledgerをSIGTERM、最終TECHNICAL_INCONCLUSIVE。
新16合成・72 root診断・2562 binding・61 prefix law/rowを保存し、全resource mapは未完成。
148 read-only provenance checks PASS。source/contract/承認/marker/旧結果と親dirty worktreeを保持。

新key worker CPU合計約7.94秒。保存fragmentごとのguard/ディレクトリ走査を有力な瓶頸として整理したが、関数ごとのCPU寄与は未測定。
修正・上限緩和・再実行・prefix勝敗判定は行っていない。別sourceでの技術修正案と別承認の必要性だけを提示する。
[技術報告と原証拠](../../tracks/algorithm_codesign/parent_allocation_v2_results_and_gpt_handoff_20261015.md)。
科学的採否/新規性/次実験は未判断、retry0、mandatory STOP。
