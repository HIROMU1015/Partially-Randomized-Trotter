# 2026-10-06 Track B SP-1後GPT reviewの仕様化

SP-1 result `9d2bb1fa439748b02084bd9fbc9b10a705328f8a`のmandatory STOP後、利用者からGPT reviewと研究計画を受領。
旧result/分類/markerは変更せず、actual finite RTEではないweight1符号coin・D/R confound・十分shots/加法費用のscopeを保持。
四maskの限定negativeをselective placement一般の不可能性や二層RTE×PAIの失敗へ広げない。

GPTの次RQ案は有限精度coherent-signalのblock合成・資源設計。
finite-RTE補正後平均Mのphase-aware直接合成は未実証候補。一般joint/first-moment/LCU/ISの新規性は採択しない。
独立branch/worktree `track-b-post-sp1-block-synthesis-design-20261006`、baseは上記SP-1 result。
添付reviewとrootの計画一文書だけをbyte-preservingでinputsへ保存し、identityをmanifestに記録。
root/A/旧B worktreeには書き込まない。

[数学・実装仕様の入口](../../tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)、
[claim/強い対照表](../../tracks/algorithm_codesign/block_synthesis_claim_and_baseline_matrix_v1.md)、
[小型pilot未承認案](../../tracks/algorithm_codesign/block_synthesis_small_pilot_proposal_v1.md)を作成。
TP channel/first moment/unitary atomを分離し、negative time/phase/finite normalizationと接続normを保持する。
Pauli-LCU、task-tuned ordinary、positive mixture、同辞書standard LCUを対照へ戻す。
2-qubit三family×二係数比×二時間の12 target、有限辞書、precision/capsをレビュー用の案として具体化した。
値の採用・accuracy配分・solver・primary/materiality・authorizationは未固定。

current rte.pyは固定git blobのtextとして参照し、関数を実行しない。
一次文献の該当定義・節だけを確認し、全証明や引用網の不存在証明としない。
新science/実装/tests/matrix/solver/library/synthesis/trajectory/compile/GPU/NPZ/DFは0。
BF/BM closures、SP-0.5/SP-1、Aの原稿/source/statusを保持。
必要資料をcommit/pushして利用者/GPT reviewへ戻しmandatory STOP。追加実行は未認可。
