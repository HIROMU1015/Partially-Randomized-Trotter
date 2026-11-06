# V5 depth保証の限定補強：scope

2026-11-07 JST、SERVER 2。基準 `285b77023542c3b39caea767e2a937dffa046a3b`。独立branch/worktree `codex/pr-reference-depth-v5-reinforcement-20261107` / `pr-reference-depth-v5-reinforcement-20261107`、出力 `artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/`。

Linear H4 1Å/STO-3G/DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r、energy精度1.5936e-4 Ha、全失敗率.05、phase cap .4rad、Qiskit1.3.0/compiler A/RZ-SX-X-CX/全結合、B′4-CX RZZ、reference既存F/G・actual inverse・vacuum P・8-CX CSWAP、D融合・basis共有を固定する。shots8598/8606、qubits9/17、既存CX/RZ・誤差・成功条件は不変更。

Stage Aは、既存builderのD0〜3半step(.01)とD0 full-step(.02)の**source回路をinverseしてから**compiler Aを適用する。正方向保存nativeのadjointを代用品にしない。5種類×4保存reference contextの20 local inverse recordを、barrierで区切った4 batchにまとめてcompileする。正方向Dとbasis/frameは保存結果を再利用し、再compileしない。source反転・符号・basis inverse閉包・q1/融合後の多重度も確認する。

Stage BはB′全4 contextの保存D/frameを調べる。下界に用いたrecordがX-freeなら、その非X依存鎖とCX fence、SOURCE初段pass、最終CX→X処理順序を結びつける。pre-last内部Xが未保存であることをX=0の観測へ置き換えない。必要ならXを全て捨てた保護部分のδを用いる。実際のCX削除後にXが消え得る人工例で、一般的なX保護の主張を否定する。新しい物理trajectory・whole compileは作らない。

Stage A/Bで安全な境界を得た場合のみStage Cへ進む。維持なら保存式・値との対応照合だけを行い、18round総計を再計算しない。修正なら変更local boundだけを差し替え、既存の有理確率・shot・lane penaltyを再利用する。reference下界は正方向G laneだけでも支えられることを区別する。

指定名 `PR_reference_V5_depth_independent_review_2026-11-07.md` は基準treeと既知のローカルpathで未提供・未確認。添付されたCodex指示の2論点を対象とし、独立レビュー本文を読了したとは扱わない。今回の検査も著者の技術確認であり、新しい独立科学レビューではない。

実行前の有限対象は `PREDECLARED.json` に保存する。科学条件/compiler/gadget変更、安全な境界の不成立、又は限定scope完了でmandatory STOP。旧SOURCE/raw/結果/grant/ledger/marker/STOPは変更せず、新探索・最適化・T合成・巨大18round compileへ進まない。
