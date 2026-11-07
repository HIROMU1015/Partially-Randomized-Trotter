# 2026-11-07 Track D C7-3 q32 paired価格検証

確認済み。[結果](../track_d_c7_3_q32_20261107/RESULT.md)／[固定PLAN](../track_d_c7_3_q32_20261107/PLAN.md)／[承認済み引継ぎ](../track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)。

同H4/1Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2/S2/s1/r32/K0/h=.1、q32/cut16、独立8 whole-path。元H厳密ground仮定・準備費除外。sampling/raw SOURCE35bfd0423d08114b0a24107501ce545cdfb28f32、compile/post SOURCE94e94e8c951a16487c5b1bc1e30369d595626d31。

旧controllerだけを許可済み調整停止し、投入済みraw worker8件を自然完了までdrainした。旧KeyboardInterrupt/STOP_FAILUREと未完了の集約raw0/source_input_preserved=falseを保全。一方、全8 local raw PASS・旧compile0を確認して固定135 member hashを封印し、新SOURCEでraw8を再利用、sampling/raw再実行0、未着手32 P0/postのみを実施した。人工/回帰50件の実行前検証、追加高速化・handoff等を含む61件の切替前検証はPASS。

新SOURCEの実行コマンドは固定runner `scripts/run_track_d_c7_3_accelerated.py --execute` にPLAN所定のinput位置を与えたもの。full-wrapper compile32各一回、postcheck32/32・最終pairedχ8/8 PASS。SOURCE/input継承hashの前後照合一致。小演算子評価は公式Aer double/unitaryへ変更、Reference17qは既存double/statevector全256入力列の検査を維持。科学精度/P0/seed/cut/モデル変更0、閾値緩和/科学retry0。

全8観測pathでΔCX=−6960、候補208−224qの整数残差0。RZ差平均−1435、total depth差平均−51728、qubits+8。方式別絶対価格とpaired共分散を保存した。一次12量は探索的幅内、別population-lineはRZ/CX/total depth×XYの6量幅外。モデル再fitなし、確認的95%保証/全path公式/q≥64/actual N/全RPE優位は未取得。

停止済み結果の収録・報告は `scripts/collect_track_d_c7_3_q32.py` と `scripts/summarize_track_d_c7_3.py` で保存scalarを投影しただけで、回路/検証/統計を再計算していない。原本160 memberはローカル、公開は科学scalar/価格/共分散/features/hash/SOURCE/旧新STOP。公開SHA/tree/bytes照合後mandatory STOP。次の担当はユーザー/GPTが長qモデルの目的・構造bridge/追加検証/校正の必要性を判断する。Codexは別途許可後だけ担当し、自動継続しない。
