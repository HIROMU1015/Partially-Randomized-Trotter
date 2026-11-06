# C3 run02：中断時点の保存結果

SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535`。[run02 PLAN](RUN02_PLAN.md)の同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/LD2/S2/s1/r32/K0/h=δ=.1、q4=保存00..03/cut2、q8=00..07/cut4を使用した。元H厳密ground仮定・準備費除外、全256 system入力列による状態非依存検査である。

両qのraw各22 check＋paired X/YはPASSし、8 P0 wrapperのcompileが各1回完了した。q4の両方式X/Yとq8 symmetric X/Y・Reference Xの7 postcheckはPASSした。最後のq8 Reference Yについてはcompiled QPYとcompile PASS markerがあるが、observable/checkは未生成で、最終paired/phase/SOURCE-input照合・result/STOPも未生成のまま該当processを確認できなくなった。終了原因は特定していない。correctness不一致の記録はなく、全8本完了とも扱わない。

中断時の61科学ファイルは[固定入力一覧](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run02_interrupted_inputs.json)にbytes/SHA256を保存した。旧run01の失敗・STOP、元SOURCE・入力、run02の全保存ファイルは変更しない。run02を再起動せず、利用者の修正指示に従う[補完PLAN](COMPLETION_PLAN.md)で最後の保存回路postcheckと最終照合だけを別出力へ実施する。

|固定fixture・方式（各X/Y、q8 ReferenceはXのみ検証済み）|RZ|CX|RZ depth|CX depth|total depth|qubits|size|
|---|---:|---:|---:|---:|---:|---:|---:|
|q4 symmetric|19867|7686|4746|3744|11116|9|40572|
|q4 Reference|20092|6998|2516|1570|5583|17|40301|
|q8 symmetric|39610|15360|9492|7515|22259|9|81015|
|q8 Reference X|39602|13776|4531|2870|10093|17|79615|

この表は中断時の保存値であり、最終の全wrapper判定は[補完結果](COMPLETION_RESULT.md)を参照する。新sampling・ground solve・DF refit・N評価は0。固定fixtureから全law期待価格・q単独効果・全RPE費用の優位性を主張しない。
