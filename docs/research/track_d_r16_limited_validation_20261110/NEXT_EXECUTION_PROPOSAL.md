# 次に必要な判断と最小作業

今回の対象はLinear H4／1.00Å／STO-3G／Nα=Nβ2、元H exact ground仮定・準備費除外、number-CDF8／元順LD2／ordered partial S2/s1/r16/K0／h=.1。新A Q、Closed P3、α最適化を混ぜていない。

S1の13round条件付き十分条件は成立した。r16ではr32に比べて減衰・RTE誤差が増え、特に長roundの必要Nが増える。したがって、局所的な回路価格低下だけでr16のPR＋RPE総費用優位は決まらない。今回のq4単一path内のStrong／Reference差と、別のr32 pathとの参考比較を区別する。

1. **ユーザー／GPT：** 本REPORTを受け、r16の期待価格校正へ進む価値があるか判断する。主資源指標（CX、RZ、depth、幅、あるいは明示的重み）をどう扱うかも研究判断であり、Codexは正式採択しない。
2. **Codex（新しい明示的実行許可がある場合のみ）：** 対象law・q・標本数・seed/cut・precision・compiler・価格family・STOP条件を事前固定し、必要最小限のr16 paired価格校正を行う。今回の単一pathを母平均とせず、X/Yを独立標本と数えない。既存Strong最適化とReferenceの追加register／CSWAP／cut損失を含むfull-wrapperで比較する。具体的な追加標本数やqは今回採択していない。
3. **Codex（さらに範囲が承認された後）：** r16固有の期待価格・長q移送資格が得られた場合に限り、今回の固定Nと接続する。旧r32のA/F/Sモデルをr16とラベル替えせず、全13round総費用更新には別の承認と十分な価格証拠が必要。

一方、研究速度を優先してr32の既存A/F/S条件付き評価を参照軸として維持する方針も選べる。r16の追加校正コストと、長round N増加を埋める価格改善の見込みを比較してから決める。新しいP0互換compilerやReferenceアルゴリズムの開発を先行させる必要はない。

S1の限界は数値norm＋engineering guard、ideal IID/exact-native、元H exact ground仮定である。PCG64 joint-law保証、all-law compile／合成誤差、routing／実機noise、fault-tolerant T価格は未資格。topology-free depthは実行時間ではなく、RZはT countではない。

本書は次段階の提案だけで、追加sampling・compile・fit・RPE成功確率評価・総価格計算を開始しない。公開照合後はmandatory STOP、次の担当はユーザー／GPT。
