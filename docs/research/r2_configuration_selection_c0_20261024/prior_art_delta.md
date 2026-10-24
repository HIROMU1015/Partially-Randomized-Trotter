# 構成選択研究の差分：既存文献を再利用した限定調査

旧501b4232…の39文献索引と8a2fbe94…の10文献差分を再利用した。全件再調査ではない。今回は下記のsectionだけを一次本文で再確認し、旧R3が引用するerror proxy/MF選択の直接関連箇所を追加確認した。全証明・全実装の精査や詳細科学レビューではない。

|一次資料・固定版・確認箇所|既知の内容|本研究に残るかもしれない差分と限界|
|---|---|---|
|[Güntherら 2503.05647v2](https://arxiv.org/pdf/2503.05647v2)、App.A3 A32–A41、App.E E21–E23|一般次数PR、signed stage、RTE負担、反復込み資源モデル|高次PRそのものは既知。有限K・allocation・wrapperを安価な情報選択へ結ぶ有効性は別途必要|
|[Hagan–Wiebe 2206.06409v3](https://arxiv.org/html/2206.06409v3#S5)、§5.1 Def.7 Eqs.57–58、§5.2–5.3|高次outer/inner composite、partitionとquery/errorの交換関係|deterministic/random混合を新規性にしない。law/taskの差だけで新選択法とは言えない|
|[Kissら 2212.05952v2](https://arxiv.org/html/2212.05952v2#S3.SS3)、§3.3 Eqs.34–41、§4.2|個別実装cost、samplingとpartitionのcost-aware選択|固定法での安価なscreeningの情報量・取得費用は検査課題。ただしcost-awareという目的自体は既知|
|[Moralesら 2210.15817v3](https://arxiv.org/html/2210.15817v3#S4.SS2)、§IV.1–IV.2 Eqs.40–48|公式長・誤差定数・次数・T/epsilonによる比較とthreshold|本研究はsignal/shotとnative価格を含む。しかし誤差定数を安価に得る問題は残る。v3の異次数thresholdはEq.48、旧索引の範囲だけで補完しない|
|[Kanasugiら 2603.22778v2](https://arxiv.org/html/2603.22778v2#S2.SS2.SSS4)、II B4 Eqs.17–18|PRの単価・shotsを含むcost、partition最適化|DF/PR/RPEを使うことは差分でない。条件付きsingle-time Gとenergy/FT taskを混同しない|
|[Casaresら 2606.30741v1](https://arxiv.org/html/2606.30741v1#S4)、§IV Step1以降の構成guide、III.5|factorization、grouping、誤差とstep cost、randomization等を組み合わせるSPRINT設計|最も近い統合設計の一つ。DF prefixに適用しただけでは不足。今回の読取はguide・関連節で、全比較の優位性は未認定|
|[Mehendaleら 2312.13282v3](https://arxiv.org/html/2312.13282v3#S2)、§II Eqs.8–9、III.2|摂動誤差推定をpartitionとresource比較へ利用。Eq.8の正解ground stateをEq.9の近似状態へ置換し、本文ではCISDを使用|候補全signalの代わりにerror proxyを使う着想は既知。正解状態の利用はOracle、近似状態でも取得費用が必要。eigenvalue errorと本研究のsingle-time signal biasは同一でない|
|[Poianiら 2406.03033v1](https://arxiv.org/html/2406.03033v1#S2)、§2、§3.1 Thm.3.1、§4|異なる取得cost/fidelityでbest armを同定しcostを抑える。既知のfidelity bias制約を仮定|評価削減・regret・保留は新規でない。本資産には対応するknown bias envelopeがなく、同論文の保証をheuristic proxyへ転用不可|

P32（2608.07720v1）の次数・整数step・精度比較、高次PF/乱択の先行性は[前回比較](../prior_art_higher_pf_delta_20261024/papers_comparison.md)を参照した。本C0で同論文全体を読み直したとは扱わない。一般のerror proxy・MF最適化全分野を網羅したとの主張も行わない。

## 内部R3との重複が最優先の論点

[R3既存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a2fbe9461d9cc262326fa30f9778b295adeb3f8/docs/research/r3_prior_art_and_minimal_contract.md)は、ordered DF prefix、partial-RTE、安価な診断、abstain、regret、詳細評価数、未使用条件という組合せまで既に検討し停止していた。

したがって本研究に残る候補は、(a)その停止理由を変えるquantum固有の新関係、または(b)方法的新規性を求めない限定的な選択benchmark/否定的結果である。現C0で(a)は見つかっていない。(b)に独立研究として十分な価値があるかはGPT・ユーザー判断へ返す。

H1/H2は会計恒等式の拡張だけなら説明研究、H3は既知探索法の適用だけなら工学研究となる。予測モデル・新理論・世界初の構成最適化法の成立は認定しない。無理に追加sampling/compileで新規性不足を埋めようとしない。
