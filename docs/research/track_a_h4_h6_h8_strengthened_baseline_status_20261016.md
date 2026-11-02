# H4/H6/H8：強化決定論的対照の取得状況と次の判断

2026-10-16 JST。完了済み保存結果の索引・比較整理であり、新しい科学レビュー、統制されたsize-scaling解析、新規計算の実行承認ではない。
[H8の新結果・全表・監査](track_a_h8_native_b0i_full_family_results_20261016.md)、[正式共通ノート](track_a_shared_research_status.md)、[既存統合科学レビューの固定原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)。

## 取得範囲と条件を分ける

|系列|固定条件と比較範囲|B0+I取得状況|固定根拠|
|---|---|---|---|
|H4 geometry|linear/STO-3G/legacy DF12、6距離.70/.80/.90/1.10/1.40/1.60Å、T=.8、prefix3/4/5/6/9、q1/2/4/8・δ=.8〜.1、ordinary control/structured Gaussian、32 paired費用、ε=.005〜.1の302点/距離|120構成・240軸。新234＋検証再利用6。全1,812点でB0+I397、B2 1,415が点最小、tie0|[結果98cdb35c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md)|
|H6 native|linear1Å/STO-3G/DF19、12 modes/sector400、T=.8、prefix5/10/15、q1/2/4/8・δ=.8〜.1、symmetric-directional、経験的u、PR確認union8/独立32|登録12構成、費用取得済み9構成18軸（旧pilotの全条件不適格1構成を含む）。新12＋旧pilot6、残6軸は不適格で未測定|[結果34efc914](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)|
|H8 native|linear1Å/STO-3G/DF28、16 modes/sector4900、T=.8、prefix7/14/21、q1/2/4/8・δ=.8〜.1、symmetric-directional、経験的u、PR確認union6/独立32|登録12構成、適格な8構成16軸を新規取得、残8軸は不適格で未測定|[今回結果・execution SOURCE54ed98c5](track_a_h8_native_b0i_full_family_results_20261016.md)|

native系のtol-only1e-8/cutoff0/weighted Hermitian projection/signed generation orderと、H4 legacy条件・control・数値政策の差を保持する。H4 native anchorの固定16panelをgeometry全domainの代用にしない。下表も分子サイズに対する統制scaling則ではない。

## H6/H8の主表示：P=0・factor1・RZ点費用

分母は各系の登録内で適格かつ取得済みのB0/B0+I/full S2/full S4最良。PRもそれぞれの確認集合内の点選択である。εはcomplex signal精度でありenergy精度ではない。

|signal ε|H6 PR点費用低減率|H8 PR点費用低減率|H8決定論的最良|
|---:|---:|---:|---|
|.05|29.35%|28.77%|B0+I p14/q1|
|.01|49.69%|52.72%|B0 p21/q4|
|.005|53.27%|52.73%|B0 p21/q4|
|.001|56.14%|56.12%|B0 p21/q8|
|.0005|50.37%|23.83%|full S4 p28/q4|

両系とも登録集合・保存標本のP=0点費用では全5精度でPRが低かった。厳しい.0005ではH8のfull S4が強く、低減率が小さくなる。一般的なPR優位や「高精度ほど有利」を認定しない。H4は粗い精度で強化対照が勝つgeometryがあり、native H6/H8の5点だけでH4の負の結果を消さない。

仮想準備費用Pを加えれば各系で決定論的な候補が下包絡へ入る。P=0だけの点優位をstate preparation込み・全P・energy/QPE/RPEの優位へ移送しない。H6/H8の経験的allowanceと母平均winner未保証、H4の同一標本選抜/推定、未観測希少event、旧compiler同一性の限界は保持する。

## 採択済み方針と未認可の次段階

既存10月15日レビューではnative B0+Iの各登録範囲を完成させ、その後H5中性ROHF pilot、H6固定δ=.2でT=.4/.8/1.6の限定比較、H5確認後にH7という順序を採択した。今回のH8結果はその実施証拠として整理し、Codexによる新しい科学的採択にはしない。

- 今回：H8結果・監査・比較表・共通ノートの更新と公開準備まで。追加計算、grant発行、原稿改訂は0。
- 次候補：H5中性doublet/ROHF共通軌道のinput/sector/native correctness・実資源pilot。計算実行は未認可。
- H6のT比較は独立した別scope。既存anchor費用を利用できても、新Tのsignal/trajectory/compileは別途認可。H7はH5の正しさ・実資源確認後。
- native対照により中心的な解釈が変わる、reference/allowanceの意味変更が必要、確認外PR候補を主要claimにする等の場合は原review§14.3へ戻す。通常の結果公開だけで同じ詳細科学reviewを繰り返さない。

H4/H6の元結果・R01〜R10原review・旧STOP・消費済みgrantを変更しない。新しい科学計算や詳細研究設計へ自動移行せず、公開準備後mandatory STOP。
