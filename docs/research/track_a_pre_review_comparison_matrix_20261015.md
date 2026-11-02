# H4/H6/H8 比較条件と証拠matrix（2026-10-15）

全て既報値・保存metadataの転記であり新再会計ではない。commit/blob/bytes/SHAは[固定索引](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/fixed_reference_index_v1.json)、今回prototype SOURCEは`3c513d1ad52ce99f0d2df7d8a300e4d58d15a553`。[入口](track_a_pre_review_technical_closure_20261015.md)に実装・検証・禁止範囲をまとめた。

## 条件

|系列|H/state/DF条件|PF・prefix/q/r/K|control/Gaussian・numerical allowance|precision|
|---|---|---|---|---|
|H4 native fixed16|1Å/STO-3G/T=.8、8mode α2β2/sector36、legacy explicit DF12、固定H/state|B1 full12 S2/S4×q1,2,4,8。B2 p3/S2、q1,2,4 r4、q8 r2、K2/4|symmetric_directional、native Gaussian/number-phase。EMPIRICAL u＋logB margin、u/headroom≤.01|旧.05/.01/.005/.001＋D0 .0005、factor1/10|
|H6 exploration92|1Å/STO-3G/T=.8、12mode α3β3/sector400、tol1e-8/cutoff0 signed generation-weighted Hermitian DF19|B0 p5/10/15 S2、B1 full19 S2/S4、B2 p5/10/15 r1/2/4 K2/4、全q1/2/4/8|同native control family、spin-sector/spectral/forward allowanceはEMPIRICAL|同上|
|H6 confirmation union8|探索と同一入力・科学scope|事前選抜B2 union8、det20 registry（費用16）|same compiler/circuit binding、独立cost32。uはsignal側の旧保存値|同上、.0005/metric/Pでunion外候補を確認済みにしない|
|H8 exploration92|1Å/STO-3G/T=.8、16mode α4β4/sector4900、tol1e-8/cutoff0 signed generation-weighted Hermitian DF28|B0 p7/14/21 S2、B1 full28 S2/S4、B2 p7/14/21 r1/2/4 K2/4、全q1/2/4/8|symmetric_directional/native Gaussian。independent occupation CSR/expm、EMPIRICAL u。数値algorithmはH6と異なる|旧4ε/factor1/10。.0005は未再会計|
|H8 confirmation union6|探索と同一入力・科学scope|p7/r4、q2/4/8各K2/4、det20 registry（費用16）|旧signal/uをreuse、独立cost32。ordinary anchorはprimary cost外|旧4ε/factor1/10、D0新ε/metric/Pは準備のみ|
|H4 geometry＋I（別条件）|.70/.80/.90/1.10/1.40/1.60Å。**1Åなし**。STO-3G/T=.8/DF12、距離別H/state|元各218候補、LD0/3/4/5/6/9/12、q1/2/4/8、r1/2/4/8、K2/4。Iはp3/4/5/6/9×4q×6距離=120構成240軸|ordinary/structured Gaussian cost系列。旧headroom=ε/√2−bias、native D0のu-aware規則とは別|302点ε=.005〜.1（.05含む）×6距離、1812比較点|

Native固定compilerはQiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし、hash1239304c…を参照する。オプション一致と、runtime/library/profile又は旧hostとのoutput bit一致は別。prototype人工検査/D0準備Python3.12.3とH8 production profileも同一とみなさない。負Yoshida時間、全global phase/axis/control、signed coefficientsの省略は許可していない。

## 選抜・対照・費用証拠

|系列|選抜/確認|決定論対照|六metric/shot/P|rare情報と比較限界|
|---|---|---|---|---|
|H4 native16|固定panelのB2 n32。広い探索92とは異なる|B1 8 n1のみ。B0/native I未登録|D0六metric/paired SE、.0005、fixed pair＋envelopeあり、P=0基準|Taylor詳細は旧receipt。本作業でtrajectory別の再監査/新集計なし（NOT_REAUDITED）。母平均tail boundはなし。固定候補以外の最良PR/最良detを認定しない|
|H6 exploration92|B2各n8、旧4εのRZ上位和集合を選抜|B0 12＋B1 8、B0 p5の4qは全旧ε不適格で未取得。I未取得|D0六metric/paired SE/P。CX depth .0005最小p5q8r4K2はunion8外|K4 order4は広い探索で1回。別候補へlayer費用を移植しない|
|H6 union8|B2各独立n32、探索n8をpoolしない。確認内の事後点最良|同det n1をreuse。I未取得|D0六metric/P。.0005の六metric点最良p5q8r2K2|既済conditional q8r2K2 s2 pilot8/16は局所補助証拠。全support boundなし|
|H8 exploration92|88適格構成、16det＋72PR×8=592group|B0 p7の4qは全旧ε不適格/未取得。I未取得|1572wrapperの全六metric保存。primary探索1184＋確認384、ordinary4。旧map/P固定pairあり|探索5040 event：0/2/4次=5024/15/1|
|H8 union6|旧各ε RZ上位2のunion6を確認前固定、6×32=192group。確認内の点最良は事後選択|det16保存をreuse、I未取得|旧map720行、paired SEあり。D0 .0005・全metric選択・P envelopeは未実行|3584 event：3580/4/0。q4K4/q8K4確認で全event0、未観測高次の母平均費用はUNKNOWN|
|H4 geometry＋I|PR/B3各n32を選抜/推定で同時使用。Iはdet n1|元B0/B1＋実測I120構成。240軸の六metricΔ0はこのordinary系列のみ|RZ＋paired SE、六metric axis means保存。他metric paired covarianceは公開projectionにNOT_SAVEDの場合あり。P=0、旧u-aware D0の対象外|event次数NOT_SAVED、172候補のnormalization丸め。B=1表示から未観測確率0を推定しない|

MISSING costとINELIGIBLE candidateは別。native Iは未取得であってcost0/旧B0同値ではない。H4 native16にB0がないことから「最良deterministic family全体に対するPR削減」を主張しない。

## 既報点比較の読み方

|既報domain|比較対象とP|既報観測（再計算していない）|
|---|---|---|
|H4 native .0005/factor1|登録full S2/S4の点最小、P=0|PR/full PF .429664、約57.03%低い。B0なし|
|H6 exploration .0005/factor1|登録full S2/S4の点最小、P=0|PR/full PF .497374、約50.26%低い。n8探索値|
|H6 union8 .0005/factor1|同full PF、P=0|PR/full PF .496333、約50.37%低い。独立n32の有限union|
|H8 union6 旧.05/.01/.005/.001|登録full S2/S4及びB0の各点最小、P=0|full PF比54.29–65.76%、B0比39.48–56.12%低い。.0005の値はUNKNOWN|
|H4 geometry 全302ε×6距離|旧B0/B1/B2/B3＋実測I、P=0|B0+I397/B2 1415点。旧集合のみ1812点B2最小と区別する|

これらは異なる候補集合・DF policy・精度domain・標本/選抜・numerical guardに条件付けた観測。percentを並べてもサイズ依存性の証拠に昇格しない。Pは共通状態準備のhypothetical cost/shot、実state-prep回路を取得したものではない。energy/chemical accuracy/QPE/RPE/noise/FT overheadを含むend-to-end費用ではない。厳密な母平均winnerや同時保証なし。

## SOURCE/result/入力の参照

|系列|科学・実行SOURCE|固定result|
|---|---|---|
|H4 native|217c971dd537b1cc7fbc13742bc0f6c263ff7f45|0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41|
|H6探索|科学5a903aea2492c2d211c1e15e7737df0d50694c12、実行78a39051d8b6d41c33d592ecfe172f8824adfaeb|df99d6ee28942df8d12aac3c4db7cd202df55d77|
|H6確認|87b6b9f46b7e8ba9abe132ee21478c114b6f46f7（旧確認SOURCEをlineageに保持）|321cd8c3066194fe98741cbebbcdba1515d1deaa|
|H8|c31ef424175c02379eafebe8fc6c7548c1d9aabe、seed cb8a56c6b2a772827275f9200b02638030f33f72|681ec23c93fdd1090d325908f4b155a79115afdf|
|H4 geometry|d0726e946f89f8c30a08d383fb8ff48b363629ec、dataset5aa6c3685afb9c3afcfbdc36f80244da0a7fe525|7c4483378047b57f514c47dad98905e3f2a9a756|
|H4 geometry I|924b779c2bc6f6e844d6283c355f91c73aed6445|98cdb35cd08469714f20d6cc3b982d9c80908f46|

H4/H6の全H/state/input hashは[固定D0 input inventory](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_input_inventory_v1.json)とcontract/provenanceを参照する。H8はH `1a5ebefa18fc1c9f01ad644b81d13f49c7bf944bccfe3837d459c35d3325dbe2`、state `50a8fcb4cd29d60349db8e8fe728cb609eaaa903d9531833ef63e2711fa363fb`、inputNPZ byte SHA `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`。今回NPZ未読、旧input receiptを固定参照している。
