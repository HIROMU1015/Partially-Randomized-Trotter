# 条件別充足判定・段階C最小計画（未実行・未承認）

2026-10-21 JST。全8条件について、保存点・層別経験値・未知費用条件付き境界まで充足。全support母平均/compiled上界の保証は8条件とも未取得。数学的上界と費用順位の正式採択は行わない。

|条件|Bで扱えた範囲と残る不確かさ|Cを必要と判断する場合の最小候補|
|---|---|---|
|H2 p1q1r2K4|S0 n79/single n1、multi/S4未知。single SEはnull。未知混合境界は保存感度表参照|同じp1q1でsingle精度補強またはmulti/S4の必要範囲。全support保証には別推論が必要|
|H2 p1q4r2K4|S0 n12のみ。single/multi/S4未知、PR方向は境界依存|同じp1q4のsingleを中心に必要な層を選ぶ。q1標本や旧Aをpoolしない|
|H4 DF10 p3q1r4K2|S0 n1071/single37、multi未知、S4支持外。観測mean固定ならPF方向は非負未知費用で維持|今回の限定問いにはC不要。母平均保証を要求する場合だけmulti/component情報を追加検討|
|H4 DF10 p2q8r2K4|S0 n71/single7、multi/S4未知、境界依存|DF10 p2q8の必要層だけ。legacy DF12 p3の費用を転用しない|
|H6 p4q2r4K2|S0 n38/single10、multi未知、S4支持外。条件付きPF方向維持|今回の限定問いにはC不要。p4のcomponent/層推論の要否は別判断|
|H6 p4q8r4K2|S0 n103/single1、multi未知。singleのSEなし。境界はその経験平均固定に条件付く|p4q8r4のsingle精度補強と必要なmulti。旧p5q8r2費用を移送しない|
|H8 p8q2r4K4|S0 n48のみ。未観測single/multi/S4。今回正境界はS0費用の約5.22倍|同じp8q2のS2_SINGLEから必要範囲を検討。multi/S4は別層として保持|
|H8 p7q16r4K4|S0 n48のみ。未観測single/multi/S4。正境界はS0費用の約4.88倍|同じp7q16のS2_SINGLEから必要範囲を検討。旧q4/q8をq倍して転用しない|

優先候補はH8の2条件。続いてH6 tightのn1問題とH2 tightの全希少層欠測を検討し、必要な保証に応じてユーザーが対象を選ぶ。これは新しいPR候補の再選択・再ランキングではなく、今回固定8条件に対する次の情報取得候補である。nや受入精度を今回の結果から自動設定しない。

正しい条件付きlaw、次数countと一様位置、元component確率、同一event両軸full-wrapper、元scalar/control/preparation/analytic shots/compilerを保つ新契約が必要。S2_SINGLEならN2=1,N4=0、S2_MULTIならN2>=2,N4=0、S4_PRESENTならN4>=1,N2任意。multiをN2=2、S4をN4=1へ勝手に制限しない。各層本数・科学call・資源・受入/STOP・seed namespace・新SOURCE/grantは別途承認する。

既存pilotはH6旧p5/q8/r2、H8旧p7/q4,q8/r4、H4 legacy DF12 p3/q4,q8。取得済みのpilotを重複せず、今回Pに移送しない。sampleの最大値・経験SE・n8 pilotを全support上界にしない。

今回のλ_R復元は保存event scalar照合で、全tail norm再認証ではない。canonical lawと実PRNG bit law、compiler閉包、観測層の母平均推論も残る。これらのD経路の実行を今回Bから自動承認しない。**mandatory STOP**。次の担当：ユーザーが主張・必要保証・Cの範囲を判断。Codexは新指示まで停止。
