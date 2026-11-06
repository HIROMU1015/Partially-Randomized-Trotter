# C6b独立q8 holdoutへの提案（未採択・未実行）

これはC6aで許可された技術的な次段階案であり、q8実行許可・長qモデル資格・RPE精度/成功確率の変更ではない。

- 固定候補：C6aの同一endpoint/law/P0/strong融合/half-step cut政策を維持し、予約済み未使用domainからfresh q8 whole-pathを生成する。C6aのcalibration標本や旧固定q8 fixtureをholdoutへ混ぜない。係数・閾値・標本数はholdout前に固定し、結果を見て調整しない。
- 資源：6加算価格（RZ/CX count、各depth、total depth、size）を両方式・X/Yで扱い、paired差も併記する。主資源を今回勝手に選ばない。qubitsは加算せず、幅9/17と利用可能幅の別条件にする。
- 提案する価格幅：各方式は絶対1 compiled単位＋calibrationの探索的q8予測規模の5%。paired差の幅は差そのものを分母にせず、symmetricの同資源予測規模の5%＋1単位にする。RZとCXを換算せず、depthはDAG depth。幅はRPEのenergy誤差幅と別の古典的価格許容差。5%は校正pilot後にholdout前の案として示す設計判断であり、計算済みの保証ではない。採択はユーザー／GPT。
- 古典信頼度案：family 36（6資源×2axis×2方式＋6資源×2axisのpaired差）、family alpha=.05をBonferroni配分した名目的二側normal計画。X/Yが同価格でもnを2倍にしない。n16でnormal近似・finite-bit IID距離・tail supportは保証されていないため、これを有効な同時CIと呼ばない。厳密な信頼度が必要ならsupport/tail/IID/実装law距離等の追加根拠か、別の明示された統計契約が必要。
- calibration平均の共有依存：q2/q4 domainの擬似独立を条件とすると、affine q8予測は3*mean4−2*mean2、共有校正共分散は(9*S4+4*S2)/16。方法間・axis間・資源間のSを残し、各roundへ独立に新fitしたと扱わない。paired差ではS_RR+S_SS−2*S_RSを使う。共有固定input/codeの不確かさやモデルbiasはこの式に含まれない。
- 必要数の案：未知q8 SDを暫定的に2*SD4と置いた条件付きplanningのみ。zを上記familyのnormal quantile、vcalを校正分散、tauを固定幅とすると、n >= z²*s8_plan²/(tau²−z²*vcal)。分母<=0ならholdout数だけを増やしてもその幅を満たす計画は作れない。分散transport自体は未検証であり、得られる数を保証数とはしない。具体数・floor条件はC6aの完全32標本の統計取得後に示す。pilot規模の追加確認としてn>=32を候補とするが、32を十分数と認定しない。
- 将来の判定案：事前固定したモデルと幅に対し、holdout meanと探索的予測の差、および校正＋holdout統計の不確かさを別々に報告する。名目的な差の幅を使うなら |mean8−prediction8| + z*sqrt(s8²/n+vcal) <= tau を全familyで確認する案。全wrapperのcorrectness PASSが前提で、raw/phase/law/baseline/compile不成立でSTOP。モデルbiasや未知tailをnormal近似が消すとは主張しない。モデル不適合は失敗として保存し、seed/cut/fit/閾値の事後変更をしない。

q2/q4の2平均で2係数をfitするため校正残差自由度は0。native boundary、lane critical path、full-wrapper compilerの非加法性は未分離であり、holdoutは1つのq8 domainを確認するにとどまる。q>8、13-round期待価格、Nとの乗算、RPE総費用へは自動移送しない。

## 完全pilotからの条件付きplanning

計36量のfull値は[提案JSON](../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/C6b_planning_proposal.json)。下表はX軸の代表で、Yもfamilyに含める。nは保証ではなく、sigma8=2*SD4とnormal近似を置いた条件式の結果。floor blockerに対してnを捏造しない。

|資源|量|提案幅tau|校正だけの名目的halfwidth|条件式n（minimum32）|
|---|---|---:|---:|---:|
|rz_count|symmetric_directional|2068.05|2416.81|floor不成立|
|rz_count|reference|2067.19|2416.48|floor不成立|
|rz_count|paired_delta|2068.05|7.7212|32|
|cx_count|symmetric_directional|794.319|744.98|41|
|cx_count|reference|715.119|744.98|floor不成立|
|cx_count|paired_delta|794.319|0|32|
|rz_depth|symmetric_directional|499.778|598.19|floor不成立|
|rz_depth|reference|247.481|359.658|floor不成立|
|rz_depth|paired_delta|499.778|395.26|32|
|cx_depth|symmetric_directional|388.859|326.283|32|
|cx_depth|reference|153.588|194.779|floor不成立|
|cx_depth|paired_delta|388.859|213.796|32|
|total_depth|symmetric_directional|1163.75|1311.98|floor不成立|
|total_depth|reference|546.969|783.643|floor不成立|
|total_depth|paired_delta|1163.75|866.451|32|
|circuit_size|symmetric_directional|4222.99|4868.97|floor不成立|
|circuit_size|reference|4152.53|4868.67|floor不成立|
|circuit_size|paired_delta|4222.99|7.7212|32|

36量中floor不成立は20量。保証された必要数はnull。古典alphaは価格モデル用の未採択案であり、B4-2/RPEの成功確率条件を変更しない。実q8データは0、確認的PASSなし。

幅5%の技術的根拠は、今回のCX paired差の平均（q2 −240、q4 −688）を、それぞれのsymmetric平均4296.125、8152.875と比較する際の粗い分解能候補である。より大きいdepth差とRZ増加も個別評価する。目標の採択・十分性の認定ではない。CX差の標本SD=0をlaw/q8分散0へ外挿せず、条件式n32を保証数としない。今回の20/36量のfloor不成立は、追加holdoutだけではこの名目的幅を達成できないという計画上の限界であり、Reference correctnessの失敗ではない。
