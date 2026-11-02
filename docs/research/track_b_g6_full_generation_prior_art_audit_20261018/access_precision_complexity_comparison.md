# 同一target・access・精度・費用条件

2026-10-18 JST。既存M1〜M12、D1〜D5の条件を比較へ適用する。新しい保証・計算量定理・oracleを追加しない。

## 1. Targetと同値性の水準

共通targetは`M=P_m(-iσxR)`、`R=Σp_iQ_i`。mは有限奇数、pは明示された正の有理数、0<x≤1。x=0は既存identity経路の別扱い。一般命題のため分子・geometry・basis・DF rank・L_Dは非該当で、新しい物理条件は選定しない。

次を別々に判定する：同じHilbert空間のoperator／同じfinite mean／同じevent集合／同じproposal・weight／同じ有限precisionのbias・range／同じconfidence・費用。最初の一致だけから後ろの一致へ進まない。異なるlawで同じmeanを作れることも、アルゴリズム全体が同じという認定にはしない。

WANは1 microstepについてeven n≤m−1へ切断するとpairedな有限targetに揃える対応がある。r段の積をP_m一個と同一視しない。PTSCはS0=Iのzeroth-orderと有限cutoffを比較の入口とし、一般Trotter remainderをMに置換しない。Proposition 3のcutoff条件はs_c≥2なので、奇数cutoff s_c=mとの直接照合はm≥3に限定する。m=1の先頭pairはEuler恒等式として区別し、Proposition 3の適用域を広げたことにしない。CTSもcutoff・左右operator・normalizationを明示してから比較する。Zhao–Yuan §4.2は高次tailを取り込むmodified targetのままでは一致しない。

WAN原文のpositive-time `e^{+iĤt}`とG6の`e^{-iσxR}`、式のoperator順序とAlgorithm 2のappend順序は定義式に戻す。角度を絶対値だけにせず、raw eventの同一性を無断で主張しない。一般wordを並べ替えることも許さない。

## 2. 全方式に同等に認めるaccess層

| 層 | 共通に利用できる情報・操作 | 無料としないもの |
|---|---|---|
| A0 基本p/Q | 明示p表、label、Q²=I、signed時間、同じcontrolled Q/rotation条件、必要precision | p表のread、Qの物理実装、角度取得、state preparation/readout |
| A1 共通Green query | A0に加え、既存の切断Green再帰・指定wordの係数queryを**双方**に許す | O(Lm²)等のsetup、局所係数算術、root/bit処理。G6完成済みevent law全体を無料oracleにしない |
| A2 既存DF/native情報 | 取得済みfactor/basis/係数、同じPauli化やGaussian操作、合法な共有・streaming/cacheを双方に許す | DF/Pauli情報の新取得、未実装thin adapter、全word価格表、未確認native certificate |

非Pauli Packet N02–N04が支持するのは、既存Rの共役Z/ZZの代数とprimitiveのSOURCE。G6実接続・DF float→rational binding・native総費用は未確立である。演算子クラス自体は独立差としない。A2を採用する新しい実装・実験を今回行わない。

SPS/ISがprocess/observableを対象にすることだけで対照から除外しない。controlを含む拡張系とX/Y測定への対応を検討する余地はある。ただし、operator meanのscalar phaseを失わないdictionary・weight・取得法を確認する必要がある。「数学的に埋め込める」と「既存algorithmが同じ情報から局所生成する」は別である。

## 3. 比較経路のaccess・精度profile

この表のprofileとP1〜P12の各行を合わせて読む。各行の機械索引にはprofileも展開して記録する。

| Profile | 既知文献の入力・normalization・取得 | 有限precision・confidenceとの比較境界 | controlled/費用境界 |
|---|---|---|---|
| WAN | Pauli係数と回転、ordinaryの次数重みとIID label。全語辞書を要求しない | App.E.2にfinite truncation bias。G6のroot/dyadic相対誤差・実π保証は別 | Pauli Clifford実装を一般Qの無料実装にしない。共通角利用も対照に認める |
| AK | cyclic-group walkとGreen/resolvent。Z2への既存特殊化、形式次数切断 | Greenの既知式であり、event sampling・finite-bit estimator・confidenceを与える資料ではない | quantum回路providerは対象外。Green算術の費用は両側で課金 |
| MT | Pauliと可換/反可換関係、modified tail係数、LCU routine | tail変更の誤差と、G6の同じfinite Mへのdigital biasを混同しない | LCU/SELECT等のaccessとG6のcoherent単一eventは直接同価格でない |
| PTSC | Pauli係数、次数・label law、remainder formula | truncated LCU誤差の保証と、G6の同じ実dyadic保証は別 | zeroth-orderと一般PF補償を分離し、controlled位相を揃える |
| SCU | Pauli収集とnormalizer、layerを分けた生成。全高次表への固定はしない | truncated operatorと左右draw、有限bit π・c̃・bias/rangeを区別。掲載されていない丸め保証を補完しない | channel表現のleft/right費用とHadamard一つのcoherent operator費用を揃える |
| SPS | gate/process libraryと分解係数、L1、offline最適化 | process近似精度とmeasurement varianceの保証を、G6の係数bias・strict phaseへ無断移送しない | controlled拡張dictionaryの取得・位相保持・library生成費用が別に必要 |
| IS | 取得済みensemble、p/q、費用c。原理上は未正規化係数/実proposalの補正も同じweight恒等式 | Theorem2の既存bias保存は、新しいtarget係数近似の誤差を消さない。§IIの二次moment/漸近costと有限Bernstein Nを区別 | 費用付きproposalの正規化・queryは有料。error-detection後zeroと量子前rejectの実行費用を区別 |

profileの一次節・式とURLは[literature_scope.json](literature_scope.json)、具体行のlocatorは[逐次表](generation_pipeline_alignment.md)。

## 4. G6/G7の有限bitとU：既存保証の境界

実dyadic proposal πと近似event係数c̃に対し`W̃=c̃/π`、`πW̃=c̃`。この等式は既知IS型補正で、meanは一般にM̃である。p、acceptance、childの丸めを補正することと、root由来の係数近似を消すことは別である。

既存BIT/M9–M10は、各存在する確率因子の相対誤差ηと係数の相対誤差ρ、十分な固定bit数・正supportを条件に、bias≤ρB_o、range≤(1+ρ)B_o/(1−η)^(m+2)を与える。固定K/Hが任意入力で成功する保証でも、自動precision selectorでもない。symbolic tangentから物理角度・native列への接続も別条件である。

Uは当初G6から区別した後続の既存公開レビュー§7とG7 SOURCEにある。`U=d_empty+Σ_even l≥2 b_l r_l`のr_lは**raw語が既にreducedである確率**で、reduce後の長さ分布ではない。UはB_RAの上界で、samplingにB_RAを渡す経路ではない。G7はUとB_oの外向き区間hi、digital moment上界κB_oU、range、残余誤差幅を使う。

一般式のconfidenceとSOURCEのAPIを区別する。`G.budget`のαは1/320固定で、任意αの実装ではない。provider誤差・angle/native bias・係数bias・Taylor bias・PF bias・QPE/RPE統計誤差を別々に会計する。今回新しいNや上界の値は計算しない。

## 5. Moment、量子呼出し、古典費用

ideal G6のZ=B_RA/B_o、全trial出力X=B_o·1_accept·Yとすると、既存M7のE[X²]=B_oB_RAとrangeがある。Nは全trial数、期待量子実行回数はNZ。実際の費用はevent依存価格を含むので、二次momentだけから実行費用は決まらない。NZもhard capではない。

ISはnormalized p/qを記述するが、**ISという原理が必ず未知B_RAの取得を要求するとはしない**。未正規化c̃と実πが取得できればc̃/πという補正は直接対応する。G6で重要なのは、それらを取得する局所手順と、未知量を無料代入しない認証予算である。費用最適proposalを数学的に書けることと、同じaccessから安価にsampleできることも分離する。

| 費用の層 | 既存証拠で記録できる範囲 | 除外・未取得 |
|---|---|---|
| input/setup | p表read、切断F/G、既存O(Lm²)有理演算・O(Lm)係数 | DF acquisitionや全物理入力の費用は含まない |
| local query | word係数・parent/全child。既存O(m³+Lm²)等の限定算術count | native回路・全online selector時間のcountではない |
| bit | 既存の係数bit長・精度取得の多項式論証、isqrtと有理丸め | arithmetic操作をunit-costの実秒としない。新tight boundなし |
| proposal/reject | G7のraw draw、reduced判定、局所coin、child。失敗trialも古典費用 | `dyadic_index`は線形走査、roundingにsort。全生成O(m)実時間とはしない |
| controlled/provider | CPはexact CQ・実adjoint条件付きIR。外側controlとhelper | 一般physical providerの誤差・T/CX/depthを0にしない |
| angle/native | symbolic ratio、既存の条件付きnative誤差規約 | 未取得角度の合成、online cache、whole-circuit費用 |
| statistical/task | 全trial、weight、range、bias、confidence、accepted quantum cost | prep/readout、長時間PR/RPE全cost、実測shotは未取得 |

**構成的な保証の価値と量子総資源改善は別の判断である。** 今回どちらも採択せず、新しい性能数値・順位・下界分離を作らない。
