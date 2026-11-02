# G3 Phase A：保存eventからの有限law検査仕様

2026-10-09。GPT [G2 review §8](../../research/track_b_G2_scientific_review_20261009.md)の指示を実装する。
旧G2 commit b260189b020ab7dfabb16bf424f49a6efff40d75を基点とする独立branch。
本仕様は科学的RQ/target/baselineの変更ではなく、認可されたoptimizer/rounding/検査の技術設計。
新registered LP・旧run再実行・DF/分子等は0。科学的採否はGPT G3へ返す。

## 対象と公平な限定集合

distinct-basis controlled finite P3、p=(3/4,1/4)、σ=+1、x={1/8,1/4}。
既存21 columns/x・3 precisionを保持。ordinary/PTSC-K0/Aの63 pure profiles/xとJ1の81/x、計288。
各profileに同じproposal生成器・同じparameter集合を与える。
この有限比較集合は旧B2任意混合の最適値、全proposal・全precision混合の最小ではない。
pure-profile補題のfinite-confidence目的への完全性を自動主張しない。

zero費用がないaxisはcost-ISとcanonicalの混合η={0,1/2,1}。
zero費用のあるaxisはzero group内をcoefficient比例、positive group内をcost-ISとし、
zero massをcanonical massまたは{1/64,1/16,1/4,1/2,3/4}から選び、同じηを使う。
T/CX/1Qは別々のcandidate選択。実costの0を変更しない。
同じdyadic lawを重複計算しない。最大15,000 law、wall600s/CPU480s/AS512MiB/output32MiB。
その上限・入力・算術はscope JSONへ結果前固定する。

## 有限samplingと平均

各ideal係数α_i=γ_g norm_g p(i|g)をintervalで囲む。
group normを先にmidpointへ丸めてからexact IID shareを掛け、rational係数a_iとする。
これでa_i/p(i|g)はgroup内で厳密に一定。
proposal q_iを分母2^60のlargest remainderへ丸め、全q_i>0、Σq_i=1を要求。
corrected weight w_i=a_i/q_iはexact rational。有限60-bit uniform整数とcumulative countsでeventを選べる。
実際のrandom samplingやmeasurementは行わない。

Σq_i w_i U_i=Σa_i U_iはexact cancellation。
ideal α_iを使う数学上の平均は元P3と厳密一致する。デジタル係数a_iを使った実装平均は
intervalで認証する近似であり、無誤差のexact ideal meanと偽らない。
group shareを保持するのでsaved normalized columnsとtargetのdegree残差ξを独立に計算できる。
ξ≤1e-12、Σ|a_i−α_i|≤1e-25を確認。controlled phase/word/orderは既存IR signatureへ結び付ける。
source矩形/量子operatorを新たに評価しない。合成error上界は既存sourceを信頼する条件付き。

## 共通confidenceと資源

m2=Σq_i w_i²、L=max|w_i|、bias上界=coefficient L1 error+Σa_i·2δ_i。
s=1/200−bias>0、ell=ln(10560)の外向き上界。
n=ceil{ell(2m2/s²+(4/3)L/s)}とし、n≤10⁹/axis。
別verifierがq/weight/eventからmean、bias、m2、Lを計算し、
n s²−ell(2m2+(4/3)Ls)≥0をexact rationalで確認する。

two axesのT/CXは2n·EC、1Qはn(2EC+5)。workspaceは1 qubit beyond2 system。
statistical accuracyはsame finite P3。Taylorからexponentialへの誤差、QPE total costは外。
全288 profileへの新しいfamilywise実測保証ではなく、各taskの共通sufficient予測。
別queryのnonobjective capsは新設しない。各resource座標を保存し、明示capsのverifierもテストする。
旧RA-D0のq,y constant-weight certificate、query budget witnessと同じclassとは呼ばない。

## 条件付きPhase B

有限certificateを通るJ1がTまたは1Qで有限対照集合の最小をstrictに下回り、同集合に支配されない場合、
known return比較を行う技術的必要条件が成立したものとして記録する。
このgateは科学的materiality/new-method GOではなく、同targetの強い対照が判断を変え得るかの条件。
差が消える・technical failureならPhase Bを実行しない。
Phase Bへ進む場合は、reviewで認可された既知12 returned event条件だけを別のsource/scopeで固定し、
strict phase-preserving実装・費用・error・workspace・有限lawを同規則で比較する。
新angle grid/dictionary/precision追加、旧55k/111k登録LP、CTS/分子移送は行わない。
終了後に結果・source・実行回数を公開しmandatory STOP、科学判断をGPT G3へ返す。
