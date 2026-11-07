# Stage C7-3：独立q32価格検証の未採択案

2026-11-07 JST。**提案のみ。q32 sampling/回路生成/compile・新fit・確認的保証の実行許可ではない。** 今回のC7-2は[技術報告](REPORT.md)と人工検証で終了する。次の担当ユーザー/GPTが目的・仕様・標本数・価格幅/古典信頼度を決め、採択した固定範囲を別途指示した場合にだけCodexが実行する。承認証明書や新grant発行は不要。

## 1. 目的を分ける

1. **構造反例探索**：q32のpaired整数CX残差 `ΔCX−(208−224q)`、basis/cut/P0 interfaceに反例があるか。有限標本で残差0でも全support定理にはならない。
2. **価格の探索的移送**：凍結affineのpaired/絶対価格と新domainの有限価格を比較し、RZ/sizeやlane/depthの補助特徴を保存する。
3. **確認的な母平均価格精度**：必要な幅・古典失敗確率・law資格・範囲/分散の根拠と固定nを持つ別設計。1/2のpilot成功から自動認定しない。

推奨する最初の候補は1/2の**探索的n=8 whole-path、最大32 full-wrapper compile**。各path32 stepなので総occurrenceは256、過去q16 n16と同程度のevent生成総量にしつつ、長いwrapper数を減らす負荷案である。8本は統計的powerや母平均精度から算定された必要nではない。目的が3ならこのnを継承せず、下記の精度設計を先に行う。C6bの32本/C7-1の16本も必要nの根拠にはしない。追加標本を結果に応じて無断投入しない。

## 2. 実行前に固定する科学・law・SOURCE

同linear H4/1.00Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、S2/s1/r32/K0、h=δ=.1、元H厳密ground仮定・状態準備除外。packet/endpoint/lawはC7-2報告の既存identity。ground solve/DF fitは不要。

q=32、**complete step16直後のglobal cut**を価格前に固定する。whole ordered V=U31…U0=A B、B前16/A後16、gadget一つ。strong q−1=31境界、Reference内部q−2=30境界、cut損失1。tail共有、head relative fusion、analytic Gaussian位相、actual inverse、scalar/identity集約を両方式で維持する。

fresh whole-pathは全32 occurrenceを同じ固定lawから生成し、同pathを両方式/X/Yで共有する。過去cohortの巡回、短path反復、prefix使い回し、保存eventの連結標本へ代用しない。未使用master tuple/domainを採択後sampling前に固定・公開し、旧C6a/C6b/C7-1の既使用seedと非重複を照合する。**今回seed数値の生成・選定・samplingはしていない。** PRNG seed分離とideal joint-IIDを区別する。

SOURCE/PLAN、sample ID/occurrence順、cut、P0、precision、モデル/family/価格幅/n/STOPをsampling前に固定する。event rawは一回だけ保存してhashを価格評価前に束縛。canonical request/binding、C3 prepare_steps、C5 wrapper/_checked_price、B2/C6a CAR cache、Stage A/C2 postcheck、C6b fast state-action、C7-1のresource pipeline/統計を再利用する。旧q8/q16 guardは変更せず、必要なq32薄adapterだけを次の許可内で追加する。今回は実装しない。

## 3. モデル最小仕様案

|対象|sampling前に凍結するもの|q32評価と禁止する解釈|
|---|---|---|
|比較用affine|C6a既存method/axis/resource係数48と共分散、price scope、P0|q8/q16を新fit点にしない。既存predictorのq32使用許可を新段階で明示する|
|CX構造候補|`208−224q`、整数残差の定義、非零primitive/例外/同basis rule|残差0は観測一致、残差非零は保存。5%幅で整数反例を隠さない|
|counts補助|shell/primitive数、event support/sign、basis run、cut/端部/identity feature|今回未校正のfeatureを新係数でfitしない。全P0価格への代替登録をしない|
|depth補助|max-plus wire interface、CSWAP順、lane境界/transition、compiled critical path|shell＋max単体laneを恒等式としない。新DAGモデルの係数fitが必要なら別scope|

q32の凍結affine外挿の代数重みはq2/q4に対し(−14,15)。校正平均が独立なら係数由来予測分散は `196 Cov(mean_q2)+225 Cov(mean_q4)`、相関があればcross項も必要。method/resource/axis間共分散を保持する。**今回は保存数値へ代入したq32価格予測や共分散再計算をしていない。** 新段階で採択された仕様と既存関数で作成する。

paired12量（6資源×X/Y）を主family候補、方式別絶対24量を補助family候補、qubitsは9/17の幅制約として別記する。X/Yは同じwhole-pathの一つのvectorであり、n8をn16にしない。主指標の重み付けやRZ→T換算をしない。familyを増やすなら同時誤差配分も事前に定義する。

## 4. n・価格幅・古典的精度の設計

探索的なら既存価格幅`τ_j=1+0.05 C^S_{pred,j}`とclassical α=.05を**比較用の未採択候補**にできる。これはRPE成功確率やエネルギー精度の設定ではなく、実行前にユーザー/GPTが採否を決める。新モデルに都合のよい値へ事後変更しない。

一次の「凍結済み数値予測に条件付けた価格差診断」と、二次の「未知population affine lineの推定誤差を含む診断」はestimandが異なる。前者は新q32 meanのばらつき、後者はその分散と校正共分散/モデルbiasを明示して扱う。二重計上せず、一次だけ幅内の結果を二次保証へ読み替えない。

確認的なnを求める場合、各量の許容幅τをモデルbias余裕b、校正誤差e_cal、新標本誤差e_newへ事前配分し、`b+e_cal+e_new < τ`を必要条件にする。残余幅が非正なら標本を増やすだけでは解決せず、モデル設計へ戻る。概算`n ≳ z² σ²/e_new²`はpilot分散を用いる計画近似であり、小nの正規近似を有限標本保証にしない。

rigorousな同時平均誤差が必要なら、同law独立性の資格と事前の価格範囲[L_j,U_j]を得た上で、例えばHoeffding/union boundによる

\[
n\ge \max_j\frac{(U_j-L_j)^2}{2 e_{new,j}^2}\log\frac{2J}{\alpha_{new}}
\]

等の十分条件を候補にする。保存min/maxや観測SD0をsupport幅/確率1の根拠にしない。価格範囲のSOURCE上の有効な上界、calibration側の誤差資格、finite-bit/joint-law差が未取得なら確認的必要nは**未確定**。幅が保守的すぎる場合の別 concentration/事前固定sequential設計も採択前に決め、結果を見て固定nから切り替えない。回路価格のclassical αとRPEのαは別。

## 5. 最小実行・補助情報

採択されたnの全pathについて、raw独立CAR ordered積、AB=V、actual A†、vacuum ray/χ、scalar/identity/signed angle/normalizationとattenuationの会計、strong/control-V coherence、X/Y誘導observable/complex coherenceを先に確認する。元Hのground vectorは使わない。

各path raw PASS後だけstrong/Reference×X/Yの4 P0 wrapper。q16と同じP0/位相保持・component1e−10/observable2e−10/rtol0を候補とし、q32での残差を報告する。精度が足りない場合に閾値を緩めてPASSとせずSTOP。precision/source変更が必要なら保全後に別判断へ戻す。

postcheckはCAR/raw/P0、両方式X/Y、χとreference leakage、全256 system入力のstate-actionを再利用する。RZ/CX counts・filtered depths・total depth・size・peak qubitsを別に保存し、同一pathの方式/資源/axis covarianceを保つ。

最小補助featureは、basis run数/同異basis隣接、supportとsigned/zero angleの例外分類、head/tail・step・cut界面のnative表現、A/B raw命令境界、scalar/χ、CSWAP順とwire role、compiled gateごとのwire listとRZ/CX/total max-plus終端時刻、critical path操作分類。**追加の単体A/B/shell compileを標準で増やさない。** full raw/compiledから抽出できるfeatureを使い、単体compileが必要なら別途対象/目的/件数を明示する。

P0全体bridgeはC7-2で得られなかったため、full-wrapperを局所代理へ置換してcompile件数を削減しない。重複car/head/atom生成のcache、batched state-action、独立pathとraw PASS後compileの並列pipelineを使う。CPU/RAMは実空き資源で決め、過去worker数を上限にしない。旧正常job/共有環境/他userを変更しない。実行開始前の専用環境で必要なsynthetic/regressionだけを行う。

raw event/QPY/NPZは既存どおりlocal保全。公開はscalar結果/共分散/feature/hash索引/SOURCE/STOPで、機微rawや内部資源情報を含めない。SOURCE/input前後とremote SHA/tree/public bytesを照合する。

## 6. GO/STOPとq≥64

GOは採択済みSOURCE/law/domain/cut/model/n/family/width/precision、既存資産との適合、共有環境非干渉を確認した場合。correctness/SOURCE/compiler/位相/公平baseline不一致、重複実行、入力破損、範囲外処理が必要なら該当科学処理をSTOPし、partialを保全、無断retry/resumeなし。

価格候補の反例や幅外は有効な結果であり、correctness等が正常なら事前固定した診断まで完了する。標本除外、cut変更、価格幅/family/nの変更、C6a係数再fitで隠さない。全pathが揃わなければcomplete-only統計をPASSにしない。

q32で観測一致でも、q64〜4096の全域保証、全13-round総費用、actual共通N/RPE成功保証へ拡張しない。q≥64へは、モデルbias/校正誤差/新標本誤差の分離、CX bridgeまたは残差包絡、lane/interface移送条件、独立domainと必要n/precision、資源実行性の判断が必要。全support定理がなくても限定探索を行うかはユーザー/GPTが別途判断する。

**今回の次作業**：ユーザー/GPTが、探索的n8案か確認的精度設計を先に行うか、主要な評価目的・価格幅/古典信頼度・モデル仕様を選ぶ。Codexは採択済みの明示実行指示が来るまでq32その他の科学実行を開始しない。
