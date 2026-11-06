# 次の最小科学検証案：一層・一特徴の層別＋CV

2026-11-07 JST。**提案のみ・未採択・未実装・未実行。** 本書は実行許可ではない。
今回の指示が許可するのは静的調査・数式・文書作成だけ。以前の一般的な継続許可を、本書の新control計算や測定へ適用しない。

## 1. まず確かめる問い

「h=1でrotation fragment群の一つのcontrolを学び、全5層を維持した推定へ加えると、fresh評価で層内分散が下がる兆候があるか」を最初の問いにする。
最初から全層・全候補・保証付き総費用最適化へ拡張しない。h=0はpath固定、h=2–4はN1で希少という構造を利用し、最初の案はこれらのλ_hを**取得前から0に固定**する。学習失敗後のfallbackではない。

対象は保存H4/1.00 Å/STO-3G/legacy DF12、8qubit/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y、N1 744e1a0のactual Q/selected full、N2保証ground、認証済みexact engine。
B1-OLと層別＋CV（仮称SCV）だけを対象にし、現行A1/B0/B1-Oを追加しない。target・5層・positive supportを変更しない。

## 2. controlの固定案とmetadata-only事前段階

候補族は u_S=1{rotation fragment∈S}。S={fragment-6}は非定数性を証明できる例だが、少数trainで十分な出現頻度を持つとは確認していない。
最初の具体的な固定ルールとして、以下を提案する。

1. 保存LAW/catalogの同一性と、label→fragment対応を確認する。
2. 正質量を持つ保存fragment集合を固定canonical順に並べ、非空のproper subsetを有限列挙する。新しいfragment/分子/roundは探索しない。
3. exactなν_1(S)=Q(F(L0)∈S|L0≠a)について、1/8≤ν_1≤7/8を満たす群のうち|ν_1−1/2|最小を選ぶ。同点はcanonical bitmask順。
4. 条件を満たす群がなければ**測定前STOP**。別featureへの自動変更・新しいthreshold探索は行わない。
5. S、ν_1、Σ_1=ν_1(1−ν_1)、range[0,1]、全input hash、selection ruleを保存して、最初のtrain前に凍結する。

この選択は未知bit・旧752 data・state・Born・path信号を見ず、既知のlawだけを使う。既存5層の中をさらに取得層へ分割する操作ではない。
新しい平均/Σ/Sの数値取得・balance guardの成否は今回UNKNOWN。計算は次回の明示科学scopeに含める。群質量が適度でもbitとの共分散は0かもしれない。

実装SOURCEとこの有限ルール・入力hashを先に公開固定し、次回許可されたmetadata数値段階でS/νを得る。その出力identityをtrain前に固定する。出力を見てSOURCEや規則を選び直さない。追加の承認発行手続きは不要だが、今の文書を実行許可と扱わない。

## 3. 提案取得量：合計224 fresh physical shots

| 段階 | 一軸の件数 | X/Y合計 | 目的 |
|---|---:|---:|---|
| h=1 conditional train | 32 | 64 | scalar slopeとintercept、rank2/range確認 |
| 全5層の共通fresh評価 | (4,64,4,4,4)=80 | 160 | B1-OLとSCVのpaired診断 |
| 合計 | 112 | **224** | 一条件、一候補族、一campaign |

旧案の512 calibration/8192 production capは継承しない。allocation probeやcalibrationはこの初回案では行わず、quotaを最初から固定する。
32 trainは1係数に対して既存16/層より反復を確保する低予算提案で、十分な学習精度・powerを保証する数ではない。
64 evaluationを主なh=1へ置き、他の全正層も4以上保持する。固定quotaはNeyman/期待費用最適ではない。baselineはこの固定quota版のB1-OLであり、既存のprobe付き配分や最適層別法を上回ったと判定する設計ではない。

feature出現率q∈[1/8,7/8]なら、32 trainでfeatureが全同値となる確率は q^32+(1−q)^32。
これは正確なrankを必ず得られる保証ではなく、balanced特徴を事前に求める理由。実H4でのqや応答相関からshot数を逆算していない。

両軸で独立なh=1 trainを一回だけ行い、intercept付き既存OLSの手順で係数を凍結する。
提案guardはrank2、finite slope、D_1=2+|λ_1|≤4。h=0,2,3,4はλ=0・幅2。この新guard/shot数も未採択である。
rank不足・range違反時はSTOP。clip/refit/追加shots/feature変更で救済しない。

## 4. paired評価と独立性

同じfresh Q_h bit/metadataをB1-OLとSCVの両方の**評価だけ**に使う。追加量子測定なしに差を診断できる。
これは2方式の独立campaignを二つ実行したことにはならない。baselineとSCVの推定値は相関する。差の不確かさを求めるならpaired covarianceを含める必要がある。

- trainと評価はpath/bit stream・trial identityを分離する。
- X/Y、全cellのstreamは独立domainを固定する。候補選択はどのbitよりも先。
- 共通評価recordを二つの推定器が読むことは、明示したpaired設計の内部再利用である。学習への再投入はしない。
- 旧397、新B1339、752-shot pilotのbit/係数/word/cacheを一切混入しない。
- train2 domain（X/Y×h=1）と共通評価10 domain（X/Y×5cell）、各path/bitを分離する12 domain/24子streamを提案する。PCG64/SeedSequenceと独立uniform53-bitモデルを再利用し、新campaign/seed/spawn順・domain schemaを将来SOURCEへ固定する。今回は候補seedを設定・生成していない。
- 比較用pathは直接Q_h取得。同じfeature値を強制するためのpath選別や追加条件付けはしない。

## 5. 取得する指標と会計

凍結λで、各層のraw Bとresidual Zについて平均、不偏sample variance、all-equal/feature-zero/range、層別quotaを保存する。

$$
\widehat V_{\rm OL}=\sum_h p_h^2s_{B,h}^2/n_h,\qquad
\widehat V_{\rm SCV}=\sum_h p_h^2s_{Z,h}^2/n_h.
$$

h≠1は同じrecordsかつλ=0なので差に寄与しない。追加削減の診断の中心はh=1のpaired差。
inactive層のsample variance=0も母分散0とはしない。全targetの精度やscoreを保証する根拠には使えない。
active h=1でraw/residualがall-equal、学習のfeature出現が乏しい、軸間傾向が不一致などの場合は診断保留を許し、結論を得るための追加shotを足さない。
候補を実際に有望と分類する閾値・scientific GO/NO-GOは本書で採択しない。

量子会計を分ける。

| 会計 | 計上 |
|---|---|
| 実際のpaired campaign | train64＋共通評価160を一回ずつ課金 |
| B1-OL単独を同quotaで行う評価費の参照 | 共通評価160の全RZ。CV trainを無料と呼ばない |
| SCV全手法 | train64＋共通評価160の全RZ |
| 両手法の評価point score | 同じC_evalを使うVhat×C_eval。独立費用の和ではない |
| startup込み探索的score | それぞれに必要なstartupを足した費用とVhatの積。期待費用保証ではない |

一つの共通測定を実支出で二重課金しない一方、baselineの費用を0にもしない。古典feature table/平均・fit・cache/compile・engine・生成・wall/RSSは別記録。
full selected no-prep価格は既存Qiskit1.3.0、basis(rz,sx,x,cx)、opt2、seed17、全段階同profile、cache hitもfull課金。
旧pilot実現RZを新価格の予測・代替値にしない。新しいRZ総額/variance/N/期待利得は今回未計算。

## 6. 将来必要となるcall範囲と最小技術差分

以下は**将来許可する場合の上限提案**で、現在実行したcallではない。

- 固定LAW/catalogのbyte/identity確認、metadata-only群選択/ν/Σ作表を一回。CDF/preparationは再生成しない。
- 保存bundle decode一回、state solve/spectral/engine再認証0。
- train64＋共通評価160＝224 logical trial、exact engine/Born各224以下、selected full build448以下、compile224以下。
- path word64/trial＋別測定word1。総word上限14,560。全実pathの列挙・未取得pathの実compileは0。
- post-runは保存結果・metadata・全費用の整合確認だけ。科学retry/第2seed/追加campaign0。

既存sampler/facade/price provider/exact kernel/manager・journalを再利用し、新しいconditional control bank、cell別model、paired result/fee serializerだけが将来必要。
元freezeのrank4、旧METHODS、confidence_cellsの4方式定数をそのまま使えない。old sourceを編集せず独立SOURCEで最小adapterを用意する。
先に人工lawで、条件付き平均、h=0定数、層別中心化の不偏性、範囲、train/eval独立、paired会計、identity/STOPを検証する必要がある。今回その実装やtestを実行していない。

実行前の資源確認・持続manager・SOURCE/環境/入力固定と公開照合を維持する。worker数に恣意上限を置かず、前pilotの127 workers対128 trialの境界待ちも技術設計で考慮する。exact kernelをfloat/GPUへ置換しない。

## 7. STOPと次の判断

metadata群/平均/範囲・入力identityの不整合、balance不能、rank/range失敗、stream/word/bit/価格不一致、非干渉不成立はSTOP。
保存bitは価格処理より先にfsync。取得途中の費用や結果をUNKNOWNのまま保全し、再測定で埋めない。
完了/中断後は公開照合してmandatory STOP。新control正式採択・224-shot条件/seed/stream・実装/人工検証・科学実行の範囲は、今後の具体的指示で確定する。本書だけで起動しない。

次の保証付き段階は、追加削減が有限診断で検討に値するかを判断した後に、独立cal/production、ε/α、range、返却率・全期待費用を別に設計する。小pilotのsample varianceやpaired point scoreを十分Nへそのまま流し込まない。
