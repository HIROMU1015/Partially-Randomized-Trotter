# 初回LD1/LD2 whole-shot評価の有限科学案

2026-11-13 JST。**未採択・未実行。今回の許可は計画まで。** 新MC、保存mean再解析、新N/RMSE/G計算を始めない。以下の候補をユーザーが選び、必要な具体値とSOURCEを揃えた後、別途その有限範囲だけの実行許可を受ける。承認証明書/grant発行は不要。

## 1. 固定入力と評価target

| 構成 | canonical configuration | mean/config/数値SOURCE |
|---|---|---|
| PR-LD1 | `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d` | A91349のB_MEANS_LD1、a9abf8deのCONFIGURATION_LD1、provider fefd548e |
| PR-LD2 | `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae` | 同LD2 |

locatorの完全SHAは[SOURCE_AND_READINESS](SOURCE_AND_READINESS.json)と[HANDOFF](HANDOFF.md)。各13raw/corrected/normalization/誤差の保存recordを読取専用で使用し、AのSignalBatch・reference/normalizer・旧P0〜P4を再実行しない。H4 DF10/sector(2,2)、保存H/state/snapshot、full DF-H target、h=.1/M12、S2/.1/K2/exactκ、元prior/decoder/shift0・budget/P0を保持する。

**evaluation_targetの採択が先に必要**。最初の条件付き診断では、Aが保存したreference_solver_energyを同一のproxyとして使う案があるが、厳密groundへ昇格しない。別のtarget包含を使う場合はそのSOURCEと適用scopeを指定する。A targetにはenergy_haがなくproxy-adopted=falseなので、旧packetを修正せず、後述の別descriptorをsummaryへ渡す。

## 2. whole-vectorの具体案と初回範囲

最初の科学質問を二つに分ける。

1. **同じ量子shot予算でLD1/LD2の全decoder診断がどう違うか**：2構成×1共通whole-vector、計2 MC callで調べられる。
2. **その差が全shotスケール変更でどう変わるか**：両構成に2 vectorずつ、計4 MC callが最小の二水準対照案になる。必要な検出力を保証する数ではない。

「最大3本」という既存interface案は上限の一例に過ぎず、3本を必要条件にしない。現在のtaskはLD比較とshot配分感度の両方なので、**2構成×2本を本来の初回対照案**、開発量/主張を絞る場合の**2構成×1本を縮小案**として提示する。どちらも未採択で、科学的な必要N探索ではない。

### 配分候補

新しい必要Nの理論がないため、初回は実験用の共通baseベクトルbを事前指定する。

| 候補 | 全round X/Y vector | 選ぶ根拠・制限 |
|---|---|---|
| 共通予算control b | **候補としてのみ**旧S3で実際に使った`[64,60,56,52,47,43,38,34,30,25,21,17,12]`を両軸へ同じ形で明示採用する案 | 既存実行可能な予算形状を対照にできる。新LD1/LD2のnominal・十分N・最適Nではなく、旧RMSEや成功資格を移送しない。ユーザーがこのcontrol用途を明示採択する場合だけ使う |
| 別の共通base b | ユーザー/A/Bが供給する13行のexact `(N_X,N_Y)` | 旧S3形状を使わない場合の代替。全値・SOURCE・軸制約を事前固定し、A meanからその場でNを生成しない |
| 二水準目 N_hi | `N_hi[m,a]=ceil(s b[m,a])`。**s=2は二倍予算contrastの候補** | 全13round同時のスケール応答を調べる実験設計。必要十分倍率や単調改善の仮定ではない。sとceil規則もユーザー採択。値の配列を今回算出していない |

最初のcontrolでX=Yを提案するのは対応条件を揃えるための選択で、decoder/APIは軸別Nを扱う。一般の最適配分にX=Yを強制しない。原rawはX=Re、Y=Imとして別binomialに渡し、両軸や価格標本数を追加乗算しない。

allocation_id/sourceは各config・vectorを一意に指す通常の研究metadata。physical configuration IDは変更しない。旧nominal形状をcontrolに採用しても`USER_SUPPLIED`の実験配分であり、`sufficient_N`やQUALIFIED flagを作らない。

N増加に対する単調性は仮定しない。count格子・偶奇でzero/tieの確率が変わるため、二分探索、失敗trialの再draw、配分を混ぜたroundごとのwinnerを行わない。

### 片方から始める案との比較

| 案 | call範囲 | 得られるもの・限界 |
|---|---|---|
| LD1のみ×1又は2本 | 1又は2 | 新callerの科学入力受入/そのLDの診断。LD2比較はできない。既存S3/人工接続があるため、単なる技術shakedownだけを新MCの目的にする必要は低い |
| LD1/LD2×1本 | 2 | 共通予算でのLD対照。shotスケール感度は未評価 |
| LD1/LD2×2本 | 4 | LDと全shotスケールの対照。point bestを十分N/G最小にしない |

段階案を選ぶ場合、後段LD2や配分追加を初回結果を見て無断開始しない。最初から同一scopeに両構成を含めると、多重性・seed・精度を対称に固定しやすい。

## 3. 実行前にユーザーが採択する項目

| 項目 | 判断材料・未確定事項 |
|---|---|
| claim | [risk計画](RISK_AND_N_EVALUATION_PLAN.md)のA、B1/B2、Cのどこまでか。初回にA又はA＋B1感度を選ぶ案はあるが正式採用していない |
| target | 保存solver proxyとして評価するか、資格付きtarget域を用意するか。両LDで同じdescriptorとsource。shift/prior/元target Hは固定 |
| 全outcome loss | UNDEFINEDのconditional診断か、明示bounded補完familyか、特定energy補完か。元decoderは変えず評価規約を別定義。旧[0,4096]を継承しない |
| menu | 上のbase b、軸制約、s、2 call又は4 call等の固定範囲。供給vectorのexact整数・allocation ID/SOURCEを保存 |
| precision/confidence | MSE半幅η、未解決率p_tol、主張する同時confidence/failure budgetとfamily、必要loss support。その根拠から有限n_MCを決める |
| seed/joint law | 有限call分の明示seed表、各trial/shot/軸/roundのfresh IIDを仮定するか。PCG64/binomial/binary64 probabilityの元規則を維持 |
| 系統/数値誤差 | 入力rawに条件付く範囲、経験的allowanceの限界、target/law距離の未取得、Minkowski個別項に必要な基準/coupling |
| SOURCE/出力 | 最小callerとmetadata受入を人工確認して固定。新run directory、全trial/status・summary・契約・消費signal_inputを保存 |

これらを数値的に埋める前に科学callへ進まない。計画書の更新やcommit許可だけを科学実行許可にしない。

## 4. 試行数を決める手順

**Aだけ**なら、点RMSEの希望精度を有限nで保証する方法は現コードにない。未解決率のCP精度からnを設計する場合、それはそのmetricだけのprecisionと明記する。conditional RMSE CIも必要なら、risk計画の分子/分母CI追加を先に人工確認する。

**B1/B2**で有効なloss supportとMSE半幅ηを採択した場合、既存`trial_budget`の式でnを固定する。共通target/hなら同じ保守的supportで比較できる案があるが、数値境界の根拠が必要。ηはε²との判別余裕又は研究上見たい差から選び、保存S3の点推定を見て都合よく設定しない。

2構成×2配分等の全評価とmetricへαを事前配分する。単一run95%を全比較95%とはしない。nが大きすぎる場合は「精度未達/実行案再設計」で止め、恣意的なtrial capで資格を作らない。rare branchを覆うより有用なrisk法が必要なら別技術scopeへ戻す。

初回は独立したcall seed表を提案する。同一seedを各LDへ流用しても、NumPy binomialの乱数消費は確率/Nに依存するため、検証済みcommon-random-number couplingとはならない。cross-LD共分散を0と仮定した比較や、coupled Minkowski項の推定へ流用しない。新しいcoupling法を使うなら別SOURCE/人工検証が必要。

## 5. 最小の技術実行形

`W.prepare`で固定config/全mean/whole vectorsを受入れる。原targetを変えず、別の採択済みevaluation_target scalar/sourceを`S.summarize`へ渡す薄いcallerを使う案が最小。元S.simulate_preparedとR3を直接呼ぶので、現W.diagnoseのenergy_ha欠落を無断fallbackで埋めない。評価targetと全契約はMC開始前に検査し、C型の`signal_input`も実際のrisk入力として事前固定する。

A/B診断は`n_packet=None`、入力Nはevaluated allocation欄。全status・conditional RMSE・未解決率/CI・可能なenvelopeと入力scopeを残す。Cの科学的資格producerは別作業であり、点RMSEが目標より小さいだけでN_X/Yを正式十分Nへ書き込まない。

## 6. 資源・保存・STOP

R3 batchを使い、同meanをallocationごとに物理再計算しない。元binomial生成はshotを1発ずつ列挙しない。count配列は概ね`2(M+1)n_MC×8 byte`、decoder配列/全trial Python記録は別にO((M+1)n_MC)を要する。実wall/RSSは今回測定しておらず、過去速度を新runの保証にしない。

独立config/vectorはseed・同時confidenceを保持して必要に応じprocess並列化できる。CPU/RAM/workerを恣意的にcapせず、共有設定/他担当processへ干渉しない。大きなnのためchunk化が必要なら、単にsimulateを小分け再seedすると乱数順・実験定義が変わるので、先に新callerのseed/RNG保存規則と人工同値性を固定する。

SOURCE/config/target/law/physical binding不一致、mean欠測、raw範囲外/非有限、N非整数/int64 backend domain外、loss/support違反、保存不能はSTOP。元数値gate・closureを再計算して穴埋めしない。通常zero/tieやprior外は観測として残し、都合の悪いtrialを捨てない。未解決lossが未定義なら全risk資格は未取得の結果として終了する。

固定call数・反復数の完了でSTOP。precision未達、優位不明、RMSE上端が大きいことを理由に同run内の反復増/seed変更/N変更/他candidate追加をしない。全trial・summary・SOURCE・入力/評価target/統計契約を保存し、旧結果/KEEP/grant/ledger/marker/STOPを上書きしない。

価格未取得中はrisk/N診断と不足coverageだけ。適格E[C]・N証拠・物理入力資格が揃った後の実G/方式選択には別の有限実行許可が必要。今回の計画公開後もmandatory STOP。
