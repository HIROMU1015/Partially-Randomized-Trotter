# Mechanism input gap report

状態：**METADATA_BINDING_COMPLETE_WITH_EXPLANATION_GAPS**。入力の存在・候補・schema・保存shot原値の同一性を確認した。Gの数値再構成・費用比・shot比は未実施。

## core

対象32比較について、PRの同cohort paired軸、PF軸単価、両側の保存整数shot、保存exact G、元保存SE、直接実測/modelの参照を結合した。H4/H6/H8の元A/D sealの完全bindingを登録表と照合し、native側の省略schemaはHamiltonian/state/compilerとsnapshot（H6/H8ではinputというfield名）を照合した。省略fieldを不一致と誤認した初回準備は停止して保持し、新しい準備範囲で訂正した。

H4 PRE_D_Pはlegacy DF12・測定PF subset16・新A探索n8・旧fixed anchor n32。H6/H8はDF19/28・PF136/184、新A探索n8と旧確認union n32。H8固定2候補n48は公開比較表・192軸表のscope/candidate/replica/axisで結合し、元探索や旧n32をpoolしない。

H8 n48のpublic表にはseed/eventの生値がないため、その原本を再監査したとは主張しない。public pairingと元完了報告の来歴を保持する。H8原compiler profile bytes／旧compiler output完全同一性も引き続き未確認。これらを数値一致PASSへ読み替えない。

## 説明入力

PR/PF合計64 method recordのうち60でlog_B・signed bias・allowance・headroom・αの対応fieldを結合できた。これはfield存在の件数であり、EXPLANATION_COMPLETEという数値検査の結果ではない。

|contrast ID|系／候補|残る未結合field|影響|
|---|---|---|---|
|A18_OLD_PR|H4 B2_p3_q8_2nd_r2_K2、ε=.0005|allowance、headroom、α_axis|B/log_B・signed biasは保全。対象精度の保存precision rowが元signalにないため逆算しない|
|A58_OLD_PR|H6 B2_p5_q8_2nd_r2_K2、ε=.0005|allowance、headroom、α_axis|同上。別の保存会計を接続する場合も出典・shot・bindingの再固定が必要|
|A82_OLD_PR|H8 B2_p7_q2_2nd_r4_K2、ε=.05|log_B、signed bias、allowance、headroom、α_axis|今回の既知signal参照範囲では対象recordが未結合|
|A98_OLD_PR|H8 B2_p7_q8_2nd_r4_K4、ε=.0005|allowance、headroom、α_axis|B/log_B・signed biasは保全。対象精度の会計は未結合|

上記もcoreの対象から除外しない。coreのexact照合が通ればcoreを完成させ、説明側はEXPLANATORY_INPUT_MISSINGを保持する。保存shotの再計算、headroomからのbias推定、G/Nからの単価推定、B=1等の無根拠な補完はしない。

## 探索・取得範囲

前回private byte lockと元A/D seal、native configが列挙するold_signals／A_origin_signal／D_saved_signal等、固定prepare_pr_cost_batch.pyが指定する旧H4 D保存signal、H6 V1 configのadditional/old/B0I signal、固定fc6bd039…のH8 Phase A保存signalを使用した。取得byte・hash・origin・commit/path・candidate recordはprivate bindingに保存した。raw NPZ/QPYをdeserializeせず、未知の保管先やprojects全体は走査しない。

残り212比較は前回のmetadataを保全し、今回の数値permitへ追加しない。H6非相殺の全登録未確定、新A factor10の保存shot欠測、H4旧full対照の軸単価欠測はそのまま。追加科学入力の生成や補足の数値解析は別承認。
