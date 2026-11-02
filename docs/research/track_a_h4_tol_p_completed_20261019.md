# H4 tol-only DF10 独立確認P 完了（2026-10-19）

**承認された固定2条件のP one-shotを完了。ε=.05ではPFが低費用となり探索時のPR低費用方向は再現されず、ε=.0005ではPRが低費用で方向を再現した。native exit0・全worker回収・終了監査PASS、mandatory STOP。** 不利な結果を含めてそのまま保存した。追加sampling・自動retry・候補再選択・旧標本混合は0。詳細科学レビューやH2/H6/H8最終統合へ進まない。

## 2条件の費用順位と不確かさ

linear H4、隣接1 Å、STO-3G、tol-only1e-8 DF rank10、T=.8／P=0、通常signed bias／allowance倍率1。PRのL_Dはε=.05で3（q1/r4/K2、δ=.8）、ε=.0005で2（q8/r2/K4、δ=.1）。全値はこの固定条件と保存shotの範囲に限る。

費用はfull cosine/sine Hadamard wrapperの実測compiled RZに保存shotを掛けた`G_RZ`。表は完了resultからの転記（平均表示のみ小数2桁丸め）。`PF−PR`が正なら新PのPR平均が小さい。paired SEは同一trajectoryの両軸をまとめた標本平均のSEで、保存PF費用を固定値として扱う費用差にも同じSEを付す。CI・p値・powerの追加計算は行っていない。

| ε | 新P n | PR平均 G_RZ | paired SE | 保存PF G_RZ | PF−PR | 有限標本の順位／方向再現 |
|---|---:|---:|---:|---:|---:|---|
|0.05|1,108|71,128,344.93|174,917.74|70,108,214.00|-1,020,130.93|PF < PR／NOT_REPRODUCED|
|0.0005|78|4,964,066,816,492.41|14,140,828,222.91|7,856,558,002,614.00|2,892,491,186,121.59|PR < PF／REPRODUCED|

ε=.05のPRは`B2_p3_q1_2nd_r4_K2`、PF対照はD190登録内の`B0I_p5_q1_2nd`。新P平均のexact値は `39405103093/554`、PF−PRは `-565152537/554`。固定探索方向+1に対して新Pは−1であり、`NOT_REPRODUCED`を保存した。

ε=.0005のPRは`B2_p2_q8_2nd_r2_K4`、PF対照は`B0_p7_q8_2nd`。新P平均は `193598605843204/39`、PF−PRは `112807156258742/39`。新P方向+1で、`REPRODUCED`を保存した。登録外のglobal optimumや他分子へ一般化しない。

## 希少eventの観測

trajectory分類と個々のRTE event次数の件数を区別する。以下は保存された新Pのeventラベルの整数監査であり、旧標本の再集計、law確率再構成、追加のsamplingではない。

| ε | S0_ONLY trajectory | S2_SINGLE | S2_MULTI | S4_PRESENT | order0 event | order2 event | order4 event |
|---|---:|---:|---:|---:|---:|---:|---:|
|0.05|1071|37|0|0|4395|37|0|
|0.0005|71|7|0|0|1241|7|0|

RTE event総数5,680。複数S2 trajectoryは両条件で未観測。K2候補ではS4は有限cutoffの支持外であり、K4候補ではS4を観測しなかった。未観測層の費用・分散を0で補完せずunknownとして保持した。観測頻度をlaw確率や母平均保証に置き換えない。

採用n1108／78はS_STATSのfamily α=.05・planning power=.8・SD倍率4・相対検出差.02に基づく有限標本の点方向再現用の計画値である。実power、母平均優位、選択後coverageを認定しない。探索n8、保存FP n32、legacy標本は新P統計へ混ぜていない。

## 固定SOURCE・入力・承認

最終seal commitは`0ca8bf68dc154418781bb102cd48cd14d66bdb18`、科学SOURCEは`3c664ca4a9f003880b9f6d56f2a0e55ed5b56603`。科学SOURCE closure25ファイルとprovider/kernel251ファイル、runtime34,273ファイルのbytesは実行前後と終了監査で一致した。最終sealの29入力を保持し、実行承認receipt・fresh資源receipt・seed refreshとその所有snapshotだけを追加結合した。元seal・元INVALID grant・全既存結果・grant・ledger・marker・STOPは保全している。

実行contract digestは`99bb967819a18a625e74f5ff77d0e6b27d2a05930702d08a8d0113dd20005474`。ユーザーの「H4独立確認Pの科学実行」「P科学実行one-shot1回を明示的に承認」をP限定receiptへ固定し、別stageの承認へ拡張していない。新1,186 master／1,810 seed整数は封印値を維持した。最新の既知70,270除外整数とgeometry登録の同一性をrefreshし、衝突・seed引直し0。並行H2は観測時に環境準備のみで科学seed登録0、そのhandoffを所有snapshotへ保存した。liveな他worktreeのHEAD/statusや将来更新を凍結条件にしていない。

保存rank10入力・状態を再利用し、元の2 prefix preparationだけを復元した。H4の保存dense eigh／canonical phase／追加の明示的な事後再正規化なしという来歴、およびH6/H8のsolver・正規化との差を保持した。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing指定なし、`symmetric_directional`、original constant＋tail identityを一度だけ加え、referenceを回転しない契約を変更していない。

## 実行と技術監査

| 項目 | 実績 |
|---|---:|
| 無条件trajectory／native evolution | 各1,186 |
| compiled cosine/sine軸 | 2,372 |
| occurrence／RTE event | 1,732／5,680 |
| control probe／独立event oracle | 8／2、全PASS |
| 保存入力からのpreparation復元 | 2 prefix |
| 新P endpoint会計／比較 | 1 pass／2比較 |
| worker枠／累計所有worker全回収 | 60／1,186 |
| witnessed実行wall | 287.504592秒 |
| native exit／同一親回収 | 0／PASS |
| 中断・retry・追加sampling・再選択・旧cohort混合 | 全て0 |
| 新PF compile・signal・shot再算出・bootstrap・SCF/DF/state/reference | 全て0 |

freshのper-core実稼働量、physical topology、既存affinity/quota、空きRAM約466.81 GiBを観測し、空きphysical core60個を選んだ。過去のworker数を固定上限にせず、CPU/RAM/worker/wallの恣意的capを設けていない。各worker1 thread、parent preparationをfork/COW共有し、paired evolution・fingerprint reuseを維持した。最初の2 trajectoryでcontrolを検証し、全回収後に残1,184を並列処理した。自己所有processのaffinity・cache・保存先だけを設定し、共有設定・他ユーザー・H2のファイルやprocessを変更していない。

固定監査はSOURCE/input/compiler/runtime、ONE_SHOT・grant・承認receipt、native exit0・同一親wait/reap、1,186件のtask結果と2,372 measured軸、master/step seed・event digest・元の無条件law／normalization契約、nonrefundable ledger、先頭control/oracle、固定n、sole accounting、旧9,216記録と最終sealの保全を検査してPASSした。新しい9,556記録のSHA保全manifestもprivateに保存した。これはprimary agent自身の技術監査であり、別AIによる独立レビューではない。

統計は固定科学SOURCEの一回会計で保存した値を保持している。追加の`record_h4_tol_p_completed.py`は終了監査・保存結果の転記・保存平均からの要求された費用差の減算・eventラベル整数監査・seedを除いたprojection作成だけを行い、その独自SHAをexecution bindingに記録した。科学SOURCEやmean/SEを作り直していない。

result SHA256：`85fb838688bee5c64e264a394257c9272b290686a6e4b3de73292364a58fb13a`。

## 保存先と次の担当

[保存result](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json)、[費用差・SE・順位](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/comparison_projection.json)、[eventラベル監査](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/event_label_audit.json)、[実call](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/actual_calls.json)、[worker回収](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/worker_receipts.json)、[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/completion_audit.json)、[SOURCE・実行binding](../../artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/execution_binding.json)。元seed・raw event・入力・所有process証跡・全ledgerはprivateに保持する。

新しい公開push対象の明示がないため新規pushは行わず、成果物をローカル独立commitへ保存する。既存sealはユーザーの手動push後にorigin追跡へ反映されている。

次は**ユーザー**が2条件の結果と有限標本・希少eventの限界を確認し、受入可否と必要なら後続の指示を判断する。**Codex**はmandatory STOPを維持する。追加sampling・再選択・詳細科学レビュー・H2/H6/H8最終統合は開始しない。
