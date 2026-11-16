# H4方式別RPE探索の並列化・高速化：通常入口への接続と技術検証

2026-11-17 JST。独立branch `codex/h4-rpe-search-throughput-20261116`。基点はB2 `d8e806265c3ded8e1184181c37c2d7ec71f08ade`。実装と統合はroot、保存運用分析・適応探索・人工DP計測は担当sub-agent、rootの入口変更の限定的な別担当技術確認は [TECHNICAL_REVIEW](TECHNICAL_REVIEW.md) に記録した。作者自身の検査、別担当の技術確認、重要な科学レビューを区別する。研究概要589行の全文読取は保存分析担当への委任である。

実H4の新規信号・較正・compile・Q3、round05本番は **0件**。固定SOURCE、原科学結果、旧STOP、他担当worktreeを変更していない。今回の完了は、技術実装・保存記録の分析・人工検証・引渡しである。

## 1. 結論と優先順位

| 優先 | 通常入口に接続した改善 | 根拠・採用範囲 |
|---|---|---|
| 1 | D2のtotal-state DP prefixをB2の広域driver→通常CLI→実route workerへ接続。設定例は境界 `[5,11]` | 過去4,783起動のうち3,761件に、起動時点で完了済みの入力互換prefixがある。人工3配分列は3.927→2.522秒、CPU3.926→2.519秒、全DP出力一致。実H4の速度は未測定。 |
| 2 | D2の共通非同期queue・DP worker設定をB2のV2、prior結果再利用、準備後継続へ接続 | 独立候補優先・有限予約・真の起動件数・採用順を維持。人工4独立DPで4slot×1workerは1slot×1workerの約4倍のthroughput、CPUはほぼ同じ。小fixtureの結果であり実H4の最適profileではない。 |
| 3 | 較正等の準備待ち中に、既に登録済みの候補を1batch処理する `overlap_ready_batch` | 広域driver→通常CLI→人工workerを通し、準備待ちとの重なり・総予算・STOP伝達を確認。投入順と予算消費先が変わるので既定false。実較正との速度効果は未測定。 |
| 4 | `adaptive_hybrid`、同一構成内のround間振替 `allocation_exchange` | 両方とも既定無効。適応先読みは人工1構成で起動88→59件、12構成では563→128件だが待ち時間は延びた。振替は人工の減算局所最良を脱出したが、実H4の改善は未確認。 |

科学的な比較対象は、固定H4直線1Å/STO-3G/tol-only DF10、固定problem・各routeの精度契約、Q3のMSE上下界、条件付き `Σ N·E[C_no-prep]` である。状態準備費用を含めない。未資格MODEL価格・ideal/raw IID・保存DF targetのproxyという資格を引き上げず、実backendや物理ground energyへの保証・大域最適性・新しい科学的winnerを主張しない。

## 2. 既存成果の再利用と変更点

- D2 scheduler公開 `1eda4ba3f468dfba89823941222843ef5cfe0402`、prefix公開 `0742a750be7559d2532b3059aa75bc5279dbe7aa` の後続branchを照合し、追加後続がないことを確認した。
- `rpe_optimizer_concurrent.py`、`rpe_dp_prefix_reuse.py`、人工test/benchmarkを再利用した。D2の古いCLI/worker全体でB2を置換せず、B2のPR-DF共通入口、複数A入口互換、process終了gate、保存collectを維持して必要部だけ接続した。
- round05の物理schedule集約、価格下界による除外、FINE、独立候補優先HYBRIDは既存実装を使い、作り直していない。
- 既存 `transfer_count=2` は近いκから適合shot vectorを種として移す機能であり、同一構成の高価roundから安価roundへshotを戻す振替とは異なる。振替だけを任意設定で追加した。
- SOURCE・raw signal・契約・h/target/M・scope・数値runtimeが合うprefixだけを再利用する。元AST/hash/置換数guard、残余mass/loss/moment会計、pair予算、suffix検査、最終MSE計算を保持する。途中状態の利用でも新しい配分はQ3予算1件である。

## 3. 保存運用ログ：どこで待ったか

[SAVED_OPERATIONAL_SUMMARY](analysis/SAVED_OPERATIONAL_SUMMARY.json) と [各jobの時刻・因果donor](analysis/SAVED_OPERATIONAL_ANALYSIS.json)。正確な4run locatorだけを読み、REQUEST/LAUNCH、JOB/DP stages、PROCESS_RETURN、保存CPU/RSSを用いた。MSE・count table・DPを再計算していない。

| round | 実起動 | 規則採用 | parent終了receiptあり | 最初の起動→最後のreturn 秒 | terminal job wallの和 秒 | 保存worker tree CPUの和 秒 | 平均同時数 / 枠 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 4229 | 3334 | 4211 | 29423.99 | 528477.66 | 838779.01 | 17.961 / 18 |
| 02 | 185 | 43 | 178 | 3022.15 | 25247.32 | 42260.37 | 8.354 / 32 |
| 03 | 85 | 49 | 85 | 5543.39 | 9978.05 | 15958.67 | 1.800 / 32 |
| 04 | 284 | 82 | 284 | 4083.96 | 41084.90 | 63344.29 | 10.060 / 32 |

round03の「270件」にはround02の185件が含まれ、新規実起動は85件。原報告は書き換えず今回の数え方を明示した。round01/02にはparentのPROCESS_RETURNがない18/7件があり、terminal intervalの和・占有率はその件を除く。worker CPUは全保存RESOURCE_FINALの和なので標本範囲も違う。receipt欠落を成功終了や異常signalと推定しない。

round04のdriver log先頭→末尾は4,702秒、Q3起動spanは4,083.96秒、job wallの和は41,084.90秒である。これらを同じ待ち時間とは扱わない。driver spanは呼出し間のgapを含む。対応するPREPAREは37秒、SIGNALSは107秒、V2 logの2区間の和は4,558秒だが、丸め・再開・未対応STARTがあるため、各段階の和を総時間へ無理に一致させない。

round04の1job平均はcase build0.260秒、metadata2.560秒、起動→Q3呼出し5.391秒、call→DP3.002秒、DP134.128秒、DP後評価0.645秒、route return→adapter終了0.778秒。主要な仕事はDPである。CPUの利用可能数・host飽和・controller/外側adapter/結果取込のCPUは旧記録から確定できない。round04の個別job treeの観測RSS最大は4,668,528 KiBであり、共有pageの重複もあり得る。個別jobのRSS peakの和を同時最大メモリにしない。

**「実行可能な候補があるのに空いた時間」と「前結果待ち」は旧logでは未測定**。低い占有率だけから原因を決めない。今回の通常入口は次を記録する。

- `ROUND_STAGES.jsonl`：入力/価格準備、準備API、探索/Q3、比較export、保存。
- V2 `selection_timing`：候補生成・分類、preview整列、backend回答待ち。
- `PROGRESS.jsonl`：高分解能ENQUEUE/LAUNCH/ADOPT、選択結果待ち、再開live、未起動予約の取消。metricsは「queueあり」「queueなしで選択結果待ち」「どちらもなし」の空きslot秒を分ける。これは投入済みqueueの範囲の帰属であり、未生成の全候補の不存在を証明しない。
- `DP_PREFIX_EVENTS.jsonl`：HIT/MISS、再利用round、hash/read/write秒、保存bytes。元workerのDP stage・子process exit/signal記録も使う。

### 費用水準への過去の到達

時刻のあるterminal結果に条件付けた履歴である。到達CPUの正確な時系列は未保存なので、完了済みworker CPUの下界と、その時点で起動済みworkerの最終CPUによる上界を示す。controller/adapter CPUを含む総CPUではない。全roundを跨ぐ経過時間はround間の人手gapも含む。

| 費用水準 RZ | 最初のQ3起動からの秒 | 到達までの起動 / terminal完了 | 保存worker CPU下界〜上界 秒 |
|---|---:|---:|---:|
| 501233190.43 | 1056.24 | 128 / 111 | 27461.79〜31576.96 |
| 437894306.43 | 5581.72 | 700 / 683 | 160271.81〜164075.69 |
| 414301229.37 | 18043.04 | 2465 / 2448 | 518877.03〜522154.56 |
| 410199237 | 47562.66 | 4499 / 4463 | 890105.45〜896998.05 |

## 4. 起動時点で可能だったprefix再利用

同じ構成・signal・binding/target・全評価契約/scope・kernel/accelerator・記録interpreterを照合し、各jobの起動より前にreturnした検証済みjobだけをdonorにした。後から完了したjobは使っていない。旧Pauli SOURCE.runtimeはnullで完全な数値runtimeを遡及確認できず、frontier本体も未保存である。次表は **同一runtimeを仮定した入力互換機会** であり、実cache HITや実速度改善ではない。

| 保存境界m | 機会のある起動 / 4,783 | 全起動で保存した場合の異なるkey数 |
|---|---:|---:|
| `[11]` | 532 | 4,243 |
| `[5]` | 3,761 | 986 |
| `[5,11]` | 3,761 | 5,229 |
| `[2]` | 4,616 | 125 |
| 全非終端境界 | 4,700 | 19,372 |

round数の和を時間へ換算しない。`[5,11]` では3,132起動に稼働中jobとのprefix一致もあったが、完了donorがなければHITには数えない。現実のcheckpoint容量は未測定。旧OUTCOMEだけからfrontierを復元しておらず、旧結果の再計算もしない。

## 5. 人工DPでの速度・CPU・保存コスト

[prefix/SUMMARY](prefix/SUMMARY.json)。固定guard付き元DPへ人工13round・4shot・frontier256等を与えた。3配分列は基準、m12変更、m6変更。33比較すべてでDP全出力・MSE上下界・判定が一致した。

| 境界 | 3配分のwall中央値 秒 | CPU中央値 秒 | 保存容量 bytes | 対比較速度比中央値 |
|---|---:|---:|---:|---:|
| 保存なし | 3.927 | 3.926 | 0 | 1.000 |
| `[11]` | 2.714 | 2.713 | 82,413 | 1.445 |
| `[5,11]` | 2.522 | 2.519 | 110,437 | 1.555 |
| `[2,5,8,11]` | 2.538 | 2.535 | 190,420 | 1.548 |

3反復の `[5,11]` のhash/read/write中央値は0.01631/0.00285/0.01016秒。別の対応基準との3反復でm2単独は0.987倍、m5単独は1.088倍だった。機会数が多い境界が速いとは限らない。設定例の `[5,11]` は今回の人工根拠による開始候補であり実H4最適ではない。

| 人工4独立DPのslot×DP workers | wall 秒 | 総CPU 秒 |
|---|---:|---:|
| 1×1 | 5.512 | 5.515 |
| 2×2 | 3.120 | 7.056 |
| 4×1 | 1.385 | 5.529 |
| 1×4 | 6.170 | 7.630 |

上のprofileは各1反復。少数scheduleを深く進む人工2配分では、1workerの通常2.612秒/CPU2.612秒に対し `[5,11]` は1.414秒/CPU1.412秒。2/4workersの結果もSUMMARYにあり、この小fixtureではworker増加が優位にならなかった。実H4の大きなinteger frontierやworker8の最適性へ移さない。

barrierで2jobの同時cold開始を再現すると両者MISS、同じkeyへのSAVEが2回（重複1回）となった。wall1.424秒、CPU2.850秒、最終cache41,463bytes。donor完了後のwarmは0.0586秒。atomic保存は同時coldの重複計算をなくす機能ではない。元の採否順を変えて人工的にHITを保証しない。

## 6. 必要なQ3件数と探索規則の効果

[adaptive/NOTES](adaptive/NOTES.md)、[人工時刻モデル](adaptive/artificial/CHECKS.json)、[選択後開示の保存再生](adaptive/saved_replay/SAVED_REPLAY.json)。同じ候補を速く評価するprefix/workerの効果と、候補順・件数を変える効果を分けた。

| 人工ready構成数 | HYBRID | 起動 | 採用配分 | 最後の起動jobまでの模擬秒 | 模擬job時間の和 秒 |
|---|---|---:|---:|---:|---:|
| 1 | 従来 | 88 | 32 | 250 | 880 |
| 1 | 適応 | 59 | 32 | 240 | 590 |
| 12 | 従来 | 563 | 120 | 910 | 5,630 |
| 12 | 適応 | 128 | 120 | 1,080 | 1,280 |

採用配分順は一致した。模擬job時間の和をCPU時間と呼ばない。振替fixtureでは減算だけの局所最良896から適合698へ26番目の回答で到達した。shot上限と厳密な費用減少を守り、MSE単調性は仮定しない。

保存再生は4,783 REQUESTだけをindexにし、共通の初期C保存点・ready26構成・32枠・予算400で比較した。V2が配分を選んだ後にだけOUTCOMEを開示し、保存hashも照合した。外側の広域schedule選択と準備時間は含まない。

- 従来/適応とも同じ23回答の後、同じ未知配分で打切り。再生内の最安517,151,301.57 RZ、その水準まで751.063模擬秒は同じ。水準到達時の模擬起動は122→65、次の未知配分の投入を含む停止時点では126→69。最終の起動済みjobまで待つ時間は878.18→891.57模擬秒。
- 23回答の保存worker CPUの和は5,602.51秒であり、新profile全体のCPUではない。未知の先読みを含む総CPUはnull。
- 振替ありは10回答後の未保存振替で打切り。再生内最安541,743,905.57 RZ。同予算400を消費した時点の優劣・実H4の振替改善は判定できない。
- 未知配分を不適合にせず、400件までの結果を補間しない。再生内の最安は初期知識と到達範囲に依存し、B2全保存集合の410,199,237 RZを変更しない。

## 7. 通常・再開・予算と不具合への対策

通常CLIは共通schedulerを選択でき、準備継続にも同じoptimizerを渡す。科学stageの新規許可を作らない。詳細な設定・コマンドは [HANDOFF](HANDOFF.md)。

- 新runは `continues.run_dir` からfamily constants等を引き継ぐ。元の状態は不変で、前STOPを来歴へ保存する。
- 計画専用で一度も実行しなかったCOMPLETIONだけを再開対象に戻し、実際の科学的HELD/STOPは保持する。同runの非一時的STOPは計画・準備・prefetchより前で保持する。
- 準備中のinterim batchはCompletion共有STATE/registryを別instanceから触らず、既存公開済み候補だけを使う。準備futureは選択順に吸収する。準備待ちを先行Q3に使うため、全準備後に始める場合と有限予算の使い先が変わり得る。
- batch総予算は元round上限でcap。先読みも予約時に計上し、未起動予約だけを取り消せる。V2の方法全体のpreview更新では、別構成の古い未起動予約も解消する。構成別並行探索は自分のqueueだけを更新し、他method/構成の仕事を消さない。
- 既完了prior jobは同じroute/identityでcalls=0。再開live workerは枠に数え、`wait_for_running_jobs` 時は未採用先読みも終了まで待つ。実行済みjobを再投入・取消しない。
- 適応先読みの観測counterはinvocationごとにresetする。完了OUTCOMEと起動済み予算は再利用されるが、再開後の未採用先読み件数・投入順の完全一致は保証しない。人工の通常executor再開でこの挙動を確認した。
- 当初の設定例は価格登録表をコピーせずplan-onlyが失敗した。固定価格表を新runへコピーするよう修正し、通常入口の開始・同run再開が双方終了code0、信号・較正・Q3起動0であることを [entry/PLAN_ONLY_ACCEPTANCE](entry/PLAN_ONLY_ACCEPTANCE.json) に記録した。

## 8. 検証範囲と成果物

[checks/VALIDATION](checks/VALIDATION.json) と各log。入口13、scheduler15、prefix10、保存分析8、準備API11、適応10、既存priority6、schedule7、executor21、generation11、preparation8がPASS。固定実Pauli routeはworker設定1/2/4/8のconstructor/元AST guardのみ確認しkernel call0。人工DP比較は別に33件すべて一致。

4方式に共通のscheduler/profile/worker/prefix接続を人工4方式経路で確認した。保存運用ログ・保存再生の効果はPR-Pauli中心であり、Full PF/B0I/PR-DFの実探索の改善を実測したことにはならない。実H4の同一費用到達までのwall/総CPU短縮、および同じ実計算予算での最安費用改善は次の承認済み科学実行で測る項目として残す。今回新しい科学実行は提案・開始していない。

検証済み実装は `a50aadc1f430e8c8ee6f8917a8024f1121ec913a`。SOURCEと固定公開commitは [SOURCE](SOURCE.json) を参照。原runは読取専用、人工job/重複metadataの全作業directoryはサーバー上に保持し、公開するJSON/logと再現scriptを本文から辿れるようにした。
