# C8-1：q64保存8本・残作業の限定実行PLAN

2026-11-08 JST。利用者の「Stage C8-1残作業の完了 — q64独立価格検証の限定再開」に従う。基点a708a130。旧runのSOURCE/失敗/STOPを変更せず、新branch・新namespace `run04_limited_completion`で完了済み成果をhash照合して継承する。これは旧runのretry/resumeではない。

## 固定科学条件・許可call数

Linear H4/隣接1.00Å/STO-3G/Nα=Nβ=2、元H/number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q64/complete step32後global cut。元H厳密groundを仮定、準備費除外。packet/CDF endpoint/law/512seed・8保存event/16384eventsは元6b1d613dの固定入力を使う。追加sampling0、ground/DF solve0。

sample00のraw1/4P0は6b1d613d、4 postcheck PASSはf1e89590から継承する。これらを再実行しない。sample00の最終paired X/Y・vacuumχを**最初に一回だけ**実施し、PASSした場合だけ保存sample01..07のraw7、raw PASS後最大28 P0/28 postcheckへ進む。各pathの最終paired/χは各一回・全8回。全8のraw/4post/最終pairedχが適格な場合だけ、n8の価格統計・凍結A/F/S診断・CX整数残差を取得する。

sample00の旧postcheckには価格metricsが未保存のため、同じcompiled QPYから既存metrics/feature関数で計数し、旧postcheckを変更せず別のadapter recordへ記録する。追加compileや状態伝播ではない。χ/pairedは保存observableとconditionsの既存C3/C6a checkerを使う。新たなstate simulationは不要な構成である。

strongのuncontrolled Gaussian/directional/head relative fusion/tail共有・63step境界融合・scalar集約を維持する。Referenceはwhole path一gadget、B前32/A後32、actual A.inverse、analyticχ、二CSWAP、内部62融合とcut一境界損失。P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=false/topology-free/宣言位相保持。X/Y両側をcompileし、価格scopeは準備を除くfull Hadamard interrogation。

component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fit/renormalizationなし。f1e89590と同じgate matrix/cache・Aer double **simulation fusion無効**をraw/postの両方へ選択する。全256入力列・Reference全出力を保持し、17qubit全Operatorを作らない。科学的な判定規則は変更しない。

## 再利用・最小拡張

C8-1 raw worker/whole-path builder/CAR・head・atom cache、修正済みhandoff pipeline/CPU allocator、P0 compile worker、f1e89590数値adapter、例外前診断、C3/C6a最終pairedχ、既存q64価格・共分散・モデル診断を再利用する。新規sampling器/compiler/state simulatorは実装しない。

旧runnerは元raw1/compile4の引継ぎまでで、f1e89590の4 postcheck/保存observableを引き継げないため、新しい薄いinheritance/worker入口を追加する。4旧checkのbytesを保持したまま価格metadataを別記する。

数値経路を変えず、既存BatchedStateActionsの`last_batch_sizes`追加を観測する小さな継承classで、完了列数/全列数/経過時間をtask別private progressへ保存する。phase/angle/列順序/演算や最適化を変えない。封印済み旧実装には変更しない。

## 資源・非干渉

実affinity/quotaとMemAvailableを起動時・dispatch時に確認する。CPU/workerに低い人工上限を設けず、P0は1 CPU予約、raw/postは残CPUを配分する既存修正済みschedulerを使う。memory admissionは同qの完了済みraw/Reference postの実peakを使う作業estimateで、2倍を初期目安にする。P0見積もりには完了済みP0 workerのpeakを用いる。旧raw workerを再利用した1件はrawと同一ru_maxrss履歴を持つため、raw peak未満だった他の完了P0記録と区別する。現在のMemAvailableにも再利用workerの残存memoryが反映されるため、P0予約にraw peakを重ねない。これは科学的boundや恒常RAM上限ではない。実空き資源に合わせて独立taskを並列化する。

小BLASとcompileのnested並列だけを抑える。共有OS/scheduler/venv/package/cache、他ユーザー・他job・旧workerへ変更/signal/killを行わない。別worktreeの通常更新や一時的競合だけで正常workerを停止しない。

postのRAM見積もりも方式別の完了peakを使い、strongの小さい9qubit検証にReference17の大きい予約を流用しない。CPU配分と科学的なworker処理は既存のまま、dispatch時のtask estimateだけを選ぶ。

## 統計・モデル・STOP

全8が揃う場合だけ方式/axis別RZ/CX count/depth・total depth・size・peak qubits、n8平均/SD/min/max/paired差・共分散を既存実装で取得する。X/Yをn16にしない。凍結A/F/S・校正共分散は元packetを保全し新fit0。各pathのΔCX+14128を保存するが、全path公式を認定しない。探索的/有限観測と母平均・長q・actual保証を区別する。

SOURCE/input/law/継承hash/基準比較/位相/χ/漏れ/精度の不一致は依存投入を止め、active正常taskは回収しpartialを保全する。失敗path除外・seed/cut変更・追加retry・精度緩和なし。全8未達でcomplete-case統計を全8と報告しない。失敗時も保存証拠に基づく長q戦略整理は追加科学計算なしで行う。

q≥128、新sampling、追加calibration、モデルfit、N、全13-round費用更新、ground/DF solve、routing/T/noise、actual成功保証、詳細科学レビューは0。旧SOURCE/raw/input/QPY/NPZ/result/grant/ledger/marker/STOPを保全する。

原本はlocalのみ、公開はscalar/check/統計/SOURCE/hash/STOP/報告・更新長q戦略。remote SHA/tree/public bytes照合後mandatory STOPし、次の担当をユーザー/GPTへ戻す。
