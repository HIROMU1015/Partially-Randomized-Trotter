# 測定方法と証拠境界

人工技術検証のみ。公開済みF01最終commit `07f65831f045bf1fdbd4b370e44c7f59a9901ddd`から独立worktreeを作成した。固定資料の旧「未採択」記録は原文のまま保全し、今回のユーザーによるR1方針採択と区別した。Codex Aのlive worktree・private source・lock・processは開いていない。共通AGENTS.md、worktree AGENTS.md、指定共通運用指示§8.4を適用。AI sub-agentは使用していない。

## Scheduler

元のF01 `run_pair`・Meter・Streamをそのまま使用し、endpointだけ人工実装へ置換。X→Yは同一instance、initial/X/Y/finalのstate identity検査を維持。人工namespaceは `ARTIFICIAL_B3_PERF/replica/arm/axis`。A/B・両passは同じbit prefix、X/Yは分離。SHA counterによる独自人工bit fixtureであり、実G7のbitstream再生ではない。乱数的IIDを主張しない。

light/mixed/heavyはCPUを実際に消費するinteger loop、Fraction、人工law sort、word/weight/phase構築を含み、Bだけ有理数級数とevent再hashを追加する。ZERO_R/ZERO_C/EVENTを約1/4・1/4・1/2で混ぜる人工設計で、実G7の発生率・operand幅・Full packetを再現したものではない。sleepだけのmockは使わない。heavyの32項も実F3 enclosureではなく人工endpoint surrogate。実G7/F3/ReturnKernel/G9はimport blockerで拒否する。

現行topologyは8 pair・単一pass。64-taskは8 pairの独立人工replicaを増やした機械scalingで、本番で64 pairが存在するとは主張しない。別測定の16-task pass overlapはbaselineとinstrumentedを同時に投入するR1変更候補。pair分割は実装していない。新規採択を要する候補の測定許可と科学契約の変更許可を区別する。

各設定は3反復、設定順は固定seedで並べ替える。scaling前に8 pair×16/axisをwarm-upし集計から除外。共有OS cacheはflushしない。microbenchmarkは各process内で6回のwarm/calibration、目標約0.2秒のinner loop、3反復。OS cacheはuncontrolled、cold performanceや専有機性能を保証しない。性能suite同士は順次実行する。

## 費用

Wallはdispatch開始から全owned workerのjoin完了まで。全worker CPUはjoin済みのRUSAGE_CHILDREN差分で取得し、parentのdispatch/IPC/owned-RSS observer CPUを別記・合算。workerごとのCPU/wallはrun_pair準備からcloseまでで、IPC送信・process終了は全worker CPUに含まれるが各task CPUには含まれない。worker reuseは独立process/setup/HWM条件を変える別比較。

setupはtask内のendpoint/Meter/Stream構築まで。process起動費は各task wallとの差・小task測定で観測するが、親子の重複wallを加算しない。gmpy2/orjsonの最初のimportは各fresh childの初回使用時の実費に含まれる。microでは事前importを明示し、warm arithmeticとfresh end-to-endを区別する。

RSSはworker自身のbaseline・HWM・finalを保存。schedulerは自分のparent/active childrenのRSSだけをsampleする。RSSの和は共有pageを二重計上し得るため物理使用量ではなく、連続peakでもない。HWM上位worker数個の和は上側参考値。reuseのHWMは同一processの過去taskを含む。logical output bytes、/proc/self/io差分、write throughputを記録。fsyncしないためNVMe永続write throughputとは呼ばない。

出力再読・chain/file hash照合・非決定metadataを除く正規化比較はtimed regionの外で行う。これは人工検証費用を本番費用へ混ぜないため。normalized streamから除くのはgenerator_exclusive_nsだけで、memory telemetry行はtrial streamから除外。event/evidence/angleのhash、trial ID、kind、axis、retentionは保持する。元の実測metadata込みchain/file SHAは各run内部で照合し、異なるrun同士の一致は要求しない。

各per-pair JSONLはexclusive new file。既存run directoryの再使用を拒否し、retry/resumeを実装しない。人工failure suiteは各回が別の独立fixtureであり、実one-shotの再実行ではない。測定中は固定SOURCEを変更せず、無関係worktreeの変更を停止条件にしない。

## 再現

専用venvのPython 3.12と依存gmpy2/orjsonを使う。system package・共有venv・cache・OS・quotaは変更していない。source inventoryを各suite開始時に固定。runnerの `--suite scaling|optimizations|hotpaths|stop --output <new-private-directory> --repeats 3` で各人工suiteを実行できる。新しい出力先が必要。

全172人工回帰には元110件、新しい10件、Python/native encoderそれぞれF01全26件を含む。新encoderのserialization faultは新しいencoder入口へ注入し、元の失敗prefix assertionsを維持した。

## 追加の全trial監査

測定後、既知の2445人工file（約897 MB）だけを16 owned calculation processesで再読し、1,152,889完了trialの連続index・count・bit数、chain/file SHA、重複IDを照合した。約1.18秒wall/16.06秒child CPU。実科学結果を読む監査ではない。replicaのglobal identityはrun directory/task key/local trial_idの組で、原schemaのlocal IDだけをglobalに一意とはしない。検証は著者である主agent自身のchecksであり、別reviewerの独立レビューではない。
