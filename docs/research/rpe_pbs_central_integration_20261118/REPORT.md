# 中央管理・価格検証・PBSの実統合（2026-11-18 JST）

中央移行版、価格検証引継ぎ版、PBS専用版を実際にmergeし、合流した同じSOURCEで通常入口から実PBS人工検収を完了した。
検証済み統合版は **`925985b4ea76a15f8a07c507179d4fa6c492eabd`**。
実H4信号・量子回路compile・Q3本体は0件。人工compileはtext artifactの生成である。
運用と中央担当への取込みは[HANDOFF](HANDOFF.md)。

## 取り込んだ版

| 系列 | 取り込んだ公開tip | 固定実装 |
|---|---|---|
| 中央移行 | `674dc7874f04c76a11cffb3c412f9f4570d6afa6` | `6da8554e0040f98578667973e8802c680f87982e` |
| 価格検証・工程予算 | `8ba82646eae8a013b58a956bb10995c3c9451f86` | `c4e90c9e5ffc7230583567500a99c4ec5c255f7b` |
| 既存PBS | `5e6cb8f7113636ea78f623357ba0abe57d3b7d05` | `22a428bd3` |

中央移行をmergeしたcommitは`65d989a40`、価格側をmergeし重複終了処理を整理したcommitは`a6e99f46f`。
両tipとPBS系列は固定統合SOURCE925の実際の祖先である。前回のAPI差分照合だけで統合済みと扱った記録ではない。
起点の確認と判断は[MERGE_INPUTS](MERGE_INPUTS.json)。着手時点のcurrentは`romutail-6da8554e0-r4`で、
より新しい統合済みSOURCEはregistry・固定checkout・公開branch・配置draftには存在しなかった。

中央側の利用者Stopの`None`返却、runtime状態表示、Windows Git SSH、WSLのfile-input/scp輸送を取り込んだ。
`tools/`は中央tip674とbyte一致する。価格側の共通ModelStore、巻戻さない引継ぎ、metadata更新後の再開、
明示`preparation_budgets`、科学組立てSOURCEとworker SOURCEの分離も保持した。
SCI Q3の`28d976b99`、timed adapter349、組立てd887、既存原結果・STOPを更新していない。

## 起動・終了の接続

方式viewに複写されるowner専用finishは1箇所で除外する。方式ごとの終了では中央、PBS manager、
任意の価格検証service、relay、RPCを終了しない。ownerの本終了または起動失敗時だけ後処理する。
価格serviceのcloseはidempotentで、重複していた`pop('finish')`も1つへ整理した。

中央既定の`distributed.relay`未指定時は既存の独立常駐relayを操作しない。
価格側のようにrun専用relayが明示された場合は、そのrunの`distributed/relay`だけを起動・停止する。
設定中に共有state directoryが指定されても、run所有のdirectoryへ解決される。
共有`.rpe-central/RELAY.json`やそのSTOPをPBS側の後処理で変更しない。

[既存接続検査](CONNECTION_TESTS.txt)88件（87 PASS、旧配置専用1 SKIP）と
[合流所有権検査](OWNERSHIP_TESTS.txt)4件が通った。計92件中91 PASS・1 SKIP・0 FAIL。
中央Stop、release選択、Windows/WSL、store・工程予算、通常終了・起動失敗、旧run pinを含む。
SKIPはこのhostに存在しない旧local0配置bootstrap専用ケースである。

## 同じ合流SOURCEでの実PBS検収

controller、PBS gateway、計算node worker、人工leafを925に固定して配置した。
既存の専用Python3.11.0rc1と、同一の科学/runtime/input profileを再利用した。
[profile同一性](deployment/PROFILE_REUSE.json)がPBS transport pin以外の一致を示し、
前回の保存実入力4方式Q3＋準備15件のmetadataを再実行せず再利用した。
計算nodeのworkerはPBS実割当2CPU/4GBの範囲、workq、walltime4分で動かした。

通常入口`scripts/run_rpe_schedule_search.py`から候補・taskを選び、自動qsubで
**`267195.CPUcluster-PRISM`**を投入し、計算node`CPUcluster035`で処理した。
14人工task（8 Q3、4 signal、1 compile、1 audit）を各1回採用した。
同じPBS workerがcompileと2 Q3を処理し、そのcompile artifactをRomuTailのauditへ渡した。
8件のexact rational結果と、方式別best／rule_adopted_best／confirmed_all_bestの設定・shots・MSE・costが
同じ925の単機参照と一致した。

qsub受理後の応答欠落と、compile完了後・中央配送前の切断を人工的に発生させた。
controllerだけを終了・再起動し、同じjob・token・原compile attempt
`604c1df6401f412ca9866dcf9926993c`で回収・採用した。
全taskのASSIGN/ACCEPT/ADOPTは各1回。最終の完了run再開は0.0654秒で、
追加LAUNCH/ASSIGN/ACCEPT/ADOPT/PBS割当はすべて0件だった。

原記録は[比較・再開](cluster/COMPARISON_AND_RESUME.json)、[単機参照](reference/FIRST_ACCEPTANCE.json)、
[PBS終了後の保存成果](cluster/SAVED_WORKER_HARVEST.json)。
jobはF、exit0、PBS run_count1、worker EXITED。保存3成果、未完了0、残存0。
[最終確認](FINAL_JOBS.json)と各TIMING.jsonを参照。実H4の計時・性能を測定した結果ではない。

## 配置と登録

中央用の正規固定checkoutは`.rpe-central/releases/925985b4e`。
検収用checkoutと対象3,398ファイルが一致し、既存Git objectを読み取り専用で共有している。
[配置照合](deployment/CANONICAL_SOURCE.json)、[クラスタへの不足配置と再利用](placement/PLACEMENT.json)。
cluster releaseは`/home/AbeHiromu/projects/rpe-pbs-20261118/releases/925985b4e`。

最新価格c4のSERVICEを基に、925の制御コード・Windows/WSL修正・PBS925・共通store・工程予算を組み合わせた
新しいSERVICEを生成した。科学組立てd887、Q3 28d、保存store、既存worker/profileを保全した。
テンプレートの`science_approved=false`、Q3・準備予算0を保持する。

**`romutail-pbs-integrated-925985b4e`をVALIDATEDとして非アクティブ登録した。**
[REGISTRATION](deployment/REGISTRATION.json)に検収SOURCEと証拠を結合した。
currentは登録前後とも`romutail-6da8554e0-r4`。PBS担当は切り替えていない。
通常RomuTail gatewayの明示release Previewと、既存Windows通常clientからのPreviewも925で通った。
[Windows実機結果](deployment/WINDOWS_PREVIEW.json)。client設定・WSL受付・共有relayは変更していない。

新SERVICEはc4由来のrun専用relayを使う。中央担当のcurrent切替では、旧常駐relayの所有taskを照合し、
新旧relayの所有を重複させずに移す必要がある。切替操作は中央担当に集約し、具体手順をHANDOFFへ記した。

主担当Codexが実merge、競合整理、実PBS障害操作、生成・登録・Windows実機Previewを担当した。
sub-agent `pbs_worker`が合流所有権検査・単機参照・原attempt/採用の照合、
`pbs_readiness`が中央進行状況の確認・固定配置・profile再利用照合を担当した。
これは技術接続検収であり、新たな科学的評価や独立の重要科学reviewではない。
