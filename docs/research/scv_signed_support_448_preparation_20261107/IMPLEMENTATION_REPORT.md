# Signed Support 448-shot：最小activation実装

2026-11-07 JST / SERVER1 GPUA100-gleap-01。**技術実装・人工検証・SOURCE候補の固定だけ。実H4科学実行は未許可・未開始。**

対象仕様は固定 `138d071805430ef1e556655392e4d9f572feba40` の4文書の第一候補。ユーザーが実装仕様として採択した448 shots、seed20261110、signed幅6を使用する。旧契約文書は変更せず、384-shot代替へ切り替える機構も設けない。

## 追加した境界

| 新SOURCE | 必要な差分 |
|---|---|
| `src/trottertracks/algorithm_codesign/scv_signed_restore.py` | 既知schemaの保存controlだけを復元する93行のfactory。通常momentコンストラクタを呼ばない |
| `scv_signed_pilot.py`（同directory） | Signed用purpose、64 train/軸、全5層quota、軸別一回fit/guard、Full/half診断、受領・会計・STOP |
| `scv_signed_runtime.py` | 同じworker/group/poolの周囲に有限counter、global trial受付、短いSTART/STOP排他、tail会計、timerを接続 |
| `scv_signed_activation.py` | input/profile lock、保存復元一回、親でsaved bundle一回decode、fork継承、新namespace、既存持続managerへのrouting |
| `scripts/tracks/algorithm_codesign/run_scv_signed_pilot.py` | 既存と同型の受動CLI。今回はhelp以外の実管理操作を呼んでいない |

旧signed kernel、sampler、OLS、exact engine、価格器、worker、managerファイルは変更していない。旧Driverはindicator balance、32 train、224 cap、indicator rangeを含むため継承せず、適合する処理だけを呼ぶ。

## 保存controlとidentity

固定metadata結果 `b4512675ae1d650858256547d359210361d77f68`、`artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/METADATA_RESULT.json`、SHA256 `c5895f7696711c92bf87e591a3c2264cd7672d9af70d13027db790950e305ba0` にbindする。

既存signed binderでLAW/catalog bytes、H/preparation、label/sign/support/qubit/patternを照合する。保存全5層のp/ν/m₂/Σをcanonical Fractionへdecodeし、pの既存層bindingとの一致、Σ=m₂−ν²、positive Σ1、snapshot完全一致を検査する。保存ν/m₂を使う既存fingerprintを照合し、SignedControlの既知5fieldだけをfrozen型へ復元する。label特徴の新しい有限和・path列挙はない。

metadata identity=`c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2`、control identity=`5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89`。実H4での復元そのものは今回は実行せず、人工保存結果で検査した。normal `SignedControl.__post_init__` と旧indicator選択は将来のparentでも禁止する。

## 再利用

- b23b3c0の `SignedControl.value/snapshot`、`ConditionalBank`、`fit_h1`、`SignedFrozenCV`、`SignedModels`、`paired_evaluate`。intercept OLS、binary64 slope→Fraction、signed boxを維持。
- `ConditionalSampler`、`Domain`、`Streams`、`pcg64_streams`。同じactual53-bit直接条件付きQ。A1はh1 trainの取得tagだけ。
- `SavedEngine`、`MeasurementFacade`、`SavedPrices`。CVへ渡すのはbit/path metadataだけ。state/Born/E0/gap/真信号/真varianceは渡さない。
- `Campaign.group`、`prepare_trial`、`worker_trial`、`reconcile`、`Envelope`、`cost_summary`、`account_costs`。旧run/learn/execute、fragment selectionを呼ばない。
- qualified manager SOURCE `5cb542ece4d208878a1dfc7011007b99297a5a55` のcron→outer recorder→manager→private parent、`ManagedPool`、owned-child終了回収、consume-once/Restart=no。新managerは作らない。今回cronは未変更。

SOURCEファイルの改変ではなく、新process内のscoped wrapperで受付・観測を追加し、終了時に戻す。engine/probability/build/compileの引数と返り値は同じbodyへ通す。公開lockは979 SOURCEファイルのbytesを固定し、認証済み旧SOURCEの検査も維持する。

## 順序・独立性・診断

X train64→Y train64→X fit/guard→Y fit/guard→両軸凍結→X eval160→Y eval160。width=2+2abs(λ)≤6をexact比較し、他4層λ=0。失敗時clip/refit/追加shotなし。両軸fit PASS前のeval受付を拒否し、X fit失敗でも両train費を残す。

12 domainはtoken辞書順、path child2i/bit child2i+1。mappingは[SETTINGS.json](SETTINGS.json)。global trial=(campaign,token,trial_id)、0起点quota内、共有planned/attempted bitmapで二重worker測定を事前拒否。workerは保存path/bit wordをreplayし、新PRNGを消費しない。

Full counts=(8,128,8,8,8)、half=(4,64,4,4,4)。各層trial IDで固定二分し、worker完了順を使わない。同じ係数・recordから計6集計。Half spreadはCI/SEではなく、退化/疎・不安定flagを併記する。kernel snapshotのrange_guard=NOT_ADOPTEDは保ち、外側の幅6guardを別fieldで保存する。

## STOP・有限call・観測

短い受付区間に既存fcntl.flock方式を用いる。outer区間ごとに別descriptorを開き、process消失時はOSが解放する。同一processのthread排他と再入管理、parent signalの短いmaskでSTART/STOPを線形化する。**gate/compile本体はlock外**。長い計算を直列化しない。lock所有の人工child SIGKILL後にもdurable STOPできることを確認した。

STOP_REQUESTEDをdurable保存してから共有eventをsetする。STOP後のprepare/word claim/CALL_STARTを拒否し、受理済みin-flightのRETURN/EXCEPTIONは保全する。未開始をcancel、既存gate境界で停止、同じworkerを再起動しない。bitは既存workerが価格より先にfsyncする。

静的capsとstarted/returned/exceptions/in_flightは別field。trial/path/engine/Born/threshold/price≤448、selected full≤896、compile≤448、word≤29120、apply≤427840かつ各probability≤955、fit≤2、paired≤6、decode/restore/stream book各1。RTE build_sequenceとwrapperは固定selected_circuitの内部各一回で上限896から従属し、独立した実測call数と偽らない。

部分groupも既存worker全journalをreconcile/classifyし、planned未開始・bit欠測・価格欠測を残す。`SIGNED_TAIL_ACCOUNTING.json` は将来のoutput名で、既知RZとtotal UNKNOWNを分離する。未完走の費用比較を完走扱いにしない。

callごとのinclusive wall、親path/fit/paired、親user/system CPU、worker user/system CPUの開始/終了値、RSS/PSS、cgroup祖先・memory/pressureを別記録する。worker wall和をCPUや親wallとしない。price−compileにはbuild等が含まれ、純cache/lookupはUNKNOWN。観測目的で数値kernelを置き換えない。

## 実運用の残る条件

実空きCPU二窓・affinity、memory/cgroup headroom、memory pressure、最大有用stage160 taskに応じてWを決める。任意の低い固定上限・共有設定変更はない。親RSSは初期worker footprint proxyで、Fractionの将来peak保証ではない。path投入前にもmemory状況を読み取り、不成立ならSTOPする。実H4長時間負荷と性能、全host/recorder同時消失時の記録回収は未保証。

今回は実H4復元・bundle decode・科学PRNG・fit・Born・shot・build/compile・価格・paired集計・manager起動を一切行っていない。公開照合後mandatory STOP。次回はこのSOURCEで一campaignだけ実行する明示許可が必要。
