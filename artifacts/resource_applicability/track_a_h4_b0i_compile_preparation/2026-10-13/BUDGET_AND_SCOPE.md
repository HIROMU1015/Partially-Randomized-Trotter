# Call・CPU/RAM・storage・6/10/240の未承認案

## 6件の具体案

|geometry|prefix/q|axis|元B0の参考RZ/shot|追加compiled RZ|
|---|---|---|---:|---|
|.70 Å|5/1|cosine|8454|未測定|
|.70 Å|5/1|sine|8453|未測定|
|.80 Å|5/1|cosine|8458|未測定|
|.80 Å|5/1|sine|8457|未測定|
|1.40 Å|3/1|cosine|5218|未測定|
|1.40 Å|3/1|sine|5217|未測定|

同じ追加費用を両軸に仮定する保存値の逆転閾値は約2966/1019/212。
各新axis costは独立に取得する。実Δは増加/減少/不変があり得て、共通値やΔ=0を強制しない。
pre-transpile instruction countの一致を、compiled RZ一致の証拠へ格上げしない。
新trajectory/signal/SCF/DF/state0、旧cache節約予約0。

|項目|固定したproposal|根拠・限界|
|---|---|---|
|new compile cap|6|3 deterministic構成×2 wrapper；1 task 1 call、failure reservation返却0|
|旧H4 carry|73023 consumed/reserved、13615088338 bytes、268318.34715263546 s|元MAP_COMPLETEの固定summary。未解決5件を含む保守会計を保持|
|累積actual cap|74805、追加6件込み最大73029|旧capを変更しない。新grantは未発行|
|累積charge cap|17GiB、stage追加256MiB込み最大13883523794 bytes|旧charged bytesへ追加、reset/返却なし。今回科学charge追加0|
|worker/driver|各1 process、各AS/RSS8GiB|旧閉じた8GiB observer capの範囲。新実H4 peakは未測定で、capは所要RAM予測ではない|
|observer|AS256MiB/RSS64MiB、内部thread1|旧stdlib observer実装再利用、new grantのroleとして別承認が必要|
|admission|120.25GiB、headroom16GiB|旧conservative120GiB floor＋observer AS。予算計算上は8+8+16+.25=32.25GiBだがfloorを維持|
|CPU候補|worker0、driver1、observer2|1秒受動sampleで各busy0%、異なるphysical cores/NUMA0。提案のみ、権利・予約・現在の空き保証ではない|
|internal threads|OMP/BLAS/MKL/NumExpr/VECLIB/BLIS/Rayon/Qiskit=1、parallel=false|process限定env、既存venv/Git/共有設定は変更しない|
|wall cap|stage3600秒、task600秒|未承認の停止上限。expected timeは未測定。旧累積wallを0にせず、将来stage wallを加算する|

元H4 mapの旧wall上限解除後の記録は保持する。observerへ渡すprior_wall=0は新stage timerの基準であり、
historical carryのresetではない。historical wallはplanとterminal receiptに別保存する。

## Storage/inode案

- stage charge/allocated output/control/lockの合計cap256MiB。metadata/journalは2MiBまで、1 frame/file64KiBまで。
- observer traceは32MiB。最悪3600秒・1秒周期・8KiB/frameで約28.2MiB＋terminal reserveを含む。
- 6 task結果・6 reservation・source/auth/input receipt・native終端・lock・directory block丸めを残りへ含める。
- private temp/Matplotlib/XDG cacheはcontrol配下。残り最大約222MiBを共有するが、独立した追加予算ではない。
- output/control/lock合計256 inodes以内。worker4以上・12 stress・file per tickは作らない。
- fresh availableは320MiBと256 inodes以上、実行中も64MiB free/16 free inodesを終端余裕として保つ。
- user/group/project quotaはGETだけ。UNKNOWN/EPERMはgate不合格。shared quota/FS settingsを変更しない。
- 既存入力3 NPZは既にあるため、新たな41.16MB copyを予算へ予約せずbyte-only参照する。NPZ/raw/checkpointはcommitしない。

今回のsnapshot：effective available494775336960 bytes（約460.79GiB）、filesystem available401218416640 bytes（約373.66GiB）、
219734833 free inodes、block4096、user/group/project quotaはDISABLEDのKNOWN。
各scope PSIは0。OOM counterは0/0/1をbaselineとして保持し、既存1を消去しない。
snapshotはlaunch成立・予約ではなく、承認後のfresh readが必要。

人工testsは1 process/AS2GiB/CPU120秒/wall180秒で44件PASS、実行約2.15秒。
ru_maxrss報告1214550016 bytesは人工processのhigh-waterであり、実H4 workerの所要RAMや速度へ外挿しない。

## 10/240案は比較のみ

|案|calls|追加registry/資源と分かること|残る限界|
|---|---:|---|---|
|今回の最小pilot|6|代表3固定構成のaxis別実Δ；実装/registry/未承認planを具体化済み|397点全体・補助120構成のwinnerは確定しない|
|拡張pilot|10|.70 p4/q1とp5/q1、.80 p5/q1、.90 p5/q1、1.40 p3/q1、各2axis。4 calls/2構成とinput .90のbindingが追加で必要|zero-Δ最小だった5構成を測るだけ。実Δで他候補が補助最小となる可能性|
|全対照取得|240|6距離×20 B0構成×2axis。全registry・新scope/source/grant・output/carry・q>1の資源再見積りが必要|登録q domain内の補助familyまで。未登録deterministic手法・母平均/energy資源の保証ではない|

10/240のsource/registry/grant/runnerは今回作成・実行しない。
表の追加4件は対象集合の差であり、6件の既測定結果を再利用できる保証ではない。
再利用にはSOURCE/input/profile/numeric/terminal receiptの別scope bindingと明示採択が必要で、今回の節約予約は0。
再利用を保証できない10件の別campaignでは、new actual worst caseは10とする。
6件の結果を見て自動拡張せず、残callsのcache節約を保証しない。
old full-wrapper RZ値は測定済みだが、modifiedのwall/RSS/charge/compiled RZは未測定である。
未知の時間や全240件のmemoryを6件や構造instruction数だけで確定しない。拡張には別の科学判断と資源承認が必要。
