# Pilot規模と費用の設計候補

推奨検討案はprofile各cell n=256、SとP1の二条件、fused codec・CachedObserver・同じ保存/trace粒度。16 cell=4方式×2layer×2axis。各条件にcorrectness256、baseline16n、instrumented16nを置くため、合計は2(256+32n)。旧Full-N契約は256+15,855,296+15,855,296=31,710,848のまま。

|候補|profile/cell/pass/条件|correctness合計|online合計|総calls|用途と限界|
|---|---:|---:|---:|---:|---|
|短い入口案|64|512|4,096|4,608|4個の16-call block。経路不足や短時間計測の可能性が高い|
|推奨検討案|256|512|16,384|16,896|16個の16-call block、4個の64-call blockを同じprefixから集計し、path頻度・観測CPU安定性を記述できる|
|長めの案|1,024|512|65,536|66,048|64個の16-call block。長めのprefixを最初から選ぶ案であり自動延長ではない|

256は必要十分な確率保証として選んだ数ではない。最小correctness16を維持しながらsetup/onlineを分離し、4方式A/B・X/Y・両pass・S/P1の全組合せと複数blockの記述比較を行うための、64と1024の間の提案である。実G7のpath率/CPU分散は今回測っていない。十分性を保証できないため**ユーザーがNを選ぶ**。一つのprefixを複数blockに分けても独立反復ではない。

科学repetitionは各条件1回の案。同じseedを再実行してCPU母平均CIを作らない。cold/warm cache、実行順、同時負荷、CPU affinity、runtime、pressureを観測として保存する。S→P1の固定順には順序交絡が残る。独立runや順序反転が研究上必要なら、別の事前採択された反復数・global予算・fresh claimを要し、結果を見て追加しない。

correctnessの必要条件、raw zero/coin reject/acceptedの観測数、A/B trace、pass trace、correctness prefix、bit消費・canonical・hash・completed prefixを確認する。成功receiptだけでは未出現pathをPASSにしない。未coverageなら結果をその範囲へ限定するか、後でユーザー/GPTが別実行の必要性を判断する。

CPU計測精度は、clock実効粒度/呼出し負担の人工校正とblock累積時間を併記し、短すぎるcellを分解能不足とする。block最小/最大/中央値とprefix16/64/256の記述値を示す。IID samplingや系統誤差モデルなしにSE/母平均CIを付けない。量子confidenceの転用は禁止。最終の精度要求・必要block長はユーザー/GPTの採択事項である。

|費用境界|記録と解釈|
|---|---|
|host outer|admission前から全join・出力finalization・STOPまでのparent CPU、全worker CPU、global monotonic wall。report自身の保存やoffline検証は別の後処理window|
|pair/setup/online|各pair CPU/wall、constructor/prepare、source/identity checks、generator、audit、angle/requery、evidenceを分ける|
|coarse/fine|それぞれouterへ別々にreconcile。fineはall-call/return CachedObserver、residualを明示。coarseとfineを足さない|
|保存|exact serialization、hash、write、flush、summary/receipt、state/trace IOを記録。flushとfsyncを区別|
|IPC/observer|IPC call/CPU/wall、launch、observer callback/残差はcontained diagnostic。parent+workerへ再加算しない|
|memory/IO|各worker RSS/HWM・親/workerのsampled合計RSS、shared-page重複、sampling間隔、logical output、物理保持bytes、rchar/wchar/read_bytes/write_bytesを区別|
|障害|failure prefix、drain費用、unknown、未ACK creditを別表へ。正常比較対象外。欠測はnull|

total CPU=host parent+全worker、global wall=host outer wall。pair wallの和はglobal wallではない。CPUにpair/IPC/setup/observer/fineを重ねて加えない。SもP1も同じfused・CachedObserver・new trace・入力・保存・guard条件とし、古いheavy observerのS0と比較しない。

この二条件は並列化の固定portfolio運用費用を比べる。fused単独の増分改善率は識別しない。必要ならP1/reference codecを第3条件として事前採択する追加案があり、n256ならさらに8,448 calls、総25,344 callsとなる。これは現在の二条件JSON候補へ含めておらず、選ぶ場合は契約/SOURCE bindingの改訂が必要。既存人工P→PF利益を実G7へ外挿しない。

CPU秒とT/CX/量子callは別列で扱い、価格scalarを採択しない。Fullのlocal coefficient取得、child law、rational正規化、packet/angle requeryはfine関数labelとcoarseカテゴリを対応表で結ぶ。並列時のcache/帯域/IO競合があるので一般的な方式固有費用とは呼ばない。

今回の人工normalは768 calls、parent約0.501 CPU秒、全worker約5.558 CPU秒、wall約1.593秒、log約1.68MB、result約0.995MB。これは新trace/guard/保存を含む人工proof runであり、性能比較や実G7の予測値ではない。詳細な実測値はforward_evidence/normal/observations.json。
