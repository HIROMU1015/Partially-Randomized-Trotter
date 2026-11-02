# 人工port契約の独立受入

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

固定ports.pyの実APIは `ArtificialEndpoint.sample(bits,trial_id)`、`angle_for(result)`、`shared_identity()`、`ArtificialBits.bits(n)`。exerciseは固定 `run_pair` へ接続し、Outcome(EVENT/ZERO_R/ZERO_C,event,evidence)を渡す。新しい独立adapterはこれらの実装をimportせずに保存rawを読む。

|公開case|完了trial|未完了sidecar|独立分類|
|---|---:|---:|---|
|sealed_small|1,536|0|VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE|
|sealed_n256|16,896|0|VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE|
|S_generator|224|0|VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE|
|P1_angle|992|1|VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE|

[全検証結果](artificial_port_case_verification.json)。正常は48 pair/6 phase、4方式×A/B×X/Y、correctness16＋profile baseline/instrumented、S/P1。task順序/offset、candidate/input/release、global/local trial、event/weight、F01、worker copy、caller hash、CPU partitionを照合。

人工bitsを独立に再構成し、kind・raw/reduced word・child・ratio・weight・coefficient・proposal・phase・provider_calls・reverse wordまでexact比較する。浮動小数点変換なし。B EVENTのangleは `actual_angle=null` / `not_a_mathematical_enclosure=true` とevent再照会surrogate hash。数学的enclosureではない。

profileのtrial≥16ではSOURCEがfull event/evidence/angleをconstruct/hashした後にhashのみ保存する。独立人工replayで3 documentを再構成し、event/evidence/angle SHAを比較。full documentが保存されているとは表示しない。これは人工surrogateだから可能な検査であり、実G7への無条件な移送はできない。

S_generatorはSAMPLING、result_acquired=false。P1_angleはANGLE_AND_REQUERY、result_acquired=true、trial_completed=false、inflight eventとsidecarを保持する。後者の1件を完了992件へ加算しない。F01 prefixとstream chain/hashは既存独立consumerで再照合。typed failureを確率zeroへ変換せず、retry0/resume falseを確認。

portはphaseの全pairを完了してから次phaseを止める人工exercise。これをGlobal Bundleのarrival順permit/STOP後ENTER禁止の実装と扱わない。失敗phase内の他pairは完了し得る。
