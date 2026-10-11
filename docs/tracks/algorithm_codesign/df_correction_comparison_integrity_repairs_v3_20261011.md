# DF correction comparison：限定integrity修正 v3 / GPT引き継ぎ

2026-10-11 JST。**SOURCE_PREPARATION_COMPLETE / EXECUTION_NOT_AUTHORIZED / mandatory STOP**。
旧source S=`b4aba2b698cdc81710dbdbb5769a1686f716ffe2`を保持し、その独立branchから限定修正。
branch：`track-b-df-correction-comparison-v3-integrity-preparation-20261011`。
新source S'のfull SHAは公開receiptへ固定する。receipt-only childを実行sourceと混同しない。

## 入力原本と今回の範囲

[GPT source v2 review原本](../../research/track_b_df_correction_comparison_source_v2_review_20261011.md)：
30,217 bytes、SHA256 `a866d8c6f37456e072546aba02f540f56f7d64e344a09cd1409df59e96504278`。
入力commit `6e34162c03ef5ab2b66f568daa4eed4609497374`をpushし、GitHubから同じbytesを再取得した後に修正した。
レビュー補助codeは今回添付されていないため、原本MDと独立に用意したmetadata/mock controlsを使った。
補助ZIP/codeの不足を理由に科学範囲を拡張していない。

A/Bアルゴリズム、CD/E recipe・seed・初期状態・時間・補正rate、精度、confidence、誤差会計、
compiler/synthesizer/slot選択順序、最大22 native compiles / 64 T keys、CPU/RSS/wall/output capsは不変。
[contract_v2.json](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json)と
既存の数値moduleは旧Sのbytesを保持する。v3は保存／来歴の実装修正番号で、科学契約v2を置換しない。
runner名、authorization path/status、結果logical pathもv2 interfaceを維持する。
旧結果・marker・STOP・原本・source manifestsとv2の検証記録は変更しない。

## R1：大きい結果・prefixの完全取得

新module `storage.py`をworker／parent reader／guardへ接続した。

- serializerは元の `json.dump(default=str, indent=2)` と最終newlineを維持。
  512 KiB以下は元のlogical JSON pathに保持する。
- 大きいJSONはraw＋chunkの二重常駐をせず、最大256 KiBのUTF-8断片と
  `<logical-path>.review.json`索引を保存する。JSON valueを丸めたり省略しない。
- 索引は元bytes/SHA256、順序、断片bytes/SHA256、encoding、top-level field名とlist/dict件数を記録。
  writerは保存bytesの再結合を照合してから索引を公開する。索引自体も512 KiB上限。
- `load_json(logical-path)` は小JSONと分割JSONに共通で、全断片と元digestの照合後にのみvalueを返す。
  配布された断片を索引順でbyte結合すると元のserializer outputと完全一致する。
  不正順序、破損、欠損、path escapeを拒否する。断片はそれぞれ独立にvalid UTF-8。
- prefixは新世代を完成してから索引をatomic publish。途中停止時に前世代を保持。
  cleanupが途中で止まって旧rawが残っても、完成した索引を優先して検証する。
  正常更新後に消すのは同一NEW-run logical artifactの旧断片世代だけで、過去の結果を触らない。
- chunk/index/temporary/prefixも従来のaggregate output capに算入。出力上限を拡大しない。
  保存失敗や索引超過はtechnical STOPで、prefixを科学的結果としない。

[metadata分割例](../../../artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/metadata_review_example/)
には4,097件の**架空の文字列record**を用いた。6軌道係数、Hamiltonian、科学計算結果ではない。
GitHub再取得で確認するのはこのmockのbyte identityと新sourceの取得性であり、未実行E結果のサイズではない。

## R2：freeze inventoryの元SHA256の結合

新module `inventory_integrity.py`をcoordinator／workerへ接続した。

1. 元inventoryをexclusive createし、作成bytesのSHA256を直ちに計算。
   source/authorization/contract/marker、順序付きkeys・件数・bytesをfreeze receiptへexclusive保存。
   fileをread-onlyにするが、owner変更不能のsecurity保証とは呼ばない。
2. coordinatorは元inventory SHAとreceipt SHAをmemoryに保持し、全T job payloadに渡す。
   workerはその二つのdigestとmarker/来歴、key membershipを開始前・合成前・結果保存前に確認する。
3. job-start receipt、T resultへ同じbindingを保存。parentは各batch前、集計前、各resultのangle/binding、
   集計保存前、terminal保存前を元digestに照合する。
4. 最終T集計はbindingと各T resultのbytes/SHA256を保存。terminal provenanceにもbindingと最終照合status。
   元inventory変更が検出された場合、prefixの成否によらずTECHNICAL_INCONCLUSIVE / STOP。

現在のfile内容へ合わせてjobを作り直すだけでは元receiptのdigest条件を満たさない。
key削除・順序変更・角度変更・whitespace変更・同path置換・receipt変更をmockで拒否した。
合成対象、順序、seed、precision、呼出し上限を変えない。再合成や追加keyなし。

非blockingの堅牢化としてcoordinator自身も固定numeric runtimeの自己identityをmarker前に確認する。
source gateは新manifest v3に切り替え、旧manifest v2を履歴としてそのまま残す。
authorization-only直接子のpath条件は不変。

## 検証と不変性

[focused tests](../../../artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/focused_tests_final_v3.json)：
**88 passed（既存62 + integrity26）**。最終suite約1.51s wall、1.22s CPU、peak約140.58 MiB。
metadata4,097行、長い有理数文字列・Unicode・端数chunk、破損／順序／欠損／path、prefix中断、
分割readerとoutput guard、元inventory～job/result結合、改ざん、terminal STOP、親runtime拒否を確認。
初回88件PASS後、prefix cleanup中断時のreaderを補い最終sourceで88件を再確認した。
本番の数値性能やimmutable CI・外部再現の証拠ではない。

元suiteに含む小さいoff-domain compileは各1回、今回の準備全体2回。
**登録CD tasks、登録E compile、T合成、登録frame生成、分子/NPZ/GPU/LP/trajectory/量子shot、G10/RA-D0再開は0**。
source/code/旧evidenceの不変性、contractのbyte identity、protected source hashes、各review fileサイズを
新[準備artifact](../../../artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/)へ記録。

## 次の確認とSTOP

次はS→S'の限定diff、上記testsと契約不変性だけをGPT/利用者が確認する。
数理設計・研究方針を今回再採択しない。別の明示one-shot認可までauthorization=false、runs0、markerなし。
将来はS'から実行branchを作り、既定authorization-only直接子Aを固定する。
公開receipt-only HEADから実行しない。

既存の条件付きIEEE/BLAS numerical guard、未取得CD native費用、A T coverage、registered margin/cap/fingerprint
の未確定性はそのまま残る。失敗時にprecision/seed/fixtureを変えずSTOPする。
今回の取得性・来歴修正だけで科学的GO、優位性、algorithm採択を認定しない。**mandatory STOP**。
