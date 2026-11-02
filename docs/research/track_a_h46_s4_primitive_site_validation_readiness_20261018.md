# H4/H6 S4：負stage primitiveの実block別確認

利用者の「次の作業に進んで大丈夫」により、先行報告b7385211…の最大66 primitive評価案が承認された。
元の未承認plan・SOURCE・結果・grant/ledger/STOPは履歴として保持し、新branchに独立した実行を固定する。

対象はH4 full DF12/p12/q1/S4とH6 full DF19/p19/q8/S4、T=.8、symmetric-directional。
元の科学SOURCE536a3671…、native provider217c971d…、H4 codec/H6 archiveの固定bytes/SHAとpreparation hashを照合する。
typed DAG metadataから選択diagonalだけを読む。実回路・live scientific preparationは復元しない。

`validate_h46_s4_primitive_sites.py`は元producerを変更せず、各blockにmodel/nativeの負時間を各1回与える。
最大33 block×2=66予約を事前journalへ記録。q反復と両軸は再利用する。
保持RZのindex/angle hex・非zero RZZ・phaseを保存し、局所差を元8行の保存差へ照合する。
一致しなければSTOP、自動retryや範囲拡大はしない。

新trajectory/control/reference/compile、full-wrapper生成、理論費用表の再計算、実装修正は0。
全結果・callbackはprimary agentの技術確認であり、GPT/Claude詳細科学レビューや独立別担当reviewではない。

既存private venvを変更せず利用。小規模な1 process、NumPyのみ、内部thread1。
固定SOURCE公開後、fresh空きmemory/affinity・source/input・未使用outputを照合して一回起動する。
新たなAS/RSS/CPU/wallの恒常上限や共有環境変更は加えない。
native childのexit/waitとpeak RSS/wallを保存し、結果公開後mandatory STOPする。

次の担当：CodexがSOURCE固定・限定実行・結果監査と公開まで実施する。
