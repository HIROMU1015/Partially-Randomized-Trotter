# 2026-10-10：AX-2B fingerprint性能改善・v5準備STOP

前日のH4 v4 pilotはwrapper phase900秒で停止し、fingerprintの完了区間がwall時間を支配した。
利用者の「その作業に進んで」に基づき、性能改善・固定toy検証・実行前固定を行った。
元v4実行worktreeの2,022ファイルを別v5 worktreeへbyte一致でコピーし、旧source/artifactは変更していない。
開始日は2026-10-09で、branch/artifact batch pathは開始日を維持した。

controlled/nested toyをprofileし、v4のgenerator/JSON token経路に大量のcallsがあると確認した。
直接writerと、1 hash call内に限る容量制限付きdefinition canonical bytes cacheを追加。
同じbyte列を全てhashへ渡し、definition subhashへの置換、phase/numeric情報の省略、mutation検査削除はしない。
cacheはcallごとに破棄し、強参照でobject ID再利用を防止する。process memory cap8 GiBは不変。

固定5 toyすべてでlegacy/v4/v5 hashが一致。
3回交互測定のmedian speedupはflat2.05、shared nested123.12、controlled/nested3.64、distinct2.29、oversize2.41。
shared nested toyのPython追跡peakはv5 150,707 bytes。toy構造に強く依存し、分子時間/RSSへ外挿しない。
最終167 local synthetic tests passed。初回2 testの前提誤りを修正し、初回結果も保持した。
production sourceはその修正で変更せず、science/native全non-import ASTはv4と一致。

schema/implementation以外の科学plan、8 cell/28 wrapper、seed/compiler/cap、回路解放/diagnosticsを維持した。
157 science hashesで新manifestを固定。digestは
`e7b6d1fd05d5946f9ecaf3833655e78c9373d640e65d7757151fea24be3edf24`。
science_authorized=false、launch_allowed=false、assigned_cpu=null。
分子signal・trajectory sampling・compile・H4再実行/H6/H8/GPUは0。
総数値allowance・accuracy・shot/total cost/方式順位の研究判断は更新しない。

判断：v5の合成準備gateを通過した状態でSTOP。
次は同じH4 scopeの一回実行を判断する段階。新schema認可・CPU・exclusive outputを必要とする。
今回の認可を科学実行へ転用せず、全28件・四次baseline完了前にH6/H8へ進めない。
元worktree、旧結果・原稿・Track Bを保持し、commit/pushなし。

[v5準備・証拠・検証範囲](../track_a_ax2b_h4_fingerprint_preparation_v5.md)に詳細を記録する。
