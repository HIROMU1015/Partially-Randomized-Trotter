# 横断索引：結果・終了・未完了を区別する

詳細な問い・構造・数学・費用・利用情報・結果・差分・再利用価値は [CSV](algorithmic_evidence_crosswalk.csv)。各ref IDの固定commit/path/SOURCEとSHAは [SOURCE索引](source_catalog.json)。新しい性能比較は行っていない。

## 優先branchの現行状態

|読取入口|観測したremote HEAD|本調査で優先する後続資料|
|---|---|---|
|pr2-v4-s2-parallelization-20260928|b2e1bf65…|M1-B1完了→M2 TRANSFER_SUPPORTED→PM-0/PM-1/PM-2結果。旧M2実装資料の「未開封」は履歴|
|track-b-algorithm-codesign|09c9e895…|当該HEADはBF1_INCOMPLETE。後続BF_CLOSUREが限定negativeとして主仮説を閉じる。元runを成功へ書換えない|
|track-b-g3-finite-law-diagnostic-20261009|3b0fa70b…|G3の有限law/既知return比較完了。後続D/G6では同じtoyの保証・価格・生成scopeを更新|
|track-b-de-closure-and-next-direction-preparation-20261015|101bf342…|Parent allocationはS4訂正後の正式集合を保持しD＋E採択済み。旧v3の判断待ちを現行にしない|
|track-b-c1-freeze-and-manuscript-synthesis-20261021|185ec4e9…|C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT。G6数学維持、C1追加科学実行は凍結|

## 分類別の読み方

|分類|該当する代表系列|再検討しなくてよい範囲・残すもの|
|---|---|---|
|実用情報で中心gate不通過|FR-1 AVAILABLE、BF有限目的、P-A interval層|真値半径、同じfamily/budget、同じ退化proxyを使う救済を繰り返さない。機構とrun-level利益は保持|
|既知式との同値/方法差不足|BM三次BCH、Gaussian active-span案、ordinary Euler端点|同じ列/集約規則/既知compiler部品を名称変更して新アルゴリズムとしない。Gaussianの性能不利を実証したのではない|
|全量化した命題への反例|G6 P_U|既存Uが全benchmarkを支配するという命題は再提出しない。真のB_fullや全G6方式の劣位を意味しない|
|限定positiveとtrade-off|PR-2 M1/M2、G3、Parent S4、cluster/P5|物理条件・指標・access・有限候補・不確かさを保持。RZ/T/CX/1Qのwinnerや母平均を統合しない|
|技術停止後に別来歴で前進|BF R0復元、DF E/T保存復元、C1 R1/R2/P5R|旧failureは保持。復元・新実行の完了と、未取得native/全task保証を区別|
|科学的効果の判定に必要な証拠なし|C1認証native総G、Gaussian期待利益、候補別安全なbias proxy|UNKNOWNを性能negativeや費用0へ変換しない。C1休止を解除しない|
|使用不可の古いscreening|partial_randomized_pfの隔離出力|ground-state不一致で撤去されたCgs入力由来。モデル学習・正解labelへ使用しない|

PR-2の「proxyが失敗した」は、保証付きfrontier全保持に失敗した意味と、primary RZ最小のpoint regretを分ける。PM-0の保存値では後者は0である。ここから旧barrierの解除や一般的selector成功は導かない。

toyのG3/D/Parentは2〜3 system qubitsの既知provider。C1はH4 DF12だがtail＋scalarの探索logical計数。H2/H4/H6共通nativeのfull-wrapper費用と同じ母集団へ結合しない。K（paired cutoff）とm（Taylor polynomial degree）の意味はSOURCEごとに確認し、一律に同じ整数として結合しない。
