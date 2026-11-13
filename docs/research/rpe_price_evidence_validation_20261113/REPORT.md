# RPE平均compiled価格：証拠coverageと有限検証案

2026-11-13。**現在のPR-LD1/LD2・旧Full PF recipe catalogを208軸セルへ静的に対応付けた。モデルの再実装・実S/C再解析・fit・compile・実Gは0。** 既存C1914のfalse資格を維持する。新しい価格の値や係数は生成していない。

[COVERAGE.json](COVERAGE.json)／[D向け有限検証案・必要精度](VALIDATION_PLAN.md)／[HANDOFF](HANDOFF.md)／[人工・回帰ログ](TESTS.txt)。[BUILD_COVERAGE.py](BUILD_COVERAGE.py)は固定Git JSON/CSVからmetadataだけを投影する。入力snapshot、Dの実constant reader、kernel配列、private event/cacheを読まない。NumPy/SciPy/Qiskit/PySCFもimportしない。

## 固定した対象・SOURCE

基点C `1914d0764115d82ec860e08810dafeb50f8d9a42`、同リソース評価系列のC branchを継続。A/B/Dの作業中worktreeを読取・変更せず、公開commitのGit objectを参照した。

|用途|今回確認した固定SOURCE|
|---|---|
|現PR構成|A `a9abf8de75bf3432fcd714e868e5c14c8884b8f8` のLD1/LD2。完了signalはA91349を再取得しない|
|Bの現状と精度要求|`d71721f30c7b7fc13ba0789eefffd6238aca3ebc:docs/research/rpe_deterministic_risk_20261113/`。MC無しriskは技術実装済み、実H4への適用・具体N/十分N資格・価格precision budgetは未取得|
|AのFull PF選定状況|`2f0de95304000af251a179fefdddf2a10b8fdc14:docs/research/pf_effective_hamiltonian_20261113/SCIENCE_PLAN.md`。Full PF一候補・一幅の科学pilotも未実行で、公式/Δ/格子を正式選定していない|
|構造費用の既存優先材料|B `23d74855135e718554d824a269427de9943213d1`。nominal×ideal E[S]のP0–P4であり正式G/価格重みではない。再集計しない|
|構造平均|C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` の26値。値/区間の再計算・新価格への数値結合なし|
|直接paired/cohort|D `39ddd59edf54a3a901e5302207b2765ddbd03fbb` P2の保存summary。H4 DF10/S2/LD1/K2、m0/m6、fresh n4、role DEVELOPMENTを原本どおり保持|
|Full PF direct|D `e7dca5f7301c8633b17a59b6a7161331303aef2d` の確定receipt catalog。73件の再監査・価格再計数はしない|
|C1/価格モデル|C1 `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b` と凍結7eeabf0a、mean_compiled_price、C1914。係数・role・資格を変更せず、quote_modelは呼ばない|
|D corrected scalar|`e827367629ca6c2920d9d908fc1631314399db86`。expected total=c_H+s_RのSOURCE付きselector修正。旧748/T2のIDを上書きせず、新科学runの価格が得られたとはしない|

共通物理条件はLinear H4/1Å/STO-3G、保存tol-only DF10、同H/state/snapshot・sector(2,2)、元full DF-H target、h=.1/M12/P0、元fragment順。PRはS2/Δ.1/K2・Aのexact共通κを維持する。保存stateやsolver proxyを厳密ground保証へ昇格しない。

## 候補×round×axisのcoverage

|対象|静的セル数|利用可能な保存証拠|今回認定していないもの|
|---|---:|---|---|
|PR-LD1 全13round XY|26|同physical ID/q/R・H/state/snapshotのE[S]26軸。m0/m6のP2 paired有限cohortは計4軸、各n4|現B identity/law_bindingは元E[S]の宣言と異なり、全26でそのまま一致しない。P2もDEVELOPMENT/source bridge未受入。全law E[C]/CIはnull|
|PR-LD2 全13round XY|26|A/Dの全schedule metadata、修正済みscalar selectorのSOURCE|同prefix2のE[S]・C1校正・価格cohortは未取得。T2は価格取得前STOP、corrected metadataだけで価格を埋めない|
|Full PF 6旧recipe×13round XY|156|同physical candidateの確定direct receipt 16軸。別8軸は同catalog内MISSING、残りは照合catalog外|6 recipeはS2/Y4/M5×Δ.1/.4の参照catalogで、今回採択したA menuではない。旧native lowering以外の再利用と現V2 traceは未受入|

208はconfiguration/round/axisセル数で、208個の独立回路・独立標本という意味ではない。early roundで実幅が縮まり、別のrequested Δから同physical candidateになる場合もある。compile重複排除はphysical/source/profile/axisの同一性で行う必要がある。

|Full PF参照recipe|m3(T=.8)|m4(T=1.6)|他round|
|---|---|---|---|
|S2/Δ.1|q8 両軸EXACT receipt|q16 両軸EXACT receipt|照合catalog外|
|S2/Δ.4|q2 両軸EXACT receipt|q4 catalog内MISSING|照合catalog外|
|Y4/Δ.1|q8 両軸EXACT receipt|q16 catalog内MISSING|照合catalog外|
|Y4/Δ.4|q2 両軸EXACT receipt|q4 両軸EXACT receipt|照合catalog外|
|M5/Δ.1|q8 両軸EXACT receipt|q16 catalog内MISSING|照合catalog外|
|M5/Δ.4|q2 両軸EXACT receipt|q4 catalog内MISSING|照合catalog外|

上のEXACTはD既存receiptの判定の参照で、Cが今回受入を再認定したものではない。将来Aが別Δ/公式を選べば、6 recipeに丸めず新metadataへ同catalogを照合する。

C1のH4校正はprefix1/q8,16/R4,8・旧lowering。現LD1は当該qでR16,32等となりdomainが違う。LD2はprefix自体が違う。固定係数を移植しない。S0aの8件は別Rの保存path S parityで、平均Cの証拠ではない。加法モデルは208セルともIMPLEMENTED_NOT_EVALUATED、prediction=null/qualified=false。

Truncated B0I/p7/S2/.1のm3両軸には旧carrier155529e2の保存価格SOURCEがあるが、Full PF73 receiptはtruncatedの資格ではない。Full-H旧light n9はK4/T.8/q16/R16等で、未承認seed K2/q=2^m/R448やPR tailとは異なる。両者はCの既存c546計画/coverageを参照し、今回新規scheduleや価格モデル評価を追加しない。

## 比較単位と接続の確認

full controlled evolution＋X/Y Hadamard＋measurement、scalar一回、状態準備P=0、compiled RZ gate_countを維持する。normalizer Bは価格へ乗算しない。旧Full PF native policyとD新ZZ loweringをcompiler hash一致だけで混同せず、同loweringでのreceipt bridge又は別の直接証拠が必要。X/Yは独立の価格応答で、同pathの二軸を2n標本としない。

D最新版はinput_source.scalar_bindingへconstant/tail/totalのquantity・SOURCEを追加するが、from_b/from_a_job/check_requestはC固定D748と同一のASTだった。module全体はobserver差分があるためbyte一致とは言わない。C3802の既存転送・anchor/hex gateで足り、production変更・pin更新は不要。新1件＋既存13件＝**14人工・回帰PASS**。source-bound totalをそのまま転送でき、tail-only resolved scalarは拒否された。実D selectorのconstant readerやpreparationは起動していない。

C1914のモデル/MeanKernel/B接続38テストは固定証拠を再利用し、重複実行しない。新しいモデル推定系、N solver、費用optimizerを作っていない。今回のcoverage readerと人工metadata確認は科学的な独立レビューではない。

現時点ではBの適格N・比較marginとAのFull PF選択が未確定なので、**必要な数値価格精度と実compile標本数はUNKNOWN**。[VALIDATION_PLAN](VALIDATION_PLAN.md)はこの条件を明示した最小bootstrap案で、実行許可ではない。保存S/Cの新fit・再集計や正式G/winnerへ進まない。公開照合後mandatory STOP。
