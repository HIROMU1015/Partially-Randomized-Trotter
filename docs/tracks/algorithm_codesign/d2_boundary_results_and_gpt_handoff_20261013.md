# D2 共通ステップ境界相殺：一回実行・保存結果監査・GPT引き継ぎ

2026-10-13 JST。**D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW**。
六方式・219,303 ordered positive pairを一回の実行で完了し、mandatory STOP。
研究継続・algorithm採択・新規性・主要claimの科学判断は行わない。

## 固定来歴

- 新science source S：`28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`。
- authorization-only直接子A／実行HEAD：`9e753a0d807df683c86f7742ed0c14ddcd4e58ba`。
- 実行branch：`track-b-d2-boundary-one-shot-execution-20261013`。結果公開commitはこのreportを含むcommitであり、実行HEAD Aと区別する。
- 利用者原文：「次の作業に進んで大丈夫です」。直前の報告が固定SへのD2 one-shot承認を次工程として指定していた文脈で、[receipt](d2_boundary_execution_authorization_receipt.md)とJSONへ正確に記録した。source SHA等を利用者が同じ文中で再引用したとは表現しない。
- [contract](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json) SHA256：`fe85771bc3b4dbd9f1e9d485cc95cc046b6bc47fc1748b6c691270ff0fbb9e91`。
- [authorization](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/authorization.json) SHA256：`bec34d732af6e3c3de7de31dd0c5aa5c23242f735e96aa907844f7933fc54968`。
- [one-shot marker](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/one_shot_consumed.json) SHA256：`49d06c74b0dd081632d72d535dc655ca794fea8573951a63dad6af4c4be4c508`。消費済み。削除・再利用しない。
- [terminal](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/terminal.json) SHA256：`f6a7aa1c249c4f2ff935e1360d184a83360153634e81056761b25affe8300647`。
- 元D1 source `874ca12d347f8f35e5d8d64f65f24abc4a2df523`、result `fc96571881f301435a655f057709df4d6c1678e0`、保存分析 `2dcd56737be35526f6436a52835f59d817adf30c`は不変。

## 比較条件と実行範囲

3-system-qubit explicit synthetic involution、p=(1/5,3/10,1/2)、τ=5/7、σ=+、state=000、degree5、r=2、x=5/14。Q0=Z0、Q1=V1†Z1V1、Q2=V2†Z2V2、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)、R_P(θ)=exp(−iθP/2)。分子geometry/basis/DF rank/split L_Dは適用外。既知development入力で、独立validationではない。

保存済みstep IRを左step→右stepの時間順に扱い、D1と同じexact adjacent inverse cancellationを境界にも全方式共通で適用した。非可換因子の並べ替え、新angle、合成sequence内部の追加簡約、汎用compilerは行っていない。

proposal・weight・acceptance・N・Taylor/native誤差上界・precision・既存sequence価格は不変。どちらかのstepがzeroなら古典draw後、量子初期化前にwhole zeroとする。新平均費用は保存law上の全positive pairのexact集計であり、Monte Carloではない。

各axisのε=1/200、α=49/24000、各row resource failure=1/12000を保持。r=2 subsetの12axes+6rowsのallowance和は1/40。Nを新varianceや真の信号で縮めていない。

compile／新synthesis／新angle／matrix／sampling／trajectory／LP／DF／molecule・NPZ／GPUは0。新しい登録処理は219,303 pairのliteral境界相殺費用評価だけ。実量子shot数は0。

## 完了と資源

| 方式 | positive step support | ordered pair数 |
|---|---:|---:|
| ordinary |273|74,529|
| partial-return-tail |264|69,696|
| closed P3-tail |258|66,564|
| local full |63|3,969|
| closed P5 |63|3,969|
| matched CTS |24|576|
| 合計 |945|219,303|

外側observerを一回のみ呼び、outer exit0・signalなし・terminal COMPLETEの整合を確認した。runs=1、retry=0、次stage認可なし。

- outer wall 4.012546秒、CPU 1.729110秒、peak RSS 26,772 KiB（約26.145 MiB）。
- guard記録 wall 1.627554秒、CPU 1.586695秒、peak observed RSS 26,677,248 bytes（約25.442 MiB）。outerとguardは計測範囲が異なる。
- 1 process・numeric threads1、CPU affinity32、実行前MemAvailable 54,172,260 KiB。固定runtime SHA一致。GPU query0。
- CPU cap900秒／RSS512 MiB／AS1536 MiB／出力16 MiB／log64 KiB／review512 KiBを維持。wall capなし。
- 原run保存物13files、197,891 bytes（監査追加前）。stdout54 bytes、stderr0。memory/call/output cap failureなし。
- integer mass共有・suffix価格cache・streamingを使用し、全pair IRを保持していない。高速化前の登録比較は行っておらず、速度倍率は主張しない。

## 保存資源の比較

以下はh=0の期待native費用切片。単位10^6 gates、表示のみ丸め。全exact値は[資源表](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/saved_output_audit/resource_rows_exact_v1.json)が正本。Tは1Qに含まれるため足し合わせない。

| 方式 | T：旧→相殺後 | CX：旧→相殺後 | 1Q：旧→相殺後 |
|---|---:|---:|---:|
|ordinary|515.022952 → 512.961038|27.904618 → 23.780790|1344.170250 → 1337.355225|
|partial|456.252299 → 454.483823|23.480168 → 19.943216|1163.682829 → 1157.837668|
|closed P3|443.162079 → 441.402602|23.591285 → 20.072330|1081.327155 → 1075.509423|
|local full|430.039337 → 428.282208|23.532129 → 20.017871|1108.697460 → 1102.887640|
|closed P5|429.626133 → 427.870692|23.509518 → 19.998636|1107.632166 → 1101.827928|
|CTS|643.093987 → 643.093987|15.227487 → 14.781004|1623.187930 → 1622.073157|

六方式のh=0順位はT/CX/1Qの各coordinateで旧順序と同じ。T最小は登録集合内のP5、CX最小はCTS、1Q最小はclosed P3。科学的なalgorithm採択・全資源優越の判定ではない。

P5−partialの相対差は次のとおり。

| 指標 | 旧D1 | 共通境界相殺後 |
|---|---:|---:|
| T |−5.835843%|−5.855683%|
| CX |+0.124998%|+0.277891%|
| 1Q |−4.816662%|−4.837443%|

Tの低費用関係はこの固定モデルで残る。P5だけが相殺を得た結果ではない。relative T差の変化は約0.01984 percentage pointで、P5自身の旧T切片からの減少は約0.40860%。異なる百分率を混同しない。

prep/readout slopeは旧2NZ²のまま。各metricに独立のh≥0を加えたG(h)=A_after+B_old hで、P5対partialのCX交点は約6.637185。旧約3.514950から移ったが、これは二方式の交点であり全六方式の最小包絡線の交点ではない。T/1Qはこのpairについてh≥0でP5の切片・slopeが共に小さい。

N・moment・range・biasを再最適化した比較ではない。旧accepted-tail/hard費用の保守的上界もそのまま保存した。全回路compiler、分子PR、RPE/QPE総費用、classical online費用への一般化はしていない。

## 保存値監査と保護

[監査](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/saved_output_audit/saved_output_audit_v1.json)は1,218 checks PASS。これは実行後saved-onlyの会計/provenance監査であり、immutable CI・第三者再現ではない。

- 六行とresource mapが同一、coverage219,303、depth mass和=Z²、budget/N/slope/workspace/旧tail/hard/biasの不変を確認。
- 各coordinateでold−saving=after、oldG−newG=2N×saving、0≤after≤old、worst-price上界をexact rationalで照合。
- source182 critical・旧保護対象344 files、contract S・authorization A・marker・terminal・outer・STOPのidentity一致。
- 原run13filesのbytes/SHA256は監査前後で不変。外側stdout/stderrをbyte不変で保存。
- [実行前確認](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/saved_output_audit/independent_preflight_v1.json)：Aの唯一のparent=S、許可二pathのみのdiff、remote=A、clean、runtime・未消費markerを実行前に確認。
- 45登録外testsはsource準備時の記録を参照し、今回再実行していない。登録pair kernelも監査では再呼出ししていない。
- [親保護](../../../artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/saved_output_audit/parent_worktree_protection_v1.json)：今回の実行開始時のHEAD・status・index・未stage差分hashおよび27 dirty tracked pathのbytes hashと一致。本作業から親へ書込み・reset/clean/stash/stage/commitは行っていない。

監査script [audit_d2_saved_boundary_result.py](../../../scripts/tracks/algorithm_codesign/audit_d2_saved_boundary_result.py)は新しい結果監査用の追加資料であり、Sの登録sourceを変更していない。結果公開commitのHEADを実行Aとして扱わない。

## GPTへ戻す論点と停止

科学的レビューは利用者の開始承認後、D1原結果・key価格機構分析・今回D2を一束で扱う。

1. 固定r=2でT関係が残り、CXの相対差/交点が動いたことを、成立条件の研究としてどこまで主張できるか。
2. 現key価格に依存する差について、別合成条件の前向きな検証が必要か。そのscopeはまだ採択していない。
3. whole compiler・未知classical cost・独立provider等のうち、次の主要claimに必要なものは何か。

新r、provider、precision、seed、SP組合せ、分子、G10/RA-D0再開、追加pair診断へ自動進行しない。既存closure・原D1分類・source・authorization・消費済みmarker・STOPを保持。

**mandatory STOP。Codexは科学判断を代行せず、次の研究方向・追加実行の必要性と範囲をGPT/利用者へ戻す。**
