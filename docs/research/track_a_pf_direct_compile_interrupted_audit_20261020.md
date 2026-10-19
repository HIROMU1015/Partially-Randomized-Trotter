# PF直接compile：中断後の保存結果・終了証拠監査

2026-10-20 JST。現在、計算は停止している。coordinator・workerが残っておらず、元のterminal.jsonと契約最終gateがない。**task単位の保存受入572軸、明示的compile前STOP200軸、native終了未確認16軸**を区別して保存した。run全体PASS・全worker回収済みとは認定しない。

[packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_interrupted_audit/2026-10-20/README.md)。実行SOURCEは`220b9644df6e114116737fd041973973fcf86cf5`、承認基点は`d510d7beda94d293432328a17fcc60d4286dc637`。新しい監査はstdlibによる保存記録の照合だけで、新科学実行・retryは0。

| 対象 | 承認軸 | 保存task受入 | compile前STOP | native終了未確認 |
| --- | ---: | ---: | ---: | ---: |
| H2 DF3 | 72 | 0 | 72 | 0 |
| H4 DF10 | 122 | 0 | 122 | 0 |
| H6 DF19 | 214 | 214 | 0 | 0 |
| H8 DF28 | 314 | 292 | 6 | 16 |
| H4 legacy DF12 | 66 | 66 | 0 | 0 |
| 合計 | 788 | 572 | 200 | 16 |

対象はlinear H2/H4/H6/H8、隣接1 Å、STO-3G、tol-only DF3/10/19/28、T=.8、P=0、symmetric_directional。legacy H4固定DF12は別契約。費用指標は状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ等の軸別整数ゲート数で、shot重み付けや費用順位は再計算しない。旧profile等の照合待ち48軸は今回も対象外。

## 保存した成功とSTOP

572軸は元workerのexit0/reap・task ID・result SHA・SOURCE・compiler hash・compile前後fingerprint・整数metrics・完全pairを再照合したtask単位の保存受入である。H6の指定候補B0_p11_q4_2ndはcosine/sineとも86,749 RZ。モデル誤差や新しいfitは計算していない。

200軸は本番loaderの形式処理不備によるcompile前STOPであり、元入力の破損ではない。

- H2/H4 DF10の194軸：元のarray_recordがnp.ascontiguousarrayの後に保存したscalar shape=[1]に対し、新loaderが変換前shape=()を比較したSAVED_ARRAYS。native/control/compile未実施。
- H8 B0I p21/q1・2・4 S4の6軸：scalar receiptのpolicy/receipt入れ子形式を新loaderが展開せず、top-level C参照でSTOP。native/control/compile未実施。

確認済みnative終了証拠は386 pair（成功286＋明示STOP100）。確認済みcompile失敗は0だが、終了未確認分の成否は不明。未確認16軸をこの0へ含めたり、全て未実施だったと扱ったりしない。

## 中断と未確認16軸

最後の資源観測は**2026-10-20 03:43:21 JST**。その記録はworker8、投入待ち0を示す。今回確認時点では元coordinator PIDの同一processは存在せず、元workerも見つからず、tool sessionはUnknown process id 25019だった。元の終了・signal・回収証拠はない。停止理由や実際の停止時刻は**未確定**で、チャット終了・tool timeout・OOM等の原因を推測で確定しない。

未確認pairはH8 S4/q8の次の8件で、各cosine/sineの計16軸。

- B0 p17、p18、p19、p20、p21
- B0I p18、p19、p20

B0 p17とB0I p18にはraw cosine metricsが保存されているが、pair終了・native exit/reapがないため受入表へ昇格していない。他の部分結果もworker証拠archiveへそのまま保存した。既存の結果・call消費・marker・STOPを保全し、再起動・再実行・予約返却はしていない。

当初の終了用スクリプトはterminal.jsonがある通常終了を前提としていた。その前提を補うため、別の[中断監査](../../scripts/audit_pf_direct_interrupted_execution.py)を追加した。元terminalやnative exitを捏造せず、元SOURCEのcodeは変更しない。今後の実行方式では、coordinator自体の独立した終了証拠と、チャットsessionに依存しない継続・監視を最終結合する必要がある。これは後続の技術修正事項で、今回の中断原因を断定するものではない。

## 時間・資源の確認できる範囲

資源観測区間は01:07:58〜03:43:21 JST、**155.38分**。正確なrun終了時刻・全体wallは欠測。確認済み386 workerのCPU合計は**50.98 core-hour**、その単一process peak RSSの最大は**4.970 GiB**。未確認8 workerの終了時資源量を含まない。

同時稼働workerのRSS合計の最大観測値は**54.201 GiB**、worker数の最大観測値は64。RSS和は約10秒間隔の観測で、物理的な独占memoryや厳密なaggregate peakではない。CPU/RAM/workerに新しい固定上限を加えず、共有環境・他ユーザーの設定やprocessを変更していない。

## 保存物と限界

accepted_gate_counts.csvとaccepted_measurements.jsonを将来のq依存model比較に使える。system/candidate_id/axis/compiler_hash/SOURCEで結合し、Hamiltonian/state/snapshot・DF系列/fragment順・T/control・provider/runtime/profileを追加照合する。旧モデル式・係数・予測は元の計画表へ残す。穴をmodel値で直接実測として埋めない。

中断後のSOURCE/input/provider/runtime byte gateは両契約PASS。ただしこれは欠けた元の最終gateを代替する証拠ではない。旧計画・技術準備・契約分割の44ファイルはSHA不変。元の科学SOURCE/入力/raw/grant/ledger/marker/STOP・Figure 2・論文主表は変更しない。

これは主担当自身の構造・来歴・数値形式監査で、詳細科学レビュー・外部再現・最終evidence freezeではない。sampling/fit/shot再計算/費用再ランキングは0。

次の作業：**ユーザー**が今回の新SOURCE/結果の公開と、loader修正200軸・終了未確認16軸の新しい実行範囲を判断する。**Codex**は追加承認のある範囲だけで技術修正・新SOURCE/新契約を準備し、無断retryしない。前回の22ファイル公開承認を今回へ拡張しない。

**mandatory STOP。**
