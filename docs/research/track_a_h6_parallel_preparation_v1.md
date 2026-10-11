# H6 group間並列化：次回142探索の準備 v1

2026-10-12 JST。利用者は「次の計算で複数groupを同時に処理する」を選択した。
[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_parallel_preparation_v1/2026-10-12/README.md)にSOURCE、seal、Phase A、資源、budget、限定tests、未承認条件を集約する。
SOURCE `67fed3285acd757226fada25278d9a569cac76b8`。Science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` の224 paths不変。

linear H6/1Å/STO-3G/tol-only DF rank19/sector400/保存state/T0.8、prefix5/10/15/full19、92候補・4精度を維持。
PF/PRの構成、numeric allowance/normalization/shot、seed/trajectory、full-wrapper/native lowering、
Qiskit1.3.0/compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、RZ/CX/depthの費用定義を変更しない。
Python processを2つ並行に動かす実行制御のみを追加。各group内はserial/internal thread1、同じfrozen cost_taskを使用する。
古典wall/RSSの改善は量子回路資源の改善ではない。

Phase A447は正常完了しnative metadataから受理。旧449＋1=450 group/904 wrappers、残142固定。
旧H4/H6 run・環境・証拠・dirty/untrackedは元worktreeで保持。独立home worktreeで変更した。
既存233 runtime pathsのblob/hashを保持し、新4 control/runtime pathsでclosure237。
新SOURCEと別REVIEWに軽量資料を固定し、real runtime/NPZ/cacheを新規commitしない。

候補はworker2、CPU20/21各1、AS32GiB、RSS18GiB/worker、coordinator22/23・soft2/hard32GiB/RSS2GiB。
Aggregate40GiB・RAM64GiB・余地2GiB、従来memory/PSI/OOM/5秒guard・output limitsを維持。
RSS18GiBはsampling STOP threshold。2 workerの完走とtransient使用量は未測定。
旧8GiB受理を偽装せず、既存admit_groupの全科学checksをそのまま保持し期待AS receiptだけ32GiBにする。

34限定tests PASS、zero scientific reads/new transpiles/new science launches。
completion orderと独立task seedを分離し、保存/受理/集計はoriginal index order。
first STOPを維持し、別worker cleanupを継続、全dispatchがreapされなければ不合格。
正常終了後もprofile/input/sourceとparent receiptを再照合してからcompletion receiptを保存する。

technical seal済み、approved=false/runtime_authorization=false/allowed_cpus=[]。
Phase Aのgrantは142 groupへ再使用できない。新exact grantとfresh launch条件を要する。
142探索の完了/STOP後mandatory STOP。自動retry/resume・selection・独立32確認・H8は実装入口から呼ばない。
このPhase Bの科学結果・real speedup・旧compiler output bit equivalence・全142 wall/容量完走は未検証。
公開は利用者のpush待ち。次の承認範囲と具体的CPU/memory案は資料入口に記録した。
