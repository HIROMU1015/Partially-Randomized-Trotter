# 4方式×3ホストの準備：残る変更セルの検収

現在の集計は10 PASS・2未確認。Full PF/B0I/PR-DFはserver2・server1・localで保存証拠PASS、
PR-Pauliはserver2が新しい固定routeでPASS、server1とlocalは同routeの実機dry-check返送待ち。
local0は中継・検査指令を担当し、計算hostに含めない。

- [現在の割当表](HOST_ELIGIBILITY.json)
- [server1 PR-DF専用Python](SERVER1_PRDF_IMPORT.md)
- [RomuTail PR-DF source解決・固定adapter](ROMUTAIL_PRDF_IMPORT.md)
- [Pauli route変更の科学条件・source](pauli_descriptor_route/ROUTE_CHANGE.json)
- [Pauli server2の公式dry-check](pauli_descriptor_route/server2_dry_check/READINESS.json)
- [SOURCE・辞書・評価契約の拒否検査](pauli_descriptor_route/NEGATIVE_CHECKS.json)
- [local0の変更2セル検査と返送](pauli_route_update/local0_tools/README.md)

今回のPauli科学job IDは805877c2ab6c3e03f7f1のまま、REQUEST全体hashは新routeにより
ed04775d03240eee59d639fd039842652f28eaf4472f59cf4f12f9192a35f693へ変わった。
SOURCE28d、CID、shot配分、費用、評価契約は維持し、受理だけを既存の固定33e routeへ接続した。
別の配分で保存済みのLD4 PASSは今回の未評価点の科学的適合へ転用しない。
候補選定と旧履歴読込みは従来routeで先に行い、選ばれた候補のREQUEST組立で限定route更新を適用する。
生成するDISTRIBUTED_OPTIONSのroutes_overrideを通常分散roundへ渡す。

RomuTail DFは実装e37・証拠09efを取り込んだ。元assignment SOURCE28dは変更せず、
新adapterの実装e37と配置09efを別manifestで固定する。adapter本体とprefix helperは両commitでbyte一致。
controllerのassignmentにprofile/adapter manifestを固定し、workerが同じHEAD・hashを実行前に確認する。
workerの新transport supervisorは保存profileから固定adapterを呼び、profile.route_viewsを外部JSONで渡す。
server1 DFの専用Python、他2hostの元adapter、PF/B0Iのprofileは保存された条件を維持する。

通常prepareはEVIDENCEのroute/profile規則を自動解決する。利用者が候補ごとにPythonやhostを選び直す必要はない。
新REQUESTは全体hash・入力・SOURCE・route・runtimeを別途照合し、過去PASSの一律転用はしない。
新しい分散設定のrequired_hostsは空集合にし、一部hostの準備不足で既に実行可能なhostを止めない。
既存attemptと完了結果の再利用・回収は従来どおり同host/attemptで照合する。

## local0へ依頼する操作

[RUN_PAULI_DELTA.ps1](pauli_route_update/local0_tools/RUN_PAULI_DELTA.ps1)は既存alias qg1・RomuTail・qg2を使い、
新しい専用directoryへ検査用ファイルと必要なroute viewを置く。
実行対象はPR-Pauliのremote2セルだけ。既存10 PASSの再dry-check、worker/relay起動、実Q3、signal、価格較正、compile、queue操作は0件。
現在のZIP SHA256は5c5273436d2fafd65c2be5294cd67313a8dc7fcafee3839398b49601d66c1a70。
返送失敗時の--return-onlyは同じ保存結果の転送だけを行い、dry-checkを繰り返さない。

この段階は準備途中であり、12 PASS完了とは記録しない。remote2件の公式READINESS・受領hash・host runtimeを確認後、集計を更新する。
実H4のQ3完了件数は0。dry-check PASSと科学計算完了を区別する。
