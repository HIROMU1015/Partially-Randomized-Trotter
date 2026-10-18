# 既存fixture・入力・結果の可用性

存在、固定commit、path、bytes/SHA256、型・scopeだけを確認した。[evidence_index.json](evidence_index.json)と[source_inventory.json](source_inventory.json)が固定URLを与える。**NPZのdecode・配列取出し・科学値の再集計は0件。** file hashとGit blobの確認は数値再解析ではない。

`AVAILABLE`は記載の資料があることだけ。`NOT_AVAILABLE`は指定された監査対象で利用可能な形を確認できなかったこと、`NOT_VERIFIED`は内容・保証を検証していないこと、`NOT_APPLICABLE`は現在の接続条件にはそのまま使えないことを意味する。repository外・Codex Aの領域の不在を証明しない。

| 必要資料 | 可用性／固定の所在 | 型・利用できる証拠 | 残る条件 |
|---|---|---|---|
| G6有理形式fixtureと既済test | AVAILABLE：`28cfabb1…`、T01、`artifacts/track_b_g6_return_generator_audit/2026-10-10/focused_tests.json`とscope/result/STOP | finite odd m、小有理p/x、形式独立oracle、digital eventの既存記録 | 分子DF label registry、測定state/backendとは結合されていない |
| G7 event/U/budget/provider fixture | AVAILABLE：`ab2549f4…`、T02、`artifacts/track_b_g7_budget_control_preparation/2026-10-10/`、結果は`a6896940…` | exact rational p/x、event/None、conditional CQ、既済18件、保存one-shot結果 | 独立DF input binding、実backend shots、一般provider certificateではない |
| G8 on-demand・resource-cap記録 | AVAILABLE：`cb60a4a1…` SOURCEと本基点の既存preparation/result metadata | cache/zero/accepted、条件付きprovider誤差、stub/固定取得記録 | signed-time一般bindingと測定accumulatorなし。旧capacity/precisionは新taskへ採択しない |
| G9 P5/native例 | AVAILABLE：`0ef2b927…` SOURCE/T10、G9 v1/v2 manifests・STOP・consumed | 固定3-system-qubit provider、P5・phase/actual-adjointの限定証拠 | 原23件とv2 API19件を別履歴にする。分子DF全eventのcertificateとしてNOT_APPLICABLE |
| DFBlock／registryの人工定義 | AVAILABLE：`321cd8c3…`、T03/T04。RY等を使う小basis fixture | c/sign/support・basis ID/hash、I/Z/ZZ、controlled builderの検証仕様 | RY fixtureがあるだけでGaussian分子実証にしない。新testは未実行 |
| partial-S2／repeated／Hadamard人工定義 | AVAILABLE：同commit、T05/T06/T07 | 小DFHamiltonianをSOURCE内で構成。D/R、phase、controlled、bit mapping、metadataの検証仕様 | ordinary law用。G6全trial、実W、zeroを保存する既成fixtureではない |
| H2分子test | SOURCEはAVAILABLE：T03の149–197行 | `build_df_h_d_from_molecule(2)`を呼んで構築するtest | 保存済み分子inputを読むだけのfixtureとしてNOT_APPLICABLE。実行すれば新しい分子計算を伴うため今回禁止 |
| 保存H4 DF snapshot | AVAILABLE：本基点、`artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` | byte identityのみ確認。S20のwriter/loaderはschema/hash、constant、one_body、lambdas、g_matrices、metadata_jsonを記録/検証する | 内容decode、保存hashの内部照合、basis U_ops生成、G6有理p/x生成はNOT_VERIFIED。filenameの旧scopeを将来の選定条件にしない |
| 旧ordinary H4結果 | AVAILABLE：同stemの`.json`、本基点のbyte identityと関連validation文書 | 旧ordinaryの保存来歴。値の再解析なし | G6の新law・受理率・weight・costへ転用不可 |
| 直接使えるG6-DF入力binding証拠 | NOT_AVAILABLE：今回確認した固定Packet／SOURCE／manifestで、target・正確なp/x・label↔component/basis・誤差を全て結ぶ保存bundleを特定できない | 実装の候補型はDFRTEEventPreparationとG7 event fields | 外部・未指定bundleの存在はNOT_VERIFIED。勝手にDF生成・decodeして補完しない |
| G6-DF backend全trial測定結果 | NOT_AVAILABLE：監査対象の統合結果なし。既存G7/G8／Hadamard metadataはquantum shots/measurements=0 | 既存結果は条件付き会計／回路検証の証拠 | 実量子測定・新しいweighted estimatorの実証と呼ばない |
| State preparation契約 | 一般DF-G6接続用としてNOT_AVAILABLE | Hadamard wrapperはstate preparationを明示的に除外 | state/sector/basis ordering、正規化、成功率・費用を別途固定。保存oracleや厳密ground accessを自動移植しない |

保存H4 snapshotは既存の候補であり、今回採択したpilot入力ではない。H4 1.00 Å／STO-3G／rank12／旧L_D=3等の文字列は上記保存物の来歴を示すだけで、新しいgrid・precision・r・seedを選んでいない。

## 科学値を扱わずに解消した技術事項

必要な型・API、固定SOURCE、過去の検証仕様、保存snapshotの所在は特定できた。一方、manifestの存在と、内部数値が新G6入力へ正しくbindingされたことを区別した。可用性不足を理由に旧実験を再実行せず、保存値の一致を得るための新たな数値処理もしていない。
