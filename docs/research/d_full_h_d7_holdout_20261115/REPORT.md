# Full-H κ2 D7独立holdout：取得・記述的比較完了

**DESCRIPTIVE_COMPLETE。指定3 fresh path・6 compile・4比較が有効、欠測/無効0。** 必要な取得/実装gateは全PASS、親/子/compile workerは全exit0。モデルPASS/FAILは付けず、model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE、tolerance=nullを保持する。結果と比較・C向けHANDOFFを指定branchへ公開し、remote/公開成果物照合後mandatory STOP。

## 固定条件・観測前結合

H4直線1.00Å/STO-3G、保存tol-only DF10、one-body込みFull-H pure RTE/L_D0/K2/κ2、group時間0.1。元scalar/event位相、symmetric-directional control、新ZZ lowering CONTROLLED_ZZ_CX_CRZ_CX_V1、測定付きfull controlled Hadamard wrapper、P0除外、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持した。

- 科学実行SOURCE/input：`9cbabc617e95a81810924413c1b2b06ced329514`。固定checkoutで取得し、稼働中の変更・停止・再起動はしていない。
- 公開後処理高速化SOURCE：`03cc993eeff827e87fc120a1aedb14a34fefcdf2`。科学native/provider/compiler/profileを変更しない別の後処理。
- 採択規則：f0765067a5a8b39888e36f3a8b80f1bdbda69223、docs/research/full_h_d7_holdout_rule_proposal_20261115/PROPOSAL.mdとHANDOFF_D1.md。
- C観測前凍結正本：b5165f11bb63f9eb2a6bfe385ba75d8003b7e724、docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json。models.X/Yとpredictionsの4行を使用。
- D7固定plan：7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f、d_full_h_p2_calibration_resume_20261115/FROZEN_JOBS.jsonのplan。
- 較正完了・実装来歴e628c0d69c7da559369d99f323e3656a96c0728a、旧SOURCE/STOP/rawを保全。
- 全6軸結果SHA：`b5fac0ee4eac83f027f883e5ea5630623a98bf29`。m5先行公開c511f6b4f521f0e1dd7853bea02e0b0058272640を含む。

観測取得/開封前に既存frozen_model_gateを実行してPASSし、4予測を(cell_id,axis)で一意に選んだ。physical/cohort/path/seedとfeature順を固定計画へ照合。predicted_RZの分子/分母を直接読み、係数からの再予測・fitは0。model selectorとprediction selectorは[COMPARISON.json](COMPARISON.json)の各行に固定した。

§10.5はCの公開前照合を引継ぎ、その後の関連remote ref・既知保存先の名前・稼働jobを、観測値を開かずに確認。[PREOPEN_CHECK.json](PREOPEN_CHECK.json)。動いた関連refはCモデル凍結/公開metadataのみで、同一holdout取得出力なし。事前受入可能な同等結果0、重複科学job0、対応不明0。fresh namespaceと固定holdout cohort/seedを使い、較正seedとも非重複を確認した。未知host/archiveを網羅したとは主張しない。

## 取得件数・個別観測価格

| 項目 | 予定 | 再利用 | 新規/有効 | 欠測/無効 |
|---|---:|---:|---:|---:|
| fresh path | 3 | 0 | 3 | 0 |
| 個別X/Y compile | 6 | 0 | 6 | 0 |
| 記述的比較 | 4 | 該当なし | 4 | 0 |

| cell | q/R/T | physical ID | path | cohort |
|---|---|---|---|---|
| M5_ROUND | 32/283/3.2 | FULL_H_bc8aab34256b73b68025 | 0,1 | D_FULL_H_P2_D7_M5_ROUND_PROPOSED_V1 |
| M6_ROUND | 64/566/6.4 | FULL_H_4ec8d73571760a6830b8 | 0 | D_FULL_H_P2_D7_M6_ROUND_PROPOSED_V1 |

各pathは固定step_seedsで一度samplingし、同じpathのX/Yを各一回compileした。個別compileのcache reuseは全false。科学retry・seed差替え・追加path/compileは0。seed全文は[FROZEN_JOBS.json](FROZEN_JOBS.json)と各価格receiptに保持する。

| cell | path | 軸 | compiled RZ | 必要gate |
|---|---|---|---:|---|
| M5_ROUND | 0 | X | 1384413 | PASS |
| M5_ROUND | 0 | Y | 1384413 | PASS |
| M5_ROUND | 1 | X | 1401483 | PASS |
| M5_ROUND | 1 | Y | 1401483 | PASS |
| M6_ROUND | 0 | X | 5551296 | PASS |
| M6_ROUND | 0 | Y | 5551296 | PASS |

X/Yは同一pathの別測定軸。m5は独立2 path、m6は独立1 pathで、軸を合算して標本数を増やさない。

## 四件のexact記述的比較

PはCの保存予測、Oはm5同軸path0/1の算術平均、m6同軸path0の値。符号はP−Oで、負は過小予測、正は過大予測。

| 比較 | P（exact） | O（exact） | P−O（exact） | (P−O)/O（表示%） |
|---|---|---|---|---|
| M5_ROUND-X | 27894546580642075/20115111656 | 1392948 | -124757970359813/20115111656 | -0.445257% |
| M5_ROUND-Y | 27894546580642075/20115111656 | 1392948 | -124757970359813/20115111656 | -0.445257% |
| M6_ROUND-X | 111815248527103419/20115111656 | 5551296 | 150309651597243/20115111656 | 0.134608% |
| M6_ROUND-Y | 111815248527103419/20115111656 | 5551296 | 150309651597243/20115111656 | 0.134608% |


P/O/P−O/(P−O)/Oの分子・分母をJSONとCSVにexactで保存した。上表の百分率だけ6桁に丸める。4行のOはすべて正で、相対誤差の未定義0、comparison_status=DESCRIPTIVE_COMPLETE。これは記述的完備性でありモデル精度の合格ではない。数値許容差はnull、モデルPASS/FAILなし。再fit・事後閾値・全law期待値/CIへの昇格は行わない。

## correctness・公開証拠

M5 path0とM6 path0はlow-control/high-control leakage/independent occupation vector、Hadamard X/Y、compiled-vector X/Yの各7数値gate PASS。M5 path1は所定どおり数値gateを追加せず、全3 pathでparity/new lowering・one-body・scalar/event phase・X/Y・元compile fingerprint/metrics対応を確認した。元閾値とphase_alignment=falseを保持。全pathのresolved/expected scalarは-0.33147781341681615で一致し、総Tで一回の適用を確認する既存gateを維持した。

各cellのAXIS_PRICESは個別軸の価格・physical/cohort/index/seed・実行SOURCE・元compile記録を、AXIS_EVIDENCEはその実gate・資源を保持。COMPILE_RECORDS/p<i>_<axis>.jsonは元compiler JSONのbytesをSHA付きで保存する。元raw RESULTはlocal保全しlossless gzipを公開、manifestに元/圧縮bytes/hashと復号一致を記録した。

公開結果SHAから2 cellを別processで復号・照合し、元callerのC_INPUTS/PRICES/sidecar/implementation evidenceを再生成して完全一致を確認。各6軸の元compile bytes/hash/fingerprint/compiler/metricsも確認した。[PUBLICATION_CHECK.json](PUBLICATION_CHECK.json)。各比較行のobservation_sourcesから個別価格・実装証拠・compile record・raw manifestとdecoded path selectorへ辿れる。prediction_sourceはC b516の固定JSON selectorへ辿る。[HANDOFF_C.json](HANDOFF_C.json)にまとめた。

これはD1本人による技術照合で、独立reviewやC担当の科学的受入ではない。C_scientific_admission=NOT_PERFORMED。モデルの固定原本や採用資格は書き換えていない。

## 並列化・資源

独立cell/pathを同時実行し、各X/Yは別processで個別compile。既存exact controlled-gate cache・同一canonical bytesのfast fingerprintを再利用。nativeの状態vector/一部回路走査は逐次であり、そのために科学workerを途中変更・再起動しなかった。共有OS/package/library設定を変更していない。

| cell | 全job wall | user/system秒 | GNU time maxRSS KiB | 親/子exit |
|---|---|---|---:|---|
| m5 | 22:05.61 | 2352.36 / 43.07 | 4278444 | 0 / 0 |
| m6 | 1:26:15 | 6200.26 / 69.05 | 16338552 | 0 / 0 |

GNU timeのmaxRSSは同時workerのRSS合計ではない。path別native資源と各compile worker実時間は[HANDOFF_C.json](HANDOFF_C.json)のresourcesと元AXIS_EVIDENCEに保存。例：m6 X/Y compiler実wallは972.278/966.101秒で並列。外側compile_receipt.wall_secondsは親のwait/QPYロード等を含む時間で、compiler worker実時間と区別する。

完了cellの検査・保存を他cellの取得と重ね、pigz64 CPUで圧縮。m5 raw585753898 bytes→gzip23953379 bytes、圧縮2.096秒/全archive検証2.970秒。m6 raw1170083541 bytes→gzip47630156 bytes、4.024秒/5.738秒。公開SHAからのcell別照合は7.761/13.239秒、別PIDの並列。全jobのserial対比速度倍率は未測定。[手法・適用条件](PUBLICATION_ACCELERATION.md)。

人工/回帰32件とimport-only23 moduleに加え、後処理人工3件PASS。実H4の追加試運転0。人工の速度比を科学全体へ外挿しない。

## 未確認・ロードマップ・終了

未確認/範囲外：モデル精度の数値許容差、母law E[C]/CI、m6の分散/SE/CI、rare寄与・sampler/ideal距離、境界一般化、追加round/長時間資格、N/G/winner。m6の2 path化、再fit、観測後の閾値設定を行わない。

ロードマップ差分は「Full-H κ2 D7指定m5/m6独立holdout6軸を取得、4件の固定予測比較をDESCRIPTIVE_COMPLETEとして公開。モデル精度のPASS/FAILは未判定、資格は自動昇格しない」。次はCが固定結果を受け取る。閾値・追加検証等が必要なら別の研究判断であり、D1は進まない。ロードマップ本体は統合担当が更新する。

指定成果物のcommit/push・remote/公開照合をもってmandatory STOP。計算・archive・比較を再実行しない。
