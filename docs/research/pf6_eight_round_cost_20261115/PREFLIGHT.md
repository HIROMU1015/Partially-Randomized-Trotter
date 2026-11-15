# PF6：既知8roundの部分コスト・逆転条件の固定範囲

2026-11-15 JST。ユーザーは固定保存値による軸別積、候補別部分和、候補間差、未取得部分の逆転条件の数値解析を明示許可した。Q3・shot最適化・予測・compile・全13round正式G・winnerは対象外。

既存B branch `codex/rpe-total-dp-acceleration-20261114`、着手HEAD422db93fcを継続。B8dba096dのINPUTS/C_RESULT_JOIN/HANDOFF、C0b4b2d94のALIGNMENT、ユーザー指定D/C6入口を照合した。関連履歴・C/統合roadmapの索引・関連公開refに、同一6候補・vector・K={0,1,2,3,4,5,6,9}の部分和／15逆転条件は確認範囲で同定されなかった。旧2026-10の費用逆転研究は別candidate・時刻・shot条件であり代用しない。自ユーザーの関連Python script名/cwdを読取確認した範囲で同等集計jobはなかった。今回出力namespaceは新規。全非公開資料の不存在を証明したものではない。

## 実際の経路とSOURCE

`analyze.py prepare`は固定Git blobを読み、各入口のHANDOFFが示す軸別receipt・sidecar・guard・原価格を辿る。各blobはcommit/pathをkeyに読取cacheし、96軸分の同一JSON・gzipの重複読取を省く。新しい価格provider/fit/compiler/decoderをimportしない。

- m0/m1/m2：元AXIS_PRICES→各軸PRICES・SIDECARS・元trial。
- m3：C ADMISSION→受入済みprice/sidecarと元JSON.gz／JSON／CSV軸receipt。旧生成ID/cohort未記録のnull、generation_SOURCEを保全。
- m4/m5：D C_HANDOFF_INDEX→C_INPUTSの元calibration record、price_reference又は保存M5/m4のC admitted receipt。元EXPLORATION cohortを保持。
- m6/m9：保存RESULTの**observed_RZと元receipt**だけを使用。predicted_RZやモデル相対誤差は算術入力にしない。

各軸をB original configuration→C alignmentのphysical descriptor→元price identity/law→B original/total candidate joinに照合し、既存B wrapperがSOURCE固定で読むC `price_implementation_guard`（c628e9e8）を実際に呼ぶ。完全な元configurationを保ち、検査する1軸priceとsidecarだけを渡す。guardは他の欠測をskipする既存仕様なので、必ず**checkedが当該1軸**であることを確認する。全13価格取得やN資格の成立を意味しない。

結果は96軸受入、0件のmissing/duplicate/identity/law/receipt不一致。未取得U={7,8,10,11,12}の60軸はC=nullを保持。元receiptのconfiguration/candidateをtotal IDで上書きしない。INPUTSには96軸の元入口・元receipt selector・実装証拠・profile/cohort・物理contextと、未取得60軸を保持する。

## 会計と検証

既存controllerのcost/assess/selectは全scheduleと十分N資格の判定を含むため、今回の部分和に適用して条件を緩めない。元コードの供給値`N*C`と加算という算術規則に従い、薄い有限集計だけを行う。Python任意精度intでX/Yを個別積算し、normalization・古典標本数・state preparationを乗算しない。P=0、quantityは測定付きfull controlled Hadamard wrapperのcompiled RZ数。

X/Y各軸の固定vectorは`[59,55,51,47,43,39,35,31,27,23,19,15,11]`。新しいnominal/sufficient Nは生成しない。元Q3は保存componentのモデル閾値適合、元API return/CPU/RAM欠測の部分保存・運用UNKNOWNをそのまま参照する。

人工testは個別軸積・整数部分和・逆転不等号/同額/入力順反転/同額部分和、missing/duplicate/null/非整数拒否、巨大int保持、固定metadata96軸・未取得60軸の保全を確認する。実行コードとINPUTSをcommitで固定した後、`analyze.py analyze`を一度実行し、AXIS_PRODUCTS/PARTIAL_SUMS/PAIRWISE_CONDITIONSのJSON/CSVを保存する。欠測・不一致があればissuesを明示し、ゼロやモデル値を補完しない。

比較単位はこの固定vectorの既知部分K_iと未知のU_i。15組の全体差は`K_i-K_j+U_i-U_j`であり、未知差の閾値に `<`、`=`、`>`を分ける。既知部分順位を全13roundのwinnerとしない。

関連する既存関数・記録は実際の入口で再利用し、大規模な新controller・承認管理は作らない。96軸の小さい整数演算にworker増設は不要。主agentだけで作業し、他担当branch/worktree/processと既存SOURCE/raw/結果/grant/ledger/marker/STOPを変更しない。

有限再解析の結果公開・remote照合後mandatory STOP。ロードマップ差分はHANDOFFに記載し、本体は統合担当が更新する。
