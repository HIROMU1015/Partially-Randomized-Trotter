# A1：旧opt2 SOURCE・payload・実行環境の限定静的調査

2026-11-06 JST、SERVER1（GPUA100-gleap-01）。専用branch `codex/o2-a1-opt2-provenance-20261106`。
出発点：[3354b8baa70b5ed7a798069db8f2f4c4fdd4b526](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3354b8baa70b5ed7a798069db8f2f4c4fdd4b526)。[前回STOP](../o2_a1_q1_q2l_20261106/REPORT_V2.md)を維持した。

**記録されたSOURCEの回収と来歴の区別は進んだが、旧hashを生成した完全payloadとbinary環境は回収できなかった。科学再実行は0件。**
旧hashの再現を追加で試すことや、新targetの採用はしていない。科学的targetの不一致も確定していない。

## 対象と探索境界

固定対象は保存linear H4、1.00 Å、STO-3G、legacy DF rank12、4電子/8qubit、L_D3、m0/q1/δ=.02/r1/K2、threshold0、X/Y共通law。
旧期待identityはpartition `903439…`、preparation `50c77f…`、tail `c51e35…`。完全hashは[PAYLOAD_SEARCH](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/PAYLOAD_SEARCH.json)に保存した。

既存artifact/archiveの旧hash carrier 60 path（JSON59件とsource1件）、payload必須key、元WP11のvenvを除く1,514既存ファイル、元opt2 checkpoint53件、関連SOURCEの既知commit・path履歴・archiveを静的確認した。
`.venv311`の記録と現存venvの`pyvenv.cfg`・dist-info・requirementsも読み取った。別host、未接続store、全てのbinary形式まで網羅した不存在証明ではない。

[collector](collect_static_evidence.py)は標準ライブラリ、read-only git/rg、JSON/AST/bytesの確認だけを使う。研究モジュールをimportせず、NPZをloadしない。
準備・軌道基底再構成・CDF・moment・bias・RNG・state solve・回路build/compile・測定・variance/N/cost評価はいずれも0件。人工科学テストも再実行していない。

## 回収できたSOURCEと回収限界

[SOURCE_LINEAGE](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/SOURCE_LINEAGE.json)は、記録SHA256に一致する各ファイルの固定commit/pathを保存する。

| 記録 | 記録対象の回収 | 主な固定SOURCE |
|---|---|---|
| WP05aのSOURCE manifest | 4/4件 | `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c` |
| WP06bのSOURCE manifest | 4/4件 | 同上 |
| WP06aのSOURCE manifest | 2/2件 | 同上 |
| 本実行のopt2 checkpoint51件のSOURCE set | 10/10件 | `7753eb71844e429facdb6f32c2def6e37379541f`、同内容のWP11 `8b574c54…` |
| 初期smoke/timing checkpoint2件のSOURCE set | 9/10件 | `research_direction_full_opt2.py` の `f169ec2b…` 版は今回の固定commit/path履歴/archiveから未回収 |

WP05aのmanifestは `df_partial_s2.py` の元版SHA256 `2a9ff548…` を明示しており、archiveにも同内容がある。
元hash生成資料のprovenanceが記すHEADは `efa90e0b…` だが、dirty/untracked生成時の各SOURCE hashを優先すると、記録対象を回収できるsnapshotは後の `26aa95c7…` になる。
HEADだけをcheckoutして原実行の全SOURCEが得られるとはしない。上表の回収は**manifestに列挙されたファイルだけ**であり、未記録の全推移的依存やbinary環境の完全復元ではない。

## 旧期待hashの記録とSERVER1 opt2を区別する

旧3hashはWP05a/WP06bの`physical_instance`に保存されている。特にWP06bの生成SOURCEは、そのrunで得たpreparationのhashを直接記録する。
これらのprovenanceは `.venv311/bin/python`、Python **3.11.0rc1**、NumPy1.26.4、Qiskit1.3.0を記録している。

一方、元SERVER1 opt2のrun_provenanceと実checkpointの`dependency_versions`は、Python **3.12.3**、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0を記録する。
元CPUはAMD EPYC7742、記録thread設定は各1である。環境差の存在を確認しただけで、その差をhash不一致の原因とは認定しない。

[CHECKPOINT_METADATA](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/CHECKPOINT_METADATA.json)は元WP11内のignored checkpoint53件のpath・bytes SHA256とprovenance抜粋を保存した。
全53件にpartition/preparation/tail hash、完全component、basis操作は保存されていない。これは新たに生成したrawではなく、既存保存物の静的抜粋である。

元opt2の `_worker_preparation` は同NPZからsplit/preparationを作り直し、WP06b artifactのfingerprintを検査するが、旧3hashとの一致検査を持たない。
workerの返却値にもその3hashを出力しない。このため、**WP06bの旧3hashとSERVER1 opt2 workerが実際に生成したpreparationが同一だったと、保存記録だけからは確定できない**。
保存SOURCE・NPZの対応は回収できても、workerの未保存identityを後から認定できたことにはならない。既存opt2の科学的正否や費用結果を今回改判してはいない。

## Payloadの回収結果とbasis metadataの差

旧hash carrierに、全fragment weightを含むpartition、全basis操作、pre-thresholdを含む完全component catalogは見つからなかった。
WP06aには代表eventの係数等があり、WP06bには324件のsupport-basis証拠があるが、後者はsource/selected basis hash・support・操作数・残差等の記録で、basis操作列や全component係数ではない。
324はこの証拠表の行数であり、実53-bit lawのpositive support数ではない。他研究のcatalogを旧opt2の完全catalogとして流用していない。

[BASIS_METADATA](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/BASIS_METADATA.json)で、WP06bの9種類の`source_basis_hash`と、前回保存された現preparationのtailの9種類の`basis_hash`を文字列として比較した。
共通は2種類、各側だけにあるものは7種類だった。新たな演算子・固有値・信号を計算した結果ではない。

basis hashは順序付きlocal operation metadata（parameter文字列・local matrix hash等）を含む。
したがって、tail差をpartition metadataの差だけで説明して済ませる証拠はない。一方、浮動小数点表記・局所行列・基底表現の差が、物理的に有意な演算子差やfinite target差になるかは、このhash比較では分からない。
元操作列が未回収のため、許容差内のoperator同値も検査していない。

## 版差2ファイルの位置付け

[静的diff](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/SOURCE_DIFFERENCES.diff)とAST比較を保存した。

| SOURCE差 | hash入力との関係 |
|---|---|
| `df_partial_s2.py` | explicit ordered partition対応を追加。既定weight-ranked経路のpolicy文字列は維持。前回診断では同じpayloadに対する旧/現hash関数は一致した。これだけで数値差を説明できない |
| `pf_delta_validation.py` | 変更は`validate_pf_delta_grid`のLD末端条件 `>`→`>=`。今回のload/split/preparation経路はこの検証関数を呼ばず、LD3/DF12でも当該末端条件に当たらない。直接原因を支持する差ではない |

重要な未回収入力は、partitionのexact weight、pre-threshold coefficient、basis操作metadataである。
rank側の`np.linalg.norm`、基底側のeigh・fermionic Gaussian分解・Qiskit operation metadataへ依存するが、今回はこれらを実行していない。丸め、BLAS/LAPACK、libm、Python等のどれが原因かを特定したとはしない。

## 実行環境について分かった範囲

[ENVIRONMENT](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/ENVIRONMENT.json)に旧記録・現存設定を保存した。
旧hash生成時の `.venv311` は、今回調べたmain/WP11の両worktreeには存在しなかった。元のwheel/build hash、BLAS/LAPACK、完全lockは回収できていない。
requirementsのversion pinと同じversion表記だけでは、原binary環境の完全保存を認定しない。
現存 `.venv-server-opt2` はPython3.12.3で、NumPy wheelもcp312のmetadataを持つ。これは現時点の観測であり、原実行以来一切変更されていないことの証明ではない。
共有venv・package・cacheの変更や再構築は行っていない。

## 調査後の判断

今回の限定探索はここで終了する。旧hash一致を目指した追加準備計算は推奨しない。
旧元環境の完全再現をA1の目的にせず、次のどちらかを判断する材料は揃った。

| 方針 | 必要な条件 |
|---|---|
| 元targetの回収を継続 | 別host等から元`.venv311`/完全payloadという新しい具体的な証拠を取得できる場合に限る。新たな準備計算の要否と限定scopeはその後に決める |
| 保存DFから新しい一貫したfinite targetを固定 | 旧opt2再現とは分けた新科学実行契約を作り、同じlaw・target・state・価格policy・誤差条件でraw／層別／CVを比較する。旧opt2の価格やbiasを無条件移植しない |

現時点では後者の契約検討へ進むことが合理的な選択肢である。ただし今回採択・実行したわけではない。
元payloadと新payloadの同値も、科学的target不一致も未確定であり、既存A1を終了する判断ではない。詳細科学レビューは今回実施していない。

## 証拠・保全・STOP

[EVIDENCE_INDEX](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/EVIDENCE_INDEX.json)は調査対象のbytes SHA256とcollector hashを記録する。
[保全照合](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/PRESERVATION.json)で元mainのHEAD・status・既存dirty/untracked 2,136ファイルと旧A1 artifactの不変を確認した。
詳細証拠は自分のlocal branchにcommitし、公開要約は別branchで本文・treeを照合した。元main、旧SOURCE/raw/STOP、C1凍結、S2/R1判定を変更しない。
[STOP](../../../artifacts/o2_a1_opt2_provenance/2026-11-06/STOP.json)：**mandatory STOP。科学retry・新target採用・Q1/Q2-L・Q3–Q5への自動進行はしない。**


## 公開状態の追記

全証拠一式の公開は自動承認審査が拒否した。理由は、内部path・未公開checkpoint/環境情報を含む大きなtreeの公開GitHub搬出に明示許可がない、というもの。拒否された一式を別経路で公開していない。

公開済み資料への参照と非識別的な結果に絞った[公開要約](REPORT_PUBLIC.md)を別branchへ保存した。公開commitは`0a4ea78f7af60cd405787f98c33d25798d468461`、本文8件とtreeを照合済み。元一式を含むlocal commitは公開branchの祖先ではない。
詳細証拠一式はlocalに保管する。全証拠の公開にはその搬出への明示許可を待つ。科学再実行の許可とは別である。
