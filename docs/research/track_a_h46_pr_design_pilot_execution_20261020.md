# H4/H6高次PF：33本/群 DESIGN_PILOTの実行契約

基点 `3d0b5477ce04604ce873c079d9c66d02f2782239`。利用者は33本/群、byte可逆level1保存、H4/H6両系それぞれ一回の実科学取得を明示承認した。準備後の再承認は不要である。**この文書の契約記述だけでは実行完了・PASSを意味しない。進行・終了の証拠は別の実行snapshotと終了監査に記録する。**

## 科学範囲

| 項目 | 固定範囲 |
| --- | --- |
| 系・入力 | 線形H4/H6、1 Å、STO-3G、保存tol-only 1e-8 DF10/19、保存Hamiltonian/state/preparation |
| 時間・状態準備費 | T=.8、P=0 |
| PR | prefix H4=3/H6=5、K4、ABS_TIME_PER_OUTER=32 |
| 公式・q | S2/Yoshida4/m5_bestの保存係数・signed schedule、各q=1/2/4/8 |
| 標本 | 各群33、各系396 trajectory/792 compile軸、総792/1584 |
| cohort | DESIGN_PILOT専用の新campaign・新seed、既知registryと両系間の衝突拒否 |
| 回路 | symmetric-directional Hadamard full wrapper、cosine PASS後sine、paired trajectory |
| compiler | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、num_processes1 |
| 検証 | compiled raw state差は厳密に1e-7未満、その他native/control/oracle/Hadamard/log_Bは1e-9未満、位相合わせなし |

各系の科学call範囲は専用plan/grantに固定する。396 trajectory、792 compile、7425 piece sampling/native piece composition、47520 RTE event、低枝・高枝・独立oracleは各396、uncompiled/compiled wrapper作用は各792。sampling lawを変えずS0/S2_single/S2_multi/S4を排他的に記録し、強制・棄却・自動top-upはしない。

同じ保存入力・係数の[統合データ](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/README.md)、[付属記録](track_a_h46_higher_pf_supplementary_protocol_v1.md)、[標本設計準備](track_a_h46_pr_design_pilot_readiness_20261020.md)を再利用する。既存pilotや探索・確認標本へ混合せず、新しいsignal/SCF/DF/state/referenceの生成はない。

## 採択された新規出力保存

[容量調査](track_a_h46_pr_pilot_storage_optimization_20261020.md)の圧縮prototypeを、今回の新SOURCEにだけ接続する。旧契約・原本は変更しない。

1. QPY/NPZはserializerの原byteを新しい専用raw一時fileへ保存し、fsyncする。NPZの再serializationによる圧縮ではない。
2. zlib/DEFLATE level1で別archiveを作り、原byteのSHA/sizeとarchiveのSHA/sizeを別々に保持する。
3. archiveを展開して原SHA/sizeを照合し、atomicな非上書きpublicationとdirectory fsyncを行う。
4. SOURCE・plan・stage・task・logical raw pathに結合したmanifestをfsyncして公開する。
5. manifestとarchiveの整合・復元SHAを確認した後、新run内の登録されたraw一時fileだけを解放し、そのreceiptを残す。既存原本、外部path、symlink、他taskは対象外である。
6. archive-aware resolverは新しいrestoration領域にだけ復元する。byte監査はQPY/NPZを科学的に再生せず、streaming SHA/sizeで完全復元を検証する。

**byte復元PASSと科学native PASSは別である。** 圧縮のPASSで数値FAILを受け入れない。manifest以前のcrashはraw/partial/archiveをquarantineとして残す。manifest以降のbyte復旧は科学retryではなく、途中の科学結果をPASSにしない。過去のSTOPは保持する。

## 並列化と容量gate

H4/H6は独立契約・grant・ledger・marker・supervisor・STOPを持つ。両系を起動する前に全static gateを通し、空きCPU/RAM・保存DETのworker時間proxyから両系の予測終了時刻を均衡させるworker配分を求める。固定の4-worker等を引き継がない。各系でもfresh gateを行い、空き資源が減っていればadmissionを減らす。worker内BLAS/transpilerは1とし、trajectory間process並列、入力/prefix context cache、cosine/sineのevolution共有を使う。登録q×piece数の降順にdispatchし、標本結果に依存した順序変更はしない。

採択された最終保存proxyは、各系の最大保存圧縮pair×4＋16 MiB/trajectoryで、H4 27,780,359,904 bytes、H6 104,734,517,184 bytes、合計132,514,877,088 bytes。これに各admitted workerのraw/partial/quarantine proxyと独立復元領域を加える。既存原本の保存量は削減しない。inode・I/O観測も保存する。**これらはPR/rare出力の上界ではなく、切り捨て条件でもない。** 実際のENOSPC等はSTOPであり、n9への切替えや出力省略はない。

所有processのRAMと専用scratchを定期観測する。観測peakと厳密な最大値は区別する。一時的な上位PSI・CPU/I/O競合や無関係なworktree更新だけでは正常workerをkillしない。共有環境・他job・既存venvは変更しない。

## STOP・終了・公開

数値FAIL・seed衝突・SHA不一致・容量不足・native異常・台帳不整合は該当stageをSTOPし、retry・top-up・条件変更をしない。共通SOURCE/storage/inputの整合性問題では、他方も起動・継続しない。単独数値FAILを無関係なstageへ拡張しない。

外部supervisorのnative exit/wait4/reap、全workerのexit/reap、exact call台帳、paired axis receipt、8数値gate、イベント/seed/signed schedule、archive全SHA、結果SHA、保存byte数、wall/RAM/scratchを終了監査する。両取得の終了後に実行するのは技術監査だけであり、科学統計へは進まない。

seed一覧・private原本・QPY/NPZ・認証情報・private絶対pathを公開しない。公開するのは実装・人工検証・許可済み範囲の取得metadataである。作者自身の技術検証であり、独立したAIレビューではない。

**今回は分散/SE/MCSE評価、n_E採択、本探索、独立確認、shot政策、会計、再ランキング、AD03採用、詳細科学レビュー、final freezeを行わない。**

次の担当：Codexが承認済み両取得と技術終了監査を実施し、成功・失敗を保存してmandatory STOP。ユーザーがその後のpilot限定数値評価を別途判断する。
