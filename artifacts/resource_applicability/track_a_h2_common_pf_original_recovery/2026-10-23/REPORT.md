# H2共通Full PF原本の限定回収・静的identity照合

**対象4軸の原plan、実行入力、wrapper/compiled QPYおよびvector archiveをGPUA100-gleap-01の既知の本人所有private領域で確認し、固定結果のSHA・byte数と照合した。新科学call・compile・simulation・COMPARE・正式採択はすべて0。ローカルcommit後mandatory STOP。**

2026-10-23 JST。専用branch `codex/h2-common-pf-original-recovery-20261023`、基準 `ca29dd9ddc8a09cace84310f0dfdf7124652e07a`。監査SOURCEは [SOURCE.json](SOURCE.json)。Codex主担当自身の技術確認であり、独立科学レビュー・CI・外部再現ではない。元科学実行hostnameは保管場所から推定しない。

対象条件はlinear H2、隣接1.0 Å、STO-3G、signed tol-only DF1e-8/rank3、Full prefix（L_D）3、T=0.8、P=0。S2/q4（δ=0.2）とYoshida4/q2（δ=0.4）の各cosine/sineに限る。元のsigned schedule、symmetric-directional control、original-plus-tail-once scalar、system4/ancilla4、full Hadamard wrapperを保持する。保存単価は回路1軸当たりのcompiled RZ。旧Full 1367/2033とは別cost objectであり、u/N/G、総費用、比、順位は算出していない。

## 回収状態

| 対象 | 分類 | 今回確認した範囲 |
|---|---|---|
| 原 `plan_APPROVED.json` | 原本byteまで確認済み | 144,924 bytes。SHA `6eef028aff888dd64d8c6bb1833f01112df1026e452f8b89318c866f5e1a1892` は固定公開projectionの原plan SHAと一致。pathを除いたclosure・全共通fieldも一致 |
| 当該実行の入力5資料 | 原本byteまで確認済み | snapshot/reference NPZはhashのみ。receipt/comparison/reference receiptと各sealを照合。配列decode・reference再生成0 |
| 原result 2件 | 原本byteまで確認済み | 元ファイル全体SHAを固定native_result_inventoryとBのoriginal_PF_result_hashesに結合。固定gzip内packetとも内容一致 |
| wrapper 4、compiled 4、vectors 4 | 原本byteまで確認済み | gzip 12件のstored/raw SHA・byte数一致。QPY8件は既存静的parserで全byte消費。raw書出し・回路object化・vector数値decode0 |
| 原planの365 closure entry | 原本byteまで確認済み | 宣言path上の現存byteと全seal一致。236 provider、8 loaded-runtime-identity、21 compiler/array binary、Python1、SOURCE90 entry等。重複entryを含み、365個の独立fileという意味ではない |
| 固定GitHub資料13件 | 固定GitHub原本まで確認済み | GitHub本文全体（gzipはbase64）を取得し、固定local Git blobとの全文byte・Git blob OID一致。既存資料の再登録・再push0 |
| 完全なhistorical actual-load/dependency manifest | 未発見 | 宣言closureと完全な過去worker実loadを区別。限定run root内に追加runtime/load/provider/seal名のreceiptは見つからず、元SOURCEも完全load追跡を生成していない |
| 旧snapshot容器 `62c6481b…` | 保存receipt・SHAまで確認済み | 当該共通PFが実際に使用した再構築容器 `8bb1aa35…` とは別。旧容器原本を復元・再hashしたとはしない |

「元資料の所在のみ判明」「mismatch」に分類した対象原本は0。未発見は上記の探索範囲内の判定であり、他環境での不存在の断定ではない。private原本はこのhostで検証済みで、GPUA100-gleap02へは未転送。公開原本であると表現しない。

## 原plan・QPY・runtime・cost・correctnessの結合

| 候補 | axis | wrapper raw bytes | compiled raw bytes | 保存RZ/軸 | 原result SHA256 |
|---|---|---:|---:|---:|---|
| H2_FULL_3_4_2nd | cosine | 103874 | 179283 | 1655 | `ba2d0185f5c421eea7b5971ccee1568adbcb60fd1bb601ed26be810810f5ccef` |
| H2_FULL_3_4_2nd | sine | 103919 | 179285 | 1655 | 同上 |
| H2_FULL_3_2_4th | cosine | 155610 | 266755 | 2463 | `2881bd3251447e7a3a7567fb5d2b3ddd53bd8c70173118d320b66e5b981adb32` |
| H2_FULL_3_2_4th | sine | 155655 | 266757 | 2463 | 同上 |

上表は保存fieldの転記・identity照合で、費用の再測定ではない。[4軸の結合表](axis_identity_4.json)、[12 archiveの完全SHA/byte表](archive_identity_12.json)、[原plan・入力・result](original_identity.json)に、token化相対path、raw/archive SHA256、計算したGit blob SHA1、固定commit/path/JSON pointerを収録。計算したblob SHA1はGitに原本が登録された証拠ではない。private QPY/raw/archiveに固定Git commitは確認しておらずnullを明示した。

全軸のFull preparation hashは `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9`。Hamiltonian `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、state `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf` は原plan/result/receiptに一致。当該実行snapshot容器のSHAは `8bb1aa35cf80266dd4aa1552cc65097770dfe67c690c2e292860923a5b4ae2f9`。旧容器との科学配列同一性は既存receiptの主張を保持し、今回配列を再照合していない。

compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing指定なし。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。原SOURCEのcost_oneがpinned providerの`transpile_and_measure_cost`→`_gate_counts`→`count_ops`を使うことを静的に確認。guardは `H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1`、unchanged gate stream、state fitなし、postcompile phase correctionなし。compiled QPY内guard metadataは原resultおよびBの公開4軸表と一致。wrapperのancilla4・classical0・cos/sin終端も静的照合した。QPY stream digestはserialized streamのidentityでありQiskit fingerprintやoperator同等性証明とは別。

[runtime表](runtime_identity.json)は原planのpath→role→SHA/bytesを結合する。Python binary SHA `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` はBで確認済みの元PR binaryと同じ。今回元PR側原本を再探索・再hashしたわけではなく、Bの固定比較証拠を参照した。PF宣言closureの全現存byte一致は、完全な過去actual-load・transitive dependency証明、3.12対3.11出力同等性、現在共通native全operation streamのbindingを自動成立させない。

correctnessは原resultのcompiled_vector各軸がstrict <1e-7、phase_alignment=false、single saved input stateの既存PASS。vector archive自体のraw byte identityを確認したが、誤差の再計算は0。low-control/high-leakage/independent-occupation/saved-signal/Hadamard検査も元receiptの範囲で回収し、all-state/operator保証へ拡張しない。原COSTの2対象PIDを固定worker_exit receiptのexit0/reapedへ結合し、4 stage completion、原plan/grant/CONSUMEDの結合と旧STOPを確認。過去監査がPASSであることと今回の静的確認を分ける。

## 既存資産の再利用

着手前にBのSOURCE/manifest・原本回収実装・QPY parser・SHA検証と、元PFの実行/監査SOURCEを読んだ。既存 `h2_pr_identity_recovery.py` のfile_digest/verified_bytes/loads/digest/git_bytes、`h2_pr_saved_qpy.py` のread_qpy/summaryを変更せず呼び出した。zlib専用archive verifierはgzipに適合しないため、同じ検証方針でgzip framing・サイズ上限・trailing拒否だけを小さなadapterにした。科学compiler・guard・cost counter・saved-state validatorは再実装せずSOURCEを参照するだけとした。旧PF auditor全体はshot/費用/ランキング再計算を含むので実行していない。

出所commit・path・関数・line・SHAは[SOURCE](SOURCE.json)。pinned providerの実compiler byteは、Bが索引した固定Git `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` とも全文一致した。[compiler provider binding](compiler_provider_binding.json)に両計測関数のGit blob・SHA・lineを補完した。新コードは対象2候補のreceipt join、非機密projection、保全に限る。人工8 test PASS（破損/連結gzip、展開size、誤SHA、重複候補、private path、科学import禁止）。初回の構文修正と、科学固定後に追加された旧auditorの参照commit訂正は技術修正として記録し、元科学SOURCEへの変更・科学retryにはしない。[検証記録](verification.json)。

## 保全・引渡し・STOP

指定の2026-10-23版共通運用指示を実ファイルから読み、§8.4・§10を適用（SHAはSOURCE）。空き資源を確認し、人工CPU/RAM/worker上限を新設せず、小さい限定I/Oと静的parseを実施。共有venv/OS/scheduler/GPU設定・他ユーザーprocessへの変更0。宣言closureと既知H2 run rootの2,141 fileは最初のbaselineから前後SHA/bytes一致、変更0。[保全監査](preservation_audit.json)。旧grant/ledger/marker/STOPは読み取りのみ、新grant・旧grant再利用0。

GPUA100-gleap02向け[非機密handoff](HANDOFF_GPUA100-gleap02.md)、[未取得原本・保持者への要求](GAPS_AND_REQUESTS.md)、[分類一覧](recovery_status.json)を用意した。private内部絶対pathはGit管理外のlocator mapと21-member引渡しmanifestに限って保持。原本payloadの複製・archive作成・他host転送は行っていない。承認済みの非公開経路による受渡しと、限定projectionの将来公開を分ける。

新成果物の公開承認は未確認のためpushしない。政策2a等の採択、Aの独立diagonal証明、旧Bのcross-runtime条件、COMPARE実装・実行へ進まない。[STOP](STOP.json)：科学call0、compile0、simulation0、COMPARE0、正式採択0。次段階は開始せずmandatory STOP。
