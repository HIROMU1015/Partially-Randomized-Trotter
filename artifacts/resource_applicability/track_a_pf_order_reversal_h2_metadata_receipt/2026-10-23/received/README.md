# H2 outer32：保存metadataの共有

2026-10-23。**`PARTIAL_WITH_GAPS`。対象PRの67 task／134軸の元receiptは取得・静的照合済み。runtime・nativeの完全同一性、共通allowance政策、比較Cの科学的成立は認定していない。**

これは保存metadataのprojectionである。新SCF/DF/state、sampling、signal、simulation、回路build/compile、数値誤差・allowance・shotの再計算、費用会計・ランキングは全て0。旧科学runner・科学audit・pytestは実行していない。元の75行監査は参照のみで再実行していない。

## 対象と固定来歴

|方式|候補|取得した範囲|
|---|---|---|
|Full PF S2|`B1_p3_q4_2nd`|native control、両軸測定・fingerprint・compiler receipt|
|Full PF Yoshida4|`B1_p3_q2_4th`|同上|
|PR S2|`H2O32_c7ec590a32a859bc`|EXPLORATION n33、66軸|
|PR Yoshida4|`H2O32_8a2ecdd0294ffb74`|EXPLORATION n34、68軸|

linear H2、隣接1.00 Å、STO-3G、保存tol-only 1e-8 DF3、T=.8、P=0、対象精度ε_sig=.0005。Full PFは全DF因子/prefix3、PRはprefix1・K4・ABS_TIME_PER_OUTER=32。symmetric_directional、cosine/sine full Hadamard wrapper、compiled RZ。方式固有のpreparationやsigned scheduleを無理に同一化していない。pilot・旧PR・独立確認cohortの混合はない。

固定commitは[前段監査366b2fd2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/366b2fd24f7e86aa82b5b518b6e3774520f65596)、[新取得SOURCE4ec26921](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4ec26921c6872ed6e8667f167074e1e405cf1a3f)、[取得結果a05f4ed1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a05f4ed19e87e1eb65f7fc346385b5b4eae69845)、[保存会計SOURCE9a77d2b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/9a77d2b0a0b87e246e85f085b37f5923c18e27bb)、[保存会計結果8613b3ec](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8613b3ecda2030a677b1690fe7aa3da23dec4873)。旧Full PFは[結果829d697a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/829d697ae7905b3ea90a61da9d2d4da15d8b757a)、[compile SOURCE cf99e445](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf99e4452df96291d7e9d37d76d2e91be5cbc413)、[provider217c971d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/217c971dd537b1cc7fbc13742bc0f6c263ff7f45)。固定commitの資料を参照し、今回新しい費用を生成していない。

新探索の元plan SHAは `4dda5e68815f8359b3890dc7a9e301956a631d6cea500e8a90703224db719373`、元終了audit SHAは `38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723`。元config・plan・auditを実読取し、この値と一致した。新signal原本SHA、runtime inventory、各result/receiptの原SHAは[origin_inventory.json](origin_inventory.json)と[各ファイルinventory](file_inventory.json)で追跡する。

## ファイルと読み方

|ファイル|内容|
|---|---|
|[acquisition_context.json](acquisition_context.json)|元pipeline/configからのSOURCE、plan、input/runtime/preparation、正確なtoken locator|
|[runtime_provider_manifest.json](runtime_provider_manifest.json)|2,012封印file、保存hashと現在のhash、package identity、Git対応、旧runtime保存manifest|
|[preparation_metadata.json](preparation_metadata.json)|fullprep3/cut1と旧PFの別hash、Hamiltonian/state/snapshot binding、SOURCE上の構成宣言と欠測|
|[signal_records_2.json](signal_records_2.json)|対象2候補の元signal、bias成分、log_B、u/margin、forward誤差、allowance診断を無加工数値で転記|
|[rte_schedule_metadata.json](rte_schedule_metadata.json)|67 task・336片の保存signed time、RTE step配分、有限law・cutoff。seed値は除外、再計算なし|
|[reference_metadata.json](reference_metadata.json)|reference原SHAと条件付き精度宣言、旧reference scalar、snapshot receiptのshape/dtype/hash等。配列本体なし|
|[task_acceptance_67.json](task_acceptance_67.json)|67 resultと原audit accepted SHAの結合、134 cost、8種の保存correctness check、両軸識別|
|[compiler_guard_receipts/](compiler_guard_receipts/)|task/軸別の134 guard。policy、各passのstream digest、scalar、元/guarded phase、補正・state fit有無|
|[stage_end_metadata.json](stage_end_metadata.json)|原525-task全stageの終了audit・completion・exit・guard audit。今回実取得したreceipt範囲は67 taskに限る|
|[storage_records.json](storage_records.json)|402 archiveの保存manifest・release・原SHAとarchive SHA。現在archive byte SHA照合済み、復元・decodeは今回0|
|[old_pf_receipts.json](old_pf_receipts.json)|旧PF2候補のnative control・測定receipt、保存reference、終了記録とcompiled検証欠測|
|[missing_evidence.json](missing_evidence.json)|欠測・不一致と認定しない事項。0補完なし|
|[technical_verification.json](technical_verification.json) / [final_static_verification.json](final_static_verification.json)|抽出時検査と、原本JSON pointerに戻る別の転記検査、保全・公開情報検査|
|[completion_status.json](completion_status.json) / [STOP.json](STOP.json)|実coverageとmandatory STOP、後続科学stage未承認|

`{origin, pointer, saved}` は、`origin_inventory.json` の原本IDとJSON pointerから取得した値である。元JSONの小数tokenは丸めずそのまま保存し、転記のためにfloatへ変換していない。公開projectionはパスtoken化・秘密項目除外を行うため、**原本とのbyte一致を意味しない**。原SHAはorigin inventory、公開projection SHAはfile inventoryに分離した。

`${NEW_EXECUTION}` は新outer32元実行root、`${NEW_PROVIDER}` は外部provider、`${ACQUISITION_WORKTREE}` は科学SOURCEのworktree、`${RUNTIME_ENV}` は専用venv、`${OLD_H2_PRIVATE}`/`${OLD_PROVIDER}` は旧H2側を表す。同じtokenと相対pathは同じ原fileを指す。`${OTHER_ROOT_894d731f69bb}` はsystem usr tree。その他tokenも原rootを一意に識別し、絶対pathとの対応は非公開で保存した。別環境では固定Git path・hash・roleとこの相対pathで照合できる。

## runtimeとpreparationで分かったこと

新探索の封印inventory2,012件のうち、現在byteとの一致は2,011件、不一致はliblzmaの1件。保存SHAは `696e868dd0700a19a6d65fc01608ec2d70d3cb91f65710e89180cd2e688f30cb`、現在SHAは `26b37d25f7ca2a2d348de8ecabacda12e921cd933b21fa069a179760db77219d`、sizeはいずれも202,904 byteである。これは時点差として報告するだけで、原因・変更日時・過去結果への影響は認定しない。現在の環境を当時の実行環境として代用しない。

新取得の封印Python binary SHA `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` と、旧specのPython SHA `ebc2fae9fa0845b8c1fc09ed10196c0e93c26cb8fbd2a84d64548326ef93209b` は異なる。package表示が一致してもruntime全体の同一性とはしない。新封印metadataのQiskit1.3.0、NumPy1.26.4、mpmath1.3.0、symengine0.13.0と、各METADATA原SHAを収録した。

新providerの278封印fileはprovider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の固定Git byteと全て一致する。同じ278 pathを旧provider217c971dへ照合すると260一致、2不一致、16 pathは存在しない。内訳をmanifestで公開し、旧providerへの完全同一性を主張しない。科学SOURCE4ec26921の70封印fileは固定Git byteと一致したが、それだけで外部provider全体を結合済みとはしない。

明示的に調べたdf_partial_s2、df_hamiltonian、pr2_s0_s1_validation、product_formula、rte、rte_compiled_cost、ax2a_preparation、ax2b_native_df_v5は、新外部providerの封印inventoryで保存hashを確認できた。per-workerの実import完全manifestは取得できず、保存import/runtime inventoryとSOURCE上のimport元検査を分けて記録する。

保存fullprep3は `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9`、cut1は `85b8243d752d3390e214b429a80e3899f0cc8b0166e0958fc1d73a930a77597c`。旧Full PFは別hash `8e32e931047087a4a9c1f613d406b6fe353c0aa4601b7e5328750ab6ab2c9a42` を保持する。入力bindingは対応するが、complete preparation objectのblock/partition/basis全体を復元照合したわけではない。

## guard・signal・旧PFの限界

新PRのguardは `H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1`。134軸全てで保存policy、unchanged_gate_stream、state fitなし、各passのpostcompile補正なしを転記・静的照合した。**元accepted resultにはguard fileそのもののSHAがない**ため、guardとの結合はtask/axis locator・原stage guard auditによる。今回取得した原SHAを、過去にauditへ封印されていたhashとして扱わない。

新PRのu=1e-9とmargin=1e-12は保存floor付きの経験的適格性結果である。元のvector closure・中間norm・独立forward誤差・normalization誤差も転記した。uから診断値を逆算していない。reference誤差を全て含む保証、正式共通shot政策、厳密bias certificateとはしない。独立したpath-vector差やwork/action countの名前付き記録、reference誤差上界はnullと理由を残した。取得時の「shot政策未採択」というheaderを、後続会計の採択状態へ書き換えていない。

旧PF2候補はcontrol PASSと4軸の測定receipt・fingerprint・compiler hashを取得した。しかし正確な2 worker output内にcompiled-action/phase correctness receipt、保存QPY/vectorの実在locator・hashは見つからなかった。native control PASSをcompiled-vector gateへ格上げせず、新PRから検証結果を移送しない。存在しないと全環境について断定するものではない。

## 技術検証・保全・次の担当

抽出・照合実装SOURCEは `4f1e4ebd46a5e03876971a5374d390bd4ead921a`。新人工テスト8件PASS。stdlibのみの抽出時静的検査2,447件に加え、別検査器が原file/pointerの転記・受理SHA・coverage・公開禁止情報・保全を確認した。6,000件の既知原fileを前後SHA照合し不変。元SOURCE/raw/grant/ledger/marker/STOPへの書込・使用済みgrantの再利用・新grant発行は0。原科学auditやnative検証を再実行したとは呼ばない。今回の検証は担当Codex自身によるもので、別AIレビューではない。

専用branch `codex/h2-outer32-metadata-export-20261023` の公開metadataのみをcommit・pushする。private原本、QPY/NPZ、seed一覧、grant本文、credential、絶対内部pathは公開しない。既存branch・mainへのmerge、force pushはしない。

**mandatory STOP。次の担当：ユーザー／受領側Codex**がこのmetadataを使って既存証拠の静的照合を再開する。共通allowance政策、数値再会計、追加科学検証、比較Cの認定は今回行わず、必要なら別の指示・承認で扱う。
