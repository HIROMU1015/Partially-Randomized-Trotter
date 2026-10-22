# 保存PR単価：67 paired tasks / 134 axesのidentity監査

**PARTIAL、COMPARE_REUSE_PRはBLOCKED。** 新しいsampling/compile/operation stream生成、保存u/N/G再計算、順位判定は0。

S2 `H2O32_c7ec590a32a859bc` 33 tasks、Yoshida4 `H2O32_8a2ecdd0294ffb74` 34 tasks。[全task監査](PR_C_task_identity.json)にcandidate/sample、SOURCE/input binding、prep、scalar、event digest、保存event要約、signed-piece metadata、axis fingerprint/compiled SHA/paired RZ、保存checks、cohortをコピーした。平均単価・u・総費用は新しく計算していない。

## 証拠の対応

| 確認対象 | 保存証拠 | 新系列との判定 |
|---|---|---|
| Hamiltonian/state/snapshot | 保存bindingは固定H2と対応 | 保存側のmetadata一致。新setupによる再検証は未実行 |
| preparation | prefix1 hash85b8243d…、fullprep3 hash98c1beb1…、SOURCEのprepare(ham,1/3)宣言 | 実prep object/rotation/operation streamが欠測。新prep hashは未取得 |
| signed schedule | task別piece_request_index、signed_step_time、rte_steps、K4 law/cutoff、unconditioned保存値 | 操作列全体とのbyte identityまでは未成立 |
| events | 67 event_digest、両軸のdigest join | events欄はN2/N4/stratum要約。eventごとのorder/component/phase/順序のfull bodyではない |
| rotation / Gaussian stream | explicit QRのSOURCEとラベル | 保存rotation byte/hex gate streamと新streamの同一証拠が不足 |
| scalar / wrapper / lowering | 保存scalar、symmetric-directional宣言、native fingerprint、固定取得/provider SOURCE | scalar最後一回を含む実行stream・loweringと新系列の対応証拠が不足 |
| compiler | hash1239304c…、Qiskit1.3.0宣言、134軸のcompiled SHA/fingerprint/RZ、guard資料 | settings一致だけではimplementation/runtime identityを証明しない。guard保存hashも後日の現在byte由来という限界を保持 |
| runtime | 保存Python3.12/新provider metadata、sealed file hashes | worker実load manifest欠測。今回のfresh3.11と同一とは判定しない |
| correctness | native_scientific_PASS、saved checks、DIRECT_MEASURED_CT01_SINGLE_SAVED_INPUT_STATE | 保存state限定scope。全state証明・現runtimeでの再現・新common系列受入ではない |
| cohort / paired単価 | sample0..32/0..33、cos/sin pair、conditional=false、pooling=false、seed_cohort | 保存cohortを保持。母平均保証・独立確認cohortへの格上げなし |

`projection_sha256` は受領保存result projectionのcanonical JSON digest。元実行時の個別receipt原本hashと同じ証明とは呼ばない。prepared hashesやcompiler settingsの名前を新policy名に改名して受入れる処理はない。

## 回収したGit証拠と正確な欠測

受領metadataは基準commit内の固定Git blobと照合し、[metadata_sources.json](metadata_sources.json)へ保存した。固定Git object `4f1e4ebd46a5e03876971a5374d390bd4ead921a:scripts/export_h2_outer32_metadata.py` もbyte回収し[evidence_sources](evidence_sources/export_h2_outer32_metadata.py)へ保存。実行していない。exporterはpipeline/exploration/old-spec/private-mapをCLIから受ける構造で、元private canonical rootをhardcodeしていない。

既知366b2fd2監査の固定extract/build資料は既存Git資料を抽出するもので、元event raw locatorにはならない。受領origin inventoryのtoken化path、取得SOURCE/結果資料、旧H2の既知D/FP/F private locatorまでを限定調査した。projects全体・他研究private・別server・agent履歴の推測探索はしない。

次の必要資料は[retrieval_audit.json](retrieval_audit.json)にhash/byte数を記録した。

- **O00002**：`${NEW_EXECUTION}/stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json`、原本SHA `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7`。
- **O00001**：`${NEW_EXECUTION}/config_PIPELINE_APPROVED_PRIVATE.json`、原本SHA `f4da8233474bf8e9f43784e80028948d62d8dd8f43a3f8d82f4c1136ccbdfc0d`。
- これらのcanonical locator、またはsenderが保有する`${NEW_EXECUTION}`等のprivate token map。旧confirmation configの既知locatorが得られれば `/origin_config_path` を追う。

locatorを得た担当Codexは原本byte/hashを先に照合し、そこからsealed task receipts/storage manifestへ進む。既存archiveが見つかればraw SHA/archive SHA・task/axis対応を確認し、許可されたmetadata回収としてevent/stream witnessだけを取得する。保存vectorの新しい数値評価・回路生成・compileはしない。原本rawが残ると未確認のまま断定せず、公開hashのみからfull bodyを復元したとも言わない。

欠測はfull event列、rotation/operation stream、scalar/wrapper/lowering実装とのidentity join、当時loaded runtime、correctness scope受入。locators不足も明示的BLOCKER。DIAGNOSTICSだけには保存PR Cを使わないため、これらは診断の前提に追加しない。

新しい科学witnessが必要なら、元event/影響軸を確定した別契約を作る。[COMPARE_CONTINUATION.md](COMPARE_CONTINUATION.md)の最小案を使い、全67 taskの自動再sampling・再compileは行わない。
