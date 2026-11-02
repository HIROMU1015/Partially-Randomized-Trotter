# H2共通native・共通allowance検証基盤：技術campaign

状態：**TECHNICAL PREPARATION PASS / SCIENTIFIC EXECUTION NOT READY / mandatory STOP**。
対象はH2・ε_sig=0.0005の固定4候補だけである。逆転の原因、比較Cの科学的成立、共通政策の正式採択は未認定。

## 実装した範囲

新規module `src/trotterlib/h2_common_native_policy.py` はstdlibだけで動作する。共通Gaussian/nativeの固定SOURCE binding、候補別の遅延call recipe、mock構成adapter、診断値schema、保存metadata転記adapter、人工入力専用allowance生成器、費用再利用のidentity照合、人工correctness/provenance schemaを備える。runner `scripts/prepare_h2_common_native.py` は非実行draftを検査し、`--execute` は入力を読む前に拒否する。

実装済みのnative adapterは**mock operation logと静的call recipe**である。実H2を処理するproduction backend、科学grant発行器、分子loader、科学dispatcherは未接続。これらの完成や実回路の作用等価性を、人工テストのPASSで代替しない。将来の承認済みrunnerは、このinterfaceと固定kernelを結合し、新SOURCE/runtime/input/call budgetを封印する必要がある。

人工テスト36件PASS。signed piece順、scalar-once、wire/basis/control、4候補外拒否、欠測・単位・非有限・overflowのSTOP、実保存値の演算拒否、結果provenanceとstrict gate、無承認dispatch拒否を確認した。科学libraryのimport試行を拒否するguard下で実行した。検証は実装者であるCodexが行った技術チェックであり、独立科学レビューではない。

固定技術SOURCEは `bcc05ef0a028c6bbd18ccf9c7a260734078ca5bf`。Git CLIの認証が利用できないため、接続済みGitHub APIで同じcode treeを公開し、この公開commitへ契約案を結合した。科学kernelの固定commitとは別の、mock/静的基盤のSOURCEである。

## 固定比較

| 方法 | formula | ID | prefix / q | PR条件 |
|---|---|---|---|---|
| Full PF | S2 | B1_p3_q4_2nd | 3 / 4 | samplingなし |
| Full PF | Yoshida4 | B1_p3_q2_4th | 3 / 2 | samplingなし |
| PR | S2 | H2O32_c7ec590a32a859bc | 1 / 4 | K4、各outerの全piece合計32 steps |
| PR | Yoshida4 | H2O32_8a2ecdd0294ffb74 | 1 / 2 | K4、各outerの全piece合計32 steps |

共通条件はlinear H2、隣接1.00 Å、STO-3G、tol-only 1e-8/rank3、T=0.8、各軸α=0.025、ordinary signed bias、P=0、system4/ancilla4、symmetric_directional control、cos/sin full Hadamard wrapper、shot込みcompiled RZ費用。q再探索・他精度・他系へ拡張しない。

## 保存証拠の照合で進んだ点

- 必須3系列を固定commitから読取。元公開exportとの150 fileのbyte一致、新providerの封印278 fileとのSHA一致、PR67 task/134軸と候補別sample indexの対応を確認した。元acceptance metadataの検査でありcorrectnessを再実行していない。
- providerの旧新同278 path比較は前段の260一致・2差・16旧側欠如を保持する。同provider runtimeを認定する根拠にはならない。
- 前段環境で不在だった旧H2既知metadata3原本は、今回の環境で存在とhashを確認した。成功したrecovery specも公開execution bindingとhash一致。旧Full2候補のcontrol/result原本は受領projectionと一致した。原本本文は新たに公開していない。
- 調べた2つのcompile worker直下にはQPYやvector archiveがなかった。compiled correctnessの数値受入証拠は回収できていない。これは検査したフォルダに限る観測である。
- 旧FullのGaussian通常経路、新PRの隣接complex QR経路、共通化のcall境界を固定Git/ASTから整理した。科学codeをimportして操作列を生成していない。

証拠は[metadata照合](metadata_verification.json)、[private原本照合の公開要約](private_recovery_metadata_check.json)、[固定SOURCE索引](SOURCE_INDEX.md)、[source manifest](source_manifest.json)にある。manifestは604個のcommit/path記録を持つ。GitHub確認は[3入口のblob確認](github_input_receipts.json)と区別し、全604記録をGitHubから再転送したとは扱わない。

## 残る差と最小の次段階

共通nativeは新PRのadjacent QR preparationとsigned-piece composeをFullにも適用する第一候補である。prefix、tail、係数、負時間、scalar、event lawは方式ごとに保持する。preparation hashの同一化を目標にしない。新Full nativeは旧Fullとは別系列となり、旧Cを自動転用できない。PRの保存67 taskのCも、kernel/preparation/wrapper/runtime/event-lawの対応証拠を条件とする。

共通allowanceは旧Aの経験的適応規則を第一候補とする。同じ生成規則であり、同じuを強制する政策ではない。PRのpath-vector差、両path実work数、全stage norm、corrected-signal単位のclosure、共通referenceとの対応が不足する。旧Fullの診断値が保存されていても新共通nativeへの結合は未認定。値の欠測を0やcorrectness閾値で埋めない。

senderのruntime照合にはliblzmaのhistorical sealと観測値の不一致1件が残る。変更時点・原因・計算影響は未確認。過去結果を直ちに無効化したり、影響なしと認定したりしない。将来使う共通runtime closureを新規封印し、無関係な旧環境全体の復元を要求しない。

新規診断は欠測分から選ぶ。旧FP referenceとその経験的診断の再利用を優先案にし、新reference・新sampling・全134軸recompileを自動要求しない。正確な科学call数は未確定で、契約案のcountはnull。現段階は一括実行を直ちに承認できる状態ではない。この未完は隠さず、次の政策判断とcall recipe確定に接続する。

## 成果物と承認境界

- [native設計・Gaussian差分](NATIVE_DESIGN.md)
- [allowance schema・単位・元規則](ALLOWANCE_SCHEMA.md)
- [Gate 0〜4・最小科学実行計画](EXECUTION_PLAN.md)
- [非実行config](preflight_config_DRAFT.json) / [検査結果](preflight_result.json)
- [4候補call recipe](deferred_native_recipes_4.json) / [kernel binding](kernel_bindings.json)
- [実行契約案](execution_contract_DRAFT.json) / [診断coverage](diagnostic_coverage_4.json)
- [技術検証記録](technical_verification.json) / [テスト出力](technical_tests.txt)
- [STOP記録](STOP.json)

ユーザーの判断対象は、共通native仕様、共通適応allowanceの正式採択、reference再利用・診断範囲・C再利用条件である。正確なcall recipe、使用runtime/input seal、未使用one-shot grant等を別途確定した後に科学実行承認が必要。今回の実H2 build/compile/simulation、signal/sampling/reference評価、保存値による新u/N/G評価はすべて0。

今回の技術成果は専用branchでcommit/pushしremote照合する。過去SOURCE、raw結果、grant、ledger、marker、STOPは変更対象にしない。公開確認後も**mandatory STOP**。次campaign・詳細科学レビューへ自動進行しない。
