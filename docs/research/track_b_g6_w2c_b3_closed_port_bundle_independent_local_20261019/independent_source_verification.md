# 固定SOURCE・原証拠

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

|役割|commit|確認|
|---|---|---|
|直接独立基点|`5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`|merge/rebaseせず直接worktree作成|
|統合公開/測定|`62967269e611b61e73b70abe20e998d9c7eaaf50` / `deade78a19f5b23859d16816d3a7bfb37765dd80`|68 pinsのGit/public/測定blob一致|
|固定Port|`82896d53fa97235fa848d6c0833eb193af8b99d6`|6 fileの元bytes一致|
|固定Bundle|`d24cccbecdf0928fe977080dd7b28c591df0089e`|元runtimeとversioned派生の差分を独立生成|
|旧Bundle測定|`e099dcc461247e81eff981867dc28c3f3b16df5e`|基点内の固定参照を保全|
|独立Bundle/Storage V2|`6bffb5b5819918698b522270ff19b255766d3072` / `596781daeba253418ba35f4b7964312a54caf233`|既存独立consumerと原証拠を保全|
|科学参照|`76ca3c4089b201d55da0fede08e1f77733ac38a5`|科学moduleはbytes/ASTのみ。importなし|
|共通正式指示|`7eff8a33bef4ee1f29698ab10c6da46988dc8db5`|取得し§8.4適用|

[固定Git inventory](fixed_git_inventory.json)、[公開bytes295件](fixed_public_byte_verification.json)。大きな `global_byte_accounting.json` は1,432,698 bytesの全文を取得・parse・SHA照合し、空/省略とは判断していない。公開/測定/source rolesとpostprocessing helperを混同しない。管理helperは測定68 pinsに存在するものだけを測定runtimeとする。

raw SHA256 `6bacee98898f0aaeab4b15e2a5513867f14afc240f3c7b2e451f90ad6fab9d77`。10 parts、39,731,962 bytes、2,706 files。part/archive/member全件を照合し二度の専用復元を実施。既存結果への上書きなし。

ローカルは32CPU/Python3.10.12/64GiB級RAM。起動時のload・available RAM・I/O、実行中CPU/memory PSIを確認。公開case検証は32process、SHA/I/Oは32threads、新規独立mutationは14methodを個別processへ分け最大17process。新規人工3scopeは同時実行し、各scope内のS→P1/固定8workerを維持した。任意の低いCPU/RAM/wall capを新設していない。共有OS/package/processを変更していない。

serverはPython3.12.3。サーバーruntimeの宣言は固定するが、その全executable/library bytesをこのhostで実証したとはしない（historical_runtime_bytes_proven=false）。新規人工3件ではローカルSOURCE/input/executable/library bytesを独立再照合しlocal_runtime_bytes_proven=true。別物理host独立性はNOT_ESTABLISHED。
