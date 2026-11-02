# 人工受入結果

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

| 対象 | 結果 |
|---|---|
|新規unit tests|73 PASS / 0 FAIL / 0 ERROR / 0 SKIP|
|新port正常grid|n16:1536 calls、n256:16896 calls、各48pair/6phase/16cells|
|新port失敗grid|S generator:224 completed、P1 angle:992 completed、いずれもtyped technical failure。失敗prefixを正常比較へ混入しない|
|新offline mutation|22拒否：INTEGRITY_MISMATCH18、INCOMPLETE_EVIDENCE4|
|変更していないBundle43cases|bundle complete2、condition-only2、technical failure13、incomplete26|
|Bundle旧tests|107 PASS、うち35 overlaysを含む|
|既存回帰|補正版623＋Storage V2 41＋既存独立V2 57＋Bundle107＝828の別versioned ID|
|旧原版label|Python3.12 原版622 PASS/1 FAIL、補正版623 PASS。原版とcompiled差分を保持|

Local独立845 PASSは固定参照6bffb5b5の証拠で今回新規実行した件数へ足さない。旧件数/新unit/ケース/mutationは重複加算しない。テストPASSは期待した拒否を含み、evidence不完全ケースが完全受入になったという意味ではない。

全人工raw/overlaysは分割archive 12 parts、46,334,109 bytes、5912 members、SHA256 `884a442b394fc16888eb6662c236b0788da2449f19c1a7f6765e2734e5650925`。fresh restore後の104分類を再照合。rawの全member SHAはevidence/archive_manifest.json、復元手順はverify_restored.py。新offlineは同じ主担当の自己検証。検証した人工範囲で未修正の技術不整合なし。

追加postprocessing controls4件もPASS（approvalのみ不一致、grantなし、fake bits/authority実物のreal port転用拒否）。73 unit testsへ加算しない。
