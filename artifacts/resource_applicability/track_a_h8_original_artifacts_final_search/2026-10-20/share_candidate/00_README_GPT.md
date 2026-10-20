# H8原profile・保存PR回路の最終探索補足（共有候補）

2026-10-20。ローカル保管のみ。外部送信・ChatGPT添付・GitHub pushの承認は今回含まれない。

**原profile本体を過去の原契約内JSONとして取得した。保存PR回路は0/192軸のまま。** 旧共有パケットを変更せず、この補足だけで欠測statusの変更と根拠を記録する。詳細科学レビュー・最終evidence freezeは未開始。

## 対象と固定資料

linear H8、隣接1 Å、STO-3G、tol-only1e-8 DF28、T=.8、P=0、二次partial PF・r4/K4、通常signed bias・allowance倍率1。p8/q2（L_D8、δ=.4）とp7/q16（L_D7、δ=.05）、各n48、計96 trajectory・192軸の既存監査に限定する。p8 replica0・両軸は元SOURCE `1549b6f891a15ff48379996c34b344118ba2606c` の完了軌道を再利用し、残り190軸は `745a674546f2cb6258526fbfde1d9a00601689f8`。探索n8・旧n32・conditional pilotは混合しない。

- 結果：`e6eea5387a6e84d1e39a9adb3b61975acc74e200`
- 前回ローカル回収：`253356942845dddf05c2bb6f4582b026e0bce942`
- 前回ローカル共有候補：`363199c83b4cebde3d144f173e0e0aae67f7b2df`
- 前回公開パケット：`a6246f03c6297438ab624fc62e5061a58e53145f`

## 原profileの発見と意味

過去のH8 B0I契約 `final_contract_v1.json` の `/profile` が原本。原契約7,727,137 bytesのSHA256は `2c1e12d5e543a2038d2815a5a8a134a2b241b6aafba21514fd6502d6f5f16fad` で、既存固定Git資料 `b19c337f0b16513d68b59eb3366dc6eff175c313` の原契約witnessと一致した。原契約SOURCEは `54ed98c5fa5b7190f77255271f4e3bec8db15630`。このB0I契約はprofileの来歴に使用し、PR独立確認の実行receiptを代用しない。

H8識別子 `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12` は、埋込 `/profile` のcanonical JSONをUTF-8でSHA256した値と完全一致した。単独ファイルのSHAではない。profile本体にはPython/executable・runtime 34,273ファイルのpath→SHA・full critical profile参照が含まれる。両PR seal→payload→native config→既存runtime receiptの34,273項目と完全一致することを新たに静的照合した。インストール全体の再hashは行っていない。

`01_original_profile_receipt.json` は原ファイルSHA、literal切出しのSHA/offset/bytes、canonical hash規約、過去の公開witness、両条件のmap照合を保持する。原契約内profileのliteral bytesはローカルに保全した。識別子622ce…をraw SHAとする単独過去fileは同定していない。一方、profileから参照される環境compiler profile（raw SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`、8,926,400 bytes）は固定Git `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` から取得・bytes照合済み。このraw file SHAと埋込profileのcanonical digestを区別する。生成時刻の専用fieldはなく正確な生成時刻は不明であり、mtime/ctimeを生成時刻として扱わない。

内部pathを含むprofile本体は同梱しない。受領者はこの匿名化版だけではcanonical digestを再計算できない。public witnessの原契約SHA・192軸との対応は確認できるが、原profile digestの独立再計算には、別承認された原契約/profileの安全な閲覧が必要。

## 保存PR回路の結論と探索範囲

既知の実行領域・sealed store/provider/stage・元clone・来歴で特定した原profile契約領域、計10 scope・31,847ファイルを読取専用で確認した。拡張子と先頭magicによるQPY/QASM候補0、archiveの目録23件に対象回路なし。既存Git refsとreflogs、元実行clone2件にもQPY/QASM pathなし。既知scope内のアクセスエラー0。絶対path付き探索目録は非共有のローカル索引に保持する。

`04_source_and_provenance.json` は固定SOURCEの該当関数とfile SHA・行番号を保持する。元p8再利用経路と並列経路ではwrapperを生成し、pretranspile wrapperのfingerprintとcompiled metricsをJSONへ書く。providerはtranspiled circuitをメモリ上で返すが、このPR呼出経路は回路を永続化しない。fingerprint reuseもprocess内でclearされる。SOURCE上の機能と実行時PASSの証拠を区別し、今回SOURCEの関数は実行していない。

したがって、**追跡した実行経路に保存処理がないこと**と**記録した探索範囲で回路が取得できないこと**は確認した。全世界での未保存、削除、記録外の保存先不存在は証明していない。削除receiptはなく削除有無は原因不明。未接続host/backup、pruned/unreachable Git objects、symlink先、無関係なworktree・他ユーザー・agent履歴の横断探索は対象外。

preparation block/basis操作export、seed/event trace、費用/深さJSON、現在の設定、PF回路は保存PR回路の代用にしない。`03_axis_coverage.csv` に192軸すべての元task SHA・measurement SHA・fingerprint・SOURCE・再利用区分と、回路未取得を記録する。

## 旧パケットと科学レビューの制約

変更する欠測statusは原profileのみ：「未取得」→「原本で確認済み（原契約内JSON）」。単独過去ファイル未同定、保存PR回路未取得、元回路からのfingerprint再導出不能、過去compiler出力のbit同一性未検証を保持する。profile回収だけで最終gate角度・T費用・compiler挙動の完全再現や数学的なcohort独立性を追加認定しない。

旧パケットの結果・192軸・receipt、grant/ledger/marker/STOP、全ての既存科学結果は変更していない。平均・SE・費用・比率・順位の再計算、回路読込/再生成・sampling・simulation・compileは0。

## ローカル検査

```text
python3 -B verify_supplement.py
```

標準libraryだけでfile SHA・192軸・再利用2軸・SOURCE・欠測の制約を検査する。7件の改変検知を含む。原本真正性の外部認証や科学レビューではない。

共有候補：本補足ZIPと既公開 `H8_GPT_SHARE_PACKET_20261020.zip` を組にする。今回の補足ZIPを外部へ渡すにはユーザーの別承認が必要。原profile全文・内部path索引・原契約全文は別の指定資料/相手/経路の承認が必要で、repository公開を推奨しない。

**MANDATORY STOP。** 次担当はユーザー：補足候補の外部共有範囲を承認し、その後に科学レビューを開始するか判断する。
