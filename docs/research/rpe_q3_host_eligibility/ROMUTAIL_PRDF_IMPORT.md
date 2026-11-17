# RomuTail PR-DFの保存PASSと専用adapterの取込み

固定証拠 `09efb068f242d5d6ff88296df3e50283b2f951bf` の
`docs/research/rpe_prdf_romutail_source_20261117/` を
[evidence/romutail_prdf_source_09efb068f](evidence/romutail_prdf_source_09efb068f/HANDOFF.md)
へ16ファイル、原byteで保存した。[固定locator/hash](checks/IMPORT_ROMUTAIL_PRDF_09EF.json)。
新しい実dry-check・科学計算・遠隔接続は行っていない。

元REQUEST `48494ae31276a64c46ce`、そのhash、SOURCE28d、外側route23160cc87、
X/Y配分、入力hashは元のまま。RomuTailで公開済みの公式READINESSは、
`e37f6544dea3e021d3694264f7dae87add7ad30e` のpath解決adapterによるPASSである。
この実装は元callerのbytes/ASTを変えず、内側27e70efeをホスト別mappingへ解決する。

現在の09efは証拠記録を追加したdeployment commitである。
adapter本体とprefix-cache helperのbytesはe37と09efで完全一致することを確認した。
[adapter binding](checks/ROMUTAIL_ADAPTER_BINDING.json) に実装e37と配置09efを分けて記録し、
元READINESSの `adapter_source.commit=e37` を書き換えていない。

[専用profile](profiles/local.prdf.adapter_09ef.json) は公開RAW PROFILEのコピーへ
`adapter_root` と別の `adapter_manifest` を追加したもので、元SOURCE28dのmanifestは変えない。
manifestはadapter本体と、実行時prefix-cacheで使うhelperの2ファイルを固定する。
server2とserver1は既存28d adapterのままで、RomuTailの当該routeだけが専用adapterを使う。

公式のRESOLVED_REQUESTは元28d validatorのまま保持され、mappingは外部
`ROUTE_VIEWS.json` と `--route-views` で与えられている。
そのmap、元SOURCE/inputの照合記録、公式adapter_result.runtime、未変更AST guard、
内側SOURCE/entry hashと `source_resolution` を突き合わせてPASSを採用した。
workerも同じ外部flagを使い、共通adapterへ黙ってfallbackしない。

将来のclaimでは専用adapterのHEAD09ef・両ファイルhashと、元SOURCE/route/input/runtimeを
実機で再確認する。今回、現在の遠隔HEADを取り直してはいないため、HEADが違えば
新規claimを保留する。既存の異なる作業branchを強制checkoutする手順は含めない。

この取込み直後のcheckpointは10 PASS・2 UNCHECKED（remote Pauli2セルのdelta待ち）。
PR-DFは3hostの保存証拠が揃った。[取込結果](checks/ROMUTAIL_IMPORT_RESULT.json) と
[現在の集計](HOST_ELIGIBILITY.json) を区別する。
元のRomuTail不足・誤った内側checkoutの拒否記録も保存している。

focused testsは16件PASS。追加3件で元REQUEST/解決後copyの維持、専用adapterの
実装・deployment分離、未指定fallback/誤hash/外部mapping改変の拒否を確認した。
著者自身の保存証拠・人工検査であり、実Q3の完了や科学的性能の検収ではない。
