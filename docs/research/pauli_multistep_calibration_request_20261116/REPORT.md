# PR-Pauli較正要求入口の複数実step対応

既存`dd81b846f29262b14ef235b9885bf7f4b9e0100e`の入口を拡張した。B2が選んだcanonical configurationとD2の26軸応答から、**保存価格の再利用、保存較正の再利用、新しい実step別較正要求、SOURCE不足**をroundごとに返す。h・T・q・R・pieceを要求幅から再生成しない。新しい実signal・path・compile・fit・quote・Q3は0件。

[API/HANDOFF](HANDOFF.md)／[保存・人工検収](ACCEPTANCE.json)／[TESTS](TESTS.txt)。既存branch `codex/a-pauli-on-demand-preparation-20261116`は着手時に清浄だったため継続した。B2の新LD準備・現幅の価格不足解消の実装や実行領域は変更していない。

## 既存成果と実行経路

§10.5として旧入口・テスト・保存LD3/LD5のBINDING、D2 `c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa`の`pr_price_rules`、既存split driverと`13ab1acfa`のexplicit-operators取得coreを確認した。対象範囲に同等の複数step要求生成器は見つからなかった。D2の`point`はcanonical q/T/実stepとpieceを使い、既存モデルは実stepが違えば不足を返す。その判定をAでやり直さず、**渡されたD2応答の値・SOURCEをコピーする**。

旧A入口の不足は、`scope_reasons`の実step=.1制限、cell T=.8、driver metadataのT/step=(.8,.1)固定だった。workerと既存`acquire_path`はrequestをそのまま使うため、取得演算の再実装は不要だった。旧LD2 requestのidentity再現検査、保存SOURCE、旧SPECS、seed照合・scalar一回・full wrapper・compile/gate/STOPは保全した。

## roundの分類と較正要求

1. B canonical validatorと元H4 H/state/snapshot・profile・hを照合する。元configurationのhと各Tは不変。
2. D2のconfiguration/physical ID・m/X/Y・q/R・policy・SOURCE・26軸の完全性を照合する。filled軸は元の数値・分類・SOURCEを保持する。
3. 欠測roundのS2/K2・1 signed piece・整数allocation・scalar/sampler/control/compilerを確認する。実step=T/qをexact有理数で扱い、現driverのbinary64 operandと厳密に対応できない場合は必要なSOURCEを返す。
4. 同じ実stepのroundをまとめる。既存S0-r1のq=8を維持し、そのstep groupのcanonical Rの最大値を較正Rとする。cellのTは8×当該実step。これは非configuration較正cellでありm=null。元roundのTやqを変える処理ではない。
5. 元の`cell_request`と辞書identityが完全一致する保存BINDINGを優先する。違うstep/Rの較正を同じものとしない。既存cellと一致しなければdriverが読めるrequestだけを返す。

`round_actions`に元ID・T・q/R・実step・不足軸と次action、`calibration_requests`にdriver入力・実stepのexact値/hex・covered round・提案domainを保存する。実行承認は常にfalse。新しい較正の後にもS0-r1分解、全requested Tのwrapper margin、D2の登録/適用/検証が必要であり、同step grouping自体はモデル採択や全round価格の資格ではない。

`prepare_many`は、実物理cell request ID・law・cohortが一致する複数候補の要求を1つにまとめ、各canonical候補をconsumerとして残す。候補・配分・価格recordは混合しない。Rやlawが違うcellは自動統合しない。出力base/privateはcellの物理hashを含むため、同じQ8R1という短い名前でもstep/Tが違えば衝突しない。明示overrideが衝突する場合も拒否する。

## 検収したケース

|入力|保存再利用／必要な要求|状態|
|---|---|---|
|現行幅、LD3/LD5|元26価格をそのままコピー。保存Q8R660/Q8R4のrequest・辞書・scalar・16 seed・profile・operator SOURCEを再現|旧経路と一致|
|binary64 ΔPF=.2（人工D2応答）|m0を再利用、m1～m12を実step .2・Q8R8/T1.6の1要求へ集約|READY、未実行|
|binary64 ΔPF=1.6（人工D2応答）|m0を再利用。実step .2/.4/.8/1.6の4要求。最後のgroupはm4～m12|複数group READY、未実行|
|厳密ΔPF=1/5（人工D2応答）|m0/m1は再利用。m1のq=2はbinary64 .2のq=1と異なる。m2～m12は実stepが現binary64規約と一致しない|round別にSOURCE不足。丸めて代替しない|
|.2とその直上binary64の別canonical候補|同じ実schedule/cellに対する較正要求を1件、consumerは2候補のまま|最大2 path/4 compileの要求1つ、未承認・未実行|

上記の別幅caseはcanonical metadataと人工D2応答の検査であり、新しいH4価格や信号の科学結果ではない。厳密1/5のm2以降に必要なのは、canonical実stepと物理binary64への変換を定義・検証するSOURCEである。既存C診断`dbeb6480db0fb76e951a7e23116a39135712ba59`の問題を価格側だけの丸めで回避しない。

## テストと保全

旧入口6件、複数stepの接続6件、driver人工3件、計15 PASS。通常CLIから複数requestを出力し、実driverの`--request ... --metadata`、worker入口のspec/binding選択まで検査した。科学leafのload/input_preflight/acquire/worker/executeは禁止した状態で確認した。readerのbatch入力と同名cellの出力衝突も修正し、該当経路の回帰に含めた。

primary agentがadapter/CLI・保存照合・統合検査、sub-agentがdriver差分と人工3件を担当した。独立科学レビューを行ったという主張ではない。新しい性能倍率は測定していない。既存module cache・SOURCE照合・seed/driver資産の再利用と、物理cell単位の重複削減のみを実装した。

科学計算の実行はしていない。新較正のinput_preflight・freeze・取得・audit・モデル登録には別途の有限科学実行指示が必要である。旧入力・価格・BINDING・SOURCE・raw・cohort・STOPは変更せず、公開照合後に技術作業を終了する。
