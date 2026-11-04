# H8 p8 false flag：原因と科学的影響

2026-11-03 JST。担当は今回指定のLOCAL Codex、primary agentのみ。既存Git objectと、この作業環境にある既知の本人所有・封印済み保存資料を読取。別hostへのSSH、科学moduleのimport、配列decode、数値再解析、RNG replay、回路build/compileは行っていない。

## 結論

**分類は `EXPLAINED_NO_SCIENTIFIC_IMPACT`。対象は `saved_exact_export_match=false` が生じた原因と、そのflag自体の影響に限る。** p8の保存exportキーが存在しないためfalseであり、二つの科学的データを比較して不一致だったという記録ではない。flagは書出し用metadataで、Hamiltonian・state・reference・回路・価格・統計の選択を変更しない。p8の科学計算全体が完全に正しい、独立export照合が実施済み、母平均優位が保証されたという認定ではない。

前段[SERVER2監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/COMPLETION_AUDIT.json)と[LOCAL統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/879a28940392a1f2ea58f32f1a320d4b6822a6dc/docs/research/h8_g04_o1_evidence_integration_20261103/STATUS_AND_REMAINING_GAPS.md)は比較対象・原因を未解決としていた。本書は、その限定項目をSOURCEと保存metadataから解明した後続記録である。元のfalse、結果・status・SOURCE・旧STOPは変更しない。

## 1. 生成箇所とSOURCEの結合

生成処理は[H8 loaderのload](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_ab_native.py)（同固定fileの行98–115）。封印実行がimportした依存fileのSHA-256は `89d6fc478692cf1f0b52a7acdd00d1a05fc42449e961247ac09c5668a7d4e02b`。元native configのsource_hashesと一致し、公開carrier commit `0c49a88134066177816feca2741f36f6b88d23ba` の `scripts/h8_ab_native.py`（blob `9cfe2868b7c8edcd9008739b546dd054a0ffbbff`）とbyte同一だった。

このmain carrierは元科学SOURCEではない。確認の実行SOURCEは `745a674546f2cb6258526fbfde1d9a00601689f8`、元run/dataset SOURCEは `1549b6f891a15ff48379996c34b344118ba2606c`。両版の `scripts/pr_cost_native.py` は同じblob `1bc658ba554b63db0060f5ffb6b2b796292197fc` であり、[load_context](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/pr_cost_native.py)から同じ封印H8 loaderへ接続する。[SOURCE再利用表](SOURCE_REUSE.csv)に役割とhashを分けた。

flag生成式は次の通り。

```python
saved_exact_export_match = ('faithful:p' + str(p)) in export
```

**これはキーのmembership判定であり、内容の一致判定そのものではない。** 実際の任意比較は、その直前でキーが存在する場合だけ行う。

```python
key = 'faithful:p' + str(p)
if key in export:
    observed = preparation_export(prep)
    c.require(observed == {k: export[key][k] for k in observed},
              'EXACT_SAVED_PREFIX_COMPONENT_BASIS_BINDING')
```

比較するなら、再構成したprepのexportと、保存された同prefixのexport部分である。対象はcomponents、bases、constant_hex、extracted_hex、lambda_r_hex、preparation_hashの6項目。辞書・list・保存hex等の厳密なPython equalityで、数値許容差はない。再構成後、compile前のcontext準備中に比較し、実不一致ならValueErrorで停止してこのbinding書出しへ到達しない。「不一致を検出してfalseを保存する」という動作ではない。

## 2. 実際の保存資料

元runとresumeが使うexportは同一SHA-256 `771137828552b858361b6e2b25d2b675fb83f908aad6209f757e46464b817420`。封印済みnative configも同一だった。内容を数値評価せず、トップレベルキーを読み取った。

|保存キー|存在|実行時の意味|
|---|---|---|
|faithful:p7|あり|p7は任意の同prefix比較を通過し、保存flagはtrue|
|faithful:p8|なし|p8は同prefix比較を実施せず、保存flagはfalse|
|faithful:p14 / p21 / p28|あり|元exportに含まれる別prefix。p8の直接比較対象にはしない|
|discard:p7 / p14 / p21|あり|元discard系列。faithful:p8の代わりにはしない|

p8は保存[Full p28の29ブロックを復元](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_b0_anchor_adapter.py)し、先頭one-body＋8 fragmentと残りtailからprefix preparationを構成する。[prefix_preparation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_ab_native.py)は元の順序、係数・basis操作、scalar・tail identityを使い、SCF/DF/state解法や新Gaussian分解を呼ぶ経路ではない。この復元・再構成処理を今回は実行していない。

元runとresumeのp8 preparation hashはともに `52bdd60b64c3823f13cb5bf0ee7621a2051b1b9e2b0db50dfeb0c8f058168108` で、[固定design](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json)・保存preparation_receiptとliteral一致した。p7は `f794c14366f6081237cafd80a91b6659c98450f3eb720c90073ab5ed2263abc4`。これが `matches_fixed_design_hash=true` の意味である。後段のG04 projectionで付けた照合metadataと、元loaderのexportキー存在flagを同じ検査と扱わない。同じ構成経路のhash一致は、独立したp8 export照合の代わりでもない。

## 3. 元STOP・resume・旧1軌道

元p8のSTOPは `ValueError:NATIVE_START_IDENTITY`。JSON listと生tupleのidentity比較に関する受入不具合で、exportキー欠落によるものではない。修正SOURCEの[native_identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation.py)は型付き4整数をtupleへ正規化する。所有・task・値の検査を省略する修正ではない。

[resume collect/accept](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/prepare_h8_pr_confirmation_resume.py)は、元runの完全なworker exit0/reaped、task-result hash、control、固定task/seed/event/law等を要求して元replica0の1軌道・2軸を受け入れた。これは同じ独立確認48本の最初の1本であり、探索n8や旧研究n32からの混入ではない。再利用は47本に足して受入48本とするもので、49本目の新標本ではない。

今回、元1軌道と最終結果中のreplica0の保存pairがliteral同一、同一H/state/snapshot/compiler binding、同じ確認cohort、保存control PASSを確認した。価格や平均・SEを再計算していない。原本の価格・固定shot・受入件数にこのflagを使う分岐はなく、任意export比較の欠測が標本選抜や費用補正を発生させた証拠はない。旧STOPと予約非返却はそのまま保持する。[公開192軸の再利用対応](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/AXIS_PROVENANCE.csv)

## 4. 科学的影響と残る限界

|対象|今回確認できた影響|
|---|---|
|Hamiltonian・DF fragment・state・reference|flagはこれらを比較・変更しない。元とresumeは同じ保存binding/configを使う。新たな共通入力不一致は見つかっていない|
|p8 preparation|Full exportからの再構成で、同prefix独立exportが欠ける。直接照合のcoverage不足は残るが、falseは科学的不一致の証拠ではない|
|回路・compiled価格|flagはbuilder/compiler/価格処理に使われない。原回路の再build・fingerprint再導出は今回行っていない|
|標本・G・SE|flagによる追加・削除・価格置換・再抽選はない。元1＋新47の固定確認を保持。母平均の保証は増えない|

分類の「影響なし」はこのmetadata flagの原因についての結論であり、復元builderに未知の欠陥がないことまで証明するものではない。原profile bytes、旧compiled出力完全同一性、全履歴seed独立性、全support費用、厳密biasや母平均優位の限界は残る。

**flag原因の解明に追加数値計算は不要。p8を修復するための旧条件再sampling/再compileも提案しない。** 同prefix独立exportとの照合済みという強い主張は使わない。将来、独立なpreparation作用検証を強める必要が生じた場合だけ、保存H8/p8と既存sparse/direct-tail oracleによる限定作用比較を別scope・別許可で設計する。今回の長時間・高次adapterには、新しい時間・符号・公式の必要な科学検査を別途含める。

## 5. 技術確認とSTOP

[test_p8_flag_semantics.py](test_p8_flag_semantics.py)は固定SOURCEからmembership式と任意比較guardだけを抽出し、人工metadataで欠落・一致・不一致の挙動を確認する。実loaderはimport/実行しない。identity型修正も人工4整数で確認。7検査PASSはflag意味論の技術証拠であり、H8数値性能の証拠ではない。 再実行はrepository rootで `python3 -B docs/research/h8_p8_long_time_preflight_20261103/test_p8_flag_semantics.py`。固定carrierとSOURCEのGit objectが必要で、未取得の場合は当該commitだけを取得してから実行する。[VALIDATION.json](VALIDATION.json)

元falseを変更せず、新しい解釈を本書へ追補した。旧SOURCE・input・結果・grant/ledger/marker/STOPに変更なし。**本調査の公開・照合後mandatory STOP。**
