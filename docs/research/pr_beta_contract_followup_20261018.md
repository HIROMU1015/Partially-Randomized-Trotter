# β契約監査の技術追補：B03・B08・B02/B06

2026-10-18 JST。利用者が承認した技術修正・SOURCE/metadata調査・文書整備だけを実施した。Codex自身による検査で、別担当の独立科学レビューではない。

|作業|完了した技術作業|残る証拠・科学結果への影響|
|---|---|---|
|B03|座標幅の二乗underflowを明示的`OverflowError`へ接続。人工回帰9テストPASS|新SOURCEのみ修正。既存WP11での発生や結果値への影響は認定していない。WP11再実行なし|
|B08|物理grid／費用provider／WP11会計の生成経路・既定値・hash・18行対応を記録|`MODEL_MISMATCH`を維持。同一preparationの物理検証を確認できず、移送誤差・科学的影響の数値評価は未実施|
|B02/B06|引用時の保証分類とbinary64予算境界を文書化|旧JSONへ保証欄を追加せず、保証状態の再判定・50 scenarioの再計算・費用順位の変更なし|

起点は前段監査 `3fd8fca966e6b5a6a972aa71de68b31a0129f5ab`、科学snapshotは `f29eafbf4412193e8e8660d9f051889c2cba684b`。前段の79 blob・人工38検査・50 scenarioの監査結果を再利用し、同じ監査を再実行していない。

## B03：最小修正と新SOURCE

新SOURCE tree：**`d87d217a2c4044fbe62e5aa4d3bc44bfb864ff00`**（コード修正commitは`e0ae7a675d2db7445ab805242207de57d9089310`、その後に既存assertionのexpected基準許容差を厳密に維持するテスト整備を追加）。変更は[`_hoeffding_shots`](../../src/trotterlib/rpe_resource_accounting.py)と[人工回帰テスト](../../tests/test_rpe_shot_underflow_20261018.py)だけ。このcommitの関数file／test fileのSHA256は[検査記録](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/test_results.json)に保存した。

変更は、元と同じ乗算を一度変数へ格納し、0の場合だけ例外を送ることに限る。

```python
epsilon_squared = epsilon_coordinate * epsilon_coordinate
if epsilon_squared == 0.0:
    raise OverflowError("Hoeffding coordinate square underflowed.")
```

正常入力の式は`(2.0 / epsilon_squared) * log(2.0 / alpha)`のまま。対数化、代数的な並べ替え、shotの0/1への置換、比較許容差の変更をしていない。既存の上位`except OverflowError`が`shot_count_overflow`を付け、候補を不適格かつ`not_certified`とし、費用providerを呼ばない。

直接呼出しでは二乗underflowとshot値の上限超過を例外messageで区別できる。非finite・0・負の座標幅は従来の`ValueError`。候補レベルの不適格理由は、互換性のため既存の`shot_count_overflow`へまとめる。

人工9テストは以下を含む。

- `epsilon_coordinate=1e-200`と、極小正のβ_statを与えた候補。旧SOURCEの`ZeroDivisionError`、新SOURCEの明示的不適格とprovider未呼出しを確認。
- 二乗が0／正のsubnormalへ分かれる近傍5入力。underflowと、二乗は正でもshotがoverflowする経路を区別。
- 正常な整数shot18 fixture、ceil境界の前後9 fixtureで修正前後が完全一致。
- 正常候補・費用出力16 fixtureが`asdict`で完全一致。小さいshotから計算可能な非常に大きいshotまで含む。
- 非finite・0・負の座標幅、異常に大きいshot、feasibleと保証状態の区別。
- 元の`test_rpe_certification_boundary.py`のassertionを変更せず、π/3の直下／等号／直上、厳密な予算比較、加算順、summaryの保証伝播・unionの計10条件を実行。

環境にはpytest／SciPy／Qiskitがなく、共有環境・venvを変更していない。標準libraryのrunnerで実会計定義を読み、preparation・cost型だけを人工オブジェクトへ置換した。元境界テストのscalar `approx`は同じ既定許容差のshimを使用した。これは依存付きpytest suiteや実分子の統合試験を実行したとの報告ではない。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_rpe_shot_underflow_20261018.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_pr_beta_followup_20261018.py
```

両commandとも人工テスト／保存metadataのみ。B03修正はWP11の科学結果の再検証ではない。

## B08：完了した調査と未解決事項

[独立したpreparation来歴調査](pr_beta_preparation_lineage_20261018.md)に、三つの生成経路、state/operator/Hamiltonian/partition/preparation hashの区別、既存データを再利用できる条件を整理した。[機械可読の調査記録](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/b08_investigation.json)には、前段から転記した18行、既知の他grid、公開provenanceと欠損を保存した。

物理gridの閾値は関数既定値`1e-12`、会計・費用側はpreparation関数既定値`0.0`。記録された生成commitでも、この既定値の差は確認できた。ただし、当該依存fileの実行時SHAが保存source集合に含まれておらず、dirty生成時の完全closureは未証明。

旧WP04費用datasetには実preparation hashがあるが、WP11の公開aggregateはtask/result IDのprojectionで、実preparation・tail・state payloadを持たない。確認した公開checkpoint pathにもraw worker記録はない。したがって「SOURCE既定値0」と「当時の実行payloadを完全に復元できる」は別の判定である。

`MODEL_MISMATCH`を無理に解消していない。小さいthreshold errorや高いoverlapを移送上界へ昇格させず、保存値からshot・費用・順位も再計算しない。この不足は同一preparationの物理signal検証やRPE成功保証の主張を制約する。WP11の宣言モデル条件付き点費用を、今回の調査だけで誤りと判定するものではない。

## B02/B06：引用上の整理

[WP11の引用用注意事項](pr_beta_wp11_citation_notes_20261018.md)に、研究者が併記すべきSOURCE・経験的PF・状態仮定・費用範囲・保証分類をまとめた。

前段監査が保存したβ6/50、α20/50の厳密binary64予算和超過を、そのcommitに結び付けて引用する。今回これらの和を再評価していない。許容差付きfeasibleは厳密な予算不等式とは別で、費用順位の変更や再会計の必要性を意味しない。古い最終round行に`certified`等を後付けしていない。

## 検査・保全と停止

今回読んだ固定43 blob、B03新SOURCEとテストhash、生成元commit、B08対応表は[provenance](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/provenance.json)へ保存した。実分子NPZ・private runtime・grant原本は開かず、他ユーザーのjob／共有環境を変更していない。

全体manifestは前段からの拡張root keyと既存validatorのschema不一致が残る。前段の失敗記録を保全し、基点と今回の失敗を照合する。今回の新規entryだけをcore schemaへ射影した限定検査と、元entry不変の確認を分けて記録する。全体manifestのPASSとは扱わず、validatorを変更しない。

[新STOP](../../artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18/STOP.json)は本パッケージだけを停止する。前段STOPと全旧結果は保持。科学計算・保存shot再導出・費用再会計・sampling・signal生成・native build/compile・測定シミュレーションは全て0。

- **ユーザー**：B08を厳密な物理保証へ接続する必要があるか、研究主張の範囲から判断する。必要な場合だけ、既存archive回収や最小の同一preparation検証の対象・入力・call範囲を別途指定する。
- **Codex・提案／未実施**：指示があれば、既存の実preparation/state/operator記録の回収から始める。B-1/B-2/B-3/B-4や新しい科学計算へ自動移行しない。grantは発行していない。
- **公開**：外部送信承認が未確認のため、添付指示7節に従いlocal commitを保全する。push未実施・remote SHA未照合。新SOURCEと成果物commitは最終報告で示す。

技術パッケージは**MANDATORY_STOP**。B08の科学的未解決と、技術調査の完了を区別する。
