# Bへ：候補幅ごとの保存compiled価格と不足軸

**別幅のうち全26軸が保存価格で揃う候補は0。12候補すべて部分証拠あり、候補軸312件のうち100件が同物理点の直接証拠、212件が未取得・未検証。別幅への採択モデル適用0。** 現行6候補156軸は固定B2表（直接96＋条件付き60）をそのまま参照した。新しい価格値・Q3/N/Gは計算していない。

1. 保存価格だけで別幅の任意shot配分を加重する準備が全roundで整うもの：**なし**。
2. 価格不足が最少の幅：Full/p7それぞれS2 2h、Y4 4h、Y4 32h/5（10軸あり、m5〜m12の16軸不足）。ただし完成signal packetも不足。既存13round packetがある別幅は **p7/S2 1h** で、価格6軸あり、20軸不足。
3. 完成packetがない他11候補は、BがQ3候補に残した幅だけ追加取得の優先対象にする。粗いS2 8h/3・Y4 32h/5やFull S2 1h等を価格だけで先行取得しない。構造価格・T2 bias・未検証modelで劣位認定はしない。

全幅の一覧、直接roundと不足round、信号の不足は[REPORT](REPORT.md)。最初の有限案は[4 compile案](NEXT_FINITE_PLAN.md)（p7 S2 1h m1/m2 X/Y、Bが当候補をQ3可／比較候補に残した時だけ）。未承認・未実行。保存6軸を保全・再利用し、成功後10/26軸まで。長roundのモデルを同時採択しない。

## 機械可読の結合

- [PRICE_COVERAGE.json](PRICE_COVERAGE.json)：`current_156_reference`（現行表への参照のみ）、12 `candidates`、312 `rows`、6共有`contexts`、signed schedule template、調査scope。
- [PRICE_COVERAGE.csv](PRICE_COVERAGE.csv)：同312行の簡潔なprojection。
- `candidates[].configuration_id`は旧T2正本の元canonical configuration。表示名・要求幅は探索用labelであり、joinの正本にしない。
- 行のjoin keyは `(configuration_id, physical_id, m, axis)`。`physical_candidate`はT/q/actual幅・method/prefix/formula・R/Kを保持し、Tとactual幅のbinary64 hexも保存する。metadataの生成元selectorは`schedule_source`。同じphysical pointなら他要求幅の保存receiptを共用できるが、新候補の信号/decoder identityを付け替えない。
- `signed_schedule_key`で同じ公式・実幅の保存weights/1 outer-step signed time hexを参照し、q回順に反復する。scalarはsigned pieceへ分配せずtotal T一回。`context_key`で入力H/state/snapshot・profile・split・scalarの固定根拠へ戻る。
- `price_status=DIRECT_SAME_PHYSICAL_EVIDENCE`の`saved_compiled_RZ`は元保存値の転記。元価格の固定locatorは`price_evidence`。`consumer_binding`に新configurationでのsidecar/C guardが必要な状態を明示する。**この表の生成はC guardの実行ではない。** 原recordを新候補名で再発行せず、既存source-bound接続を使用する。
- `MISSING_VALIDATION_OR_ACQUISITION`は価格をnullのまま保持する。`SAVED_LOCATOR_BINDING_PENDING`は未解決候補用の区分で、今回の既知一致では0。対応未確認の所在情報が後で得られたら、この区分へ戻して再利用・新compileの双方を保留する。
- 全別幅で`conditional_model_applicable=false`。現行のa/b、piece数比、N_ZZ、構造Sから欠測Cを作らない。X/Yの価格一致は片軸のcompile省略・軸統合の根拠ではない。
- `signal.status`は完成packet／保存meanでcandidate packet化待ち／旧mean資格待ち／新mean必要を分ける。Q3判定ではなく、信号の元producer・gate・locatorを保持した所在情報である。

100行は68 unique physical-axisへの複数候補bindingを含む。100件の新価格でも独立測定でもない。今回確認した追加原trialは公開PF common 2 cellと既知long-time result JSON6 cell（両軸計16件）で、QPY/cache circuit/NPZ/vectorは開いていない。[RECEIPT_CHECKS](RECEIPT_CHECKS.json)。現在の96直接価格の再監査は行っていない。

## 現行幅の使用と別幅へ残る条件

現行156軸：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/INPUTS.json#axes`、必要なら同commitの`AXIS_TERMS.json#rows`の価格出所だけを参照する。元の固定shot配分・総額・順位は、別配分へ移送する結果ではない。

物理的には同じ1-shot Cを複数のshot配分で共用でき、配分ごとのcompileは不要。条件付き60軸については `e4458cd7548977e6e11a5a7b0c8dca0d049535d4` の固定値と `PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1` の仮定・用途を保持する。旧採択文言は固定候補・元固定配分の比較なので、新しいjoint-searchの科学実行・使用scopeはB/C側の計画と承認で明示し、Dがここで変更しない。

旧閉包は現行S2 16h/7・Y4 16h/3・M5 16hという固定実幅でのq方向の延長。別実幅の角度・boundary・compiler相殺・周期本体・terminal runは新しい裏付けが必要。同じq、公式、scalar、compiler hashだけでa/bを移植しない。採用済み60価格を直接値や無条件の厳密上下界にせず、2%保証・CI・systematicゼロと扱わない。新幅のcompile点/保存QPYが必要なら、その後の有限scopeで別に取得・監査・採否を判断する。今回は実施していない。

## 固定SOURCE・検査・ロードマップ

[SOURCE_INDEX](SOURCE_INDEX.json)にユーザー指定T2・alignment・凍結モデル・QPY監査・閉包採用・60価格・B2・A完成報告への固定参照をまとめた。A c212cee85の710保存record index関数だけを再利用し、mean一致・power・bias・N等の算術を呼ばずに追加幅を結合した。全13round完成の現行6packetは1adb7e65を参照する。

metadata checker4件PASS（[TESTS](TESTS.txt)）。pure stdlib/Git JSON・限定した旧result JSONだけを使い、新規価格予測、private QPY監査、RZ再計数、build/compile、fit、Q3、N/Gすべて0。研究code・raw・凍結モデル・旧STOPに変更なし。作者による技術照合であり、独立科学レビューとは呼ばない。

統合担当への差分：**現行6候補156軸の価格は固定表参照で維持。幅×shot探索の別幅12候補は全価格未完成、100候補軸の保存直接証拠と212不足軸を特定。p7 S2 1hのみ全signal packetあり、BのQ3候補選別に応じて4軸pilotを提案。** m1/m2現行幅のdirect取得と今回別幅の調査を混同しない。ロードマップ本体は編集しない。

このmetadata表だけで全幅の費用順位・最適幅・十分N・G・winnerを認定できない。commit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。新科学計算・QPY監査・compileは別途明示承認が必要。
