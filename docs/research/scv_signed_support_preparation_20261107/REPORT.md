# O2-SCV Signed Support：最小実装・人工correctness完了

2026-11-07 JST / SERVER1（GPUA100-gleap-01）。**技術的完了。新規24＋既存82＝106件の人工テストPASS。実H4科学計算なし。**

## 今回の成果

- [新SOURCE](../../../src/trottertracks/algorithm_codesign/scv_signed_support.py)と[新テスト](../../../tests/tracks/algorithm_codesign/test_scv_signed_support.py)：最後のrotation labelの保存sign/supportからu=s|S|/2を構成し、actual finite-grid lawに対応する全5層のexact conditional mean・二次moment・varianceを計算する処理を実装した。
- h1のscalar OLS・Fraction freeze・他4層λ=0・fresh paired評価へ最小接続した。旧SCVのsampler、OLS、中心化、5層加重合成、stream検査、費用会計、journal/STOPを再利用し、固定済みSOURCEは変更していない。
- signed専用adapterでSOURCE box[−1,1]とactual凍結係数から正しいresidual rangeを求める。width=2+2|λ|。旧indicator range/guardの流用、rank不足・nonfinite・scope/stream混入を拒否し、clip/refitで救済しない。
- 同じfresh人工recordによるbaseline/SCVの平均・sample variance・paired covariance・全5層合成、学習費込み/一回課金を確認した。新しいsampler・科学runner・managerは追加していない。

## 証拠と再利用

| 内容 | 成果物 |
|---|---|
| 実装、既存APIとの差分、metadata identity、moments/range | [SIGNED_SUPPORT_IMPLEMENTATION.md](SIGNED_SUPPORT_IMPLEMENTATION.md) |
| 人工fixture、テスト一覧、初回不具合と修正、再現コマンド、限界 | [SIGNED_SUPPORT_CORRECTNESS.md](SIGNED_SUPPORT_CORRECTNESS.md) |
| 最終106件log・機械可読結果 | [ARTIFICIAL_TESTS.log](ARTIFICIAL_TESTS.log)、[TEST_RESULTS.json](TEST_RESULTS.json) |
| 固定参照commit、SOURCE hash、環境、科学call境界 | [SOURCE.json](SOURCE.json) |
| 次回の限定差分・採択待ち事項 | [NEXT_SCIENCE_SCOPE_DRAFT.md](NEXT_SCIENCE_SCOPE_DRAFT.md) |
| 原本保全・資源観測・停止 | [PRESERVATION.json](PRESERVATION.json)、[RESOURCE_OBSERVATION.json](RESOURCE_OBSERVATION.json)、[STOP.json](STOP.json) |

既存5モジュールはSCV SOURCE `560c6092bfb5e71af8eedc4a4be5416922764f98` とbyte一致。最新方向整理 `4741cae05af662dfc42d0519ff977756a94cb7b7` をbaseとする専用branch `codex/o2-scv-signed-support-preparation-20261107` で、新SOURCE・対応テスト・本directoryだけを追加した。旧索引や科学manifestの書換えは、今回の固定SOURCE read-only範囲外なので行わない。

## 残るUNKNOWNと次の承認

人工でのfeature・exact moment・fit/freeze・signed range・paired評価のcorrectnessは確認できた。**実H4での非定数性、ν/二次moment/variance、学習可能性、X/Y相関、分散削減、学習費回収、総RZ改善はUNKNOWN。** 旧224/752 bit・係数の再fit/数値再解析はしていない。

新しいshot数・seed・guard・科学的GO/NO-GOは設定/採択していない。次は[限定scope案](NEXT_SCIENCE_SCOPE_DRAFT.md)に沿い、実H4 metadata評価と必要ならfresh train/paired評価・価格取得の有限範囲、独立性、費用、STOPを具体化してユーザーが採択・明示許可する必要がある。旧224条件の自動移植はしない。有限confidence/calibrationは別の未完了接続である。

## 保全とmandatory STOP

元main HEAD/dirty/untracked、旧SOURCE・raw・結果・grant/ledger/marker/STOP、224/752成果物、exact engine・C1凍結を保全する。別セッションのengine高速化worktree/process/cache/成果物には操作しない。共有設定・venv・他ユーザーprocessは変更せず、科学worker/managerを起動していない。

本branchを公開し、remote SHA・公開本文・blob/treeを照合後、**mandatory STOP**。人工PASSから実H4評価、新測定、calibration、別control実装、重要科学レビューへ自動進行しない。
