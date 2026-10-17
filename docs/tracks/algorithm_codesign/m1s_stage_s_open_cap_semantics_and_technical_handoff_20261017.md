# M1-S Stage S：開端点・誤差coverage・SOURCE対応の技術引継ぎ

2026-10-17。これは新しい科学レビューではなく、GPTレビューの推奨をSOURCEへ接続した技術記録。[契約候補](m1s_stage_s_cap_edge_contract_candidate_20261017.md)は未採択。

## 開端点の実装根拠

固定A>0、K>0、0≤b<γ_c<π/2ならS(γ)=K/[A sin(γ−b)]²はγの厳密減少関数。S_cが非整数なら、capに十分近い左側でceil Sはceil S_c=floor S_c+1となる。S_cが整数kなら左側ではS>kであり、十分近くではS<k+1なのでceil S=k+1となる。従って両方でfloor S_c+1。任意の外生的γを選んだ評価やγ_cで実行可能なshot scheduleを示すものではない。

n_s^cap-とn_c^cap-がそれぞれ局所一定となる左近傍の交わりは非空である。両方を解像しn_s<n_cなら存在の十分証拠。tie／not lowerから全γの不存在は導けない。この片側性を結果schemaの区分と`all_gamma_impossibility_established=false`に固定する。整数区間が分離していても各方式のfloorが未確定なら、今回の保守的候補では`ARITHMETIC_UNRESOLVED`とし、独断で判定範囲を広げない。

## 算術とcoverage

- 保存V1のπ区間を正確な有理数で1/3倍し、πを再生成しない。保存されたA_lower／unclipped_bias_radもexact rationalとして使う。
- `arithmetic.py`は旧SOURCE `677dd6957e0e1913b70ca5f999458835ae8a2e23`の`bounds.py`から`Bounds.__init__/check/pack/sin_positive/_log_unit/log_above_one`とtext helperだけを新namespaceへコピーした。元SOURCEには触れていない。
- sinは交代級数の次項、logはatanh型正項級数の残余を使う。exact Fractionの加減乗除、外向きdyadic丸めで区間を広げる。sinの単調性には、0≤bias<cap_L≤cap_U<3/2<π/2を検査する。
- R=[(M_s,L/M_c,U)²,(M_s,U/M_c,L)²]、S=[K_L/M_U²,K_U/M_L²]の正値区間式を使い、packで外向きに丸める。floorは整数除算。near-integerをfloat epsilonで決めない。
- p128はscalar interval解像度の技術候補であって、capでの整数が必ず解像する保証ではない。8 guard bitsを付けた固定scaleを使い、結果に応じた再精度化は実装しない。
- 人工テストは式の実装回帰検査。保存IEEE／BLAS／fsum／trig等の元coverage、保存oracle、物理IID、target stateを認証しない。新SOURCEを数値certificateと表現しない。

## SOURCEの役割

|path（新namespace内）|役割|
|---|---|
|spec.py|候補統計・算術・resource、固定入力、全grid、source inventory|
|arithmetic.py|明示再利用した一般rational interval kernel|
|kernels.py|open-cap floor+1、margin/S/R区間、整数分類。I/Oなし、人工テスト用|
|adapter.py|許可後のみ保存V1/π3をdecode。全108行/48対照・provenance・直接trade-off・旧区分を確認|
|analysis.py|12cellをbatch処理し、36marginを共有、hardwareに応じたPool。結果は固定48対照順に格納|
|control.py|採択・別authority・S/C/A/input/環境binding、one-shot、prefix/failure保存、STOP|
|runner|candidateを既定として拒否。引数だけで採択やauthorityを発行しない|

新evaluatorは旧V1 evaluatorや旧π3 evaluatorをimport・再実行しない。入力は保存V1のenclosureと既存π3区分で、元finite signal／Hamiltonianを再構成しない。従って独立数値再現とは呼ばない。

## call上限と性能

主対象24対照は12cell×対照2方式。cell当たりshared/ordinary/independentの3marginを一度ずつ計算するので36margin、各marginのsin上下端2callで72call。1call当たり最大512series反復、logの内部2seriesとrange reduction、pack等の有界処理を合わせ、aggregate bound steps100000を候補とする。単純な反復上界72×512＋2×512＋range/pack等はこの上限内。512項に達して収束しなければSTOPし、回数を緩和しない。fraction field1024はπ2端点、24境界2端点、36方式2fieldを含む読出しを十分に覆う構造上の候補。

32 affinity CPU、約64GiBの物理RAM、GPU utilityの存在を環境metadataとして確認した。今回の検査時点の値で、専有資源や将来実行時の利用保証ではない。Poolは利用可能affinityと独立job数で選び、数値結果やランキングを見てworker数を変更しない。CPU/RAM/GPU/worker/timeの人工capはない。exact-rational workloadではGPU/BLASを使わない設計で、driverや共有設定の変更0。native合成backendの調査・起動も0。

artifactの`environment_metadata.json`は環境読取のみ。人工serial/parallel等価テストは2個の非H4 scalar jobを比較し、速度向上率は主張していない。

## 残るもの

技術準備は契約候補・結果schema・SOURCE・guard・人工検査まで。保存H4入力でのadapter動作、intervalの整数解像性、runtime、coverage条件の充足は未検証であり、今回の禁止範囲なので評価していない。

科学的未決は、cap screen用途、PER_METHOD_ROW/α=0.05、fresh-IID/Hoeffding/等配分、保存半径の用途限定coverage、p128とone-shot/STOP条件を**正式契約として採用するか**。Stage Cのnative費用指標・HD実装・state準備費はStage Sの採択を妨げる必須fieldではなく、別段階へ延期している。Task Bのstate/prior/全round/実装誤差も未取得のまま保持する。

次はユーザーによる候補の正式採択判断。Codexはその明示承認後に採択記録を固定し、さらに別の科学実行承認後にのみ評価する。今回、新しい科学判断や研究GOを代行しない。mandatory STOP。

初回静的検査で契約文書の`Mandatory STOP`と検査器の小文字表記の不一致1件が見つかり、意味を変えずに表記を揃えた。初回記録は保全した。科学SOURCE・保存入力・結果の修正や実行ではない。

人工scalar・schema・guard・parallel/serial経路・runtime結果metadata整合の36テストがPASS。実H4入力による数値correctness／runtime検証ではない。科学的な保存データdecodeは0。

## 最終準備監査

SOURCE S=`7a338c4eba778c4ce1f189cfa8c8f050afb5df08`、実行対象9ファイルと結果schemaを固定。候補のsource_identityと固定commitのbytesを照合した。最終静的検査120件PASS/FAIL0、人工scalar・metadata等36テストPASS。既存非index7789 Git blobs、親追跡1284件・index・HEAD・tracked status、既存214 worktreeのHEAD、レビュー原本の保全はPASS。ローカル検査でありCI／独立H4数値再現ではない。

[証拠index](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/evidence_index.json)、[静的検査](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/static_validation.json)、[人工テスト](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/artificial_test_results.json)、[保全監査](../../../artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/preservation_audit.json)。GitHub同一性は公開後の別publication receiptに記録する。
