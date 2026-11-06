# H6前処理の誤差増幅修正（2026-11-06）

基点 `d166c75c407f4b2f74f239018fe4bbc5cf9dd390`。停止済みconditionfix/run01はSOURCE・raw・STOPを保全し、別worktreeで修正した。

## 確認した問題

前処理の小特異値を無制限に反転した結果、変換Wのノルムは約4.9e8〜2.6e9となった。ENTRYの係数ノルムも最大約2.5e5へ増え、座標勾配の直接計算と独立adjoint後のpullbackの差が約1e-8〜2e-7へ増幅された。元の許容差を超え、runは約88秒でcorrectness STOPした。

また、centered costの原点に渡されたJを使う一方、実primalはpack/unpack後のJを使っていた。offdiagonalのbinary64復元で一ulp動く場合、同じ原点でも差が0にならない数値上の不整合があった。

## 修正

- 特異値のscaling floorを `sigma_max * sqrt(eps_float64)` とした。逆metricの条件数はSciPyの座標精度に基づく `1/sqrt(eps)=67,108,864` 以下となる。
- すべてのnongauge方向に正のscaleを残す。truncated SVD、新penalty、物理係数のhard bound、rank/LD変更は導入しない。固定するのは従来どおり既知LD−1 gaugeの代表だけ。
- centered costは**実際にdecodeされた原点の配列**に揃えた。渡されたJとdecode後Jの一ulp差を、暗黙に同じ点として扱わない。
- 元の独立値/勾配照合、J閾値1e-5、U精密化差1e-6、DF予算、outer64/512、全下流条件を維持する。値照合を外したり、許容差を増やしたりしない。
- 探索HessianのBLAS化・primal cache・verified head SVDは継承する。H4の既定solver経路は変更しない。

## 検証

[保存点回帰](../../artifacts/research_c_h6_stablemetric/2026-11-06/technical/saved_point_regression.json)は、失敗したrunの**正確な保存x/J**へ束縛した7点で行った。新しい最適化や別条件評価は呼ばない。全7点の座標勾配照合が元の許容差でPASSし、最大成分差は約2.93e-13。centered costは原点で厳密に0となった。

**全受理条件を通ったことは意味しない。** 旧PR/strength.1の保存点はbinary64/extendedの値照合を依然として満たさず、元の検査で拒否されるまま残した。失敗点を再分類して科学endpointへ採用しない。修正後の実trajectoryが収束・物理適格・U安定性を満たすかは別の科学実行結果による。

人工テストでは近接head、巨大な相殺係数、全nongauge方向の数値rank、pack/unpackで実際に一ulp変わる原点、full値/gradient/Hessian、元J/U gate、偽solver勾配・壊れたSVDの拒否を確認する。[最終検証記録](../../artifacts/research_c_h6_stablemetric/2026-11-06/technical/verification.json)を参照する。最終版は評価95件＋fit52件＝147件PASS、FAIL/ERROR/SKIP0。人工PASSは実H6の性能保証や独立科学レビューではない。

## SOURCE・実行条件

[新契約](research_c_h6_stablemetric_contract_20261106.json)／[新SOURCE](../../artifacts/research_c_h6_stablemetric/2026-11-06/execution_preflight/SOURCE.json)。SOURCE153ファイル、SHA256 `243c7f33af702760da7bf7a98023561d530add8c254c139d1316ef9cf3392668`。

科学条件はH6・1.00 Å/STO-3G、保存CDF14 C0、元順LD4、PR/ENTRY×strength.03/.1/.3、最大64受理更新/512trial、r32/64/128、selection32/final128、subset26/800入力/full8192出力、最大218350 unitsから変更しない。input/C0のbytes、fit/eval package pinsも継承する。

fresh出力は `artifacts/research_c_h6_independent_validation/2026-11-06-stablemetric/run01`。停止済みrunのretry/resumeは0。新実行のSOURCE/契約と来歴を分け、以前の未使用unitを流用しない。現在の明示された修正・再実行範囲を超える科学条件や重要レビューへ自動進行しない。

既存の研究結果・旧SOURCE/raw/STOP・共有環境は変更しない。実行中のSOURCEを後から書き換えない。
