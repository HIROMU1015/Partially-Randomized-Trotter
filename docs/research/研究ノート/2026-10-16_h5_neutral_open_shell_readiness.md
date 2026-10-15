# H5中性開殻の技術準備

中性H5/1Å/STO-3G/ROHF、Na3/Nb2、sector100、tol-only1e-8/weighted DF。専用input・αβ別primitive・DF state/S²・native/2段階pilotを実装、32人工/mock PASS。SOURCEf906a61415d40ec7f362d00aa76e26420e25e7dc、通常最大4並列とhash/paired/spectra再利用。実H5科学・grant0、rank/速度/実回路未測定、別承認とfresh binding待ちSTOP。

[主資料](../track_a_h5_neutral_open_shell_readiness_20261016.md)。利用者は元の資源詳細指定を撤回し、簡単な高速化・並列化を希望した。科学条件・pilot40compile範囲・別実行承認・共有環境保全は維持。旧charged helperを使わず、H8固定sourceから新packageだけ追加した。
