# P0の固定数値経路

承認対象は75fb0ba114ac518bfe5d65d0164cee07d7469bb4の[SCIENCE_PLAN](../df_head_pauli_tail_preparation_20261114/SCIENCE_PLAN.md)のP0だけ。P1は実施しない。本書とcaller/module/testをcommitしてから、そのcommitを科学SOURCEとしてfresh run01を一回起動する。

- [module](../../../src/trottertracks/df_head_pauli_tail_p0.py)と[caller](../../../scripts/run_df_head_pauli_tail_p0.py)を追加。保存loader `155529e20:scripts/h46_m5_tol_saved_input.py` の `array_record/load` を既存のSOURCE-bound extractorで再利用する。snapshot/receipt/contextのhash、各配列hash、H/state hash、sector basisとliftを保全。
- 現Aの `h6_df19_rpe_d2.make_context` と `h46_new2_core.occupation_one_body` をそのままfull256/sector36に適用する。新しいHermitian化、completion変更、fragment再順序付けはしない。元のone-bodyと10個のweighted squareを共通に一回full拡張し、各36 blockと照合。
- M1 `5e4f0048:o1_m1_matched_split.prepare_matched_a.one_body` を抽出して、保存prefix後の同じfactorの `Σ weight*(a*a)` だけをJWへ渡す。元分子Hamiltonian−head、旧LD0 tail、別のG補正は使わない。辞書collectorは現 `PauliHamiltonian.from_openfermion/from_terms`。独自の小係数cutoffなし、既存imag許容1e−12とOpenFermion内部簡約は保持し、raw JW係数と再構成残差を保存する。
- JW mode qをqubit qに写す既存Qiskit辞書はlittle-endian。保存CAR basisはmode0がMSBなので、行列の両軸にbit reversalを適用する。辞書には元Qiskit labelと保存tensor順labelを併記。人工fixtureで変換前の不一致と変換後の一致を確認する。
- 各LDでfull H分解、Pauli tail再構成、保存scalar使用時の再構成、full trace identity、36への射影、head/tail/nonidentity Pauli和のHermiticityと双方向sector offblockを確認。Frobenius gateは既定1e−9、state norm/liftは1e−12。scalar差は上書きせず両値を保存。
- 全非zero labelを保存basisのbit像で完全分類する。個別labelのsector外遷移は期待可能な観測でありFAILにしない。合計演算子のsector gateは保持する。個別eventを36へ射影してunitaryとみなさない。

`COMMON_OPERATORS.npz`、LD別operator・辞書・raw JW係数、partial checkpoint、入力/SOURCE、最終resultを新run内に保存。保存state本体は公開成果へ複製しない。gate失敗時も直前の演算子と数値を残してSTOPし、retry/fallback/閾値変更は行わない。科学的厳密同一、mean縮約の本番資格、十分N/全RMSE/価格優位は認定しない。

人工5件と既存契約4件がPASS。metadata-only preflightは元contextの全binding項目を照合し、後年のcomposition metadataを追加情報として保持する。科学実行の許可を証明する新grant等は作成しない。

科学callは保存入力1組、full拡張1組、LD1/2辞書2つとoperator検査のみ。mean/reference/eigh/finite-time normalizer/sampling/build/compile/N/RMSE/C/Gは全て0。
