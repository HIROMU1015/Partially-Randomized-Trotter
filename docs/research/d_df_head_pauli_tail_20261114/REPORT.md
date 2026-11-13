# DF head＋Pauli tail：回路・compiled価格の技術調査

2026-11-14 JST。**既存Pauli D2 builderとnative DF headの接続は可能。新しいPauli rotation builderは不要で、必要なのは辞書依存の薄い接続とSOURCE/価格domainの明示である。** Pauli stringにも局所X/Y basis変換・parityネットワークがあるため、basis費用が全て消えるとはしない。実H4での平均event単価・λ・N・Gの改善はUNKNOWN。

[builderとA/C HANDOFF](BUILDERS_AND_HANDOFF.md)／[価格定義・有限検証案](PRICE_AND_VALIDATION_PLAN.md)／[SOURCE確認](TECHNICAL_CHECKS.json)／[人工テスト](TESTS.txt)／[test source](../../../tests/test_df_head_pauli_tail_technical.py)。基点はD `3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673`。既存のcompiled価格研究branchを継続し、production module・A/B/Cのbranch/worktreeは変更していない。

## 参照した固定SOURCE

| 対象 | 固定commit・資料 |
|---|---|
| 必須P3 Stage3 | `90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5:artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03/REPORT.md`。36価格完成、元nativeは9c4006bf |
| 必須P3後続設計 | `60515a33ce6231e9ff4ffdb40124ed845373351c:docs/research/h4_p3_poststage3_cost_and_stage4_design_20261102.md` |
| P3後続の実結果 | `e44e87a306d05c8549f1a044b7fd056c2e1055a3:artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/REPORT.md`。後続32 path/arm・128 wrapper取得済み。設計だけの未実行案として再提案しない |
| 必須O1-M1系譜 | `162f3d3f0f42973048fd14a7a6624e147a1e961a:docs/research/track_a_algo_improvement_survey_20261109/TRACK_A_RESEARCH_SURVEY.md` §1.3 |
| O1-M1の実SOURCE | library `8a13539e51bb113e82cb86ddb608e2e6e27786ca` の `o1_m1_resource_bridge.py`、r8 caller `1b64e46c7305e34c2a16aba9ac2e01256153838d`、結果 `0daf01d51c022977a1c0a7d29dde34a1c4c82d7c` のr8 REPORT。C0結果50ee55bcも参照 |
| 既存Pauli/DF回路 | P3 native `9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`、provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。使用する関数はHANDOFF表 |
| D | 上記3d39f8c0、`d_s1_v1_price_alignment_20261114/`、candidate adapter、fast価格、Accounting、新DF ZZ |
| C最新公開 | `3a8b37f1a282c0c06ac1a363b070f4376e9ff41a`、`full_pf_latest_price_guard_20261114/HANDOFF.md`、`price_implementation_guard.py` |
| A最新Pauli技術調査 | `75fb0ba114ac518bfe5d65d0164cee07d7469bb4`、`df_head_pauli_tail_preparation_20261114/REPORT.md,HANDOFF.md,SCIENCE_PLAN.md`。調査中に公開されたため追補確認。辞書identity/sector/scalarとLD1/m0価格接続案は共有できるが、実Pauli辞書/λ/新canonical候補はまだ未取得 |
| A既存物理成果 | `71f04be7bf94e7773a32270f36ae29eee1b903d5` のFull PF成果。Pauliの物理descriptorの代用にしない |
| B最新観測 | `ce0517de6e7b58cf4bf4c69f99f998d86099c406`、状態集約DPの固定2診断input。Pauli候補の適格Nを提供する成果ではない |
| 横断監査 | `51d324ea9c83bc12f1e866da0fa409cef926f74a`、S1/V1と回路作用の未検証範囲 |

A/B/Cの既知関連branchのremote HEADを読取照合した。公開SOURCEの調査であり、進行中のAの未公開成果を確認したとはしない。原論文の物理分割・Hamiltonian同一性・sector/有限K signal/新Pauli辞書の成立性はAに委ね、本調査で再計算していない。

A75fb0の規約に合わせ、event/native作用はfull qubit空間で扱い、sectorへ射影した各Pauliをinvolutionとして回路化しない。Aのoperator-only段階と必要なmean検証が成立した後、最初の実接続はLD1/m0 normal1 path・X/Y最大2 compileとする条件付き案に合わせた。物理辞書とR/全seedがない現在は実行できない。P3 run04の追加確認もA/C向けの再利用証拠として本成果に示す。

## 重要な違い

P3は `append_pauli_event` でsystem-only Pauli wordを作り、`h4_d2_pauli_control.central_control_body` で中央回転だけを制御する。event/product/scalar由来のglobal phaseはancilla相対phaseへ移す。O1-M1は同じPauli eventの**全体をgeneric control**にしている。これらの価格を「Pauli単価」として混合しない。

native DF側は `build_partial_native(..., controlled_tails=...)` が既に任意のordinary-controlled tailを受け取る。D2を**Pauli tail部分だけ**へ適用して渡せる。DF Gaussian headをD2へ含めると既存validatorが拒否するので、全bodyをD2化する接続は使わない。

現D `candidate_price_adapter.NativeRuntime` とcarrier composeはDF tail preparation/builderに固定されている。candidateのλ/labelだけを書き換えて流用すると別辞書のcontext混在になる。Pauli tail dispatchを明示する薄いadapter差分が必要だが、今回production実装・科学的採択はしていない。

## 保存価格の再利用範囲

| 保存証拠 | 使える情報 | 今の価格に流用できない理由 |
|---|---|---|
| P3 run03/run04 | Linear H4/1Å/STO-3G/tol-only DF10、共通Gaussian T_eff保持/V_square残差、T=.8/S2/q8/r4/K4/P0、Pauli184/DF360、D2/旧DF control、Qiskit1.3/opt1/seed17。run03固定pilotとrun04独立32本/armの価格・SOURCE・作用検査 | 元分割は現DF-prefix LD1/LD2と異なり、DF側は新ZZでない。K・law・scalar/sourceも固定。run04の標本SEは全law CIではない。新辞書のevent単価/全RPE順位へ移送しない |
| O1-M1 C0/r8 | 同H4-A T_eff/V_square/LD0の辞書変更、ordinary-controlled full wrapperの保存価格と研究上の負結果 | 主metricはCX、seed_transpiler=7、Pauliはgeneric whole-event control。現在seed17/D2/symmetric-directionalと別domain。保存RZがあっても無条件移植しない |
| D P0/P1/P2 | 効率的DF controlled ZZのunitary/control phase・固定DF path/full-wrapper価格・性能 | Pauli tailのλ/law/非Clifford RZ/単価の証拠ではない。P2のX/Y同値や小標本をPauliへ移さない |

元の回路/条件/軸/fingerprint・compiler・producer SOURCEまで一致するreceiptは再compile不要。今回一致する新Pauli候補receiptを認定していない。旧QPYからcompiled RZ角を抽出し直す行為も保存数値再解析なので今回実施しない。後続計画は[価格・検証案](PRICE_AND_VALIDATION_PLAN.md)を参照。

## 人工テストとSTOP

9件の新しい小人工テストは、元native testの2-mode one-body＋squared DF blockと独立Fock-energy oracleを再利用した。正負のτ/時間、K0/K2（非可換productとorder phase）、ordinary/symmetric_directional、scalar一回、測定付きcosine/+Im sine、weight3 Pauli/endian、identity除外を検査した。全system入力のcontrolled行列をphase alignmentなしで比較し、scalar二重適用も拒否できることを確認した。

test内で分子NPZ読み込み/DF preparation/sampling/価格transpileを禁止した。Operator/Statevectorによる人工gate定義の展開は行うが、固定basis/opt1で価格を取得するcompilerは呼ばない。旧Qiskitの非推奨property警告はログに残る。これらは実H4の回路成立・科学的性能の証拠ではない。

新実H4/H6計算・新λ/signal/MSE/N/G・保存価格再集計・実分子回路build/compile/sampling・Pauli価格取得は0。旧source/input/raw/receipt/grant/ledger/marker/KEEP/STOPを保全。commit/push・remote SHA/tree/主要成果物照合後 **mandatory STOP**。
