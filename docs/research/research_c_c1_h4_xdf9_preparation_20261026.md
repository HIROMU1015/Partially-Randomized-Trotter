# C1：推奨案H4 XDF9の最小技術準備・人工検証

2026-10-26 JST、起点 `3a8356a7843d3cb46bb8069955640381b490ac4c`。独立branch `codex/research-c-c1-integration-next-preparation-20261026`。

**XDF9専用の表現・own-ground／alias・PF経路と未承認95-call契約を準備し、人工／回帰78 tests＋29 subtestsがPASS。原packetのbyte／環境preflightはPASS、実H4 decodeと科学callは0。** 新しいfit、DF norm、回路、shots、総費用は実行していない。推奨案は未採択であり、公開後mandatory STOPする。

## 最小差分と既存再利用

- [新library](../../src/trottertracks/c1_h4_xdf9_pf.py)：旧XDF8 moduleの入力whitelist、contexts、spin lift、raw CAR、ground／alias、分類を再利用。既存prefix matvec、一般K係数展開、`MolecularBackend._ground`、`molecule.PF_rounds`、Calls／resource_planを使用する。新しいCAR・DF分解・PF engine・eigensolverを実装していない。
- [CLI](../../scripts/run_research_c_c1_h4_xdf9_pf.py)：defaultはbyte preflightだけ。`--execute`とfresh output指定が実行入口で、科学許可の証明書を要求しない。今回は`--execute`を使っていない。
- [人工tests](../../tests/management/test_research_c_c1_h4_xdf9_pf.py)：旧XDF8のinvented packet fixture／mock PF lawを再利用する。実packet・保存J/U・実回路をテスト入力にしない。

旧XDF8の科学SOURCEはrank8を固定しているため変更せず、rank依存assemblyとorchestrationだけを独立moduleへ追加した。productionの旧module定数や関数をmonkeypatchしてrankを変える方式は採っていない。第9frameのexact spin liftを追加確認し、base＋9 frameの元順・J=2ληηᵀ・独立I/Z/ZZ展開を維持する。元g補正κは既存identityで一度確認し、prefixごとの再補正をしない。核scalarはbaseだけに一度入る。

## 人工／回帰結果

| 実施範囲 | 結果 | 確認した内容 |
|---|---:|---|
| 新XDF9 | 14 PASS | 人工4軌道・9 frame、実CAR／一般K／PF経路の95 units、26セル、own ground／spectrum |
| 既存XDF8 | 22 PASS | 原8frame経路・κ／spin／basis／order／STOP・fresh出力の非回帰 |
| 既存prefix予算 | 29 PASS | 人工matvec／guard／非単調prefix／STOP、旧pathの非回帰 |
| 既存一般K係数 | 13 PASS＋29 subtests | 人工係数・型・legacy bridgeの回帰、別processで実施 |
| **unique tests** | **78 PASS** | subtests29は78に加算していない |

[人工／回帰65件log](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/artificial_regression.txt)／[一般K別process log](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/general_k_regression.txt)。前者は14＋22＋29を一度に実行し14.96秒、後者0.22秒。ローカル人工試験の証拠であり外部再現・CI結果ではない。

新e2eでは、独立に記述したchronological S2 scheduleとexpmで短／長round m0／m12を照合し、9番目のfactorを含み10番目を含まないこと、scalarと非零元g補正、kernel時間と反復nを確認した。係数・frame順の不一致、9番目のoff-spin成分、wrong SOURCE／DF row／input、NaN／負PF誤差、alias不明、既存output／run02拒否を確認。PF十分条件NOT_MET・欠測はnormal classificationで、unknownを0へ置換しない。correctness FAILはpartial evidenceとSTOPを保存し依存callを止める。

開発初期に人工fixtureのexclusive保存と広すぎたsubprocess mockを修正した。科学SOURCE・保存結果の修正や実分子retryではない。最終testsでは本物のfitter／physical再生成／DF normを例外mockで遮断した。人工PASSをXDF9の科学的適格性やCA達成に昇格させない。

## 固定条件・SOURCE・passive preflight

[未承認契約](research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)、scope `C1_H4_SAVED_XDF9_REPRESENTATION_GROUND_PF_V1`。

| 固定対象 | 値 |
|---|---|
| 契約SHA256 | `189f4fa9f8a9d34d7a268857e78ec7459fba4e0ec05e99679fa2b30934535310` |
| [SOURCE manifest](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/SOURCE.json) SHA256 | `a3fd7d17f7059240ba70d003b4dee7ad8e3f36358cf8bd64cbbed03fb1bc373d` |
| SOURCE／保存JSON | 24 files／3 records、byte hash一致 |
| 原P-H4 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| prefix9 DF row SHA256 | `dd4bd88583a9dfc27756308dcc7a6effd7bb118a77141475dcf30c5d361a61f1` |
| 保存DF科学SOURCE／結果公開 | `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`／`e3c8b3e6550b2866b6e370a13780a0ee41ebefc8` |
| 計算範囲 | Nα=Nβ2／sector36、元順rank9、S2 s1／4、δ.1・m0..12 |
| 数値条件 | component／ground residual1e-10、normalization1e-12、rtol0、H4 window[-10,10]、PF≤sinπ/12 |
| DF／精度条件 | CA.00159、reserve.00019875 Ha、保存binary64_operation_scale_guard8_v1を維持 |
| 科学call上限 | **95**、新fit／DF norm／native／CA／shots／費用／retry／resume0 |
| fresh output | `artifacts/research_c_c1_h4_xdf9_pf/<execution-date>/run01` |

保存原packetは既存Git objectからbytesだけをtask-local `.external/P-H4.npz`へコピーした。byte preflight時はNumPy未importで、np.loadも数値評価もない。[preflight証拠](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/passive_preflight.json)。旧run／grant／STOPを再使用せず、future run01はまだ作成していない。

既存task環境Python3.11.1をread-only使用。NumPy1.26.4／SciPy1.14.1／OpenFermion1.6.1／openfermionpyscf.5／PySCF2.7.0／Qiskit1.3.0／Nature.7.2／threadpoolctl3.6.0と既存requirements.lockを照合した。インストール・共有venv・共有cache・サーバー設定・他ユーザーjobの変更なし。科学workloadに恣意的なCPU／RAM／worker／wall capを追加していない。

## 残る未資格事項・実行後の分岐

実XDF9の一般K／CAR同値性、own-ground／alias、PF26は未評価。厳密丸めenclosure、物理ground重なり、合成誤差、CA、shots、native費用、総費用もこの契約では取得しない。95-call完遂は約12%の頑健性を直接証明するものではなく、その後のnative／合成／shots比較を提案できるPF適格性を調べる。

全gate／PF PASSなら`POTENTIALLY_FEASIBLE`としてSTOP。PF NOT_METならその登録十分条件の未達、alias／欠測ならUNRESOLVEDとしてSTOP。correctness FAILならpartial evidenceを保全してSTOP。どの結果でも新条件・再試行・native以降へ自動進行しない。

ユーザー／研究Cが案Aを正式採択し、この公開SOURCE／契約／原inputの95-call範囲を明示許可することが次の担当作業。Codexはその別許可後だけ科学実行する。Bを選ぶ場合は新しい開始予算政策の採択と独立13entry準備が先になる。[A／B比較・固定call内訳・後続資格](research_c_c1_next_minimal_experiment_selection_20261026.md)を参照。

今回の技術blockerはない。commit・push・remote SHA／SOURCE／成果物照合後**mandatory STOP**。重要な科学レビューも未実施。
