# 研究B Full DF-Yoshida4 q4：固定実行scope・技術検証

2026-10-26 JST。この初版は科学実行前の固定条件。結果公開時に同じ報告へ結果を追記する。
作業起点3244f2af9c4017c57c180b9057b22e3eb19c62b1。新実装・本scopeを含むcommitを新科学SOURCEとしてpushし、clean detached checkoutで一回実行する。

## 固定条件と再利用

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation。同じphysical Hamiltonian／state、全4 block順one-body/DF0/DF1/DF2、元c0=0.17を一度だけ適用。分子geometry／AO basis／分子DF rank探索は非該当。
FD5=Full Yoshida4 q4 T+0.18、FD6=同T−0.18。Macro時間T/4、既存Yoshida4の3-piece係数・signed時間を変更しない。Full PFにはD/R splitting、内部n_D、RTE r/K・normalizationを導入しない。共通入力は旧LD2 snapshotへ結合する。
Ordinary control、mode0状態準備・readout・measurement込みX/Yの4 wrapper。Complex128、operator spectral2-norm atol1e-12／rtol0、phase fitなし。Native-v5、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approx1、qubits_initially_zero=False、free topology。

既存`full_df.py`のq引数付きschedule／inventory／system／controlled／CAR数学参照／実行ループ／preflight／CLIを少数の引数で再利用。既定q2 whitelistとscopeは旧bindingに一致することを検査した。新`full_df_q4.py`は2 case whitelist・scope・観測bias分類・保存比較だけを担当する薄いadapterで、PF・回路builder・compilerを再実装しない。
`_full_df_native.py`／`_full_df_schedule.py`の旧AX-2A/AX-2B定義、`df_trotter.circuit`、B0 wrapper／native-v5／Evidence、Exact-R signal指標を変更せず使う。旧SOURCE・結果・契約・STOPを保全する。

## 科学call scopeと順序

入口は`scripts/run_b2d_full_df_y4_q4.py`。`--inspect`／`--preflight`はmetadataのみ、科学処理は`--execute`のみ。
Scheduleから導出した登録数：2 case、4 wrapper、各case96 signed block append、32 unique block probe、22 exponential key、44 operator gate、8 expectation、4 compile要求、4費用要求、**138 call**（events276、max308）。完全なidentity一覧は`--inspect`およびrun bindingに保存する。

| 境界 | 登録数 |
|---|---:|
| fixture/split/prepare/full_inputs/input_gate/full_prepare/definition_gate/saved_comparisons | 各1 |
| block_build/block_gate | 各32 |
| exponential | 22 |
| system_build/controlled_build/reference/system_gate/controlled_gate/signals | 各2 |
| wrapper_build/wrapper_reference/raw_wrapper_gate/raw_expectation/compile/compiled_gate/compiled_expectation/measure | 各4 |

固定fixtureの入力・sector/state・非可換性、全H/scalar/DF identity、全block signed/control、system/controlled、raw/compiled wrapperとexpectation、保存physical入力整合を**全件先行**させる。その後だけ2 signalと4費用を保存する。Global phase fitting／閾値緩和なし。Gaussian actual inverse、control0 identityとcontrol1相対phaseを維持する。

## 人工検証と実行条件

q2既存6管理＋14独立toy、q4新7管理＋2独立toy、計**29件・33 subtest PASS**（約9秒）。Mockでscope・whitelist・correctness先行・fresh output・失敗/partial/STOP・retry/resume禁止を確認。独立toyでq4数学積・scalar-once・control・X/Yを照合し、既存8 toyで全4 block正負時間・actual inverseを確認した。固定研究fixtureのoperator/signal/build/compileはこの技術検証では未実行。人工PASSは科学的性能の証拠ではない。

保存q2／B1のcompiler比較でtupleとJSON listの型差を正規化し、B1統計から既存measurement_count以外の6指標を直接抽出する接続を技術検証した。係数・物理系・compiler設定の変更なし。
SOURCE固定後にread-only preflightでclean SOURCE・fresh output・依存version・fixture定義・保存22行/Exact-R/q2/B1の必要な入力identity・compiler・公開blobを確認する。
Outputは既存Followup IOに合わせ`artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/`。Fresh outputの排他的作成、Evidenceの失敗・partial・STOP保存を再利用する。科学dispatch後は失敗してもretry/resumeせずmandatory STOP。科学実行前のpreflight不具合だけ修正・新SOURCE化できる。

## 観測biasと保存比較の限定

ε=1e-9、観測b≤5e-10だけを`WITHIN_OBSERVED_BIAS_BUDGET`／`OUTSIDE_OBSERVED_BIAS_BUDGET`へ分類する。値不明・非finite等は`INCONCLUSIVE_NUMERICAL_IDENTIFIABILITY`。これは点観測の参考比較で、u_num・γはnull。Correctness atolはu_numではない。予算外でもcorrectness PASSなら単価を保存し、別qへ拡張しない。
q2 FD3/4はcommit3244f2af9、PR D1/R2/R4の6 signal行は保存22行commit33715848b、B1の72統計セルはcommitbef63edbdを直接引用する。B1は各256 IIDの有限S0推定・SEで、rare条件付き費用は未取得。旧B3/B4とExact-Rは文脈として保存参照する。新MC・再統計・shot/G・総資源順位を算出しない。

今回はq4・T±.18の2 caseだけ。新q/PF探索、分子系、γ/A1、全RPE、R2-CS再開、詳細科学レビュー、継続方針の自動採択へ進まない。公開・remote照合後mandatory STOP。次の研究判断はユーザー、Codexは別指示と必要な科学実行許可後だけ進む。
