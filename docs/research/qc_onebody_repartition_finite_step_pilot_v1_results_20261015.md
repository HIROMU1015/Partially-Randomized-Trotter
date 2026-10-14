# 一体項再配分PF：固定有限step pilot v1の結果（2026-10-15 JST）

固定一回収集と独立数値監査が完了した。登録判定は `NO_UNIFORM_REGISTERED_COMPONENT_GAIN`、4/4 target×price条件で10% component費用改善なし。独立checkerは `INDEPENDENT_JW_PADE_DIRECT_RESPONSE_COST_AUDIT_PASS`。研究の正式継続・終了・採択判断は行っていない。

## 入力・source・認可

linear H4 / 隣接距離1.00Å / STO-3G / 保存DF rank12、保存generation順0..11、8 spin orbitals / 4 spatial orbitals、Nα=Nβ=2、sector36次元。PRのL_D・delta探索ではなく、12 density fragmentすべてを決定論的PFに用いる。固定h窓は下記6点。既存development `de7a` Hamiltonianとcached exact ground state一件に限る。新分子・DF生成・rank選択・ground-state preparationは0。旧PR S0のbyte mismatch STOPやheld-out状態を変更しない。

入力pack SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。source commit `19d5bd74db020564ea9a4b1882e102911e983cdb`、source manifest SHA256 `8b4a3ed3034c0931dd33cff60cb27cdce974f8ca5dc316bab71d2d0ed29d8fe1`、contract SHA256 `c3e9d2028ca74a65d7f781d48a02033755d92919a24be3415097fa573b97161d`。準備publication `0f92af19bde69fae69e97868b063a34e5afd5ca0`。ライブラリ・runner・checkerを含むfreeze21 filesは不変。

[固定契約](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)、[source manifest](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json)、[入力identity](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/input_identity.json)、[今回の別認可](../../artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/authorization.json)、[実行前GitHub再取得](../../artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/prelaunch_remote_receipt.json)。ユーザーの「次の作業に進んで大丈夫です」を、直前の完了報告で次段階と明記した一件36PFセルのone-shot実行への承認として記録した。IDE metadataを利用していない。旧認可は再利用せず、未発行targetと準備STOPは履歴としてbyte保持した。

H = constant I + dΓ(one_body) + Σλ_l dΓ(g_l)²。追加1/2・normal-order重複補正なし。mode0はMSB、interleaved spin。先頭一体Aと最初density Bのみ、A+=X / B-=X、固定first frameでspin-paired mean-zero shift。厳密E0とfinite-h truthは候補生成に渡さず、[selector seal](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/selector_seal.json)をH truth solver前に保存した。

## 登録構成と数値範囲

一方向のprojected 1-RDM gradient、端点・zero・実根・vertex最大6候補の固定leading objective。今回のselectorはt=-0.1944523707176354、許容bound=0.5662232373666428。実際の候補は4点、反復最適化0。比較はoriginal / response-selected / known first density linear-extractionの3構成、S2と5-stage signed S4、6hの36primitiveセル。CS2と隣接Richardsonを導出し69 method records / 276 cost rows。全対照に同じscalar extraction、basis sharing、fusion、同じexact-state情報へのアクセスを許した。

Λ_common=45.53536150763562。h_s=1/(8Λ_common2^s)、s=0..5：`[0.002745119306432635, 0.0013725596532163176, 0.0006862798266081588, 0.0003431399133040794, 0.0001715699566520397, 8.578497832601985e-05]`。目標1e-4 / 1e-5 Ha、deterministic budgetは半分、statistical1/4、synthesis/information各1/8。overlap>=.99、double-precision safety marginを適用した。新h・補間・救済grid・再選択なし。

保存ground検証：E0=-2.166387448634758 Ha、gap=0.2326302151201822 Ha、cached-ground overlap=0.9999999999999998、residual=1.255e-15。これらはselectorをsealした後の一回H eigensolverで取得したtruthで、候補生成に利用していない。

## 誤差・overlap

leading dはoriginal `0.01336651855253499`、response `0.008200683552525947`、known-linear `0.01031437351283108`。responseによってleadingと最大hでのraw S2 biasは減少した。

最大h=h0=0.002745119306432635での保存値：

| method | signed bias / Ha | numerical allowance / Ha | overlap |
| --- | ---: | ---: | ---: |
| S2:original | 1.007256416e-07 | 1.000e-10 | 0.9999999999999560 |
| S4:original | -6.461498003e-13 | 1.907e-10 | 1.0000000000000009 |
| S2:response_selected | 6.179765899e-08 | 1.000e-10 | 0.9999999999999845 |
| S4:response_selected | -4.440892099e-15 | 2.202e-10 | 1.0000000000000004 |
| S2:known_linear_extraction | 7.772542032e-08 | 1.000e-10 | 0.9999999999999722 |
| S4:known_linear_extraction | -2.029487689e-13 | 1.446e-10 | 1.0000000000000004 |
| CS2:original | -1.650082696e-13 | 1.000e-10 | 0.9999999999999560 |
| CS2:response_selected | -6.809674729e-14 | 1.000e-10 | 0.9999999999999845 |
| CS2:known_linear_extraction | -3.979385828e-13 | 1.000e-10 | 0.9999999999999722 |
| EXT2:original | 6.097344851e-13 | 1.667e-10 | 0.9999999999999560 |
| EXT2:response_selected | -2.309263891e-13 | 1.667e-10 | 0.9999999999999845 |
| EXT2:known_linear_extraction | -3.397282455e-13 | 1.667e-10 | 0.9999999999999722 |

CS2/S4/Richardsonのbiasは多くが1e-13程度で宣言margin1e-10程度より小さいため、その下での手法の誤差順位を確定しない。overlapは全て実質1で、約1e-14の差を資源上の有意なoverlap改善とは認定しない。1を僅かに超す表示は浮動小数点残差で、repeat accountingは1にclipする。厳密外向きinterval保証ではない。

## 固定費用比較

以下は保存した比較結果を転記した。response対象はraw/CS2のmin、strong controlは元raw/CS2、全構成S4、known raw/CS2、全Richardsonのmin。h選択は各methodに等しく許した事後fixed-grid oracleであり、真値なしで運用可能なprecision selectorではない。

| target / Ha | price | response / h | strong control / h | response / control | 登録判定 |
| --- | --- | --- | --- | ---: | --- |
| 1e-04 | P1 | CS2:response_selected / h0 | CS2:original / h0 | 1.000741289844 | NO_REGISTERED_COMPONENT_GAIN |
| 1e-04 | P2 | CS2:response_selected / h0 | CS2:original / h0 | 1.000389256520 | NO_REGISTERED_COMPONENT_GAIN |
| 1e-05 | P1 | CS2:response_selected / h0 | CS2:original / h0 | 1.000741289844 | NO_REGISTERED_COMPONENT_GAIN |
| 1e-05 | P2 | CS2:response_selected / h0 | CS2:original / h0 | 1.000389256520 | NO_REGISTERED_COMPONENT_GAIN |

P1=(Givens4,linear-phase1,pair-phase2)、P2=(8,1,4)のstatic controlled primitive価格仮説。responseは約0.074% / 0.039%高価。登録10%改善を満たさず、4条件すべて `NO_REGISTERED_COMPONENT_GAIN`。additional-information budgetも全条件で負であり、このcomponent accountingから追加γ取得へ正の費用枠を主張できない。

CS2:originalとraw S2:originalは同じ回路価格でtieとなる。固定lexical tie規則はCS2を保存するが、raw originalも最大hで両targetを満たし、γもd0も要求しない。tie名から補正情報取得が不可避とは解釈しない。responseの追加phase数が同じquery数で小さな差を作る。

**最大登録hで元raw S2のbiasが約1.01e-7 Haと、厳しい方のdeterministic budget5e-6 Haより既に小さい。** したがって原構成と再配分とも最大hを選び、6点の登録窓ではstep削減のeligibility frontierを測れていない。これは固定窓内の費用結果とその情報価値の限界であり、一般再配分法の不能性ではない。救済用のh拡張は実施していない。

## 実行・独立監査・費用の限界

producer：pack load1、RDM1、spatial eigh15、direct leading3、H truth eigh1、PF seeds36 / Schur36。独立checkerは異なるJW/Pade/direct contractions/native accountingでPF再構成36 / expm705、RDM1 / leading3、追加eigensolver0、compile0。双方attempt1 / retry0。collector約1.011s、sequential peak RSS 79413248bytes、attempt出力2205548bytes、console合計748bytes。RSS512MiB / output32MiB / log2MiB内。wall/CPU capなし。compile/T合成/反復optimizer/新ground準備0、過去27検査・準備37人工検査の再実行0。

[一次JSON](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/raw.json)、[一次arrays](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/raw.npz)、[result identity](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/result_identity.json)、[独立監査](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/independent_audit.json)、[実行・caps・保護照合](../../artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/execution_audit.json)、[保存結果summary](../../artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/saved_results_summary.json)。

この監査は同一環境での別実装による数値照合で、外部科学再現やcurrent CIではない。coherent queries=ceil(π/(h εstat))は固定Heisenberg proxy。ground-branch repeat、Richardson両experiment/L1費用は含むが、full QPE/RPE confidence theoremではない。Givens名は計価primitiveを表し物理的全basis制御を強制しない。T合成・compile・depth/backend費用は未取得。極小のrotation synthesis allowanceも予算表示に留まる。情報取得の量子費用・state preparation費用・全ground-energy totalはUNKNOWNを保ち、無償とはしない。量子strict ground状態を使う仮定と、cached-stateの古典RDM計算を区別する。

## 停止・引継ぎ

450既存protected files、source21files、152旧manifest records/non-result fields、既知global description FAIL、root dirty137pathsを保持する。N1 STOP / N2 HOLD / N3 CLOSED / THRIFT現template CLOSED、Track A/B、C1–C4、旧結果・認可・STOP不変。

[GPT判断引継ぎ](qc_onebody_repartition_finite_step_pilot_v1_gpt_handoff_20261015.md)を用意したが、新しいGPT重要科学レビューは開始していない。commit/push・GitHub byte/SHA照合後Mandatory STOP。継続・終了・第二候補移行・新規性・全費用の判断には別明示GPT開始承認が必要。今回の実行承認は消費済みで、h拡張・追加分子・compile・再実行へ流用しない。
