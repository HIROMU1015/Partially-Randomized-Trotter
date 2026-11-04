# 研究C：B2 reference数値不確かさ・方法と接続の技術準備

2026-11-03。方法A（独立高精度とのsensitivity比較）と方法B（局所誤差からraw信号への伝播）を具体化し、
既存Arb/CARを再利用するB2専用adapterと人工検証を追加した。
**技術接続は準備済み、実H4の不確かさ評価契約はNOT_READY・非実行、Stage CもNOT_READY**。
Stage Aの観測を上界へ昇格していない。新実H4数値評価・保存raw再解析・科学Stage C・native compile・sampling・重要科学レビューは0件。

## 固定対象と基点

起点は `90a2a7baf07b98108111126504d202294a884af5`。Stage A、B1、既存IID accuracyと統合schemaを含むため選んだ。
Branchは `codex/research-c-reference-numerical-uncertainty-readiness-20261103`、作業場所は独立した
`/tmp/prt-c-reference-numerical-uncertainty-readiness-20261103`。B1・統合worktreeの変更はない。
正式共通指示 `research_project_instructions_unified_2026-10-25.md` の§8/8.4/10と適用AGENTSを確認した。
これは技術準備であり、新しい科学判断・保証方式・失敗確率方針の採択ではない。

対象はLinear H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、Number CDF rank8、元frame順LD2、
partial ordered symmetric S2、s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha、DF reserve=CA/8、own-DF ground oracle・状態準備費用除外。
Endpoint SHA256は `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
保存P0仕様（Qiskit1.3/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase/既存Gaussian・relative fusion）は変更しない。
B1のcontrolled full512 synthesisとB2のsector36 referenceは別の証拠範囲。
乱数モデルは独立53-bit IID-grid（event/coherent occurrence/measurement shot/X-Y）。実PCG64のIID保証ではない。

| 固定資料 | 出典 |
|---|---|
| 元PR科学SOURCE | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` |
| Stage A科学SOURCE / 結果 | `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3` / `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` |
| Stage A保存結果 | [result](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/result.json)、[報告](research_c_iid_grid_stage_a_execution_20261103.md) |
| B1結果 | `69c9a764aee66cf2ca79ab29bd3145a027385bcc`、[P0未資格の報告](research_c_p0_uniform_method_readiness_20261103.md) |
| 統合adapter / 報告 | [c_b1_b2_scope_integration.py](../../src/trottertracks/c_b1_b2_scope_integration.py)、[統合準備](research_c_b1_b2_accuracy_scope_integration_20261103.md) |
| B2 SOURCE・固定設計 | [SOURCE](../../artifacts/research_c_reference_numerical_uncertainty/2026-11-03/technical/SOURCE.json)、[未承認契約](research_c_reference_numerical_uncertainty_contract_unapproved_20261103.json) |

Stage Aは38/38 COMPLETE・retry/resume0・mandatory STOPのまま。保存値の引用として、TVは3.4277497476064313e-15、
m12 categorical条件付きraw上界8.985640298365403e-10、保存raw再生最大残差6.021104987719889e-15、
m12 ideal atom−CARのoperator norm観測1.0867657355169296e-10、grid−CAR観測1.58961020298557e-11。
今回これらを再計算していない。`reference_numerical_uncertainty` と `rigorous_bridge_bound` はnullを維持する。

## 実際の計算経路と数値誤差

以下は固定Stage Aのsource静的調査。NPZの数値decodeや科学的な再評価はしていない。

| 箇所 | 実際の関数・表現 | 誤差の経路と扱い |
|---|---|---|
| h/g/scalar/J/U | `c_iid_grid_bridge.prepare_arrays/execute`、`stage_b_runner.load`、NumPy binary64/complex128 | 保存bit値は正確なdyadic入力として再評価可能。元積分・DF近似・物理モデル誤差とは別。J/Uの保存一致は計算誤差上界ではない |
| κとone-body基底 | `c1_stage1.corrected_kappa`: `h-.5*einsum('prrq->pq',g)`、`stage_b.prepare_arm`、`spin_adapted_eigh`/`diag_hermitian` | binary64縮約・Hermitian処理・固有分解。基底再構成と保存generatorの差は表現bridgeと評価誤差に分ける。新ground solveはしない |
| Sector CAR | `SectorCAR.__init__/one/raw_frame/candidate`、`stage_b.candidate_factor` | bilinear符号・励起tableは整数。密度、係数縮約、積、加算は浮動小数。9因子のcomponent照合は観測であり包囲ではない |
| signed atom | `stage_b.atom_operators`: 回転密度、I−2n、Z/ZZ積 | 保存Uは正確にはunitaryでない可能性がある。literal CARと理想unitary/polar targetを同一視しない。216 atomのinvolution観測だけでは単位normを認定しない |
| 係数・λ・p/q | `general_k_coefficients`→`split_native_model`→`configure_rte`、`stage_c2.analyze_table` | general-Kはfloatで生成後Fraction化。pはその保存atom係数の正確な正規化、qは53-bit格子の正確な確率。係数生成誤差はcategorical TVに含まれない |
| angle/normalization | `rte._make_event`: atan(τ)、sin/cos、`finite_rte_attenuation`: exp(−r q log B) | τ/λ/angle/attenuationの浮動小数演算。保存angle bitへのsin/cos包囲と、数学的atan(λh/r)への距離は別 |
| head指数・積 | `c_iid_grid_bridge.heads`→`c1_stage3_reference._ordered_exponentials` | 同一cacheで3指数を再利用し左右の順序は保持、h/2=1/20をfloat化。SciPy expm、complex128積。残差・unitarity観測から局所誤差を0にしない |
| event平均 | `c_pr_aware_df_c2_bridge.event_mean`、`rte.event_unitary` | p/qをそれぞれfloatに変換し216加重加算。平均の加算・重みcast・sin/cos・行列積の誤差。p/q差自体はcategorical枠 |
| raw kernel | `c_iid_grid_bridge.kernel_bridge` | phase×S_rev×E_mean^32×S_fwd。phaseと積・power誤差。保存CARはA1×corrected、PF/RTEは再生成しない |
| round powers | `power_cache` | ideal/grid/raw/correctedの4本、各12回の依存した反復二乗。m roundのモデル上の反復は2^m個の独立kernel。12と4096は別の数 |
| raw reference | `round_bridge` | `A_m * corrected^(2^m)`と`(A1*corrected)^(2^m)`は浮動小数で異なり得る。normalization pathの差を独立の科学効果にしない |
| 状態・X/Y | 保存`sector_state`、`vdot(psi, matrix@psi)` | normの観測一致は正確なground oracleとの距離ではない。state残差/gap等の適用可能な証拠が必要。複素信号はRe/Im両方と位相を保持 |
| operator/Frobenius | `round_bridge.distance`: `np.linalg.norm(...,2/'fro')` | SVD・差分・二乗和・sqrtにも誤差がある。Frobeniusは数学的にはoperator上界だが浮動小数での算出値は自動的な包囲ではない |

Science入力rolesは契約に継承した原input、candidate、factors、own_ground、PF_RTE_referenceの5つ。
科学SOURCEを今回の閲覧・技術commitへ付け替えない。

## 方法選定と再利用

| 方法 | 具体策・再利用 | 取得できる資格と限界 |
|---|---|---|
| A 高精度・独立経路 | `pr_dictionary_validation.reference.precision_context/exact`（90/128桁）、既存Arb arithmetic、同じexact dyadic入力でexpm/積/平均/二乗/signalを比較 | Precision sensitivity。共通の入力・モデル誤り、両経路の共通誤差は検出できず、差を上界へ変更しない |
| B 説明可能な伝播 | 既存norm growthとtelescopingの設計を参照し、B2専用の局所誤差・state換算を追加 | 適用条件・局所上界・coverageが揃えばvalidated numerical bound。未検証のlibm/BLAS/指数誤差やstate誤差があればestimateまたはnull |
| C 必要部分の包囲 | `c_pr_aware_df_fixed_car.Arithmetic/Matrix/pack`、`zsxx_enclosure._real/_matrix/_norm`、既存`_ordered_exponentials`と`event_unitary`のarithmetic引数 | 指定したliteral入力mapのrigorous enclosure。全H4/全law/ground oracleを自動的に囲わない。旧polar bridgeやfull512資格をB2へ流用しない |

推奨はA+B。局所上界の不足をCで閉じることは可能だが、全演算に一律の最高精度や区間計算を要求しない。
SciPy1.14.1 expmは可変次数のPadé/scaling-and-squaring実装で、APIは入力ごとの誤差包囲を返さない。
[SciPy公式資料](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.expm.html)。
Mpmathも精度変更による一致確認を誤差上界とはしていない。
[mpmath公式の精度・誤差の説明](https://mpmath.org/doc/current/technical.html)。
Arbの行列指数APIは[python-flint公式資料](https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.exp)を確認し、
実際の人工試験は既存0.8.0/FLINT3.3.1で行った（latest文書の0.9.0へ環境更新していない）。

`c_j_u_precision`のlongdoubleは目的関数primalの評価であり、complex行列expm/BLASの任意高精度保証ではない。
`c1_h4_full_synthesis.propagate`/`c1_subspace_synthesis_bound.repeat_bound`は両operatorのnorm growthを扱うが、
native synthesis用の浮動小数pilotであり、各反復二乗の新しい丸め誤差を囲わない。
そのため共有moduleは変更せず、必要な数式だけをB2のexact-rational伝播へ追加した。CAR、指数、event、RPE engineの再実装はない。

指数への入力誤差も無視しない。Hermitian H/Gが確認できる場合は
`||exp(-itH)-exp(-itG)|| <= |t|*||H-G||`を使える。
一般の非Hermitian A/BならDuhamel積分から`||exp(A)-exp(B)|| <= ||A-B||*exp(max(||A||,||B||))`。
前提未確認ならunitary用の短い式を流用せず、既存Arb指数を使うか未評価を維持する。
行列指数本体・入力のcast・scalar/timeの誤差を局所ηと入力不確かさに分けて伝播する。

## m0..12の伝播と誤差会計

新module [c_reference_numerical_uncertainty.py](../../src/trottertracks/c_reference_numerical_uncertainty.py) の`product_error`は、
表現された入力のnorm上界a,bと入力不確かさe_A,e_B、局所積誤差ηを受け、
`a e_B + b e_A + e_A e_B + η`を返す。ηが不明ならnull。SciPy/BLASがこのηを満たすと仮定しただけでは資格は上がらない。

`round_transport`ではKを数学的raw kernel、Khatを表現されたkernelとする。
αは**両方**のnorm上界（保守的に1以上）、e0はkernel構築誤差。
exact Khat^(2^m)−K^(2^m)の上界C_mと、計算されたpower−exact Khat^(2^m)の上界R_mを分ける。

```
a0=alpha, C0=e0, R0=0
C_(m+1) <= 2*a_m*C_m
R_(m+1) <= (2*a_m+R_m)*R_m + eta_m
a_(m+1) <= a_m^2
operator_error <= C_m+R_m
```

これはモデル上のcoherent repetitionの入力誤差増幅と、計算上の12回の反復二乗を区別する。
raw kernelにphase・attenuationを一度含める。`A_m corrected^q`経路を使う場合はscalar誤差も別途上界へ含める必要があり、
このadapterが黙って`(A1 corrected)^q`へ置換することはない。非unitaryのnorm growthを捨てず、2でclampしない。
数式の上界演算はFractionを使い、各段で128-bit dyadicへ上向きに丸める。これは上界の表現方針であり、実H4計算の精度採択ではない。

状態のnorm上界s=||psi_hat||、oracleとの距離d、数学的operatorのnorm上界k、operator誤差e、期待値計算誤差η_zに対し、
`|z_hat-z| <= s^2 e + k*d*(2s+d) + eta_z`。
複素信号全体の絶対誤差であり、operator誤差との直接同一視や余分なsqrt(2)・attenuation再乗算はしない。
d不明ならraw信号上界はnull。保存stateをexact bit値として扱えばそのliteral stateへの計算誤差は評価できるが、
own-DF ground oracleへの誤差を0と認定したことにはならない。

`weighted_events`は既存K0 signed-eventをexact rational pまたはqで加重するだけ。
保存angle bitとexact atan targetの差、literal Uと理想unitaryの差は別途必要。
`enclosed_rounds`は既存Arb行列で13信号を計算し半径を保持する薄い接続。midpointへの置換・勝手なstate正規化・phase fittingはない。

| Partition | 単位・対象 | B2との関係 |
|---|---|---|
| synthesis | full512 nativeのraw complex signal | B1。変更せず、未成立のまま |
| categorical | p→q、IID ideal trajectory raw信号 | Stage AのTV由来成分を一度だけ使用。係数生成・angle・state誤差を含まない |
| reference_bridge | exact saved-atom IID mapとCAR mathematical referenceの表現差、raw信号 | 数学的targetを固定し、含む誤差を明示する。現観測だけでは独立した上界にならない |
| reference_uncertainty | 他成分に包含されていないreference演算・state/oracle関係、raw信号 | B2。既にbridge包囲へ入れた半径を再加算しない |
| PF/corrected RTE | 既存operator誤差を既存engineがrawへ変換 | B2のbridgeへ混ぜない |
| DF | Ha、energy stageで一度 | raw signalへ加えない |

中心残差rだけなら観測。`r + u_atom + u_CAR`というtriangleの総量を使う場合、そのuをbridgeとuncertaintyへ二重配分しない。
中心残差だけを「exact bridge上界」と命名しない。旧interfaceの`reference_uncertainty_included=False`を満たすには、
既に包含済みの誤差と残差を具体的に分離する必要がある。今はこの科学入力が未固定なので両成分の実数値を発行しない。

## Qualificationと統合interface

`evidence_record`はcandidate/Stage A SOURCE/model/compiler/条件を既存`integration.identity()`で照合し、
m0..12、reference target、sector36、位相依存、raw換算、error model、包含項目、未解決事項、provenanceを出力する。
橋渡し項目と不確かさのerror tagは分離する。PF/RTE/DFやcategoricalのtagは追加しない。

| B2資格 | 統合時の扱い |
|---|---|
| observed_numerical_residual | valueのみ、bound=null |
| precision_sensitivity | valueのみ、bound=null |
| numerical_uncertainty_estimate | valueのみ、bound=null |
| validated_numerical_bound | target/局所coverage/state/normalization/上向き演算と根拠が揃う場合だけnumerical_bound |
| rigorous_enclosure | 上記に加えrounding_enclosedが必要、rigorous_bound |
| unknown | value/boundともnull |

Schemaは数学的根拠の正しさを証明するものではない。モデルの適用条件と上界は将来の科学入力。
既存`integrate`→`iid.qualify`→`RoundEvidence`/`assess_round`/`assess_energy`へ渡せる形に限り、
新しいRPE/shots/energy計算式は作らない。未資格B2、未資格B1または欠落があればStage Cを呼ばないことを人工試験した。
B2が完成してもWhole-P0一様上界が未成立の間は全体READY/PASSにしない。
統合branchの固定ファイル・SOURCE・outputは変更していない。

## 未承認科学契約と次に必要な作業

契約は**NOT_READY / executable=false / call_limits=null / maximum_logical_units=null**。
新しいscience entryは作らず、[entry](../../scripts/prepare_research_c_reference_numerical_uncertainty.py)はpassive専用。
`--execute`を受理せず、Python APIも実科学を実行しない。grant・承認原文保存・承認証明は追加していない。

未確定点は、(1) exact saved-bit CAR mapとreconstructed CAR/atom targetの対応、
(2) λ/τ/atan/attenuation・literal Uの差の完全で重複しない配分、
(3) 保存groundと理想own-DF oracleの距離上界、(4) それらの精度/backend/実行call graph。
Arbはliteral mapの演算を囲えるが、これらの意味上の不足を自動的には解消しない。
このため実H4のvalidated numerical bound取得はNO-GO。単なる「高精度一致の測定」なら別の限定scopeだが、それを今回の不足証拠取得と混同しない。

必要logical callsは実行graphが未固定のため**算定不能/null**。旧Stage Aの38-callもStage Cの16-callも流用しない。
次の設計では入力・identity、各reference構築/局所評価、共有power chain（1本につき12依存積）、13期待値、summary等を
実際のdispatcherから算定する。precision違い・p/q/CARのchain数・oracle relationの評価を決める前に上限を捏造しない。
CPU/RAM/worker/wallの人工capは設けない。独立precision/chain/atomを実資源に応じて並列化し、依存squaringは各chain内で順に行う。
既存のresource_planやimmutable cacheを再利用し、precision contextはprocessごとに分離、共有venv/cache/jobは変更しない。
将来のfresh outputは固定設計に記録し、retry/resume0、partial FAILは後続STOP・原本保持・fallbackなし、最後はmandatory STOP。

次の担当はCodex：上のtarget・配分・oracle関係を既存保存証拠と技術実装で閉じ、call graph付き契約を完成させる。
新しい保証対象・数値条件の採択が必要なら、その具体的な差だけをユーザー判断へ戻す。
その後、ユーザーが完成した固定B2契約に対する実H4評価を別途許可する。今回の許可はその科学実行や重要レビューを含まない。

## 人工検証・保全・公開

人工・回帰テストは**328 PASS、既知の旧テスト2件を明示的にdeselect**。
新B2は36件、既存統合/B1/Stage A/accuracy等284件、CAR/κの関連人工8件。
[実行条件・結果](../../artifacts/research_c_reference_numerical_uncertainty/2026-11-03/technical/technical_checks.json)と
[passive preflight](../../artifacts/research_c_reference_numerical_uncertainty/2026-11-03/technical/passive_preflight.json)を記録する。
2除外は既存のStage A消費後に古いfresh PASSを期待するテストと、並行失敗の表面例外順序に依存する旧C2テスト。
消費済みSTOP拒否、主journalによる両失敗順序は既存の代替テストで検証した。旧テスト・SOURCEを変更していない。
正確な行列、既知摂動、精度差、expm、積・反復二乗、norm growth、状態誤差、complex X/Y、分離・二重計上、
資格誤昇格、null、13round、wrong identity/model、Stage C拒否、fresh/retry/resume/STOPを検証する。
人工PASSはH4の数値不確かさ取得済みという資格ではない。
Arb0.8.0/FLINT3.3.1は既存task-local siteをread-onlyで参照し、共有venvを変更していない。
小さな行列の試験ではBLAS1と独立したtask-local cacheを用いた。実H4の資源・精度policyをこれで制限していない。
初回の人工試験でPython complexの大きな整数powerを「正確なiの累乗」と扱った期待値が不適切だったため、
正確な4周期の期待値へ修正した。科学入力・計算経路の変更や再実行ではない。

元checkoutのHEAD/index/未コミット状態、旧SOURCE/raw/契約/STOP、B1/統合SOURCEを保全する。
新しいGit書込みはこのB2 branchのcommitと通常push、独立worktree登録に限る。
科学SOURCEの変更、元repoのcheckout切替、main統合、他jobの停止や設定変更は行わない。
**公開照合後mandatory STOP。**

技術実装commit：`cf8d09338faaaa548a9f5112bb93935ac4f24af1`。SOURCE manifest SHA256：`03dc1944d1a181ac7a81000fcc898fdd893f154d326dc5619d79348748fa11ff`。
未承認設計SHA256：`6936683e563818e601143b0a52d8919bb4c2fe2cd26211698cb0289ca9496eaa`。新SOURCE7項目と継承166項目のunion171、保存input33項目をbyte照合した。
固定後のpassive preflightは数値libraryのimportを遮断してPASS、科学call0、実行ready=false。
最終報告commitはこの実装commitを親にして文書・契約・技術結果だけを記録する。
元checkoutはHEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、index SHA256
`68a93531c7bc1993d49f6750ac8d98a54fafb8e153ddd0b887dd17b28ddd9c4c`、作業開始時のstatus178行と一致した。
