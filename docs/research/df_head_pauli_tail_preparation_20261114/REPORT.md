# DF head＋Pauli tail：技術調査と有限検証案

同じDF head・同じ物理tailを保つPauli辞書への接続は、既存の演算子・有限RTE・回路primitiveを再利用して検討できる。ただし、現LD1/LD2のPauli辞書、λ、科学的資格、価格優位は未取得で、本番providerは未接続である。次はLD1/LD2のoperator/dictionary検証を先行し、成立した条件だけで限定mean診断へ進む案を推奨する。正式採択・科学実行承認ではない。

今回はSOURCE調査・人工4test・設計のみ。実H4/H6入力配列のload、新λ/signal/normalization/N/MSE/価格/G、保存科学値の再解析、回路build/compile/samplingは0。固定Full PF成果71f04be7と既存STOPは変更していない。[有限計画](SCIENCE_PLAN.md)／[B/C/D HANDOFF](HANDOFF.md)／[SOURCE索引](SOURCE_INDEX.json)／[テスト](TESTS.txt)。

## 再利用する既存資産

| 資産・固定SOURCE | 関数・役割 | 再利用と差分 |
|---|---|---|
| P3 Stage2：8bcf5851f、科学4bfbbaad | `matched_partition_stage2.pauli_dictionary/build/raw_event_mean`、`paired_polynomial_action` | 全Fockで独立辞書和・有限平均・scalar/νを照合する方法。T=.8/q8/r4/K4、全10fragment残差、固定VBARのrunnerを現prefixへ直接使用しない |
| P3 Stage3：90bc02d45、科学c0d9d3bb/run02 native9c4006bf | `matched_partition_stage3.component_inputs/residual_sequence/DictionaryOracle/price/wrappers` | Pauli residualと共通DF Gaussian head、D2制御、Full wrapper、非加法性検査。保存cohort価格を新辞書のE[C]へ移植しない |
| P3後続設計60515a33c | `h4_p3_poststage3_cost_and_stage4_design_20261102.md` | 同一tupleの独立価格確認・信号精度会計と、全RPE energy会計の区別を再利用 |
| O1-M1実装5e4f0048・回路/信号424ea9d1 | `o1_m1_matched_split.prepare_matched_a/pauli_tail_coefficients`、`validate_matched_circuit_signals` | 保存factorのΣλA²→JWとidentity方針の部品。現prefix・保存completionへの差分が必要。旧full-input loader/validator全体を起動しない |
| 共通：A基点71f04be7 | `PauliHamiltonian.from_terms/from_openfermion`、`rte.finite_rte_distribution/finite_taylor_operator/event_unitary` | signed係数、global collect、Pauli label順、有限K lawを再利用。OpenFermion内部cutoffとimaginary丸めは別に記録・検査する |
| 現A：signal実装fefd548e、結果91349e70、基点71f04be7 | `pr_rpe_signal_provider.make_context/_macros/evaluate`、保存reference/loader | direct-expm DF head、独立raw/corrected、exact-tail、matrix power/cacheを再利用。Pauli SOURCE/辞書identityと検証済みtail供給のadapterが不足 |
| B：ce0517de6e7b58cf4bf4c69f99f998d86099c406 | `pr_rpe_rmse_controller`、`rpe_signal_statistics_caller`、`rpe_deterministic_risk` | canonical schedule・raw→二項測定・decoder/DP/riskを正本として使用する。Aは状態集約DPを再実装しない |
| C：d7c09ef2e2521bb1815d1d11970529fc7030c2e3 | `h4_structural_kernel.Model/Wire`、`MeanKernel` | kernel代数は再利用候補。現Modelは324成分・9 DF basis・order0/2等を固定し、Pauliラベルへの付替えでは使えない |
| D：3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673 | `candidate_price_adapter`、`efficient_controlled_zz`、既存DF head builder | head/profile/Full wrapperの部品を再利用。現tail価格経路はDF preparation/hash/eventを前提とし、Pauli eventの新contextが必要 |

P3/O1-M1ファイルは固定Git blobで参照し、branch統合はしていない。B/C/D公開headはread-onlyのls-remoteと一致した。旧Track A調査162f3d3f0と横断監査51d324eaを参照した。監査で指摘されたloweringの不整合と、長roundを直接compileで埋められない問題は新辞書にも関係する。

## Hamiltonian・identity・分割

現H4はLinear1Å/STO-3G/tol-only DF10/sector(2,2)、同H/state/snapshot・target・fragment順を保持する。DF headはone-bodyと保存順の先頭L_D個、tailはその後のweighted fragmentである。

\[
H=cI+H_D+H_R,
\quad H_R=s_RI+A,
\quad A=\sum_j c_jP_j,
\quad\lambda_P=\sum_j|c_j|.
\]

Pauli化するのは現loaderの保存completionを含む**同じtail**であり、裸の元分子Hamiltonian−headへ置き換えない。現SOURCEは`h4_pr_rmse_completion.load_context`→固定`h4_df10_rpe_batch_provider.load_saved_context`→`h6_df19_rpe_d2.make_context`を使い、loaderが構成したDF表現から各weighted blockを組む。独立Pauli経路もそのfull-Fock extensionを再現し、36次元の保存blockと照合する。旧raw tensorや別のHermitian化順を無条件に同一視しない。

O1の保存factor→one-body FermionOperatorの二乗→JWをprefix後のfactorへ限定するadapterが最小差分候補。ただし旧M1は全二体tailのLD0であり、現L_D=1/2ではない。P3のbare W−D経路も現prefixの直接producerではない。元H/stateのhash・mode順・completion・scalar一致は将来の実operator検証で確定する。

現方針はtail identityを抽出して`exp[-iT(c+s_R)]`を一回適用し、head内のidentityはheadに残す。s_Rはfull-Fockのidentity係数で、**36次元sectorのtrace/36で求めない**。Pauli nonidentityのsector制限はtraceが非零になり得る。Pauli full-H constant、核scalar、head identityを追加して二重計上しない。

phaseには保存s_Rのexact operandを使い、Pauli identity抽出との差はbinding uncertaintyとして記録・伝播する。表現の丸め差を見て旧scalarを上書きしたり、追加identityを無記録で捨てたりしない。

faithful identityをtail eventへ戻す旧O1方針は、有限Kでは抽出phase方式と同じ近似ではない。P3のVBAR定数や旧λ/Bを流用しない。辞書はHermitian Pauli stringと実係数で、同じlabelをglobal collectしexact zeroだけ除く。既存collectorの係数imag許容1e−12、OpenFermionの内部係数処理、scalar差の残差を記録し、無断の項cutoffや再正規化で一致させない。

## sector：eventと平均を区別する

個々のPauli stringは一般にNα/Nβ sectorを保存しない。ΠPΠの二乗はΠと限らず、`exp(-itΠPΠ)`もΠexp(-itP)Πと一般に異なる。36次元で射影Pauliを元のinvolutionとしてevent生成する経路は使えない。現H4辞書の各labelが実際にどのsectorへ移るかは未計算で、一般論を実測済みとはしない。

一方、全Fockでのideal finite RTE平均恒等式が確認でき、各DF head blockと合計AがΠをreduceするなら、平均の多項式はsector内で計算できる。つまり**event/trajectory/native作用はfull256、検証済みideal meanは36次元へ縮約可能**という役割分担が候補になる。

\[
F_K(A,d)=\sum_{\ell=0}^{K+1}\frac{(-idA)^\ell}{\ell!},
\qquad E[U_{\rm event}]=F_K(A,d)/b_K(\lambda d).
\]

ΠとA、headが可換ならΠF_K(A)Π=F_K(A_sector)、指数・積・powerにも同様の制限ができる。これは項ごとのeventを射影してよいという意味ではない。fresh IID events/trajectory、sector postselectionなしのHadamard測定が前提。finite-bitで確率が変わった実効平均、noise、postselectionへ同じ縮約資格を自動移送しない。

人工N=1/3modeでは、XXIとYYIが別々にはsector外へ移り、和が保存する例を確認した。射影involutionの破れ、full meanと縮約mean、raw/corrected独立伝播・scalarの整合を4testに含めた。実H4のsector証拠は別途必要。

## λ・K・κ・shot・価格の依存

\[
b_K(\tau)=\sum_{k=0,2,\ldots,K}\frac{|\tau|^k}{k!}
\sqrt{1+(\tau/(k+1))^2},\quad
\log B=\sum_{\rm pieces}q\,r_j\log b_K(\lambda t_j/r_j).
\]

Pauli回転はsigned成分signとatan(τ/(k+1))、product順・order phase (−1)^(k/2)を保持する。K=2の1 eventは回転1つと0又は2個のPauli productで、event位置数と実gate数は異なる。K0は既存rteで実装済みだが、現A `evaluate` はK2/4/6に限定される。K0を使う本番変更やそのbias資格は今回採択しない。

同じA、同じevent時間幅/count、同じKならcorrected meanは辞書に依存せず、B/rawは辞書λに依存する。これは既存P3の知見とも整合する。**同じκ**で比較すると、新λによって`r*=κλ²T²`とR_outerの整数化が変わるので、corrected finite-K biasも一般に変わる。保存DF corrected meanを新κ-Pauli候補へそのまま流用しない。固定Rでの一致検査と、共通κによる全schedule比較を分離する。

現mapperはqRPE=2^m、qPF=ceil(T/ΔPF)、signed piece数Jを区別してR_outerを生成し、signed allocationを保つ。κの定義を新設せず原exact値を維持する候補を優先する。Pauli辞書λとDF辞書λの大小には一般の順序保証はない。λ、B、event数、回路単価、Nの変化を独立倍率として掛け合わせない。

| 自由度 | signal/N側 | 回路価格側・今回の扱い |
|---|---|---|
| 辞書 | 同tailでもλ/Bが変わる。固定Rと共通κを分離 | Pauli支持とDF基底切替の違い。採用未決定 |
| L_D | head/tailとPF bias・λの双方が変わる | head費用が変わる。初回LD1/2を提案するだけ |
| K | 多項式次数K+1・normalizer・finite biasが変わる | product数の分布が変わる。既存K2を維持、K0は未採択 |
| κ | Rの整数化・attenuation・finite biasが連動 | event数と長round価格が変わる。元exactκを維持する案 |
| PF公式・Δ | headのPF/exact-tail bias、qPFとactual幅・early例外が変わる | signed stage/step数・head費用が変わる。初回S2/.1を維持 |
| N | 全round rawとstate/law/decoder/riskに依存 | 各軸価格の重み。Bに委ね、今回生成しない |

BはrawのRe/Imからp_X/Y=(1+Re/Im raw)/2を受け取り、同じlaw/state/target/decoderで全round riskとNを扱う。PF/finite-K biasはraw riskへ既に含まれる部分と別予算の部分を区別し、二重加算しない。現Cのbasis切替kernelを削除するだけでPauli構造価格になるわけではなく、label支持・基底回転・parity ladder・product phase・head境界のcompiler簡約を新lawで評価する必要がある。

## 既存の正結果・負結果と移送限界

| 固定結果 | 言えること | 現条件へ移せないこと |
|---|---|---|
| P3 Stage2/3：T=.8/S2/q8/r4/K4、共通T_eff・全二体残差、full256 | 同一残差のcorrected平均が整合。λはPauli13.979010086592508、DF13.72369864844825。固定4 path/armのFull RZはP2916–3070、DF8092–9978（原報告引用） | finite bias両軸UNRESOLVED、価格は母平均でなく、LD1/2や13round RMSEの優位ではない。run01 Gaussian preflight FAIL、run02 thread compile例外と23部分価格、run03 process分離回復を保持 |
| O1-M1 424ea9d1：LD0、faithful identity、K0/2、δ.1/.05・q2/r2 | full256でselected circuit/meanの486数値検査PASS。辞書依存のattenuationと同corrected meanを確認 | 現identity抽出、strict normalized state、prefix1/2と異なる。旧validatorのraw=corrected/Bという事後定義は現独立closureの代用にしない |
| C0 50ee55bc3：LD0/δ.05/r2/K2、m0–6、εE=.1、CX、seed7 | Pauliの低い単回価格だけでは強PFに勝てず、長round減衰が総費用を支配した条件付き負結果 | 現RZ/seed17/共通κ/13round/化学精度へ倍率を移さない |
| r8 0daf01d51：q64/r8、各8 path | Pauli平均14280.375 CXに対し保存必要単価4.335396744未満を満たさず、固定2候補の研究終了が提案された | 辞書法全体のNO-GOや現κ条件の不可能性ではない。同じr8価格MCを繰り返さない |
| 11/9調査162f3d3f0 | M1は既存matched機構で、新規性や全RPE改善は未証明。LD>0/round可変Rは旧比較と異なる未検証範囲 | 技術実装の存在を新しい科学成果や最適化済みと呼ばない |

## 到達点と次

同A branchの独立資料namespaceを使い、過去P3/O1やB/C/Dのworktreeを変更せず調査した。production変更0、人工4test PASS、既存import由来のinvalid escape DeprecationWarningは今回修正していない。

次はユーザーが[有限案](SCIENCE_PLAN.md)のoperator-only段階又はoperator＋限定mean段階を選び、明示承認する。実装時は既存primitiveを接続する小adapterだけをSOURCE固定する。現在のPauli辞書採用、K/κ変更、全RPE RMSE/十分N/G/winnerは未決定・null。公開とremote照合後mandatory STOP。
