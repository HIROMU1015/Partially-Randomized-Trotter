# PR候補探索：RPE全体の資源を目的とする設計

2026-11-13 JST。**Track Aの有限時間順位を採択せず、全RPE scheduleを比較単位とする探索を設計した。今回はSOURCE調査・metadata-only adapter・人工テストのみ。実PR signal、保存値再解析、実候補探索・最適化、N/C/G計算は0。**

[SOURCE対応](SOURCE_MAP.json)／[B・価格担当HANDOFF](HANDOFF.md)／[有限科学計画](SCIENCE_PLAN.md)／[人工13round入出力](ARTIFICIAL_IO.json)／[新7テスト](TESTS.txt)／[旧10回帰](REGRESSION.txt)。基点`b72ea6b30e48e6a2f5d805903b7bd750b965cf16`の既存A branchを再利用し、B/C/Dのbranch・worktree・processへ変更していない。

## 1. 何を最適化するか

候補θは**全ラウンド**の物理構成・lawと、同じtarget/RMSE/decoder契約を持つconfigurationである。目的は

\[
G(\theta)=\sum_{m=0}^{M}\sum_{a=X,Y}N_{m,a}(\theta)C_{m,a}(\theta)
\]

を、全事象のenergy RMSEが目標を満たすという資格の下で比較すること。Nは各軸の量子shot、Cは測定付きfull Hadamard wrapperのcompiled RZ、P=0。Nだけ、T=.8の価格だけ、m12だけで最良を決めない。finite-K meanを取り込むdecoderではbranch/aliasがラウンドを結ぶため、round別に最小N×Cを貼り合わせるだけでは同じRMSEを保証しない。

既存B `3216d27a69ac8532d2a22e78f5f94491d202d238`の`generate/assess/cost/select`を正本にする。作業時に公開branchのSHA一致を確認した。`generate`はmetadata、`assess`は資格・欠測、`cost/select`は別途許可された供給N/Cに対する全schedule計算。今回は実dataのcost/selectを呼ばない。未知N/全RMSE/価格domainを0で補わず、暫定model評価と科学的資格を分ける。

対象の最小構成は、既存H4 linear1Å/STO-3G、保存tol-only DF10/sector(2,2)/36、同じH/state/snapshotと元fragment順である。PRはone-bodyを決定論側に残し、prefix後の全tailをrandomizeする。全保存DF-Hをtarget/referenceに使い、truncated Hへtargetを変えない。「元Hamiltonian」と保存DF-Hの区別を保持し、DF前の分子Hamiltonianの厳密ground精度まで資格しない。保存stateの条件付きproxy、prior/decoder、既存Minkowski予算も未資格部分ごと保持する。

## 2. Track Aの目的と再利用範囲

Track A `155529e20a8fac39e00786c40b57830b71203c4f:REPORT.md`と実装を確認した。T=.8/1.2/1.6/2、座標精度ε別、α_axis=.025の単一Hadamard signal費用が対象であり、全energy RPEの費用ではない。`h46_long_time_native.select`は各system/T/formula/εのeligible endpointを旧shot数とwork proxyで並べ、compiled対象を絞っていた。そこでの点最良・Pareto・選択subsetは新RPEの最適性の証拠ではない。

| 既存SOURCE・関数 | 再利用 | RPE全体用の変更・制限 |
|---|---|---|
| `h46_long_time_signal.candidates/cid` | formula/prefix/q/R/Kの既存変数、physical ID | 旧有限T gridを実行せず、Bが受けた全dyadic scheduleを入口にする |
| 同`evolve_batch`、`h46_new2_core` | paired Taylor K+1、signed allocation、独立raw/corrected、scalar | deterministic headは直接expm。旧contextがreferenceを自動生成する部分は起動しない |
| `h46_m5_signal_adapter.prefix_context` | prefix固有tail/scalar/λ/preparation、元fragment順検査 | anchor tailを別prefixへ流用しない。旧contract・grant発行系を新探索へ拡張しない |
| `h46_m5_plan.weights/allocation` | S2/Y4/M5の既存binary64 signed順・整数tie規則 | 新係数や8次、PER_PIECEとABS_TIMEの混在を導入しない |
| `rte.finite_rte_distribution/finite_taylor_operator` | finite-K normalizer・既存多項式 | Kごとの別lawを保持し、correctedの非unitarityをPF数値失敗と混同しない |
| `pr_rpe_power_mean`、`h4_pr_rmse_completion` | 保存snapshot loader、巨大event列を展開しないpower、SOURCE/reference検査 | 旧spectral headを新PRでそのまま使わず、直接expmのheadへ接続する技術差分が必要 |
| `pf_power_stability`、A `b72ea6b30` | 直接expm macro-step・失敗値保持のnorm/unitarity診断 | 追加のexpm正当性研究は目的にしない。PR exact-tail/corrected/rawの組合せは必要な人工回帰だけ行う |
| `pr_rpe_rmse/preflight` | 共通κ、policy分離、field/source照合、nominalモデル | 旧Hoeffding NはRMSE Nにしない。新split/recipeのmeanは同一性照合後だけ受ける |
| B controller/integration | canonical全schedule ID、N packet、資格と価格coverage | candidate管理・decoder・G optimizerをAで再実装しない |
| `h46_long_time_native.compose/sample_requests/compile_cached` | native適用順、law、wrapper、compiler、fingerprint/paired XY | 価格担当が必要時だけ再利用。旧有限cohort/単一Tの価格は自動転用しない。今回build/sampling/compileなし |

## 3. 既存の設計自由度と探索上の扱い

以下は**変数としての既存実績**であり、今回採択した数値gridではない。

| 設計自由度 | 既存の根拠・範囲 | signal・Nへの影響 | 1shot Cへの影響・接続条件 |
|---|---|---|---|
| prefix L_D | Track A H4={1,2,3,5,7,9}、full10。元のgeneration orderで切る | head/tail、λ_R・tail scalar・commutator/finite-K biasが変わる | head block数とtail basis/law/preparationが変わる。L増加が必ず有利とはしない |
| PF formula | S2、標準Yoshida4、M5 best | PF近似・signed tail時間・exact-tail対照が変わる | piece数/負時間/制御/rotation列が変わる |
| qPF又はΔ上限 | Track A各Tのq grid、S2の一部q32追加 | 実幅T/q、PF bias、normalizerの積が変わる | head反復数、q/R grouping・融合の費用が変わる |
| R_outer | Track A S2=4/8/16/32/64、Y4=8/16/32/64、M5=16/32/64等。後の可変R研究で拡張 | piece event幅、finite-K mean、B/attenuationが変わる | event数qRとlaw・event orderが変わる。BをCへ掛けない |
| finite K | Track A paired K2/4/6、最高多項式次数K+1。K0は任意で未追加 | truncation biasとnormalizerが変わる | event内次数・Pauli product/rotation・稀事象価格が変わる |
| 共通κ | A/B既存RMSE mapperのrotation型κ | λ_RとT、qを通じR_mを全roundで結ぶ | 全roundのR/event/price domainを同時に変える |
| M | Bの既存全schedule比較用の最大round index | horizon/alias/nominal shot配分が変わる | round数と最大Tの双方が変わる。Mを変えたNは再利用しない |

prefix4/6/8も「元順prefix」という同じ分割形式では表現できるが、上記tol-only保存tail descriptorの確認済み範囲にはない。特にL_D=7/8/9を今回の正規gridとせず、8のλ/scalar/preparationを他prefixで代替しない。h、target/誤差budget、state、fragment並替え、重み再fit、one-body再分配、任意subset、独自sampling law、新PF係数は今回追加する変数ではない。

**初期の推奨構成**：outerは元順prefix・formula・Δ、各Kは明示した別stratum、innerは共通κ（必要なら既存Bが扱うM）。最初はh/M/target/予算を既存baselineに揃えてK2/S2の一因子差から始め、保存結果と接続しやすくする。これは有限pilot設計の推奨であり、L集合・κ値・Δ集合を採択してはいない。

## 4. Bとの互換性に残る具体的な差分

1. native B V2はPRのS2/M5に対応するが、**PRの標準Yoshida4を拒否**する。Track Aでは物理計算可能でも、現B正本で受けられない。Y4を含める科学menuを採るならBがsource-bound formula対応を最小拡張する。AはM5として偽装しない。
2. B `select`はfinite_K/κ定義/mapperを同一方式内で固定する。K2/4/6の外側比較を隠れた変数として渡さない。まずK別stratumの評価を返し、Kを横断した最終有限menu比較が必要ならBへ明示対応を依頼する。Aで別のoptimizerは作らない。
3. B nativeのq=ceil(T/Δmax)とactualδ=T/q、A旧固定-gridの厳密δは別。Δmax>hでは初期roundの実幅が小さい。既存baselineを再現するときはA fixed-grid adapterを用い、nativeへ黙って移さない。commonκのkind文字列・definition・integer化も元のadapterごと保持する。
4. B V2/physical IDと、raw meanの再利用keyは用途が違う。同じphysical meanはMやκが異なっても同一round recipeなら再利用候補だが、N・RMSE資格・全schedule identityを再ラベルしない。exact時刻とbinary64への変換が一致しない場合は停止し、丸め補正で合わせない。
5. `pr_r_dependent_n`/`pr_r_screening`は旧fixed prefix/formula/KやR探索に制約を持つ。数学・normalizer/metadata部は再利用できるが、split自由化済みproviderとして呼ぶことはできない。H6の初期m0–3例外は旧providerに保持し、今回H4のuniform helperへ移植しない。

## 5. 全roundのsignal評価経路（未実装の数値接続設計）

`H = cI + head_p + tail_p`、`tail_p = s_p I + A_p`とし、one-bodyをheadに含める。prefixごとに同H/stateへbindingしたs_p、full-Fock係数からのλ_p、tail/preparation descriptorを用いる。sector行列のtraceやoperator normからλ_pを代用しない。保存値がなければ、その取得を新科学scopeへ列挙する。

各roundはT_m=h2^m、q_mと実δ_mを固定し、各signed piece jのt_j=T_m w_j/q_m、整数r_j（総数R_outer、各piece最低1）を元allocationで得る。有限正規化は

\[
\log B_m=q_m\sum_j r_j\log b_K(\lambda_p t_j/r_j)
\]

で、負時間も既存finite分布へそのまま渡す。B_mの単純近似やpaper envelopeをexact finite値へ置換しない。

同じhead half-stepを**直接expm**で構成し、元のforward → tail → reverse適用順を維持する。tailには既存paired Taylor numeratorのmatrix polynomial/power、exact-tail内部対照にはexpmを用いる。数値recipe・dt/λ演算順をSOURCEで固定する。rawは各pieceでnormalizerを作用させた独立macro-operatorからq乗し、最終corrected mean/Bの代入でclosure検査を消さない。scalar(c+s_p)をtotal Tで一回だけ会計する。

macro-stepをexact-tail/corrected/rawの3経路で作ってmatrix_powerへ接続する。qR個のevent列・trajectoryは生成しない。同じ(prefix,formula,actualδ,R,K,allocation,preparation,numerical SOURCE)のoperatorを共有し、R/K違いではhead/cacheとexact-tail対照を共有できる。共通δでもR_mが変わればfinite operatorは別。独立raw/corrected closure、normalizer ULP/log感度、非有限/overflow/underflowを確認する。

**unitarity gateを課すのはhead/exact-tail対照**で、finite-K corrected/rawは一般にunitaryではない。power後norm/unitarity・reference pair/closureは元thresholdを維持し、失敗残差を先に保存する。power境界normを全eventの観測最大と称さず、旧経験的allowanceの適用条件・観測範囲を更新SOURCEで明示する。未検証の厳密上界やunknown actual-law誤差を0にしない。

出力はraw/corrected/reference/exact-tail、signal空間のPF寄与とfinite-K寄与、logB/attenuation、norm/closure/allowanceとSOURCE/law。local biasはenergy biasではなく、BのdecoderがrawのX/Y確率と全scheduleの誤差契約へ接続する。旧nominal Nを新candidateへcopyしない。BのS3をAで再実装しない。

## 6. 安価な指標・段階評価・除外の区別

| 段階 | 安価な処理と用途 | 除外としないもの |
|---|---|---|
| 0 metadata | full-round physical/source一致、保存mean/reference/tail/C domainの所在、整数R allocation、構造的head/event数 | 保存が無い、旧価格domain外、full schedule未資格は欠測として残す |
| 1 normalization | 承認後だけ全roundの既存finite式、logB/ULP感度と構造workを観測し評価順を決める | logB大、event数大、旧proxy高は科学的FAILや最適候補の排除にしない |
| 2 局所signal | 全roundを要求し、anchor/初期/中間/終端から優先取得、shared cacheで残りを埋める | m12だけ通過/失敗しても全scheduleの成否は確定しない。未評価roundは残す |
| 3 N/RMSE・価格 | Bの同じpolicyで全schedule N/RMSE証拠、価格担当の全round C coverageを取得 | nominal N・有限cohort・未検証外挿を十分N/全law期待値へ格上げしない |
| 4 全費用比較 | B cost/selectで同一target/profile/policyの全round積和・不確かさを比較 | partial G、単一T、見た候補subsetから大域最適や方式優位を認定しない |

厳密なrejectは物理identity/SOURCE不一致、非整数・piece数未満のR、固定formula/law/sector違反など「指定された候補として成立しない入力」。数値gate失敗はNUMERICAL_UNRESOLVED、適用条件不成立はそのpolicyでUNQUALIFIEDとし、全手法の数学的不可能性へ拡張しない。

将来、同じRMSE・価格契約で**証明された**cost下界が適格なincumbent上界を上回る等の証拠があればbranch-and-boundに使えるが、今回その下界・上界は未取得。N_nominal×work proxyやB²を厳密下界にしない。経験的順位は保留queueの順番だけに使い、deferred候補・未探索領域・上側R境界を台帳として新設せず既存B coverageへ残す。

共通κ/R候補はBが受ける明示有限集合として渡し、任意の上限や減衰閾値を新設しない。整数化で同じ全physical scheduleになるκはmean取得を重複排除できるが、policy/envelope/M/SOURCEのconfigurationは別に保全する。K・Δ・Lの異なる短時間proxyだけのPareto支配で除外しない。価格fitの共分散、cross-round共有誤差、shotのselection biasもB/Cが未知として保持する。

## 7. 今回の最小実装と確認

`pr_rpe_signal_plan.plan`を追加した。callerがB正本のvalidator/exact_scalarとphysical対応表を渡し、全M+1roundのB identity/law・physical recipeを照合してjob要求を作る。新configuration ID・候補menu・N/G optimizerは生成しない。complete raw contextで一致した保存meanも「metadata上の再利用候補」であり、scientific acceptanceではない。

新7人工テストは、B3216の実際のmetadata関数で作ったMOCK全13round、L/formula/K変更時の旧mean非流用、M変更時のraw再利用候補とN分離、保存FAIL/欠測・duplicate、PR-Y4未対応、初期実幅・q_RPE/qPFを確認した。実H4のspecや新candidateを生成していない。旧V2接続10回帰を含め**17技術test PASS**。実dataの候補screenや数値solverは起動していない。

helperはuniform PR/S2又はM5の現在のB対応範囲を対象とし、H6 early overrideやPR-Y4・K横断optimizerは未追加。新PRのdirect-head finite-mean runnerは設計段階で、準備済みと偽らない。[有限科学計画](SCIENCE_PLAN.md)は2全scheduleの一因子差pilotを提案し、未定L/κ/誤差契約を勝手に埋めていない。

次：**ユーザー**が最初の有限menuと科学scopeを決定、**A**が必要なnumerical adapterを最小拡張・人工確認後に承認範囲だけ実行、**B**が全schedule RMSE/比較、**価格担当**が一致するcandidate/scheduleの価格coverageを返す。今回はcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。
