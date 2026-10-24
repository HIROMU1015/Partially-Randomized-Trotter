# 研究B：B4乱択PF・B2 Corrected／Processed PFのPR接続可能性 — 限定技術調査

作成日：2026-10-24 JST。区分：一次文献照合・静的SOURCE調査。科学的GO、独立新規性、性能順位、同精度総資源優位は判定していない。

**主要な確認結果。** B4のstandard accessでも、分割した和全体のexact指数が必要である。現在のPRの個別DF fragment指数・決定論PF・有限RTEだけでは、一般の非可換な `H_D`／`H_R` に対してその前提を満たすと認定できない。B2には既知のoperator誤差改善と再利用できるprocessed PF SOURCEがあるが、相似変換によるkernel固有値の改善、D側だけの処理によるfull PRの改善、全shotを通じた無料processorのいずれも成立済みとは扱えない。

## 1. 範囲、固定起点、取得資料

対象repositoryは `HIROMU1015/Partially-Randomized-Trotter`。起点は公開済みの[初期調査][INITIAL]、commit `0dae2515bfb1e12dceacabd7be4e70a1a9df2924`。同commitから独立worktree・新規branch `codex/research-b-b4-b2-pr-connection-20261024` を作成した。本書だけを追加し、他branchとのmergeは行わない。

正式共通指示 `research_project_instructions_unified_2026-10-23.md` はローカル実体を取得済みで、前作業の全文確認と同じ版であることをSHA-256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` で再確認した。`PROJECT_MAP.md` を入口とし、前作業で全文確認した `docs/research/研究概要・現状.md` 等が初期調査の前後で不変であることを確認した。取得済み資料の再読・再監査を広げず、初期調査のB4／B2の未解決条件を追跡した。

| 実際に参照した資料 | 固定版・確認範囲と限界 |
|---|---|
| Kim–García-Pintos, *Randomized product formulas beyond optimal deterministic scaling* | [arXiv 2608.07720v1、2026-08-07][B4]。Thm.1/2、standard／mixed定義、SM S1.2、S2.1–2、S3.2–3、S4の関連箇所。証明を新たに導出・再証明していない |
| Bagherimehrab et al., *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas* | [arXiv 2409.08265v4、2026-07-28][CPF]および[出版社PDF、PRX Quantum 7, 033018 (2026)][CPFPDF]。§II–V／VIIIの関連定義。出版社紙面033018-5／6を画像で直接確認し、Eq.(2)/(9)を照合。全頁読了・全式監査ではない |
| 別PF repositoryのpublished PF実装・検証 | `20d64c2fea1ed2d2188777df4f3d962ee94df486` の [product_formula.py][PFSOURCE]、[processed_cost.py][PFCOST]、[pf_decomposition.py][PFDECOMP]、[H2 validator][PFVALIDATOR]、[response README][PFOLD]をGitHubから取得。保存値の再fit・再集計なし |
| PR実装・tests・検証文書 | 本repositoryの起点 `0dae2515…` の §3、§6記載SOURCEを静的に確認。testsの存在・assert対象を確認しただけで今回実行していない。`VALIDATION_STATUS.md`／manifestの既存資格を変更しない |
| 過去の終了・監査・研究横断資料 | 初期調査の固定参照を再利用。B-F／B-M、THRIFT型mixed access、有限RTE、Gaussian singletonの必要箇所に限定。初期調査が記録する未取得・未監査部分を取得済みに格上げしない |

本書の「不足」は、確認したSOURCEと証拠では認定できないという意味であり、一般的な不可能性定理ではない。記号は `U(t)=exp(-itH)`、`H=H_D+H_R`。B4論文の `H=A+αB` との形式的対応候補は `A=H_D, αB=H_R` だが、これを新しい分割の採択とはしない。`r_PF` はB4の外側step数、`r_RTE` は既存PRのtail分割数、`q` はPRの反復数として区別する。

## 2. B4：保証と必要アクセス

### 2.1 exact指数、乱数、signed time

論文は比較可能なnormのHermitian `A,B` と小さい `α` を扱う。小さいtailのranking proxyが存在するだけでは、そのscale分離・operator norm条件を確認したことにはならない。以下のexponentialは、論文の理想primitiveである。[B4主文・SM S2/S3][B4]

| 項目 | 論文が要求するもの | 現PRへの含意 |
|---|---|---|
| Standard access | 任意の実数 `s` に対する `exp(-isA)` と `exp(-isαB)` | 各々は**和全体**の指数。`A=ΣD_j` の個別 `exp(-isD_j)` が使えることとは異なる |
| 基本乱択step | `R1(τ,u)=exp[-i(1-u)τA] exp[-iατB] exp[-iuτA]`、`u` は `[0,1]` の一様分布（Eq.(2), S19） | 挿入時刻を乱択する。既存RTEのTaylor次数・component samplingと同じlawではない |
| Standardの対称化・高次再帰 | `R2(τ,u)=R1(τ/2,1-u)R1(τ/2,u)`。同一step内の再帰全体で同じ `u` を使用し、外側step間は独立 | 半stepごとの独立drawへの置換は論文の構成を変える。seedがあるだけでは共有条件を満たしたことにならない |
| 負時間 | 高次Suzuki再帰は負stageを含む。SMは負 `τ` の積分方向も扱う | `s` を絶対値に置き換えない。負角、順序、実現列のadjointを区別する |
| Mixed access | `B=ΣB_l` に対して `exp(-isA)` と **`exp[-is(A+αB_l)]`** | 別々の `exp(-isA)` と `exp(-isαB_l)` では供給できない。DF fragment単体が可解でもmixedな和は別問題 |
| Mixedの構成要素 | `W_l([x,y])=exp(iyA) exp[-i(y-x)(A+αB_l)] exp(-ixA)` と逆演算、ランダムなgroup-commutator補正 | mixed和の実装、反復補正列、signed intervalが必要。旧THRIFT templateの単価を無料としない |
| Mixedの乱数 | dyadicな補正label `ω` とno-correction branch。同一対称化・高次step内で同じ `ω` を再使用、外側step間は独立 | 負時間のfresh drawは、元の実現列のadjointとは限らない。高次用lawはSM S3.3のorder依存条件を要する |
| 可変長 | S3.2.4 Eq.(S96)は `L≥2` の補正lawで、補正回路長の有限平均・無限二次momentを記載 | 期待指数数は最大長・高確率実行時間・測定varianceの保証ではない。有限cutoffはtail biasを生む |

Mixedのbase lawの `1<β<2` を高次一般へそのまま使わない。SM S3.3 Eqs.(S97)–(S99)は、次数 `2k` ごとに `1<β_k<1+1/(2k+2)` を指定する。ここでは既存lawを識別しただけで、PR向けsamplerやcutoffを設計していない。S4の特定model・`L=2` の切捨て実験を、任意DF tailの費用・biasへ転用しない。[B4 SM S3.2.4/S3.3/S4][B4]

### 2.2 channel、first moment、controlled phase

| 保証対象 | 一次資料の評価 | 接続できる範囲／残る条件 |
|---|---|---|
| Standardの平均channel | Thm.1：固定次数 `2k` でdiamond誤差 `O(α² t^(2k+1)/r_PF^(2k))` | exact指数と所定の乱数構造が前提。個別trajectoryのoperator誤差保証ではない |
| Standardの平均unitary | SM S2.1 Eqs.(S14)–(S17)、S2.2 Eq.(S23)・後続の反復評価は、`E[V]` と `U` のoperator normを直接扱う | system channelから逆算した主張ではない。独立stepと、exact unitaryの平均のcontractivityが反復評価を支える |
| Mixedの平均channel／unitary | Thm.2はdiamondで `O(α³ t^(2k+1)/r_PF^(2k))`、固定次数で期待 `O(L)` primitive指数／step。SM Eqs.(S91)–(S92), (S102)は平均unitaryを直接評価 | mixed access・order依存law・共有乱数等を満たす理想構成の保証。native費用でも有限RTE版保証でもない |
| Controlled coherent signal | 論文が指定する指数列そのものの位相で平均unitaryが評価される | `diag(I,V_ω)` を同じ位相で実装すればfirst momentを照合する根拠になる。system channelだけの一致では足りない |
| 位相からenergyへ | 上記はoperator／channelの近似対象 | 固定状態の信号半径、位相の安定性、状態overlap、RPE失敗確率を別に要する。ground-state energyの改善保証へ読み替えない |

根拠は[B4の主定理と補足][B4]。SM S1.2のmixing lemmaは平均unitary誤差からchannel誤差を評価する向きであり、逆向きにcontrolled相対位相を回復するものではない。`V_ω` ごとのglobal phaseを後からfitして消すと、system channelは同じでもcontrol ancilla上の信号が変わる。scalar Hamiltonian成分とRTE event phaseを含め、文字通りのoperatorを照合する必要がある。

### 2.3 内部指数の近似化で未確立になるもの

以下は既存定理と現SOURCEの適用範囲の比較であり、新しい誤差boundを導出していない。

| 置換・有限化 | 発生する接続課題 |
|---|---|
| `exp(-isA)` をD側PFに置換 | 内部PF biasが追加される。S2で用いる `α=0` のexact `A` evolutionと一次の `α` cancellationを、そのまま援用できない。外側の `α²` scalingの保存は未確立 |
| `exp(-isαB)` を有限RTEに置換 | 個々のeventはunitaryでも、そのraw meanはattenuatedな有限近似、補正後meanは有限多項式である。exact指数と同一ではなく、Taylor bias・normalization・反復順序を別に扱う |
| Mixed指数をPF／RTEで近似 | mixed sumの内部誤差とcorrection列での反復が加わる。所定の `α³` cancellationが保存される保証は未取得 |
| 連続 `u` の有限bit化・mixed lawの有限表現 | 実装された分布／時刻の平均と理想ensembleが異なる。有限shot誤差、回転合成誤差、分布離散化biasは別の項目 |
| normalizationの導入 | 理想B4の外側lawは正の確率でunitaryを選び、PRの有限RTE補正係数を要求しない。内部RTEを導入した版は、signed durationごとのnormalizationと実際のsample lawを別途照合する。従来の一個の共通係数を自動流用しない |
| 信号補正とshot | attenuated raw signalと補正後推定量を分離する。補正後meanにexact mixed-unitaryのcontractivityを仮定しない。weightのrange／moment・必要shot数は未評価 |

S4.1の数値ensemble評価にはquadratureが使われるが、安定した数値桁は有限bit samplerのbias certificateではない。S4.3は各executionで独立に回路を生成するbounded observable測定を扱う。これはPRのweightedなcoherent信号と同じ統計条件を自動供給しない。[B4 SM S4][B4]

## 3. B4：既存PR SOURCEとの対応

以下のSOURCEは全て本repositoryの起点commitの版である。機能を静的に読むことと、B4 adapterを実装・検証したことを区別する。

| SOURCE・既存接口 | 再利用できる資産 | B4に不足する部分 |
|---|---|---|
| [df_partial_randomized_pf.py][SPLIT]：`DFFragmentPartition`、`split_df_hamiltonian_by_ld`、`select_df_h_d`、`build_df_hd_trotter_blocks` | D/R所属、固定順序、one-body／scalarを含むD block、入力identity | 複数の非可換fragmentを一つにしたexact `exp(-isH_D)` ではない。rankingの `lambda_r` はinvolution tailのexact L1とは別で、norm比の証明にもならない |
| [df_partial_s2.py][S2]：`prepare_df_partial_s2`、`make_df_partial_s2_step_request`、`QiskitDFPartialS2CircuitBuilder` | Gaussian basis変更、個別DF指数、ordered D sweep、tail、scalar／identityのcontrolled phase | `pf_label='2nd'` のpartial-S2専用。forward／reverseのD半stepは互いのinverseではない。`H_D` 全体・`H_R` 全体のexact指数、B4の乱択挿入時刻列は供給しない |
| [rte.py][RTE]：`RTEConfig`、`finite_rte_distribution`、`iter_sample_rte_events` | signed time、既存paired Taylorのevent分布、component順序・位相、seed契約 | `u` 一様law／dyadic correction lawではない。exactなsymbolic tail抽出は、tail指数のexact実装を意味しない |
| 同module：`event_unitary`、`exact_enumerated_event_mean_operator`、`finite_rte_operator_moments` | 小規模参照、raw mean／corrected mean／normalizationの区別 | 古典dense参照を量子計算上の安いoracleに数えない。新B4列のmean保証はない |
| 同module：`compose_finite_rte_occurrences` | 異なるsigned time等の有限RTE occurrenceをorderedに合成し、各normalizationを記録する参照処理 | D interleavingやB4乱数共有を含む汎用controlled builderではない。既存機能の存在だけで新列への適合を認定しない |
| [df_partial_s2_repeated.py][REPEAT] | 同一準備・step幅・RTE設定の `q` 回反復、step seed、境界共有、attenuation metadata | 既定の共通stepから、ランダムなsubduration・高次再帰・共有 `u/ω` に変える接口はない。隣接block共有で残差境界が消えるわけではない |
| [finite_rte_signal_validation.py][SIGNAL] | exact physical、ideal-tail partial-S2、corrected有限RTE、attenuated meanの比較対象を分離する既存検証経路 | `exact_tail_operator=expm(…)` は古典参照。B4に必要な実装可能なexact tail accessではない。既存の共通step／`q` の検証を新ensembleへ移せない |
| [df_rte_qiskit.py][DFRTE]、[pr_dictionary_validation/circuits.py][CIRCUITS]、[gates.py][GATES] | event位相とscalarのancilla相対位相、実現回路の `actual_adjoint`、Hadamard wrapper、位相をfitしないcontrolled照合 | dictionaryのDは固定synthetic modelの可換occupation演算。分子DFの非可換D全体には拡張できない。wrapperは既存context／列を受け取るもので、任意B4列のadapterではない |

関連testsも静的に確認した。`tests/test_rte.py` は負時間の列挙mean、signed product phaseとrotation角の分離、heterogeneous occurrenceのnormalizationを扱う。`tests/test_df_partial_s2.py`／`test_df_partial_s2_repeated.py` はdense reference、controlled zero branch、scalar、signed step、既存境界最適化等を扱う。`tests/test_finite_rte_signal_validation.py` は既存signal／operator meanの比較を扱い、`tests/management/test_pr_dictionary_compile_phase.py` は既存compilerのphase修正を対象とする。**今回これらを実行しておらず、B4／B2を検証するtestsとしても認定しない。**

**B4の最重要判断への回答：現在の一般的なDF-PR構成について、前提充足は確認できない。** 形式対応 `A=H_D, αB=H_R` に対し、standardでも両方のexactな和指数が不足する。mixedではさらに `H_D+αB_l` 全体のexact指数が不足する。`L_D` の端点、tailゼロ、あるsynthetic Dの可換性だけから、一般の分子DFについて解決済みとはしない。必要なのは、対象を特定した実アクセスの証拠、またはinexact access下での別途正当化である。どちらも本書では新規設計・証明していない。

## 4. B2：何が改善され、何が同じままか

### 4.1 operator、processed evolution、kernel、固定状態signal

本節の `K` はprocessed PF実装で反復する**unitary kernel**を表す。CPF論文は `log S` をkernelと呼ぶため、同じ記号でも対象が違うことに注意する。[CPF §II][CPF]

| 区別する対象 | 確認できること | PRで未成立の読み替え |
|---|---|---|
| Correctorによるoperator近似 | CPFはcommutatorに基づき、correctorを含めたoperatorの `exp(-itH)` への近似を改善する既知法 | kernel固有位相、固定状態signal、最終energyが同じ倍率で改善する保証ではない |
| Full processed evolution `P K^q P^-1` | processorを含めて物理operatorと比較する。固定Pと実際のinverseを隣接stepで相殺できる場合は端点化できる | kernelだけの状態overlapをfull evolutionのoverlapと呼ばない |
| Kernelの固有値・固有位相 | 相似変換 `P K^q P^-1` は `K^q` の固有値を変えない | processorを既存kernelに付けるだけでその固有値誤差が小さくなるとは言えない。processed familyのkernel自体の設計と区別する |
| 固定physical入力状態でのcoherent signal | `⟨ψ|P K^q P^-1|ψ⟩` は、同じ `ψ` に対するkernel単独の信号と一般に異なる。固有状態とのoverlapも影響を受ける | signal変化をkernel固有値改善と同一視しない。無料のexact ground stateやprocessor済み入力を候補だけに与えない |

**Corrector三種とMPFは別の構成である。** Symplecticの `e^C S e^-C` は相似変換で、同じCなら端点相殺の対象となる。Symmetricの `e^C S e^C` は相似変換ではなく、interior correctorを同じ理由で消せない。Compositeは両方を含み、型ごとに費用を分ける。MPFは複数operator近似の線形結合であり、単一のprocessed積でも、各armのchannel平均でもない。[CPF Eqs.(4)–(6)][CPF]、[既知MPFの初期調査上の位置付け][INITIAL]

CPFのperturbed系では、Prop.1／§III.Cが一重Bの誤差をcorrectorで扱い、二重B項などが残るoperator評価を与える。Prop.1記載の局所評価は `O(α²|λ|³ + α|λ|^(2k+3))` で、ここでkはcorrectorの項数、実時間では `λ=-iτ` である。ただし短時間収束、正しい定義のPF、correctorの合成とその精度が前提であり、次節の表示不整合を推測で埋めて実装仕様にはしない。§VIIIには弱いpartitionやさらにnested PFで各指数を近似する方向への言及がある。これは現PRの確率的・有限RTE置換を認定した保証ではなく、generalな非可換 `H_D` のexact accessも供給しない。[CPF Prop.1・§III.C/V/VIII][CPF]

### 4.2 初期調査で保留したEq.(9)の紙面照合

取得した**出版社PDFの紙面画像**でも次の表示を確認した。

| 紙面位置 | 表示・定義 |
|---|---|
| 033018-5、Eq.(2) | `H=A+B` の通常の `S2(λ)=exp(λA/2) exp(λB) exp(λA/2)` |
| 033018-6、Prop.1、Eq.(9) | `H=A+αB` と述べつつ、中央指数が **`exp(αλB/2)`**、両端は `exp(λA/2)` |
| 033018-8／9、Eqs.(24), (29)–(30) | 通常PF2のlogと、perturbed系の `λ(A+αB)` を先頭項とする説明 |

出典：[出版社PDF][CPFPDF]、[同v4 HTML][CPF]。前回のHTML／text抽出に限った問題ではなく、紙面の表示にも不整合が残っていることまで確認できた。**著者が意図する訂正式・正誤表は取得しておらず、Eq.(9)を推測で修正しない。** literalなEq.(9)を保証付きPR実装の仕様に使わない。一方、この表示だけで他節の構成・CPF理論全体を否定もしない。将来採用する具体式と保証の版を確定するための保留である。

## 5. B2：Dのみ、full evolution、線形結合の接続条件

| 接続先 | 改善の対象になり得るもの | 元のphysical signal・残差境界・追加操作 |
|---|---|---|
| `H_D` のみをCPF／processed PFで近似 | D内部の非可換PF誤差、D近似を含むoperator誤差の一部 | D/R分割誤差と有限RTE biasは残る。D processorの間に残差が挟まるため、full列の一組のconjugationや端点相殺を仮定できない。残差をまたぐ同値性、追加basis操作、phaseの確認が必要 |
| Full time evolutionを固定processorで処理 | 適切なkernelとprocessorを含む物理operatorの近似、固定状態overlap／signal | full kernelが元の `H` を近似すること、Pとinverseの実装、入力／測定との対応が必要。既存のpartial-S2へPを追加するだけで既知CPFの誤差改善が継承されるわけではない |
| Full kernelに確率的tailを含める | どのordered first momentを近似するかを指定できれば、検討対象は定義できる | exactな固定Pでも既存kernelの同値なconjugationと改善保証は別。P自体も確率的ならfreshな「負時間」sampleをinverseとは扱えず、実現列と平均対象の両方の照合が必要 |
| 複数近似の線形結合（MPF等） | 係数付きoperatorまたは線形coherent信号のbias | 線形信号のclassical推定にはweight／normalization／shotを、coherent operator実装にはLCU等の追加accessを要する。observableのconjugation結果の線形結合を、unitaryの線形結合と混同しない |

この表は既知法と既存列の意味論からの接続条件であり、新しい残差変換、corrector、random processor、LCU手順を提案・実装したものではない。各armが現finite targetを保つか、物理 `exp(-itH)` へのbiasを改善するかも別の判断である。現在のfinite targetに一致するだけでは同じ物理精度の改善を主張できない。

## 6. B2：SOURCE再利用とprocessor／corrector費用

### 6.1 別PF repositoryにあるもの

全て `20d64c2…` の静的確認であり、現PRの同名moduleと同一版とはしない。

| SOURCE・関数 | 確認した実装上の役割 | 再利用の条件・限界 |
|---|---|---|
| [product_formula.py][PFSOURCE]：`morales_2025_y8m10b_list`、`morales_2025_yp8m8_kernel_list`、processor係数／列 | published non-processed Y8m10bとprocessed YP8m8をlegacy Moralesと区別して保存 | 係数を再探索せず再利用できる。丸め、列順、適用するHamiltonian blockと実時間規約は照合が必要 |
| 同：`_get_kernel_s2_sequence`、`_get_processor_s2_sequence`、`_get_s2_sequence`、`inverse_s2_sequence` | kernel／processorを分離し、full列をprocessor＋反復kernel＋逆列で返す。逆列は逆順・符号反転 | 一般の非対称processor列を `_get_w_list` と対称列の仮定へ押し込まない。append順と行列積順を既存規約に合わせる必要がある |
| [pf_decomposition.py][PFDECOMP]：`iter_s2_sequence_steps` | 明示S2列を展開し、隣接する同じtermをmergeする | 全termの自由な並べ替えではない。DF basis／RTE／scalarを含むPR境界で同じmergeが許されるとは未確認 |
| [processed_cost.py][PFCOST]：`ProcessedCostComponents.total` | `kernel_steps` と **明示的な `processor_pair_count`** を受ける加算的cost model | staticなPauli rotation／RZ layer表と境界mergeを使う。PRのnative compiled cost、RTE event費、shot総費用の関数ではない |
| [validate_morales_y8m10b_h2.py][PFVALIDATOR]、[response README][PFOLD] | grouped Pauli H2/STO-3Gに対するdirect固有値とphase-rotated overlapの既存検証。validatorの時刻窓は `.45≤t≤1.20`、overlap側にはfull processed列を使用 | 分子DF split／`L_D` は非該当。PR RTEでもB4でもない。保存係数・費用値を再fit・再集計せず、原taskの局所証拠として保持 |

READMEの既存結論も、固有値ならkernel、overlapならfull処理を区別し、端点費を加算で扱っている。旧legacyだけとの比較を強いbaselineとはしない。保存された短時間fitや有限時間での零交差を、新しいPR設定の係数・運用窓・総費用へ移送しない。[既存PF response][PFOLD]

### 6.2 何回どこで支払うか

下表は費用**項目と相殺条件**の整理であり、新しいgate数・倍率・費用予測を生成していない。

| 場所・条件 | 費用と確認すべき境界 |
|---|---|
| 一つのcoherent executionで固定Pを用いてkernelを反復 | kernel費は反復数に応じて必要。隣接する実際の `P^-1 P` が相殺するならprocessor一組を両端で支払う。CPF Eq.(6)の前提に対応 |
| Symmetric／composite corrector | 同符号のinterior correctorはinverse対ではない。symplectic部分だけの端点化と区別し、step内／step間に残る操作を数える必要がある |
| step幅、order、split、Pが変わる | 境界のPが同一とは限らない。各roundの異なる制御powerを一括して一組と数えない。構成上の同値性が必要 |
| PRのshot／sample、X/Y測定 | 既存Hadamard wrapperは入力準備・evolution・readoutを持つ一execution。独立なshotで準備をやり直すならprocessor端点も各executionの費用。classicalな回路再利用は量子操作の支払いを消さない |
| RPE round間 | 状態の再準備・測定を伴うround間の相殺は既定で存在しない。異なるround／sampleを通した償却には、その実行構成自体の根拠が必要 |
| Controlled化 | `controlled_gate`／Hadamard wrapperの相対位相を維持する。Gaussian basisを全てcontrolするか等は既存providerの条件に依存し、processorの無料controlled accessはない |
| 合成誤差とinverse | CPF §Vのcorrector合成はA/B指数からの有限精度構成であり、無料のexact commutator oracleではない。隣接相殺に使うinverseが実現列のactual adjointか、別近似なのかを分ける |
| 正負時間と有限RTE | 正負の回転・列順・global／relative phaseを支払う。逆Hamiltonianのfresh draw、corrected meanの逆、実現unitaryのinverseは別。内部RTE bias／normalization／合成誤差は別項目 |
| 既存の回路共有・境界merge | 同一basisの隣接操作等、登録条件内でのみ適用。途中のtailやreadoutを越えて消去できるとはしない |

根拠：[CPF Eqs.(4)–(6), §V][CPF]、[cost SOURCE][PFCOST]、[現PRのrepeated builder][REPEAT]、[dictionary circuits][CIRCUITS]。`actual_adjoint` の再利用余地と、`hadamard_wrapper` のphase検査資産はあるが、汎用processed PR wrapperは確認できない。現partial-S2 builderが高次labelを拒否する点も、係数表をコピーするだけでは接続が完了しない理由である。

## 7. 強いbaselineと、比較対象を揃える条件

| 判断したい対象 | 必要な対照・公平性 |
|---|---|
| B4のstandard accessでの効果 | 同じ `exp(-isA)`／`exp(-isαB)` accessを使える対称PF、適用可能な既知CPF／processed PFを対照に含める。候補だけにexactな和指数を与えない |
| B4のmixed accessでの効果 | 同じmixed oracleの下のTHRIFTと適用可能な高次既知法。oracle構築費・可変長tailも同条件。standard access armとの比較でmixed accessを無料追加しない |
| 現DF fragment accessからの改善 | 現partial-S2／finite RTEに加え、同じblock情報で適用可能なpublished Morales、Yoshida、YP8m8等。全SOURCEの移植済み認定ではなく、対照候補の条件を明記する |
| 乱択法との比較 | 初期調査にあるqDRIFT／高次乱択法等も、channelかcontrolled first momentかを揃えてから扱う。同じ「randomized」という語だけで同一taskとしない |
| Ground-energy総資源 | 初期調査のDF-qubitization等、同じ表現・入力情報で適用可能な強い外側baselineも除外しない。ただし今回その実装・価格比較へ範囲を拡大していない |

対照の出典・版は[初期調査 §§4/6][INITIAL]と[published PF実装][PFOLD]を引き継ぐ。B4のdeterministic lower boundは固定有限stage・Hamiltonian等に依存しない係数の最悪例であり、特定DF、processor全般、適応的法の不能定理ではない。[B4 Prop.1/2][B4]

比較では次を揃える必要がある。operator normとdiamond norm、平均unitaryと平均channel、coherent signalとkernel固有位相、ground-state energyとoperator作用誤差を別にする。finite target一致と物理時間発展精度、理想指数列とnative回路、1-shot費とconfidenceを満たすshot総費を別にする。exact ground state、processor済み入力、事前に全compileしたcatalog、係数最適化・cacheの情報費を候補だけに与えない。B4とB2の性能順位を誤差の次数やideal step数だけから決めない。

## 8. 過去の証拠・限定negative・未検証を引き継ぐ

本書は既存成果の**資格と適用範囲**を引用する。新しい数値結果・再ランキングはない。

| 既存証拠 | 保持する範囲と今回への含意 |
|---|---|
| [有限RTE signal検証][SIGNALDOC] | linear H4、1 Å、STO-3G、DF rank12・rank truncation 0、N=4、`L_D=0/1/3/6`、主検証 `δ=.1`、`q=1/2/4`、`r_RTE=1/2/4/8`、`K=0/2/4/6`。既存partial-S2のoperator、signal、半径、条件付きphaseを調べたローカル証拠。B4 ensemble、CPF／YP8m8、全RPE総費用の検証ではない |
| [B-F closure][BF] | linear H4、1 Å、STO-3G、固定DF12、generation-prefix `L_D=3`、`T=.8`、`q∈{1,2,4,8}` に対する `δ=T/q`、`Rbudget=5/10/20/40/80`、`K=2`＋登録triggerのK4。同5-stage対称4次family・同32評価budgetの保存cell内でfinite objective固有のdecision valueなし。旧中断INCONCLUSIVEと事後recoveryを混同せず、B4／B2一般のnegativeへ広げない |
| [B-M同値性監査][BM] | 固定nested列の形式三次係数を同情報compact BCH再帰と比較した既知法との同値性。分子・geometry・basis・DF rank／`L_D`・delta窓は非該当。新しい多時間尺度法やprocessed PFの一般否定ではない |
| [THRIFT型mixed-access終了][MIXED] | linear H4、1 Å、STO-3G、実RHF canonical MO、DF全16factor保持・切捨てなし、N=4、8 spin modes、PR split `L_D` は非該当。`T∈{1,4}`、`r∈{1,2,4,…,1024}` の2冪に対する `δ=T/r`、作用誤差εU=.001、固定7法。occupation算術／可変bit回転templateではknown費用に正の追加budgetがなく現templateを終了。全mixed原理のno-goでも、実backend費用の測定でもない |
| [Gaussian singleton正式handoff][HANDOFF] | 固定4-mode／2粒子のsynthetic density-square model、`L_D=1`、残差2fragment。分子geometry／AO basis／分子DF rankは非該当。F2は `T=1/5`、`q=1`、`δ=1/5` の一点、`r_RTE=2`、`h=1/10`、K0、固定Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／free topologyのwrapper。登録正しさと1-shot費の証拠は保持するが、有限bit γ・統計資格・認証N・同精度総資源は未成立。正式な一時終了を維持 |
| [Morales／YP8m8の保存検証][PFOLD] | grouped PauliのH2等における既存direct／overlap検査と限定cost proxy。分子DF-PR、finite RTE、B4との接続は未検証。今回数値係数・費用倍率を再利用してPRの予測を作らない |

固定起点の共有概要にはGaussian singletonの終了採択前の記述もあるため、その系列の状態は[正式handoff `69f8e1c…`][HANDOFF]に従う。本書は共有概要を書き換えない。[VALIDATION_STATUS][STATUS]／[manifest][MANIFEST]の既存 `source_present_no_current_ci` 等を、immutable CI／外部再現済みへ格上げしない。`C_use` は実行delta窓の経験的包絡であり、厳密上界ではない。H12係数は未決、QPE/RPE統計誤差とPF係数Cは別で、最終総費用評価は未実施である。

## 9. 研究A・Cへの連携と、統括へ戻す判断

| 担当 | 次段階の価値判断に必要な情報／判断 | 今回の境界 |
|---|---|---|
| 研究B | B4のどのaccess modelを対象にするか。B2の改善対象をD内部operator、full physical signal、kernel spectrumのどれにするか | 対象の選択・限定設計の採択は未実施。既知法の接続と独立した新時間発展構成を分けて判断 |
| 研究Cと研究B | 対象を特定したA/B／mixed和の実装可能性、DF blockとの対応、signed time・scalar・controlled relative phase、processorのnative合成と既存共有の適用条件 | exact和oracleを仮定で追加しない。新split探索、oracle設計、compile計測を開始しない |
| 研究Aと研究B | finite-bit law、signed weight／normalization、raw signal attenuation、moment／range、confidence・必要shot数の資格 | sampling law設計・新しい数値評価なし。B4外側の正のlawと、内部PR補正を分離 |
| 研究統括／ユーザー | 不足accessや保証の解消に投資するか。どの未解決科学問題にレビューが必要か。限定設計／科学検証を別承認するか | 本書だけで科学的GO、新規性、優位性を認定しない。詳細GPT科学レビューの開始承認にはならない |

**B4について次の判断を変える証拠。** 第一に、特定した対象Hamiltonian／partitionに対して、standardの二つの和指数、mixedなら追加の混合和指数が実際に供給されるという根拠が必要である。供給できない場合、inexactなprimitiveを使って何を保証するかは未解決として、理論検討を行う価値自体を統括へ戻す。次に、同じ位相のcontrolled first moment、step内の乱数共有・step間独立、有限bit／可変長tailの扱いを特定する必要がある。これらが未確定のままideal次数だけを根拠に科学検証へ進まない。

**B2について次の判断を変える証拠。** 使用するCPF式の版・Eq.(9)の扱い、processed列の順序／actual inverse、改善対象と物理信号の対応を先に確定する必要がある。D限定なら残差境界、fullならkernel／Pとinput・measurement、線形結合なら推定量と追加accessが明確であることが必要である。その上で、端点を何executionごとに払うか、内側biasと合成誤差をどう分離するか、同条件のpublished baselineが何かを示す必要がある。新しいcost値やsimulationを必須証拠として今回生成していない。

ここで挙げた証拠は将来の採否判断のための不足一覧であり、実験手順・科学call契約・新grantではない。研究B／統括／ユーザーが次の対象と科学レビュー要否を判断し、必要な限定設計・実装・科学実行をそれぞれ別途承認する。

## 10. 保全・公開・STOP

本作業は文献・静的SOURCEの閲覧と本書作成に限る。科学計算、保存結果の再集計／再ランキング、Hamiltonian生成、回路build／compile、科学tests、新しい保証・費用値の生成を行っていない。既存SOURCE・raw結果・grant・ledger・marker・共有概要・manifestを変更せず、旧STOPと別研究branchを保全する。元のdirty worktreeの変更を取り込まない。

本書のみを新規branchでcommit・pushし、remote commit SHAと取得文書bytesの一致を公開完了条件とする。公開確認後は**mandatory STOP**。新アルゴリズム設計、移植、実装、科学検証、詳細GPT科学レビューへ自動進行しない。次段階の担当は研究B／研究統括／ユーザーであり、Codexの追加作業は次の明示依頼の範囲に従う。

## 参照先

[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/docs/research/research_b_time_evolution_initial_survey_20261024.md
[B4]: https://arxiv.org/html/2608.07720v1
[CPF]: https://arxiv.org/html/2409.08265v4
[CPFPDF]: https://journals.aps.org/prxquantum/pdf/10.1103/phk5-slkl
[PFSOURCE]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[PFCOST]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[PFDECOMP]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/pf_decomposition.py
[PFVALIDATOR]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py
[PFOLD]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/README.md
[SPLIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/df_partial_randomized_pf.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/df_partial_s2.py
[REPEAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/df_partial_s2_repeated.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/rte.py
[SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/finite_rte_signal_validation.py
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trotterlib/df_rte_qiskit.py
[CIRCUITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trottertracks/pr_dictionary_validation/circuits.py
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/src/trottertracks/pr_dictionary_validation/gates.py
[SIGNALDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/docs/finite_rte_signal_validation.md
[BF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[BM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[MIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md
[HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
[STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/VALIDATION_STATUS.md
[MANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dae2515bfb1e12dceacabd7be4e70a1a9df2924/artifacts/validation_manifest.json

## 結論：各候補の3点

### B4

1. **成立している接続条件：** 理想exact accessと所定の乱数構造の下では、補足の平均unitary評価をcontrolled first moment照合の出発点にできる。既存PRには個別DF指数、signed event、位相・mean・normalizationの参照資産がある。一般DFのB4前提充足は未確認である。
2. **最大の未解決点：** 非可換な `H_D`／`H_R` 全体のexact指数、mixedなら追加の混合和指数が現アクセスから供給されないこと。有限PF／RTEへの置換で所定のcancellationと物理signal精度が保たれる保証もない。
3. **次段階の価値判断に必要なもの：** 対象を特定した実アクセスの根拠、またはinexact accessを正当化する検討に進む理由と対象。位相・乱数共有・有限bit・tail／normalization・同条件baselineの対応を揃え、研究B／統括／ユーザーが限定設計・検証の採否を判断する。

### B2

1. **成立している接続条件：** 既知CPFのoperator改善、full processed列とkernel spectrumの区別、固定P／actual inverseの端点相殺、published PFの係数・列・加算費用SOURCEを再利用できる。相似変換はkernel固有値を変えず、固定状態signalは別に扱う。
2. **最大の未解決点：** D内部とfull PRのどの物理誤差を、残差境界・有限RTE・制御位相を含めて改善するのかが未確定である。CPF Eq.(9)の紙面不整合も未解消で、processor費のshot／round単位の支払いを無視できない。
3. **次段階の価値判断に必要なもの：** 使用式・改善対象・full列と元physical signalの対応、境界／inverseの成立条件、endpoint会計と合成精度、同条件のpublished baseline。研究B／統括／ユーザーが科学レビュー要否と別途の設計・実行承認を判断する。公開確認後のmandatory STOPを維持する。
