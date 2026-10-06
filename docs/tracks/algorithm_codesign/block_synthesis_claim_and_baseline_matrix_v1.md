# block合成：claim・強い対照・scope表 v1

2026-10-06 JST。受領GPT reviewを仕様へ戻すためのscoped比較。
入口は[数学・実装仕様](block_synthesis_design_review_20261006.md)。未実証の方法候補で、性能/新規性の採択ではない。
今回のCodex確認は下記primaryの該当定義・節に限定し、引用網全体の不存在証明や全証明の再検証を行っていない。

## claim単位の差分

| primaryと確認箇所 | 既知として除外するclaim | 次Bが追加し得る具体知見 | 今回主張しないもの／未解決 |
|---|---|---|---|
| [Sparse PS v2](https://arxiv.org/html/2402.15550v2) §II.1–II.3、III.1/III.4 | channel辞書のsigned分解、l1最適化、有限残差との交換、累積sampling burden | 同一辞書/精度でのfinite-RTE平均operatorの取得・実装条件 | block化/最小gammaだけの新規性。既知solver/reuseにも同自由度を与える必要 |
| [TE-PAI v2](https://arxiv.org/html/2410.16850v2) Appendix A | unitary matrix分解とsuperoperator分解の違い、controlled測定との違い | coherent taskに揃えた両制約の資源差を切り分ける | 再現対象の違い自体の発明。直接observable測定の数値をcoherent信号へ転用しない |
| [Granet–Dreyer](https://arxiv.org/pdf/2308.03694v2) small-angle式(1)–(3)、amplitude/shot記述 | 大角度rotationとidentityの平均による小角度unitary、attenuation/測定交換 | 非可換finite blockで実装可能な辞書生成が何を付加するか | 単一回転first-moment補間を新手法としない |
| [PR v2](https://arxiv.org/pdf/2503.05647v2) Appendix A.1–A.3、E.3(a) | Hadamard信号のunitary平均、RTE normalization、rounding-to-residual | finite cutoff/phase/接続誤差を保持した短blockの比較 | 平均RTEという上位構想の新規性、exact infinite RTEとfinite cutoffの同一視、DF優位 |
| [Resource-optimal IS v1](https://arxiv.org/html/2603.13495v1) §II Eq.(7)、Theorem 1 | expected cost×worst-case moment、q∝p/sqrt(C) | range/bias/整数shots/contextを共通化したfinite-confidence適用条件 | importance sampling式の発明、同式の有限confidence最適性、C=0特異caseの無視 |
| [Positive synthesis v1](https://arxiv.org/html/2510.05816v1) Problem 1.1、§4 Problem 4.1/4.2 | 単一qubit channelの正の混合と最大branch T最小化 | joint-space/phaseとouter correctionを保持したtask比較 | whole-wrapper期待T×shots最適性、system channelを後からcontrolする操作 |

Granet–Dreyerの出版社page/指定HTMLは該当本文を取得できず、公開arXiv PDFで確認した。
取得時PDFの印字はv2。versionlessから得た本文をimmutable bytesの再hash検証とは呼ばない。
PRは指定v2 PDFの本文を確認。HTML失敗を全文確認成功とは数えない。全文コピーや文献PDFのrepository追加は行わない。
Campbell/Structure-Aware Variance Reductionは[受領計画](inputs/sp1_post_run_gpt_research_plan_20261006.md)の背景参照を保持し、
今回のCodex独立全文確認には数えない。

## 同じtaskへの比較条件

層Aは同じM・辞書・許容residual・情報access・control/phase/workspace・取得budgetで原因を分ける。
層Bは同じexact H/time/state access/ε/αのtask全体比較であり、最初の小型pilotでは未達。
同じMの比較とfull-channel制約の比較は別tagで保存する。強い制約からの改善は制約緩和として数える。

| 必須対照 | 同じtarget・精度に戻す方法 | 分離する寄与／残る条件 |
|---|---|---|
| 現RTE＋通常合成 | 同じfinite M、outer correction、全体biasの中でprecisionを選べる | 旧native 10^-6を唯一の基準にしない。実gate列/workspace/source固定は未完 |
| gatewise PAI/Sparse PS | native joint rotation/controlled実装、同一dictionary policy、取得budget | 独立積。atom結合・phaseをbaselineにも与える。PAI/Sparseの具体採用profileはreviewで一つに固定 |
| 正のprobabilistic mixture | original bω/Bを保持し、joint native channelまたはcontrolled blockへ適用 | 符号overheadを避けて有限biasを使う比較。nonunitary Mの無補正convex unitary再現を要求しない |
| block full-channel Sparse PS | T_Bを再現し、そのcoherent cornerと外部補正を評価 | block化とfirst-moment制約緩和を分離。小型dense oracle/SDP費用を保存 |
| 標準operator LCU | candidateと完全に同じM/dictionary/residual/q/atom merging | 係数差がなくても取得時間/memory/reuse差を評価。そこも同じならnew-method claimなし |
| Pauli-LCU | 同じMのphase付きPauli表現、同じprep/context/shots | T=0部分回路の退化。shots/Clifford/CX/workspace/classicalも併記 |
| controlled unitaryの全体再合成 | unitary event/単一rotation controlだけ。phaseを保持 | 融合により旧逐次合成が弱くないか確認。nonunitary Mの単一unitary合成とはしない |
| PR rounding-to-residual | DF/chemistry比較へ進む場合、同じexact taskを再定義して導入 | 現小型M比較へ無理に追加しない。DF優位claim前に必要性とsourceを閉じる |

正のmixtureで単独controlがfeasibleでない場合はdictionary内非適格として保存する。
一般的なpositive synthesis不可能性や、signed methodの新規性の証拠へは変換しない。
標準最適化との同じ答えは正しさと整合し、method成立/不成立の単独条件ではない。

## 過去STOPとの差分と非claim

| 固定STOP | 今回の変更対象 | 復活させないclaim |
|---|---|---|
| P-D/R3/B-F | PF係数やoracle selectorでなく、coherent対象とphase付き実装表現 | finite correctionがbest係数を変える、同情報selector優位を名称変更で再開しない |
| B-M現adapter | compact BCH scoreの改良でなく、operator合成の意味論/費用 | 同backend代数的同値をnew methodと呼ばない。共通reuseを対照から奪わない |
| SP-0.5/SP-1 | 開封済みprimitiveを証拠境界として保持し、別targetの仕様を提案 | cheap-notch/crossover自体の新規性、actual RTE interaction、D/R役割の一般優位 |

今回の準備で得たものは比較仕様だけ。no science/no new algorithm adoption、mandatory STOP。
geometry/precision/dictionaryを増やし続ける探索、Track AのevidenceをBのheld-outや新結果とする処理を行わない。
