# 新A/B：科学レビュー用比較表

2026-11-06。根拠・式・固定SOURCEのlocatorは[REPORT.md](REPORT.md)に集約した。以下の「仮説」「最小検証」「STOP」はCodexの整理・提案であり、採択、正式な科学レビュー、新規性判定、実行承認ではない。

## 候補の比較

| 判断項目 | 新A：PR × controlled evolution削減 | 新B：PR × signal-aware randomization |
|---|---|---|
| 出発点 | 固定文献調査O5のreference干渉と既存phase保持builder | O2のsampling law・非加法価格、coherent mean、平均誤差解析 |
| 既知研究で成立する部分 | SE-QPEのreference/CSWAP方式と、そのcanonical QPE条件。詳細はREPORT §3.1 | importance sampling、HS平均誤差最適化、Parent/S4の有限配分。詳細はREPORT §4 |
| 今回のPRへ直接使える接口 | 順序付きDF factor、basis非制御、scalar/signの記録、実adjoint、X/Y wrapper | finite law、event係数とproposal、M₂/range、係数/角度/native誤差の分離 |
| そのまま使えない保証 | random halvesへの共通固有基底の仮定、理想N保存からnative近似への移植、旧compilerの新register構成への資格 | Haar平均の1/dを単一groundへ移すこと、generatorの不偏性からfinite PR meanの一致、channel/observable一致からcoherent signal一致 |
| 具体的な未検証仮説 | fresh-shot RPEの単一干渉blockで、trajectoryごとのvacuum phaseを補正し、強制御baselineよりfull費用を減らせる条件があるか | 取得可能な構造情報が、既知配分の変数を超え、coherent bias・測定分散・生成費を含めた総費用に有利な自由度を与えるか |
| 現在確認できる方法差 | PRへの適用条件・phase対応は調査対象。reference gadget自体は既知。具体的な新回路は未設計 | 新しいlaw・optimizerは未設計。同じmeanのproposal変更やq∝p/√Cだけなら既知IS。候補名だけで差はない |
| 検証可能性 | 固定factor familyのreference作用から始められる。DF physical basisとgeneric Pauliを分ける | 既存有限supportでmean/weightの対応は確認可能。ただしstate-awareな追加情報の取得法と有用な保証が先に必要 |
| 強baseline | diagonal/中央rotationだけの制御、basis共有、phase保持、同PF/合成精度。適合するFull/tail-free対照 | canonical PR、既知IS/Parent/S4/ALLOCATION、同groupingのHS/絶対係数則、同じwhole-path価格情報を持つ既知IS |
| 公平な情報・探索機会 | exact state供給のみ。candidateだけにcontrolled prep、exact E₀、長round価格、精度最適化を与えない | exact ground vector/path signal/gap、normalizer、価格oracleを無料供給しない。情報取得・探索機会を両armで揃える |
| 全費用への因果経路 | 制御gateの残余削減→追加reference/CSWAP/補正費→同じ信号または認証disk→半径/phase→整数shot→全round G_k | law/構造→同じまたは認証されたcoherent mean、M₂/range・per-shot価格→confidenceと整数shot→全round G_k |
| 主な不利益要因 | 追加register、CSWAP/routing、reference準備/reset、補正rotation、両laneのnative誤差、受理率 | normalization/range増大、希少path価格、生成/normalizer費、状態情報取得、有限bit bias、角度変更時の新資格 |
| 現段階で主張できる性能 | 数値的利益・全RPE優越は未評価 | 数値的利益・全RPE優越は未評価。Haar boundの低下自体は固定groundの改善証拠にならない |
| 今回の完了地点 | 静的対応とレビュー問いの公開 | 既知への還元条件とレビュー問いの公開 |

RZ、CX、T、depthは合算しない。追加qubitと古典費用を別記する。同じenergy精度・成功確率、元H ground、状態準備費除外という目的を共有するが、今回は新しい物理系・数値条件・round数を採択していない。

## 数学的な棄却条件と既知法への還元条件

| 条件 | 判断する範囲 |
|---|---|
| A：非零の対象信号を持つtrajectoryでreference overlapの絶対値が1未満 | path単位のphase-only復元は成立しない。平均の相関認証や重み変更は別方式として扱い、自動拡張しない |
| A：scalar、order/product符号、reference位相が不明、actual inverseでない | 単一blockのcoherent信号保証を主張できない。channelや物理H全体の対称性で補わない |
| A：random halvesが共通固有基底を持たない | canonical連続gadgetの既知保証は使えない。ただしfresh-shot単一blockの内積対応は別に判断する |
| A：native回路が粒子数保存から漏れる | 理想DF SOURCEのvacuum固有性だけでは不足。旧compiler Aの資格で新回路を認定しない |
| B：個別Pauliまたは同規格involutionのHS則 | 絶対係数samplingと同じになる。groupingを変えるなら既知groupingと実装費も比較する |
| B：固定eventをc/qで再重みし、価格とM₂だけを最適化 | 既知importance sampling/配分へ還元する。非加法Cだけでも新規性は生じない |
| B：同じmean・実weight・同じbinary Hadamard測定 | E[(WY)²]=E[W²]。pathの期待値だけの分散低下では、量子測定分散の改善を示せない |
| B：Haar平均だけ、support欠損、未知range/normalizer、候補だけexact signal | 固定groundでの公平なRPE比較は未資格。欠損を0や推定値で補わない |

これらは各限定方式・主張の条件であり、reference干渉やrandomization一般の科学的no-goではない。未検証であることも、新規性の根拠にはしない。

## 再利用資産と必要な差分

| 層 | 既存資産 | 将来必要になる差分 |
|---|---|---|
| finite PR / SOURCE | `rte.py`、`df_rte_tail.py`、`track_c_source_guarantee.py` | Aはreference作用と全path phaseの認証、Bはlaw変更後のmean/有限bit/重みの認証 |
| phase / circuit | `df_rte_qiskit.py`、`df_partial_s2.py`、Pauli中央制御、`pr_dictionary_validation/gates.py` | Aはlane・output swap・Y符号・actual adjointを含む新blockの照合。今回は実装なし |
| 配分 / native価格 | 訂正済み`parent_allocation.py`、既存`rte_reallocation`接口 | Bは既知法との非自明な差分と生成費の定義。新しい価格取得は別承認 |
| RPE / 誤差会計 | `rpe_resource_accounting.py`、C2-B補正済みdisk/shot会計の規約 | unit-radius/S2/±1測定の仮定を点検。同じ信号単位のdisk合算後にphase/radiusへ変換。新構成・weightの資格は未了 |
| 検査 | DF builder、repeated S2、finite signal、Parent/native、RPE会計の既存test入口 | 変更した仮定に必要な最小人工検査だけを選ぶ。旧科学runnerの再実行・mock PASSの性能認定はしない |

SOURCE版・関数・testの詳細と確認深度はREPORT §5。旧raw、grant、ledger、marker、STOPや保全snapshotは編集対象外である。

## 最小の追加検証案とSTOP

以下は**実行していない**。GPTの判断とユーザーの別途承認が必要である。

| 順序 | A | B |
|---|---|---|
| まずGPTが判断 | fresh-shot式の適用条件と既知SE-QPEとの差、phaseの取得可能性 | fixed-ground保証、既知ISとの差、利用する情報の公平性 |
| 最小人工確認 | 固定familyの非可換path・非固有target・scalar/符号/負時間でX/Y対応とreference作用を調べる | 差分が具体化したときだけ、小さいordered supportでmean、weight、実binary分散を調べる |
| 次へ進む条件 | full-space誤差と元H energyへの接続、正半径・統計予算・aliasを保証できる | 同じmeanまたは認証bias、有限range、law生成・情報取得費を保証できる |
| 科学的STOP | pathの単純復元不能、phase/inverse不明、誤差・追加費用の不足を解決できない | 既知法と同一、情報アクセス不公平、range/normalizationやcoherent保証の不足 |
| 資源比較へ進む場合 | 同じcompiler・精度・強control対照で初めて価格を取得する | 同じ候補・価格・alpha等の機会をbaselineに与え、生成費と整数shotを戻す |

A/B共通で、正しい信号だけでも安いprimitiveだけでも採択しない。必要な入力や保証が欠けた段階では未資格として止め、候補追加・探索拡張で改善を探し続けない。C2や旧研究のSTOP解除はこの表からは許可されない。

## GPT／ユーザーへの引継ぎ

**GPT**が上記の数学的適合性、既知法との実質差、最小検証に進む科学的理由を判断する。**ユーザー**が候補、条件、資源指標、追加実装・科学実行の許可範囲を選ぶ。現時点でCodexによる自動後続作業はない。

**2文書の公開照合後mandatory STOP。C2全体の一時保留と既存STOPを維持する。**
