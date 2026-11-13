# 未承認：DF head＋Pauli tailの最小有限科学案

今回の実科学callは0。次の実行を自動開始しない。本案はSOURCE調査から作った条件付き案で、production adapterや新Pauli canonical configurationは未作成。[REPORT](REPORT.md)／[HANDOFF](HANDOFF.md)。

## 共通条件・候補

保存Linear H4/1.00Å/STO-3G/tol-only1e−8 DF10、weighted-Hermitian completion、sector(2,2)/36、同H/state/snapshot・保存DF-Hの同sector target、P0を保持する。h=.1/M12、S2/ΔPF=.1/K2、元exactκ・prior/decoder/RMSE誤差契約を維持する比較案である。

初回は**LD1とLD2のみを候補として提案**する。現A91349e70で同backendの全13round DF信号・reference・prefix descriptorがあり、headを変えず辞書変更を比較できるためである。Track Aの旧順位やLD7/8/9案から採択したものではない。新prefix・K0・κ変更・M5/Y4は初回範囲に含めない。ユーザーは最小のLD1だけへ縮小して承認してもよい。以下の上限は2prefixの場合。

## 前提となる技術差分（科学計算前）

1. 保存loaderのcompleted weighted blockのfull256 extensionを、同じSOURCE/演算順・mode順で取得するmatrix-only adapterを用意する。既存のone-body CAR、M1のΣλA²→JW、Pauli collectorを再利用する。裸Hamiltonian−headの別targetや任意の対称化へfallbackしない。
2. B管理のcanonical configurationへdictionaryの種類・係数/label hash・identity方針・mapper SOURCEを含める。旧DF IDをPauli IDへ再ラベルしない。Aは独立のID生成器を作らない。
3. A providerへsource-bound Pauli λ/scalarと検証済みtail matrixを供給する最小adapterを固定する。既存DF factory・生成済みDF配置は変更しない。B/C/Dへ新dictionary fieldとprice-law bindingを渡す仕様を固定する。

新科学SOURCEは上記adapterのcommitで記録する。承認証明書・新grant発行は不要。scopeを増やす変更が必要なら実行前に戻す。

## P0：operator/dictionary、2prefix

| 対象 | 新しい処理と件数 |
|---|---|
| 保存入力 | 既存snapshot/receipt/state loader 1組、同stateのfull256 lift 1、既存DF one-body＋10fragmentのfull-Fock extension 1組 |
| 分割 | LD1/LD2のhead/tail operator bundle各1、計2 |
| Pauli辞書 | prefix後の同じtailのglobal collect/JW展開2、λ_P・identity抽出各2。DF λ/scalarは保存descriptorをSOURCE付き参照 |
| 同一性 | full H=cI+head+tail、Pauli和と同tail、36次元保存block・state lift・共通s_Rを2prefixで照合 |
| sector | 全nonzero Pauli labelのbit作用で、保存basisからsector外へ出るかを各辞書で完全分類。合計tailと各head blockのsector closureも確認 |
| 新H/state生成・固有値・reference | SCF/DF fit/state solve/eigh/reference actionすべて0 |

辞書の実term数・各support分布は未取得なので上限を恣意的に設定しない。8qubitの全Pauli label空間は有限であり、非zero成分全体を対象にする。小係数を無断cutoffで落とさない。full-Fock identityと保存s_Rの差、imaginary係数・mapper内部cutoffの残差、norm/closureとSOURCEを出力する。係数和のlambda比較だけで科学的GOやコスト優位を判定しない。

## P1：限定mean（P0成立後、別途又は一括承認された場合のみ）

| 処理 | 最大件数・理由 |
|---|---|
| 共通κのschedule | 2prefix×DF/Pauliの4構成、各13roundのmetadata。新λはBの既存paper mapperへ渡す。実signal対象は下記二時点だけ |
| Pauli mean | m0/T=.1とm6/T=6.4を2prefixで評価：4 round-candidate。初期Rの整数clampと中時間の減衰を、保存DF値へ接続できる |
| fixed-R対照 | LD1/m0だけ、保存DFのR_outerと同じRを使ったPauliの代数対照を最大1候補追加。commonκ候補と完全一致なら重複計算しない。これを新κ scheduleの候補として採択しない |
| full256対36確認 | LD1/m0のfixed-R対照と同候補をfull256で1回評価。対応する36 meanは上記4＋最大1に含む |
| 合計 | 圧縮mean最大5、同候補full mean1：最大6 mean呼出し、raw/corrected/exact-tail最大18complex出力。S2 finite normalizer最大5 unique event-time/K組を共有 |
| 保存DF値の受入 | 2prefix×二時点＝最大4保存meanのSOURCE/数値gate受入診断。新DF mean0。候補・backend・scalarが一致しなければ欠測/STOPで、別Rの値を流用しない |
| reference | 保存2時刻の独立reference pairを両prefixで共有し、新reference0。全13roundの保存参照の所在もmetadataへ保持 |
| その他 | 新sampling/native build/compile、N/nominal N、MSE/decoder simulation、価格/Gは0 |

fixed-Rでは同A/K/countのcorrected mean一致と辞書依存B/rawを検査する。共通κではλ変化でR・event時間・finite-K biasも変化するため、それぞれを独立評価する。rawはcorrected/Bの事後代入で定義せず、既存providerと同じ独立matrix propagationを維持する。

full256が必要なのはoperator/sector証拠とこの1候補の照合である。個々のeventや将来のnative作用の参照もfull256で扱う。36次元meanはP0のsector closureとfull finite-law恒等式を確認した後だけ使う。合計のsector closure不成立を「微小」として無条件に捨てない。

## 数値条件・出力・STOP

既存A数値policyを基準に、演算子/sector/closure/unitarityの診断1e−9、保存reference pair1e−10、state norm1e−12を保持する。新full-Fock辞書生成の係数残差・丸めの内訳も事前SOURCEへ固定する。人工testの2e−14を実分子gateとして移植しない。係数虚部の処理や既存collectorの内部cutoffで1e−9 closureを満たしても、無条件の厳密同一やrigorous allowanceとは呼ばない。

出力はH/state/snapshotとstate lift、prefix/head order、Pauli labels/実係数/符号/p_j/s_R/λ/SOURCE、sector分類、dictionary hash、B canonical whole schedule、raw/corrected/exact-tail/reference、logB/attenuation・normalizer precision・独立closure、数値allowance、未資格を含める。N/MSE/C/Gはnull。

SOURCE/input/completion/scalar不一致、必須値欠測、非Hermitian・非有限、operator/sector gate失敗、normalizer丸め未解決、full/reduced不一致、独立raw closure失敗ならpartial保存後STOP。新reference、新full-space追加点、閾値緩和、候補変更、精度retryへ自動進行しない。新λが小さくないこと自体は正常な結果で、STOP理由や科学的NO-GOへすり替えない。

P0だけの承認でP1へ進まない。P0＋P1を一括承認した場合は成立条件を満たす固定scopeを追加の細切れ承認なしで実行できる。許可されていないP1が必要になった場合のみユーザーへ戻す。

## C/Dへ渡す次の最小価格案（今回・P0/P1には含めない）

P0後のlabel/support/order/sign分布から、Pauli controlled rotation・product/order phase・head境界を覆う既存builderの技術確認点をC/Dが選ぶ。最初の実短回路候補はLD1/m0のnormal-lawな1 pathのfull measured X/Y、最大2 compileである。seed/law/出力SOURCE・loweringを事前固定し、eventを結果で選び直さない。この1 pathは期待価格の推定・model fit用cohortではなく、構造counterとnative/compiled接続の確認用である。

既存P3/O1のPauli eventコードは再利用できるが、旧Full価格を新head/λ/Rの平均へ流用できない。人工D2と現D loweringの関係、headのnative規約・profile/domainをC/Dが固定し、必要な追加期待価格検証は別の有限scopeにする。巨大event列・長round回路を最初から明示生成しない。

## 判断できること

P0は同じ保存物理tailをPauli化できるか、λ/scalarとsector条件を判定する。P1は二時点の局所bias・attenuation・数値安定性とB受入に必要なrawを与える。全13roundのRMSE・十分N・G・winnerは判定できない。

次段階は全round signal/riskと同じ候補の平均compiled価格を揃えてBへ渡すこと。Full PFの71f04be7で得た強い局所候補に加え、最適化されたTruncated PFを比較対照として維持する。旧B0/prefix7や単一幅を無条件に「最適Truncated」と呼ばない。ユーザーによる有限科学実行scopeの承認までmandatory STOP。
