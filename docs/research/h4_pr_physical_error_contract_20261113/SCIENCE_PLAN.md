# 次の有限科学検証案（未実行・未承認）

対象H4 DF10・保存state・S2/L_D1/K2/Δ.1・h.1/M12・exactκ・minkowski/finite_Bは固定。入力はA結果169d3316…と今回PHYSICAL_INPUTS、B canonical3216d27a…の同一13round。新しいtruth/state/decoder/budgetや候補を採択する計画ではない。

## 証拠の分類

| 分類 | 現在判断できること／不足 |
|---|---|
| 保存証拠で確定 | 13 physical identity、raw/corrected/reference/exact-tail、normalization、局所bias/disk、経験的nominalの存在と元SOURCE。m3/m4 norm欠測と未資格の保存 |
| SOURCE・技術検証 | raw↔X/Y符号、scalar/normalization一回、固定Δ exact-tail stepの反復構造、K2 polynomial、round別R、B identityの対応。実samplerのRNG・threshold・独立性の仕様照合も技術調査で可能だが、今回は新qualificationなし |
| 新科学評価が必要 | 保存rawからの確率生成、全decoderのfinite-shot risk/未解決率、reference等との比較、数値不確かさへの感度。厳密state ground・実law距離・認証数値上界には別の根拠・必要に応じた追加科学検証 |
| ユーザー判断 | 数値energy proxyを条件付きtruthとするか、未解決trialのloss、Minkowski成分の定義、数値不確かさの評価法、統計精度・信頼度、科学SOURCE・有限実行scope |

## 最小の次実行：Bの全13round ideal-law decoder統計

**理由**：局所診断からはalias/branch誤りを含む全推定量のenergy lossが分からない。必要な13 raw meanとnominalは保存済みで、最初の条件付き統計評価のためにAが新mean・reference・normalizerを計算する必要はない。Bの評価器を使い、Aで新Monte Carloを開発しない。

| 項目 | 実行条件・有限scope |
|---|---|
| 科学SOURCE | Bがdecoder/finite-shot評価器の公開commitを固定する。現B3216はidentity基点として保持。未公開の実行コードを今回固定済みと扱わない |
| physical入力 | PHYSICAL_INPUTSの13行。元N_HANDOFFのnominal各軸64/60/56/52/47/43/38/34/30/25/21/17/12。M/κ/R/Nを探索・変更しない |
| target | 保存DF-Hの同sector groundが研究target。最初の実験を保存solver energyに対する条件付きriskとする場合、そのproxy仮定をユーザーが明示採用。厳密ground保証へ昇格しない |
| 統計モデル | 13round×X/Yの26 Binomial count/trial。ideal finite-K lawのfresh IID周辺分布とBorn統計。実trajectory生成・compile・新meanは0 |
| 主評価 | rawの固定1モデル・既存nominal配分・既存prior/decoderについて全trialを保存してenergy squared lossと未解決率を報告。新しいnominal算出や十分N認定はしない |
| 有限試行数 | Bの既存手法に合わせ、ユーザーがMSE許容誤差ηとvalidation失敗確率δ、全trialの有界loss契約を指定してからnを固定。old α/βや任意のnを無断流用しない |
| 反復数の候補式 | 独立trialでloss∈[0,Lmax]が成立すれば、Hoeffding案は n=ceil[Lmax² log(2/δ)/(2η²)]。LmaxはHa²、ηはMSEの誤差幅。RMSE幅を望む場合はBがMSE区間からの変換を明示。今回実nは計算しない |
| 未解決trial | 既存decoderのzero/tie等のNoneを削除しない。固定のfailure lossまたは未解決率＋resolved条件付きMSEとして扱うか、ユーザーが決める。後者だけから無条件RMSEを主張しない |
| 数値精度 | complex128の保存mean・元empirical u/logB marginを保持。Bが数値不確かさのpropagation/感度とloss集計精度をSOURCEで固定。empirical allowanceをrigorous区間にしない |
| 成功条件 | 全13identity/Source/law一致、確率が契約内、全trialがloss又は未解決として追跡され、選んだ統計精度を満たす結果と資格の限界を記録。target未達も有効な結果で、候補変更しない |
| STOP | SOURCE/binding不一致、必要field欠測、確率/非有限値/数値扱いの未解決、decoder仕様・loss契約不一致、規定精度未達。黙ったclip、閾値緩和、試行追加・再seed・候補変更へ自動進行しない |

この段階は**保存meanの新しい科学的再解析とdecoder simulation**に該当するため、別途承認が必要。今回の技術作業承認や過去S1＋S2承認を使わない。

## 成分ごとのRMSE予算を判定したい場合

全rawモデルのriskだけでは、εqpe・εtrotの各予算は資格付けできない。BがZ（rawモデルのdecoder）とZref（full-H保存state referenceモデル等）の定義、同一N/decoder、couplingと全trial lossを固定し、比較を科学scopeに明記する必要がある。

その場合の追加scope案は、**同じ13roundの保存full-H referenceを用いる1比較モデル**、26 count/trialを追加し、固定coupling下でZ−ZrefのL2とZref−targetのL2を評価すること。common random numbersを使うなら単なる効率化と誤差成分のcoupling仮定を区別する。単一位相referenceへの置換は、保存stateの混合を消す科学条件変更なので自動採用しない。

PF対finite-Kをさらにenergy単位で分離したい場合は、exact-tail referenceモデルという**もう1つの比較**が考えられるが、signal成分の分解をそのまま独立energy biasと呼ばない。corrected meanは非unitary補正量なので、物理確率として直接samplingしない。これらの追加モデルや新qualificationは最小raw評価に自動追加しない。

## Aの追加物理計算が必要になる条件

1. **m3/m4の観測normを必須とする資格へ進む場合**：同じ2候補のcorrected/raw独立mean＋exact-tail内部対照を各1、finite-K event normalizer各1だけ、固定科学SOURCEで取得する有限案。reference2 pairは今回保存値と同bindingなら再利用し、新referenceは0。元S1の欠測・条件付き診断は保持。元norm/closure/allowanceと保存meanとの同recipe照合を行い、不一致ならSTOP、retry0。これは今回起動せず、最初の条件付きideal統計評価の必須作業でもない。
2. **rigorous ground targetを必要とする場合**：sector内最低固有値の同定・gap又は包含証拠、state residualの認証等の方法を先に科学条件として選ぶ。現在の小residualだけでは足りない。新eigensolveをデフォルトの追加callにしない。
3. **common effective Hの理論ルートを採る場合**：fixed exact-tail unitaryのbranch・overlapの証拠と、round別finite-K非unitary meanへの適用条件を固定する。13scalarだけのfitで保証を作らない。必要なoperator検証の範囲はそのモデルを採る判断後に限定する。
4. **actual sampler/compiled implementationの保証が必要な場合**：ideal有限分布と実event生成・確率丸め・独立性・回路作用のSOURCE/evidenceを担当へ要求する。単なるideal decoder反復数増加で代用しない。実sampling/compileは今回・次の最小raw案とも範囲外。

次の担当：**B**が固定SOURCEと有限統計計画、**ユーザー**が上記判断・明示実行承認、**A**が必要と判断された物理証拠だけを承認後に取得。今回の全13round物理contractは公開照合後mandatory STOP。十分N/global RMSE bound/C/Gはnull・未評価のまま。
