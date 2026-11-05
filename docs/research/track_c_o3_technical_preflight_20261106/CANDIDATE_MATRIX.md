# Track C：候補・反証・最小検証の比較

2026-11-06 JST。[REPORT](REPORT.md)の固定資料・元H厳密ground・状態準備除外・同energy精度/成功確率を全行で共有する。**未採択の条件付き候補**であり、ここで正式GO/NO-GOを決めていない。将来の検証案は未実施・別許可。STOP列は将来の縮小/停止判断を助ける提案で、過去のSTOPを変更しない。

## 1. 仮説・機構・既知法との差

| 候補 | 中心仮説・変更箇所 | 数理的機構・総費用への接続 | 既知法との差が残り得る部分／残らない部分 | 主な反証材料 |
|---|---|---|---|---|
| **C-N：PFのnative境界価格** | 元H・表現・split・canonical RTE生成規則を固定した既知PF比較で、D/R occurrence境界・control・合成後の非加法価格を含めると、stage/action proxyで選ぶ構成より小さい適格Gが残る条件があるか。係数が変わればoccurrenceの時間・分布は変わる | 既知係数・内外PF構造→phase/半径とr/K/N、D basis往復/負時間/融合→full期待価格。`N×E[C_full]`で比較。まず精度と信号が同じ条件の価格効果を分離 | 既存の並べ替え、W†W相殺、diagonal-only controlは共通baseline。追加性質になり得るのは、**同じ既知fusion後にも残る価格構造が全Gの選択を変える有効条件**。単に既知PFをcompileした成果は応用評価 | P-D強B1b/B2/B4は同じ候補。BF-AもF固有gainなし。BMはcompact同値。研究BではFull Yoshida4が強い単価対照。PF orderだけでは追加性質なし |
| **C-S：合成精度・残差配置** | 既知の安価な角度辞書/合成とresidual補完を、actual finite-RTEの各occurrenceに適合させると、通常の均一精度合成より小さいGが残る条件があるか | synthesisのcoherent biasと残差massを分離し、有限mean・B・weight range/moment・半径・十分Nまで支払う。deterministic/T削減とtail/shot増加の交換 | PR原著App.E、PAI/TETRIS、SP、Parent/S4が直接対照。新IS lawやnotch辞書の再命名ではない。追加性質は**coherent meanとenergy成功を維持する、実finite-RTEに適合した合成利益の条件**。既知要素の接続だけでも応用として評価可能 | SP1はtoy/additive T、長nでloss/cap。R1/Parentは旧目的・同情報対照との違いが限定。PAIのchannel保証だけではHadamard振幅平均を保てない。小角・負時間・global phaseを落とすと不成立 |
| **C-P：processingの有限時間信号** | 同じ既知kernelへの物理processor適用が、元H groundから見た半径・多位相混入を改善し、端点費を払ってもGを下げる領域があるか | `Z_P(q)=⟨E₀\|P†U^qP\|E₀⟩`。kernel固有値を保存したままspectral weightを変える。有限時間phase/radius→branch/許容座標誤差→N、processor端点→価格 | processing自体は既知。**固有位相bias削減をclaimにしない**。既知processed対照を上回るsignal/priceの適合条件があれば追加知見。単にPを付けたものは新法でない | YP8m8のkernel/processorは既存。研究BのA3/A4差未識別、R障壁越しの無料相殺不可。元H厳密groundでもPF固有状態とは限らないが、Pが改善する保証もない。FRのbound改善だけでは設定差なし |

ここでは新係数列、processor構成、sampling law、optimizerを設計していない。既知formulaの使い分けや既知最適化の適用だけなら、それを独立新算法と表示しない。追加性質が未成立でも、条件付きの応用的総費用改善という研究目的まで機械的に除外はしない。

## 2. 再利用・最小検証・判断条件

| 候補 | 再利用する固定資産 | 将来、継続/縮小/停止の判断に必要な最小検証 | 強baseline・全費用比較 | STOP/縮小条件の提案 | 独立性・共通依存 |
|---|---|---|---|---|---|
| C-N | `PF-ORDER` registry/decomposition、P-D内外分離、signed `rte`、DF control/basis builder、研究B native-v5の意味論、RPE provider | 同一元H/stateで既存少数formulaのphase/半径/finite meanをまず適合確認。代表的な短/長interrogationで非加法価格がaction proxyの候補差に影響するか、その差が誤差予算とNを戻して残るかを限定確認。必要な範囲だけ新価格を取る | 最適化機会を揃えたS2・Yoshida4・適合する公開高次/processed、既存P-Dの強いleading model。exact Rは実装可能な場合の対照、無料全H oracleにはしない | 同じ共有/融合を与えた対照と差が消えるなら新method claimを止める。差が価格不確かさ/未検証長qに依存、または基底energy条件を満たさなければ全G優越は保留。旧P-D/BF探索の再実行は行わない | 合成法を固定すればC-Sから独立。processorなしでC-Pから独立に評価可能。高次化自体の追加性質は前提にしない |
| C-S | `ALLOCATION`訂正版、`SP-R1`の既存反証、actual adjoint/phase保持native IR、`rte` heterogeneous occurrences、error accounting | 既存SP toyを延長せず、選ばれた実finite-RTEの一つの適合条件でtarget/mean/相対phaseを確認。通常合成・既知residual法・候補配置の少数対照で、B/weight/誤差/十分Nとfull価格を接続。情報取得費も記録 | 同じ角度辞書・precision・fusionを持つ通常合成とPR原著residual処理。適合するoperator-PAI/TETRISを対照候補にする。全PAI pipeline導入は必須でない | channelしか一致しない、unknown誤差を0補完、rare平均未取得、N増加で全G利点消失なら対応claimを停止/縮小。旧SP/Parent toy STOPは別途再開許可なしに解除しない | PF/kernelを固定して単独評価可能。C-Nの新PF価格を必要条件にしない。C-Pと同時変更すると因果混合するため単独対照を先に置く |
| C-P | 別PF repositoryのYP8m8列・processed affine helper、研究BのP/actual inverseとD/R障壁の扱い、PF phase/branch、finite mean/RPE会計 | 既存P/kernel/inverse規約を固定し、元H groundに対するPなし/Pありの有限時間phaseと半径を分離。固有値集合は対照として同じものを扱う。actual finite-RTEと端点費を含め、必要なround域でN減が費用を上回るか確認 | 同じkernel単体、既知processed formula、同精度に調整したunprocessed高次PF。両者に同じstate情報・compiler/fusion・誤差予算の機会 | 同じkernelで固有値が改善したとしか説明できない、無料のprocessed stateに置換、相対phase不一致なら比較を修正して止める。半径/N改善がないか端点費を回収できなければ仮説を縮小。研究B/FRを自動再開しない | PF族と合成を固定したsignal問題として単独に評価可能。最終Gには共通会計とprocessor端点の価格が必要 |

全行の検証は未実施。必要なadapterがあっても今回は実装していない。数値閾値・系・追加gridをこの文書で新規に固定せず、ユーザーが評価対象と科学条件を選ぶ際に決める。

## 3. 共通接続A：誤差配分・全RPE会計

これは独立新算法ではなく、3候補の主張を公平に比較する共通依存。

| 項目 | 引継ぐもの | 将来必要になる差分・情報 |
|---|---|---|
| 誤差予算 | PF bias、finite-RTE、統計位相を分離した既存β/α会計 | synthesis bias/range、必要ならfit誤差を同じenergyへ接続。任意の合成誤差をCδ²へ混ぜない |
| round/branch | 既存fixed-split r/K schedule、条件付き18round/energy接続 | 高次/processingの有限時間signalに適合するphase/半径と共通E₀。O4のround別表現/split新法は今回対象にしない |
| 価格 | 同snapshot/compiler/full-wrapperを結ぶ既存provider | 新しいPF/合成/processorを許可された範囲だけ対応。既存q範囲外の移送幅を未知として保持 |
| shot | estimatorの実range/moment、X/Y・失敗率、全trialの対応 | raw半径とBを二重計上せず、signed/weighted lawへ適合する十分N。二次moment最適だけでfinite-confidence最適としない |
| 公平性 | 同じ元H ground、同じtarget/情報、baseline側の再最適化機会 | 未知E₀/gap/price oracleを候補だけに与えない。取得/生成費、paired価格と共有較正の不確かさを残す |

共通条件を合わせた後、C-N/C-S/C-Pは個別に進められる。組合せの相乗効果を主張する場合だけ、各単独対照と組合せの追加比較が必要になる。全組合せgridや3案同時の科学実行を必須にはしていない。

## 4. 今回候補へ戻さないもの

| 系列 | 扱い |
|---|---|
| P-D S1/R3、BFの同score係数探索 | 強対照選択差なし・具体的method delta未成立を引継ぐ。未保存bridgeを新計算で補完しない |
| BM floor/internal・compact再帰/cache | 同じ固定列・同情報では既知対応。集約順序や弱いcache対照だけの差を新法としない |
| PR-3固定Richardson | variance/backbone負結果とSTOPを維持 |
| P-A interval、Gaussian singleton、C1/Parent旧toy延長 | 既知同値・正式一時終了/凍結/STOPを維持。今回の候補名で再開しない |
| FR scalar-only、研究Bの同toy内PF追試 | bound改善のdecision未成立、強Full-PF対照、一時保留を保持。資産参照と旧研究再開を分ける |
| UWC旧値・研究C own-DF state結果 | 前者の来歴未結合、後者の元H state不一致を保持。C候補の実証として転用しない |

**次の担当：ユーザー**が、必要なら候補と対象条件、既知法の応用評価か追加性質の検討かを選ぶ。**Codex**は別途明示許可された実装・科学検証だけを担当する。現時点では3成果物の公開照合後mandatory STOP。自動開始する研究作業なし。
