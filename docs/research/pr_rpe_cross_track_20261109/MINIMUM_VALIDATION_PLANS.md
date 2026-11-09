# C. 候補ごとの最小検証計画

2026-11-09 JST。[A：固定証拠・SOURCE](EXISTING_RESULTS_AND_OVERLAP.md)／[B：機構・相殺要因](IMPROVEMENT_CANDIDATES.md)。**以下は未実行・未採択の案。今回の依頼は科学実行・新アルゴリズム実装を許可していない。**

## 共通条件と承認の単位

第一の比較条件はAのC-H4：Linear H4/1Å/STO-3G、保存DF12生成順・元H exact ground、状態準備除外、18round、δ0=.02、εE=1.5936e-4 Ha、failure≤.05、phase .4、compiler Aとする。基準はc5a3c102の最終extended policy。強Fullと、同じ元HへのTruncated A/Bも参照する。別条件のD-H4/Pure-H/RCの値を混ぜない。

候補を選んだ後、**使う実行SOURCEと入力／fixture、変更する条件、科学call、出力、判定・STOPを具体化してユーザーが許可する**。現時点のSOURCE欄は再利用する固定閲覧snapshotであり、まだ存在しない新実行SOURCEを固定済みと称しない。必要ならユーザーは実装準備と、明示した固定scopeの科学検証を一括して許可できる。許可後の通常技術調査・デバッグを細切れ承認にしない。承認証明書・原文保存・新grant/ledgerを作る必要はない。既存契約固有のgrant/one-shot/STOPはその契約で保全する。

最初の局所GOは「その候補に効果がある可能性を残す」という判別であり、全RPEの科学的GOや次段階の実行許可ではない。固定scope完了でSTOP。全roundへ進む範囲を最初から明示承認された場合だけ、その範囲で続ける。

全候補に必要な最終判定は次のとおり。

1. **同じ推定対象。** 元H/state、scalar、controlled相対位相、P3又は変更後finite target、ordered mean、whole-path weightを保持／正しく誤差会計する。真のE0でdecoderを補正しない。
2. **同じ成功条件。** SOURCE/input/native/finite-bit/drop/model disk、rho>0、beta_stat>0、alias／branch、36軸failureと整数Nを結ぶ。価格MC誤差は成功確率と別。
3. **同じ実装機会。** PRと強対照に同じ合法control/fusion/統計/表現を提示する。qubit、CSWAP、reset、wrapper、D/R signed境界補正も価格へ入れる。
4. **選択と評価の分離。** 過去holdoutを新しい選択に使うならtrainingへ再分類する。新policyをfreezeした後に独立価格holdout。共有cellとX/Yの共分散を保持する。
5. **費用の結論。** PR内部のG減と、Full/Truncatedとの差を別に示す。差を不確かさから識別できなければINCONCLUSIVE。SEだけをrigorous CIへ格上げしない。負結果ならそのscopeを終了する。

CPU/RAM/workerの人工上限は置かない。実行が許可された際は、既存の並列・cache・streamingを適合範囲で利用する。共有OS、scheduler、library、他worktree/processは変更しない。以下のfixture数・候補数は科学scopeであり計算資源上限ではない。

<a id="k1"></a>
## K1：joint completion間のrelative橋

**次に変わる研究判断：** 高いRTE basis費に、既存same-basis共有の先の削減余地があるか。D C1の一pathを再実験する問いではない。

| 要素 | 最小案 |
|---|---|
| 再利用SOURCE | C c5a3c102の`build_events/catalogue/emit`、D141fa762の`StrongBuilder.transition/vacuum_phase`、既存CAR/exterior checker・compiler A |
| 先に固定する入力 | 新乱数なし。保存最終Closedのm17セルh=.32/n768を入口に、既存公開raw indexで事前指定した一trajectoryから**価格を見ず順序で最初の適格な異basis2-run**を選ぶ。actual completion、support mapping、gauge、時刻順を固定。source/codecに不適合ならここで戻す |
| 最初の科学検証 | その2-runで旧往復／relative橋／既存full fallbackを比較。全Fock projector・controlled operator・X/Y・signed/scalar・actual inverseを既存参照で確認し、同じcompiler Aの完全wrapperを価格化。D/Full側にも合法なrelative機会を提示する |
| 記録する量 | RZ/CX/実DAG depth/size/qubit、追加source/native disk。元law/weight不変の確認。局所命令数を全Gへ足さない |
| 局所GO | 同じ作用とphaseを保持し、baselineに既存compiler相殺を含めても目的資源が減る。新uniform diskの全support会計が構成可能である |
| STOP | dense relativeが高価、mapping/gauge/phaseが一致しない、既存compilerが同じ節約を既に達成、増分≤0、又は指定fixture完了。保存pathの価格による選び直しをしない |
| 必要な許可 | 新adapterの必要差分実装、固定fixtureのbuild/compile/operator・signal検証。全law価格・全18roundの再会計は別途範囲を明示する |

局所GO後の最小後続案は、一つの支配occurrenceセルの両層paired価格・uniform disk・同policy N×価格。全18round更新を初手の必須作業にしない。これは今回未実行。

<a id="k2"></a>
## K2：Closed P3×Reference

**次に変わる研究判断：** 既に強化済みのC回路にもReferenceの増分があるか。RZ目的とCX/depth目的で判断が異なる。

| 要素 | 最小案 |
|---|---|
| 再利用SOURCE | C c5a3c102と公開2e38af3bモデル、既存新A reference/transfer・D physical segment/checker。新gadgetを作らない |
| 静的binding | uncontrolled factorization、negative D、RA event phase、whole weight、joint completionのvacuum eigenphase、actual inverse、χを一つの保存trajectoryに結ぶ。controlled nativeを後から剥がして代用しない |
| 最初の科学検証 | 事前指定した一PR短trajectoryと一強決定論構成で、ordinary／Reference×X/Y（計8 wrapper）を比較。条件は元H・compiler Aで統一。Truncatedへの適用可能性も同じ静的条件で判定する |
| 記録する量 | 全入力X/Y誘導observable、complex coherence、vacuum/χ、signed/scalar、actual inverse、leakage、RZ/CX/depth/size/qubit、追加disk。weightは回路外の同じ全trial規約 |
| 局所GO | 主資源のnative価格に追加利益が残り、N悪化と追加register費を戻す後続会計の余地がある。CXだけのGOをRZ勝利としない |
| STOP | vacuum/reference前提が閉じない、強対照にも同利益で相対差が改善しない、主資源価格が悪化、又は固定8構成完了。局所T率を18roundへ移さない |
| 必要な許可 | 固定source/inputを用いた統合差分実装と8構成のbuild/compile/signal検証。全support・長round・新合成・全N更新はここに含めない |

K1と同時に変更しない。後に組み合わせる場合はK1/K2 on/offの2×2で重複するbasis/boundary利益を測る。K2実装可能性と最終総T資格は別課題。

<a id="k3"></a>
## K3：D／R／drop

**次に変わる研究判断：** 安い決定論近似と全tail保持PRの間に、ランダム補正を残す意味があるか。

| 要素 | 最小案 |
|---|---|
| 再利用SOURCE | C truncation da8b2c3f、相殺分離92495a04、Closed c5a3c102。元H public replay／state／retention／scalar／directed誤差を利用 |
| 固定入力案 | C-H4の保存m17 retention（coeff_L1_2）を入口に、既存LD3をDへ残せるか静的照合。保持集合とDが衝突する場合は勝手にDを変えず案を戻す。新subset/rank gridなし |
| 最初の科学検証 | 一つの固定時刻で、drop=0の旧PR／RなしのTruncated／D＋保持R＋dropの三者を元H groundで比較。初手は保存hとrateを固定し、finite mean・Hdrop/PF/RTE個別diskを評価。相殺ありAと保守Bを区別 |
| 記録する量 | λ、実proposal・return/range、rho/beta、十分N、同compilerのD＋whole RTE＋wrapper価格。まず一cellの固定規則によるpaired/層別価格。価格MCとtrialを重複乗算しない |
| 局所GO | 保守的に元H誤差を払っても中間方式に非自明な利点があり、既存端点を含む同一menuで比較する価値が残る |
| STOP | 有用なRが空、drop errorで予算が閉じない、N増が単価減を上回る、中間方式が両端の実行可能方式に支配される、又は固定cell完了 |
| 必要な許可 | この三者の有限mean・誤差・回路・価格・Nの新科学評価と最小adapter実装。全18roundのretention/rate最適化や他分子は別scope |

whole-H equalityを捏造せず、近似evolutionから元H energyへの誤差として扱う。原Truncatedのcombined保証を破棄せず、保守Bとの両方を報告する。

<a id="k4"></a>
## K4：表現・partitionを元H groundへ戻す

**次に変わる研究判断：** RCのown-DFでの点差、又はM2の小残差が、今回の元H推定taskでも役立つか。

| 要素 | 最小案 |
|---|---|
| 再利用SOURCE | RC70981380の保存P/E2/CDF0 export・Z再配置、又はM268ef4feeのcore/residualと独立CAR。A M1/B2元H adapter、C誤差・decoderを参照 |
| 先行する静的作業 | ユーザーが**Z再配置候補一組かcore/residual一組か**を選ぶ。元H/state、J/U、κ、identity、spin、basis gauge、近似model差の許容を対応付ける。RC own-DF stateを流用しない |
| 最初の科学検証 | 新fitなし。同じ元H exact groundで一つの長時間信号を旧分割／変更分割に評価し、model＋PF＋RTE・半径を検査。予算が閉じた場合に限り、その固定cellのwhole no-prep価格とNを比較 |
| 記録する量 | 元Hとの差、tail massだけでなくD価格とRTE basis、range、rho/beta/N。Full/Truncatedにも同じ候補表現とmodel誤差を与える |
| 局所GO | promiseを揃えた条件で予算と価格差が閉じる。RCの保存改善率ではなく、この共通targetの差として示す |
| STOP | source/state/modelのbinding不足、M2物理model未採択のままになる、誤差予算超過、D増＋N増がRTE減を相殺、又は指定一組完了 |
| 必要な許可 | model扱いの採否と、指定候補だけの元H signal・価格・N評価。新SCF/DF fit/rank探索・ground solveは含めない |

候補固有の近似H groundを新たに解く検証ではない。元H/stateは保存原本を使い、別の準備コスト問題へ変えない。

<a id="k5"></a>
## K5：confidence会計の改善

**次に変わる研究判断：** 現在の十分trial数の保守性に、同情報・同成功条件で削れる余地があるか。

| 要素 | 最小案 |
|---|---|
| 再利用SOURCE | B G6/finite-bit moment/range、C c5a3c102の`qualify/rigorous_shots/discrete_allocate`とdecoder。旧RPE短段の厳密二項関数・testsを先に照合 |
| まず固定する保証 | fixed Nのまま、認証M2/range/biasを使う既知集中界とX/Y confidence regionを整理。未知の真位相を入力にせず、許されるmean disk上で一様。Fullの±1にも同機会。今回は新証明を実施していない |
| 最小科学評価案 | 保証が成立した場合だけ、保存m17のCanonical／Closed／Full同条件に新Nを算定。価格は同じ保存値を参照して固定cell Gを比較。sampling・新compileなしでも**新N／G再解析には実行許可が必要** |
| 記録する量 | 十分条件、bias消費、exact failure総和、ceil、M2上界とrange項、全ての追加learning費。次に進む場合は36軸・alias/energyまで同じ保証を拡張 |
| GO | 真のmean/variance oracleや事後confidence選択を使わず、同taskで十分N×価格が下がる。PRの絶対減とFullに対する相対減を分ける |
| STOP | 認証M2がrange²と同程度で界が改善しない、取得/学習費で相殺、保証が未知分散・optional stopping未解決に依存、又は固定m17比較完了 |
| 必要な許可 | まず保証検討の範囲。数値評価を選ぶ場合は保存m17のN/G再解析を明示許可。adaptive測定や新samplingを自動追加しない |

新しい重要な統計保証の正式採否を要する場合は、科学計算とは別に詳細レビュー開始の許可が必要。既知界の通常適用を機械的に重要レビューへ昇格させない。

<a id="k6"></a>
## K6：次のground補正を制御する構成

**次に変わる研究判断：** 現在のleading相殺を越える具体的な追加自由度があり、D仕事量に見合うか。

最初の作業は数値探索ではなく、BのK6に示した**5段D-only対称族D_{a,b}**を既存BCH／scheduleと照合し、R正時間、保存元H groundへの適用条件、leadingと次補正、追加D指数と未知量を記述すること。次補正の非対角混合情報を無料と仮定しない。既存Yoshida／compact／processor／同v再探索へ還元されるならそこで終了する。今はその具体係数・実行SOURCEを作成・採択していない。

成立性が残る場合の最小科学案は、保存C-H4の一hと独立近傍hでleadingと次補正の挙動、dominant overlap、nonresonant累積energy bias、finite-RTE raw半径を確認すること。wrapped終端位相だけをfitしない。旧調整Dと強Fullへ同じ設計情報を与え、追加段を課金する。価格取得前にcheap work見積りだけで優越を認定しない。

**GO**は補正の追加制御が成立し、RTE range／D増が全費用機会を消していないこと。**STOP**は既知法と同値、追加自由度が独立でない、負R時間が必要、係数不安定、D負担に見合わない、又は指定組成・近傍検証完了。新構成・係数評価・BCH数値・signal・compileは新科学条件であり、別の明示許可が必要。旧研究BやC2保留をこのIDで再開しない。

## 現時点で数値実行案へ進めないもの

| 対象 | 次に必要な最小作業／科学的意義 | 許可・STOP |
|---|---|---|
| whole-path finite-state sampling | 既存O2との差となる、安価に取得できるscore部分和と実conditional lawの**具体例**を提示する。既存ISの式を書き直すだけでは不可 | 未特定のため実装推薦なし。新たな構造がなければHOLD継続。旧O2/C1/Parentを再計算しない |
| P5/K4等の既存family拡張 | ユーザーが次数変更の価値を選ぶ場合、既存G7/G9と旧r/K gridを使い、P3/P5の同exponential精度・range・event価格を対応付ける有限案を先に固定 | parameter研究として扱う。新次数の分子binding・計算は別許可。低BだけでGOにしない |
| D長qの価格資格 | 既存T0→T1案とq64の完了結果を使い、P0後の絶対価格／depth境界を閉じる対象を選ぶ | リソース評価担当が別scopeで扱う。人工29PASSや有限q64を全q保証へ移さない。全13round再会計は未許可 |
| RPE以外の推定器 | ideal unitary queryとfinite weighted meanのoracle差・query価格を静的に対応付ける | 推定器family変更の正式採否が先。今回のPR＋RPE改善の実行scopeへ含めない |

## ユーザー／GPTが次に判断すること

- 主資源を当面native総RZとするか、CX/depth又は総Tを別の主課題として扱うか。
- K1/K3の限定判別、K2の統合、K4の元H promise接続、K5の保証、K6の成立性のどれを選ぶか。複数選択も可能で、科学的に独立したscopeを別々に固定する。
- 選択した候補の実装準備と科学実行をどこまで一括して許可するか。重要レビューが必要ならその開始許可は別。

**本計画の公開後、Codexはmandatory STOP。正式採択・実装・科学計算・詳細GPT科学レビューへ自動進行しない。**
