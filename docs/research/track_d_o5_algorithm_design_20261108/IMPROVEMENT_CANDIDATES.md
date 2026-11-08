# A/B/C改善候補の比較

結論案は **B：改善候補はあるが、新規性や効果が不明**。Aを最初の安価な技術的選別に置くが、科学的GO・新たな最適化変数・数値条件は採択しない。新実装/テスト/科学計算は今回0。

## 共通の比較基準

対照はStrongBuilderのuncontrolled Gaussian・relative/head fusion・tail共有・合法step境界融合・scalar集約を有効にした強いsymmetric-directionalと、既存中央cut Reference。片側だけに新しい既知最適化を与えない。

まず一つの資源g（CX、RZ、depth等）または事前採択した目的を固定する。量子幅の実現可能性を制約にし、17 qubitsを9 qubitsと同価格で平均化しない。depthのshot和は累積会計であり、実機wall timeではない。RZ→T、CX→noise/時間への変換には別契約が必要。

| 候補 | 保存する対象 | 主な利益候補 | 実装前に必要な情報 | 既知法/否定条件 | 現時点 |
|---|---|---|---|---|---|
| A: compile前の方式選択 | 同じwhole-path V、QとX/Y条件付き分布 | 選んだ方式だけを実行する一shot価格。17-qubit不可時の制約対応 | q/r/K、event/support/sign、exact basis ID/run、eligibility、axis、固定backend/profile | round-onlyは既知mixed。paired差の符号が変わらなければpath選択不要 | 優先1、headroom未確認 |
| B: complete-step cut選択 | 同じordered V=AB、actual inverse、chi | lane depthの不均衡低減、closure/端部最適化 | prefix/suffix run量、合法境界、失う融合、vacuum phase、same-source cut条件 | 中央cutで既に均衡なら利益小。単純lane和/maxは既存反例あり | 優先2、深さ中心の候補 |
| C: control-aware接口融合 | 同じVまたは同じW0†W1、Q、PF | 未融合head-tailのGaussian接口・control modeを共同処理 | exact frame、chronological order、control branch、scalar/gauge、support | 既存relative/step fusionの再実装では追加利益なし | 優先3、対象接口の同定まで |

## A：Controlled／Referenceの選択

### 既存との重複と情報の取得時点

確認したprepare_stepsは同じpathの両方式とX/Yを構築し、B/C会計も方式別に総額を返す。これらの入口にcompile前のmethod selectorは確認できなかった。ただし全repositoryに同機能が絶対に存在しないという主張ではない。別DF route選択を、同じPR pathのControlled/Reference選択と混同しない。

Round-only切替をまず強い追加baselineにする。常時strong・常時Referenceにしか勝てない方式を、path-awareの新しい利益としない。既存価格から事後に安い側を選ぶ値はoracle下限の診断であり、実行できるpolicyでも独立holdoutでもない。

| 入力候補 | sampling後compile前に使用可能か | 注意 |
|---|---|---|
| q、r/K、axis、profile、使用可能qubits | 可。事前固定条件 | qのみの切替は既知で、複雑な特徴より先に比較 |
| eventのorder/role/個数、support、係数符号、保存角のzero | 可。既存serializer | 角の再計算/実データ分類は次回科学call。zero/special-angleを都合よく無視しない |
| C7-4 event_featuresのbasis runs、different transitions、support_work、lane imbalance | 可。O(qr)のdescriptor走査 | 現関数はtail descriptorをstepごとに数え、全head/relative Gaussian価格を表していない |
| frame ID、同rotation key、合法boundary数、cutで開くframe | metadataの範囲なら可 | 同IDと数値近似frameを混同しない。relative matrix生成は既存science処理で無料ではない |
| 既存cache hit/key | identityが完全なら可 | 比較両方式の先行compile費・storage・lookup・cache構築を記録 |
| P0 DAG endpoint、compiled depth、両方式の真の価格 | compile前には不可 | C7-4のDAG説明変数はoracle。両方式compileする戦略ならその古典費を払う別方式 |
| Born、真signal/variance、ground/gap、測定bit | 不可 | feature/選択/配分へ渡さない |

### 正当性と費用式

sampling後の同じomegaとaxisに対し、事前凍結policy pi(x(omega),m,b)がS/Rの一方を選ぶとする。適格な両方式が全入力で同じobservableを持ち、選択が測定bitを見る前なら、

\[
\Pr_\pi(Y_b=y\mid\omega)=\Pr_S(Y_b=y\mid\omega)
=\Pr_R(Y_b=y\mid\omega).
\]

したがってQを変えず、不偏target・±1分布・理想variance・attenuationは保存される。Reference不適格なpositive pathを捨てず、事前契約が許す適格Sへのfallbackを使う。新retry/rejectionやsampling law再正規化は導入しない。

同じjoint path law、fresh independent shot、元H ground・RPE誤差/半径/prior/decoder条件がそろうideal laneでは旧round/axis別Nを共通利用し得る。しかしactualの方式別数値誤差はpiの選んだ領域でも覆う必要がある。有限bitのmarginal一致だけでIIDやactual RPE成功を認定しない。これらを欠けばNをそのまま使えず、今回は再算定もしない。

Delta_g=C_R,g−C_S,gとすると

\[
G_{\pi,g}=\sum_{m,b}N_{m,b}
\mathbb E_Q[C_{S,g}+1_{\{\pi=R\}}\Delta_g].
\]

Oracleの E[min(C_S,g,C_R,g)] は一つの理想下限。compile前にその値を知れることや、posthoc minが新しい期待価格推定であることは仮定しない。

実装可能な小policy案は、既存q-only判定をfallbackにし、descriptor-onlyの凍結予測差が十分明確な領域だけ上書きする方式。もし正当な予測誤差幅u_g(x)があれば、d_hat+u_g<0をR、d_hat−u_g>0をSとする条件付き設計が可能。ただしC7-4はその一様幅を取得しておらず、bootstrap SDやLOQ残差をu_gへ格上げしない。閾値や余裕幅を今回採択しない。

選択の古典生成/feature走査/fit/compile/cache費は量子gate countへ無単位で足さず、別列で記録する。full campaignでは学習・価格取得の全費用と運用shot数への償却条件が必要。二回のcompileで安い量子回路を選ぶ方法も、compile費を支払うなら実装可能だが、安価な事前選択と区別して比較する。

### 負の条件

保存q2〜32ではCX差が全pathで同じ符号/同じ式だったとの既存報告があり、CX目的のpath-aware headroomを支持していない。RZの短q/長q差はround-onlyでも利用できる。q≥64 proxyが13-round会計を支配するため、短qだけの切替で総資源が大きく下がるとも推定しない。現在のモデルは候補選定に使用済みで、同じ88 pathを新selectorの独立性能証拠にしない。

17-qubitが使用できない、routing/reset/校正でRが逆転する、predictorの間違い/費用が利益を消す、目的間の勝敗が異なる場合もある。policyの17-qubit使用確率をpeak幅へ読み替えず、同時実行のmemory/qubit制約も別に扱う。

## B：Referenceのcut位置

既存q≥2はcomplete-step中央cut。理想Vと全phaseを変えず、A/Bがunitaryでwhole-V reference条件を満たせば単回信号は同じ。既存DF資格と最終真空復帰を維持するには各halfのray保存も検査する。actual A.inverse、ordered AB、chiの全path一回集計は既存checkerのまま。

最初の候補domainは既存が受け付けるcomplete-step境界だけ。全境界のcompile探索ではなく、descriptor prefix/suffix runの固定proxyで中央と近傍一境界を比較する小さな設計から始める案。近傍側をどう一意に選ぶか、特徴重みや対象qは次段階で事前固定する。今回は実H4のrun量を数えず、cutも変更しない。

目的例は

\[
\widehat D(c)=\max(\widehat D_{A^\dagger}(c),\widehat D_B(c))
+\widehat H_{\rm cut}(c).
\]

これは特徴proxyでありfull compiled depthの等式ではない。C7-2のmax-plus/界面反例を必ず対照にする。cut損失をH_cutと内部q−2対q−1の両方に二重計上しない。A†をAと同じcompiled価格と仮定せず、actual inverse後の書換えを含むwrapperを最終検証する。

同じ一cutならCSWAP数は変わらず、complete-step境界のclosureも同じ型になり得るため、CX/RZ countの新たな削減は不明。主な余地はlane不均衡・depth。中央ですでに均衡するpathやCSWAP/routingのcritical pathが支配する条件では効果が小さい。event内部cutや開いたGaussian frameへの拡張は既存eligibilityと証明/phase/価格scopeが変わるため、別の設計として保留する。

## C：PR構成とReferenceの共同最適化

最小候補C1は、既存にhead_tail_cross_fusion=Falseと明記される接口で、ordered列を変えずに隣接basis/relative transitionを処理すること。StrongBuilder.transitionとtail BasisProviderを再利用し、適合するcontrol mode・exact frame・scalar gaugeを持つ隣接演算だけを対象にする。S/R双方へ同じ最適化を与え、既存relative/step融合を「新しく追加した」と数えない。

検証対象はfull Uの同値またはW0†W1=VとX/Y observable。directionalとordinary、uncontrolledを無条件に同一化しない。既存step0のmixed境界がordinary controlled fullになる規則は、単純なdirectional fullへの置換が不正である例である。近似したGaussianやglobal phaseの削除で価格を下げない。

一方、fragment/eventの並替え、PFの順序変更、RTE分布・normalization変更、CDF/DF表現やselected basis政策の変更はC2という別のアルゴリズム変更である。lawが見かけ上同じでも固定pathのVや有限PF biasが変わる可能性があり、元の半径/Nを流用しない。C2はO1のPF構成・O2 sampling/return・O3誤差/総資源との担当調整と正式採択を要し、今回の最小候補には含めない。

C1はcompiler/Gaussian最適化の既知応用に留まる可能性が高く、P0が既に同じ消去を行うなら増分効果は0。未融合接口が実際に残るというSOURCE上の記述は、compiled費用の改善証拠ではない。

## 最終目的への接続

A/B/C1はtarget・lawを保持する範囲では主に一shot価格を変える。ideal条件では新しいshot削減や分散削減を発生させない。literal実装誤差が増えるなら同じNが使用できるか別に資格付けする。C2は半径/近似/必要Nを変え得るため、単価だけで評価できない。

現在の研究変数を追加する正式採択はせず、最初は固定PR条件の下で「強い静的方式選択に対して追加余地があるか」を小さく判定する提案に留める。
