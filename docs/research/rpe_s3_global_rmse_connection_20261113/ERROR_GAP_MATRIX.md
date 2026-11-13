# 誤差・不足証拠・担当の対応表

2026-11-13 JST。[REPORT](REPORT.md)の固定H4 DF10/S2/L_D1/K2/ΔPF=.1/h=.1/M12/保存state・exactκを対象とする。表の「可能」は方法の適用可能性であり、今回数値評価済み・QUALIFIEDという意味ではない。**P0は認定を直接妨げる優先課題、P1は主張水準に応じて必要、P2は追加目的を採る場合のみ**。

## 1. 統計・target・decoder

| 項目 | 既存証拠で確定できること | 不足／最小の取得方法 | 解決の種類・担当 | 優先度 |
|---|---|---|---|---|
| S3の観測値 | 2084791のPRIMARY：43,819反復、条件付きRMSE .0006847202619833286 Ha、未解決0 | 標本点推定をpopulation上界とみなせない。既存結果のラベルはSOURCE調査だけで確定可能 | 調査済み、A/B。今回再集計なし | P0 |
| 未解決loss | 主UNDEFINED、energy=None。副[0,4096]は採択した感度envelope | 全事象の固定loss、又は宣言したenergy補完familyへの一様bound。特定fallbackなら新推定器として承認 | ユーザー判断＋Bの契約・技術検証。新lossでの数値評価は科学再解析 | P0 |
| 解決分population MSE | 主CIはnull、副Hoeffding envelopeは広い | 独立性・全loss support・confidenceが有効な上側risk評価。旧43,819 trialの有限再解析又は解析的評価を別承認 | B。既存summarize再利用、CI方式選択は未採択 | P0 |
| 未解決率 | 97.5% CP上端9.999788982664716e−5、真の0ではない | 未解決loss寄与を目標精度内へ制御する証拠。pUの上界だけでは不足 | B。既存CPを再利用し、追加反復は自動採択しない | P0 |
| support | 旧計画の[−32,32] Ha・U4096は仮定、観測検査のみ | wrap/energy変換の全到達出力域とfailure補完の範囲の認証。狭いpriorを出力域としない | Bの数学・binary64技術検証。既存観測はsupport証明にならない | P0 |
| proxyの生成来歴 | 97d25ccのgenerateがsector36行列のeigh最小固有対を保存。数値residualは小さい | 行列組立・最低固有値の認証包含はない。保存eighを単にやり直すだけでは保証にならない | 調査済み、A | P0 |
| E*→E0,DF | 現契約は保存DF-Hの同sector基底。ground_state_certified=false | 同じ係数の狭いground包含[L,U]とdtarget。変分上界＋正定値下界など、1個の36次元問題で取得する有限案 | Aの新科学計算が必要。state/SCF/DFの新生成は不要 | P0 |
| state固有性・overlap | 数値state/sector/hash/residual、full-H saved-state referenceあり | 残差は近傍固有値を示すがground同定・厳密overlapは未認証。共通有効位相モデルを使う場合はgap/overlap証拠 | A。直接raw全risk＋ground包含では独立のoverlap証明は必須でない | P1／解析モデル採用時P0 |
| DF前のHmol | STO-3G分子からtol-only DF、weighted completionの生成来歴 | 同basis/sectorの||Hmol−HDF||認証。tol=1e−8やPASS_ENGINEERINGからenergy boundを代入しない | 追加targetへの拡張はユーザー判断＋A。既定targetは変えない | P2／Hmol主張ならP0 |
| alias・branch | priorの係数境界、scalar互換、全観測DECODED/prior内 | DECODED≠正しいbranch。rare misbranch lossを含むrisk評価、又は十分なbranch tail bound | Bの統計・解析。局所phase/Tや零観測で代用しない | P0 |

## 2. 物理signal・law・数値誤差

| 成分 | 既存証拠・限界 | 最小の追加証拠／再利用 | 解決の種類・担当 | 優先度 |
|---|---|---|---|---|
| PF分割 | exact_tail−referenceの保存complex差、同S2/head/scalar | signalレベルは保存済み。S3 raw全riskへ重ねて足さない。個別energy成分が必要なら基準推定量を指定 | A SOURCE整理済み。全decoder比較はB、別科学再解析 | 直接riskでは追加不要、分割予算ではP1 |
| finite-K | corrected−exact_tail、K2 law・polyとB/logB・closure | signalレベルは保存済み。paperの無条件十分Nへ移送できない。共通effective-Hの仮定は未確定 | A SOURCE／数学。新モデル採用なし | 同上 |
| normalization | 正の全interrogation B、raw/corrected独立伝播、nominalは旧式で取得済み | rawをそのまま測定meanに使用。B再乗除・B²二重適用は禁止。認証B誤差は別項 | Aの保存結果をBが利用。新normalizer取得は既定不要 | P0の会計確認、調査済み |
| 保存stateのsignal | referenceは<ψs|exp(−iHDF T)|ψs>、ψsの影響はrawにも含む | 直接riskのtarget差だけを足す際、state効果を重ねて加算しない。厳密単一位相への置換は新条件 | A/Bの契約。新state生成なし | P0の会計確認、調査済み |
| R-aware u／logB margin | 13roundの経験的allowanceとSOURCE、rigorous=false | 認証raw座標enclosure又はその仮定に限定した条件付きclaim。経験的uを真のηへ読み替えない | Aの数値証拠。厳密enclosureが必要なら新科学計算、Bがriskへ伝播 | 数学的globalではP0 |
| m3/m4 norm | 観測scalarなし、SOURCE_GATE_LIMIT_NOT_OBSERVEDの条件付き診断 | 欠測を保持。観測norm2値の取得だけでは全数値誤差は認証されない。verified mean経路で包含する案も可 | A。再伝播は別承認、最初の契約整理に不要 | P1、厳密raw認証に含める場合P0 |
| ideal finite-K生成構造 | C d72dc4：同support/τ/metadata、rte.py blob対応を確認 | exact重み・event作用・実CDF・PRNGの誤差を0にしない。SOURCE一致はlaw距離の値ではない | Cの既存成果を再利用、Aで再実装しない | P0の資格整理、調査済み |
| P/Q/F/GのTV | CのFraction/CDF helperとF0有限案あり、実H4 TVはnull | 保存324 component＋13 order分布のF0。解析重み→source重みや作用誤差は別 | Cの別承認科学再解析。今回実行0 | actual-lawへの移送ならP1 |
| actual joint law | PCG64 fixed seed・seed namespaceの仕様、IID証明なし | 独立grid又は履歴条件付き距離を明示。周辺TVだけでは全shotデータlawに移せない | Cのlaw整理＋Bのjointモデル、ユーザーが主張水準を選択 | ideal限定なら条件を明示、actual保証ならP0 |
| S3のPCG64/binomial | 保存binary64確率で数値MCを実行 | 統計CIは採択IIDモデルを条件とする名目評価。固定PRNG列を独立乱数の数学的証明としない | Bの統計契約、CのRTE samplerとは別の数値実装 | P0の資格整理 |
| 回路作用・hardware | 今回のS3は実回路実行ではない | 実backendを主張する場合の実装・noise・state preparation誤差。価格loweringや価格残差からは移送不可 | 対象拡張時の担当・ユーザー。A/BでC/Dの価格研究を重複しない | P2／実backend主張ならP0 |

## 3. 認定への依存関係

| 求める結論 | 必要な最小セット | それだけでは不要なもの |
|---|---|---|
| 現S3の正しい記述 | 固定SOURCE・標本値・条件付きlaw/proxy・CIの限界 | 新simulation、新N、追加物理計算 |
| 固定plug-inモデルの全risk上側評価 | 全事象loss又はenergy補完family、support、population評価法とconfidence・数値誤差 | 共通effective H、PF/finite-Kのenergy分解、Hmol DF誤差 |
| 保存HDF groundへの条件付きrisk上界 | 上記＋狭いground包含＋解析rawとの数値差。各条件を明示 | 実RNGの一般IID証明（idealモデル限定の場合）、新state生成 |
| 既定εqpe/εtrotの個別達成 | 基準Z0・coupling・normalization/N・全事象出力の契約と2つのL2上界。target/law/数値項の配分 | raw全riskだけでの「個別予算PASS」 |
| actual-law／数学的global bound | 上記主張に対応する認証入力・joint law・target・全事象riskの上界 | nominal式・CI field名・零観測・SOURCE一致だけの昇格 |

**十分Nの意味**：固定13round・全26軸のN配分が、特定のtarget/law/decoder/loss/精度契約で要求riskを満たすという証拠を指す。各axisを独立に「十分」と呼ぶ証拠ではなく、最小shot数や最適配分でもない。confidence付き証拠と決定論的認証を別資格にし、既存packetのN_X/Y／global boundは今回nullのまま。
