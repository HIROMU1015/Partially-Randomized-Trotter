# O2後続：研究候補と次の最小検証

2026-11-06 JST。候補の技術整理であり、正式採択・新規性・科学的GOの判断ではない。文献の根拠は[限定調査](VARIANCE_REDUCTION_LITERATURE.md)、数理/実装境界は[適合条件](PR_RPE_ESTIMATOR_COMPATIBILITY.md)。計画のみで実装・科学計算は実施していない。

**条件付きの最小検証候補としてA1を残す。** CV4Quantum型control variateを、既存canonical PRの全interrogationと実測outcomeへ適用する既知法の応用である。総費用改善を示したわけではない。A2のaugmented MLMCは測定機構の候補として記録するが、現段階では数値pilotへ進める仕様が足りない。B/Cは既存Trackへ対応付ける。

## 1. 候補の分類

| 対象 | 具体的な機構 | 最も強いbaseline | 主な障害／判断を変える最小情報 | 今回の位置付け |
|---|---|---|---|---|
| **A1：既知factorized path controls** | 同じlaw・回路・finite targetの実測wYから、既知平均の古典path関数を差し引く。非binary estimatorでpath間varianceを減らし得る | raw canonical＋適切なvariance-sensitive confidence、order/利用可能label層別とcost-aware配分、同情報で利用できる既知IS/CV | 真のcovarianceは未知、量子noiseは残る。pilot/校正費、range膨張、層別への還元を含む総費用。実outcomeによる独立校正で判断できる | **最小検証案を残す。未採択・未実行** |
| **A2：測定まで結合するscaled augmented MLMC** | L-MLの非unitary difference-stateと小norm observable | 独立fine/coarse、単純coherent分岐、既知MLMC-qDRIFT、通常PR/強PF | coherent target対応、S補正、dilation成功率/価格・moment/rangeが未取得 | 文献機構として保留。量子pilot案を発行しない |
| order/sign/constant-weightだけのcontrol | 既知有限stratumの平均ゼロ補正 | 同じstrataの条件付き測定・配分 | order-only利益は層別に吸収。constant weight controlは分散0で情報なし | 単独の追加候補にしない |
| path乱数だけのantithetic/inter-shot相関 | path平均間のcovarianceを変える | 同じlawのIID、層別、既存相関監査 | 測定noiseの結合とmean/有限confidenceが未特定 | 支持する具体候補なし |
| 未知量子referenceのcontrol | 別回路測定を補助量とする | 安いreferenceのみ/独立差分/既知CV | reference平均とjoint measurementが有料・未指定。E₀/状態vectorを無料にできない | 今回支持せず |
| adaptive confidenceだけ | 停止/統計保証を変える | 同じ既知保証を認めたbaseline | O2-S1の既知接続。期待停止費の改善は未検証 | A1等の共通部品、独立新候補でない |

A1は「最適なCVが既に得られた」という意味ではない。既存SOURCEが安価に取得できるlaw/metadataを持ち、一次文献が実測noise込みのCVを扱い、追加量子gadgetなしにcovariance・有限confidence・full費用を一度の限定検証で区別できるため残す。任意のランダムfeatureを増やせば改善するとは仮定しない。

## 2. A1：何を変え、何を固定するか

固定するものは、元H/state、finite PR target、PF/RTE条件、ordered full circuit、実sampling law、native profile、X/Y readoutである。変更は実測値に付ける古典的なcontrol補正と、正当な補正後confidenceの会計に限る。path lawを再最適化するS2や、係数/回路辞書を変えるRAではない。

L-CV Eq.(33)のfactorized関数を、既存 `rte.py` のorder/selected labelsへ対応させる。具体的な入力候補は、**既存orderを条件とするlocal label indicator、および異なる既存label位置のindicator積**である。ideal categorical lawでは、平均はそれぞれ既知order確率×label確率、既知order確率×label確率の積から取得できる。これは既知factorizationの適用例で、新しいcontrol原理・定理ではない。

使用するoccurrence、order、label集合、factor集合は既存catalogから結果前に有限に選ぶ。全path indicatorや、保存stateから作るpath平均をcontrol bankへ入れない。実finite-bit lawに対する平均・共分散・rangeが取得できなければ、この案は未適格とする。

order-onlyのcontrolは比較用に置き、その先でlocal-labelに関連する変動が残るかを見る。同じ選択coordinateを使う安価な層別対照を認める。小さなcontrol bankが完全な小strata表と同じ推定器へ還元されるなら、独立利益として数えない。一方、全path表や全stratumの未知平均をbaselineへ無料に与えた比較にもせず、取得可能な情報・前処理・学習費を揃える。

## 3. A1の最小科学検証案（未実施・別許可が必要）

### 入力・対象・比較

新しい分子を作らず、既存main系canonical PRの保存入力とscheduleを優先する。再利用候補にはlinear H4、1.00 Å、STO-3G、legacy DF12、Partial L_D=3の既存cohortがある。ただし保存DF/stateと元の物理Hの厳密groundとの結合は条件付きであり、未近似分子の保証を追加しない。[既存V1/V2][V2]。**C1のR1/324-label packetや凍結pilotは実行対象にしない。**

ユーザーが選ぶ既存schedule内の一roundについて、同一時刻・同じ保存r/K・両axisのfull interrogationを比較する。既存wrapperの受理領域に入る保存roundを選び、未対応の長qへのbuilder拡張を初手に含めない。新しいr/degree/precision/成功確率/主資源/compilerは本書で固定しない。最初の検証を全18roundの新計算に拡張しない。

比較は、raw canonical、同order等の利用可能な層別/配分、factorized control適用を共通taskで扱う。既知ISやClosed等を関連baselineに使う場合も、正当に取得できる既存情報・価格と同じtarget/phase/precisionを与える。新しいIS探索やRA実験は含めない。baselineにも同じvariance-sensitive保証を認め、旧range-only Nだけを弱い対照として固定しない。

### 一束で必要な測定・統計・価格

1. **mean/centeringの接続**：固定された元SOURCEのlaw/normalization、全path phase/scalar、control入力と実qに対する既知平均を対応させる。丸め/centering biasを明示する。未知の真signalをcontrol作成・λ選択へ渡さない。
2. **学習**：実Hadamard outcomeとpath metadataを得る有限pilotで係数を推定し、結果前に決めたcontrol集合内で凍結する。最適λのoracleを使わない。control数・range・数値安定性も固定範囲へ含める。
3. **独立校正とproduction予算**：凍結後のfresh標本で、既知rangeとL-EB Theorem 10等から補正後varianceの上界を取得し、校正の失敗予算を含めて既知Bernsteinの十分Nへ接続する。その後のproductionは新鮮な独立trialとする。同じデータでλ選択・校正・成功判定を無補正で兼用しない。
4. **費用**：学習・校正を含む全quantum trialに、状態準備を除いたD halves/中央RTE/control/ancilla/readoutの資源別価格を対応させる。pilot covarianceに実量子noiseを残す。追加control評価・moment取得・回帰・記録・memoryを別座標で保存する。
5. **対照との判断**：同じ有限confidenceと統計残余幅に対する総費用を比較する。raw/層別に不要な学習を強制せず、各方式が必要とするpilotを各方式の費用に計上する。共用データを用いる比較では、その再利用と本運用時の費用の帰属を分ける。

上記3段階の標本区分は、係数選択とconfidenceの依存を切る既知の統計的条件である。別々の承認段階や新しいgrant/ledgerを作る要求ではなく、一つの明示された科学実行範囲として扱える。所定scopeの全作業を進める許可は後日別途必要である。

必要Nは補正後variance上界とrangeから求める**十分条件**であり、最小shot数ではない。校正標本の点varianceやpath期待値から無保証Nを作らない。観測による適応停止を初手には採用せず、将来採用する場合だけL-CS等と期待累積費用を使う。

### 再利用と最小追加差分

既存RTE generator・Hadamard builder・資源provider・phase decoderを再利用する。必要差分はmetadata/outcomeの対応、既知control momentと有限bit centering、独立fit/variance校正、補正後confidenceの小さい後処理adapter。旧SOURCEや固定結果へ追記しない。新しい量子回路/normalizer oracle/全path table/汎用optimizerを作る計画ではない。具体関数・testの対応は[適合条件§6](PR_RPE_ESTIMATOR_COMPATIBILITY.md#6-静的に確認したsourceと最小差分)に記した。

既存C1 logical pathsやS2 pair-tableは、実測outcomeとのcovarianceを持つ量子測定campaignではない。そこから新しいcovariance・variance ranking・CV削減率を再解析しない。既存testはphase/support/weight等の回帰資産として使えるが、CVの利益を証明しない。

### 研究判断を変える観測・終了条件

継続材料になるのは、同情報の層別/配分・既知lawを認めても、既知control平均を取得でき、独立測定で有用なcovarianceが確認され、rangeと全pilot費込みの十分N×full価格に差が残ることである。これは今回確認された結果ではない。

以下ならその具体案の終了/縮小材料とする：controlがconstant/order-only等へ還元、within-order covarianceが弱い、量子noise床が支配、λ/rangeや校正費が節約を相殺、ν/有限bitの取得が重い、未知state/referenceが必要、同じ情報の層別対照が同等以上。新しい分子/parameter/control bankの無目的探索で救済しない。1roundで利益が見えた場合も全RPEへ自動展開せず、round別の半径/bias/branchと学習負担を含めた次scopeを判断する。

## 4. A2：残す問いと今回数値案を作らない理由

scaled augmented MLMCは、単なるpath correlationとは異なり測定observable/stateを変える具体機構である。[L-ML][ML]のLemma 2はscaled normのmoment条件付きでshot varianceを制御する。しかしPRには、既存finite-RTEのcoherent平均を保つlevel対応も、非unitary block/S補正/成功率/native価格の実装もない。

次に必要な最小材料は、新しいHamiltonian実験ではなく、既存PR full targetに対して「何を準備し何を測り、Sをどう取得/補正し、成功/失敗をどの分母・費用で数えるか」の有限アクセス仕様である。これは未完成な測定アルゴリズムの設計/検証を必要とし得るため、今回は作成・採択しない。新しい重要理論レビューや非自明証明を自動開始しない。A1の実施条件ではなく、別の問いとして残す。

## 5. B/Cと既存Trackへの対応

| 方向 | 固定既存資料と重複 | Aへの接続／今回しないこと |
|---|---|---|
| B：round別bias/半径/N/価格 | UWC、既存O4、V1/V2/energy bridge。条件付き18round成功は既存成果 | A1の各axis confidenceを同じbranch/energy条件へ戻す層。単純round配分を新機構にせず、新optimizer/scheduleを設計しない |
| O1/Track A | [O1 preflight aeb9d18][O1]のM1辞書変更とM2 core/split変更 | mean/価格/normalizationを変えるO1と、固定lawで後処理だけ変えるA1は別。組合せ利益は未検証で加算しない |
| C/O3 | [O3 pilot plans 66553d9][O3]のC-N（PF）、C-S（同位置residual/合成）、C-P（processing）。TE-PAI/TETRIS、SPRINTの既知対応 | bias/step/nativeを変える課題はO3へ引き継ぐ。A1で同時にPF/合成を変更しない。PRHS/qSHIFTもcoherent mean移送・符号負担を残す |
| S2/R1 | [S2 dc42c3a][S2]、[R1 b6305ee][R1] | S2の固定T不改善、R1の候補なし、Closed/CTS、C1凍結を維持。A1は未知νやoracle covarianceを使う形でこれらの負結果を回避しない |

現時点で独立に進める数値検証案として残すのはA1のみ。A2はアクセス検討の別論点、B/Cは他Trackの既存範囲である。「並列」の名目でAI sub-agentや科学workerは起動していない。

## 6. 次の担当と許可

**ユーザー／GPT**：A1の応用検証に投資するか、元H/stateと既存入力・一round・精度/成功条件・主資源/profile、control/比較集合と有限測定scopeを判断する。A2の理論/アクセス検討を別途扱うかは独立の判断。

**Codex**：後日の明示指示がある場合のみ、その範囲の最小実装と人工確認を行い、新しい科学計算/測定simulation/校正/N再会計/compile等は許可された固定scopeだけで実施する。重要科学レビューの開始許可と科学実行許可は独立。今回の4文書公開・取得照合でmandatory STOPする。

[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[ML]: https://arxiv.org/html/2604.26865v2
[O1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/REPORT.md
[O3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66553d948aa9df684987972789c6f3ee8a741312/docs/research/track_c_o3_pilot_plans_20261106/REPORT.md
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/REPORT.md
