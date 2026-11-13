# 全RPE総コストでベースラインを探索する計画

設計のみ。11案の採択、実H4の最適化、実G/新N/新価格計算はしていない。既存4方式controllerと担当別providerを使い、探索器・RPE統計器・compiled価格取得器をCで重複開発しない。

## 1. 最適化問題と比較の二段階

方式b、全schedule θ_b、shot/誤差配分ν_bに対し

```
min_(θ_b,ν_b)  G_b = Σ_(m=0..M_b,a=X,Y) N_(m,a)(θ_b,ν_b) μ_(m,a)(θ_b)
subject to     R_b(θ_b,ν_b ; H,state,prior,decoder,target) <= ε_E²
               legal circuit/law/allocation and numerical/price qualification
μ = E[C_full controlled Hadamard wrapper including measurement], P=0.
```

Rは採択した全事象lossに対するエネルギーMSE。高確率型を使う比較なら、その同一成功条件を別の共通契約として使う。RMSEとfailure probabilityは同じ尺度ではなく、β=.4やα=.05、有限時間signal精度を満たしただけで上式を保証しない。未定義outcomeを捨てたconditional RMSEをpopulation RMSEへ付け替えない。

signal/価格取得・探索の古典計算時間は実行可能性を決める補助指標で、上記の量子RZ費用Gへ加算しない。価格標本を増やす費用と、量子interrogationのNを増やす費用も別に扱う。

**第一段階**は保存研究を活かすh=.1/M12の共通shellで、回路scheduleと方式別N政策を評価する。**第二段階**としてh/M/誤差配分も探索する場合、全方式に同じ事前prior、alias/decoder条件、候補範囲とclaimを与える。hは現B selectの比較group条件、Mは既存全configurationの変数。h横断比較にはBが比較groupを整理する必要がある。

PF/PRのδ・qとq_RPE=2^mを混同しない。h/Mを固定した最良をRPE全自由度の最適解と呼ばず、逆にPRだけのh/Mを固定してbaselineだけに自由度を与えた結果を同一探索範囲と呼ばない。合法な異なるM同士も同じtarget精度を満たせば比較候補だが、使用roundと統計契約を全て記録する。

## 2. 候補生成の軸と既存APIの境界

|方式|最初の有限family|拡張を検討する条件/未探索域|
|---|---|---|
|Full PF|3公式S2/Y4/M5 ×外側Δ又はq。旧.1/.4をseedに、既存Track Aの時間分割候補を参照|同formulaの内側/外側neighborを検討。q別actual widthとnative templateを記録。round別q_m/formulaは将来の明示scheduleで、既存global Δ familyの網羅と混同しない|
|Truncated PF|p∈{実snapshotの合法prefix}、B0/B0I、3公式、Δ/q。p7とnear-fullをseedにする案|小pも大pも最適になり得る。p7/p9だけで全cut探索済みとしない。PRと同p対照は追加の説明軸。任意fragment subset/並べ替えは新問題なので含めない|
|Full-H pure RTE|K∈{0,2,4,6}という既存研究の開始候補、明示s_m/q_m/R_m。旧K2/K4の1本のT依存layoutはseed|R係数448、T²型、K固定が最適とはしない。neighborのs/K、groupingを変える余地を残す。global K・明示layoutは既存API、K_mは未接続拡張|

どの範囲も今回正式採択していない。科学的な有限探索範囲/最大候補数と、CPU/worker/RAMの運用上限は別物である。前者は実行計画で固定し、後者に恣意的な低いcapを置かない。無限のq/K全体の最適性は、未探索域の有効な証明がなければ主張しない。

新candidateのT/qは実幅、requested Δとは別field。ceilで同じ物理回路になった異scheduleは、物理identity/law/numerical SOURCEが同じ場合にsignalを共有し、configuration IDはそれぞれ保つ。B0/B0Iはscalar規則が異なるためラベルだけで共有しない。

## 3. 価格だけ・signalだけに偏らない評価順

1. **metadata接続と保存再利用**：同H/state/target/formula/split/scalar/schedule/law/profile/loweringを照合。Full PF receipt73件はDの既存判定を参照し、新監査しない。Track A最良候補はseedとして保持し、同RPEの資格は別に扱う。
2. **安価なscreen**：保存signal又は許可された限定mean、normalization・数値安定性・構造proxyを使う。代表m3/m6/m12はregimeを見る診断であり、全roundのfeasibility/総費用順位ではない。直接expmの新しい比較実験を初手の目的にしない。
3. **まず実行可能な全scheduleを一つずつ得る**：各方式の有望recipeの全round meanを保存再利用＋不足分で揃え、担当の同一risk/decoderからNを取得する。値のない方式を敗北とせずUNKNOWNとする。PRにも同じqualificationを要求する。
4. **価格の取得と誤差契約**：deterministic直接Cを優先。長qには適用条件付きaffine/analytic候補を価格担当が検証。randomizedはlaw平均とfinite cohortを分け、必要なdomainだけnormal/rare/残差を扱う。
5. **Nと価格の共同調整**：最初に固定Nで価格誤差の影響を切り分けられるが、それは条件付き会計。最終的には方式別shot配分を価格とriskで再評価する。全round riskが分離可能と証明されない限り、各round単独最小解の貼り合わせをしない。
6. **境界challenge**：有限menuの最良がΔ/p/K/sの境界なら隣接候補を小さな追加batchへ提案する。価格の安いsingle-round候補だけを優先せず、精度改善とN変化の可能性を残す。新候補や追加科学処理は別途明示許可scope内のみ。

signal・referenceの共通cache、直接expm因子の同template再利用、Full-Hの同s/K mean共有、価格の同物理domain cacheを使う。物理signal同値性とcompiled価格同値性を別にチェックする。保存値の新解析やscreen順位計算も科学実行なので、今回実施しない。

## 4. 除外・優先順位付けの区別

|根拠|許される扱い|禁止する飛躍|
|---|---|---|
|違法な整数allocation/別H/不正SOURCE、未対応公式|その物理requestを受入れない。原因を記録|方法全体が高費用/精度不成立と呼ばない|
|数値gate失敗・underflow・欠測|UNKNOWN/NOT_EVALUATED、対象の科学資格なし|候補のphysical infeasibilityや方式の敗北としない|
|同target/decoder/全事象lossに対する証明されたrisk下界がε²を超える|その指定candidate/domainを不成立として除外可能|norm誤差の上界が予算を超えただけで不成立にしない|
|適用前提を満たすcost下界G_Lと、実行可能incumbentの上界G_UがありG_L>G_U|同時成立する範囲で除外。confidence条件ならその条件付き除外と明記|未資格affine・event数×単価・pilot最小値をG_Lにしない|
|既存の正確な物理同値性|同じsignalを一度だけ評価できる|qRのmean共有から同じ価格/seed law/IDを推論しない|
|局所bias、attenuation、構造S、Track A順位、未検証価格model|計算順・候補提案のヒューリスティック|候補を永久削除しない。未評価候補と根拠を記録|

部分roundについて有効なN_L,C_L≥0があれば、`Σ_resolved N_L C_L`は残りを非負として全費用の下界にできる。ただしN_Lは同じ全RPE taskに対する**必要数の下界**を要し、既存の十分shot数やpilot nominal Nは代わりにならない。C_Lもcompiled full-wrapperについて必要。sampling law/model systematicを含む同時区間の成立条件を明示する。

Full-H RTEに対して現SOURCEで確認できる普遍的compiled RZ下界は0。情報量からのN下界が得られても、非自明なC下界等がなければ大きなG下界にはならない。大Bや大event数だけで排除しない。厳密なboundが役立たないなら、methodを残して限定signal/直接価格の実行可能性を後で調べる。

## 5. 誤差台帳・価格精度・探索停止

各scheduleで同じloss/targetの下にcutoff/PF/finite-K、RPE統計/分解能/branch、数値/state/lawを記録する。直接raw riskに既に含まれる項を再加算せず、経験的allowanceをrigorous boundへ格上げしない。Truncated PFのretained-H energyをtargetへ置き換えない。

価格精度はcdf6a806の契約を使う。固定Nで `|G−Ghat|≤Σ N_j b_j`、価格同時区間なら `[ΣN_jL_j,ΣN_jU_j]`。未知Nにnominalを代入して最終精度とせず、まずdomainとabsolute/relative価格誤差profileを固定する。X/Y path共分散・共有係数を0にせず、価格confidenceとRPE energy精度のconfidenceも別台帳にする。

deterministic affineは固定templateごとに直接点を再利用し、校正/独立検証を分ける。q増に対する費用単調性やT/角度独立性を未検証のまま採用しない。モデル失敗ならdirect/domain縮小の別案へ戻し、閾値緩和やholdoutを使った自動refitをしない。

探索の完了は「採択した有限domain内で、同じrisk/価格資格を満たし、G区間が判断に十分分離した最良候補」又は「不足証拠/予算/重なりのため未識別」で報告する。未探索formula/Δ/p/K/s/M/h・round適応を明示する。正規の全域証明がなければglobal optimumとは呼ばない。全G/正式選定の実行許可も本設計には含まれない。
