# 最小P2計画：平均compiled価格のためのpaired残差評価

2026-11-13 JST。**提案のみ、未承認・未実行。** 本納品は技術改善と人工検証まで。
旧[SCIENCE_PLAN](../d_compiled_residual_c_minus_s_20261113/SCIENCE_PLAN.json)を変更しない。
そのn=8、全calibration/holdout ladder、P3を実行する前提は置かない。

## 1. 推奨と目的精度

第一候補は、必要なroundでの**直接paired sampling**である。同じpath/axisの
full-wrapper Cと元AccountingのSを取得し、D=C−Sを用いる。

\[
\widehat\mu_{C,m,a}=\mu_{S,m,a}+n_m^{-1}\sum_i D_{m,a,i}.
\]

Sの係数は1で固定し、最適control-variate係数のfitも追加しない。
同一lawならこれは不偏な制御変量推定量であり、そのMonte Carlo分散はVar(D)/n。
Var(C)より小さいことは一般保証ではなく、pilotで確認する。
価格model、機構分解、校正/holdoutはこの直接推定には不要。

最終比較に必要な精度は比較対象間の差と使用するroundに依存する。現時点で十分Nと
最終比較差が未資格なので、唯一の必要精度はまだ決まらない。
**計画用の提案値**は各価格の絶対誤差2%×P、Pは当該軸の保存E[S] nominalとする。
confidenceは使用価格全体で95%（X/Yを26独立標本として扱う意味ではない）。
これは研究者による採択待ちの値で、達成済み精度・十分条件ではない。

- 直接推定の計画予算：cohort誤差1.2%×P（Sの数値enclosure半幅をここから控除）、
  law差0.4%×P、rare未解決成分0.4%×P。model誤差は直接測定点では0。
- 通常の無条件sampleを使い、全supportを扱うCIがrareも包含する場合、rareをCIに
  再加算しない。上のrare枠は別扱いの尾部評価を選んだ場合の予算である。
- E[C]そのものに対する相対2%を表示するには、最終区間[L,U]でL>0かつ
  (U−L)/2≤0.02Lを追加で確認する。S基準2%をC基準2%へ自動変換しない。
- 後日Bが固定した非負の重みwを供給するなら、価格誤差の寄与はΣw eで配分できる。
  各軸を一律に高精度化する必要はない。今回はN・重み・Gを計算しない。
  比較区間が重なる場合、winnerを出さず精度・対象点を再検討する。

残差自身を相対高精度に推定することや、Dの全機構を説明することは要件にしない。

## 2. 既存証拠の再利用

| 固定成果 | 再利用できること | 再利用できないこと |
|---|---|---|
| D P1結果 `54b64ef72bf507be25970f2d3362d3be091321c3`、実行 `20fe11942369253361068dd2b00f98b26350ba8d` | m0/m6各2 path、両軸compile成功、全4 pathのD>0、X/Y価格同値、N_ZZだけでは説明できないという既存観測 | 母平均/分散の精度、Dの常時正値、全軸価格同値、rare寄与、全13roundへの移送 |
| C全round `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | 全26 E[S]を元の数値区間・K0/K1・出典付きでcopy | compiled C認定、sampler law補正、cohort CI |
| C law調査 `d72dc4e6228bb380f4d1d000e5e030794fce7164` | 数学的生成構造、CDF/53-bit格子差の有限評価方法、PRNG条件の区別 | 実H4のlaw距離の実数値、固定seed列のIID証明 |
| D adapter `af6f27d7a6467bb355686421acd067544edcbbd2`、B接続 `bf515a976f1987ad6a19c61d14689b27b0894542` | 同path/full wrapperのpairing、role/stratum/identity/law/compiler照合、未資格和の保持 | 未知systematicを0にすること、適格価格への自動昇格 |

E[S]の対象はLinear H4/1.00 Å/STO-3G、保存tol-only DF10、L_D1/S2/ΔPF=.1/
finite-K2/h=.1/M12/P0、元exactκとH/state/snapshot・fragment/basis/qubit順・
scalar一回・symmetric-directional full Hadamard X/Y。
canonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、
preparation `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`。

新lowering `CONTROLLED_ZZ_CX_CRZ_CX_V1`は元Sのcrzz展開と一致するため、
全26値の再計算は不要。m0/m6の旧kernel_sourceと他22値のsourceの違いも残す。
S/C scopeとlawの照合は必要で、Sの区間はsampling誤差やlaw差を含まない。
旧loweringのC1係数・旧価格を新loweringへ移植しない。

P1を計画立案に用いたため、今後はDEVELOPMENTのまま保存する。
新pilotの独立標本数へ加算せず、今回P1の平均・分散を再計算しない。

## 3. 次に提案する有限実行：最大20 full-wrapper compile

対象SOURCEは本納品の公開技術commitを基礎とする。専用の有限runnerを後続作業で
追加する場合、そのcommitを実行前に固定する。P1の旧runner・PLAN・grantは使い直さない。
現状は新しい科学実行入口を追加していない。

compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、settings hash
`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`で固定する。

| 区分 | 対象・path | 新sampling | compile上限 | 目的 |
|---|---|---:|---:|---|
| 実移送の技術照合 | 保存P1 run02 m0/index0、m6/index0を新SOURCEで各1 replay、X/Y | 0 | 4 | event digest・wrapper fingerprint・6指標の保存record一致と実H4の性能測定 |
| 直接残差pilot | m0(q1/R2)、m6(q64/R128)、fresh各4 path、X/Y | 8 path | 16 | Dの平均・ばらつき、直接法の必要nと実行可能性の見積り |
| 合計 | replay2＋fresh8 | 8 path | **20** | 全law認定・fit・N/RMSE/Gは含めない |

cache hitは各軸のfingerprint/compiler照合後だけ再利用し、compile節約分でpathを追加しない。
両軸を取得し、一方の価格を無条件に代入しない。

fresh cohortは仮名`D_P2_DIRECT_PILOT_V1`、index0..3、元carrier trial_seedから
candidate/cohort/index/outer/pieceごとに実行前に固定する。P1とseed namespaceを分け、
全入力のhash・準備・request/step配分・control/lowering identityを保つ。
replayは保存eventの使用で、再samplingやcohort拡張をしない。

4 pathは**小さな分散pilotの予算**であり、2%を達成するsample sizeという意味ではない。
2組のpairを比較する程度の初期情報を得るための提案で、数学的な最小数でもない。
新規calibration round=0、holdout round=0、fit=0。

この有限実行を将来承認する場合は、技術照合→fresh pilotを一つの範囲として扱える。
正常時に処理段階ごとの再承認は不要。一致失敗、入力破損、compiler/lowering差、実際の
継続不能を検出した場合は原因を保存して停止し、科学retryや予算追加へ進まない。
pilot完了時もSTOPし、必要な追加n/roundを有限案として返す。

## 4. pilot後のsample sizeと成立条件

pilotでは軸ごとのD平均、ddof1分散、X/Y共分散、S/C分散、cohort内の異常値、
order countsを記録する。X/Yは同じpathの応答でありnはpath数。
分散が小さく見えてもn=4やrare未観測から母分散0・全law精度を認定しない。

次の独立した確認cohortの計画値は、残りの絶対MC予算bに対し

\[
n_{m,\mathrm{plan}}=\max_a\left\lceil(z_{1-\alpha_{m,a}/2}s_{D,m,a}/b_{m,a})^2\right\rceil.
\]

これはpilot分散からの**予算見積り**である。pilotのsを真のσと置いた保証ではない。
観測価格を見てnを増やして通常の固定n CIを何度も試す運用は採用しない。
pilotと確認cohortを分け、確認nを先に固定する。確認nの上限と全compile数は結果を踏まえて
別の有限計画で明示する。足りなければ不適格のまま終了する。

最終の信頼性については以下を選ぶ必要がある。

- 研究用のIID/PRNGモデルと有限分散・近似的な正規性を採択する場合、t/正規区間は
  近似の区間として表示する。n=4ではその正確さを保証しない。
- 厳密な有限標本保証が必要なら、SOURCEからの全support D範囲または同等のtail保証を
  用意し、bounded-variable concentration等を使う。観測min/maxはsupport上界ではない。
  範囲が広すぎて予算に達しなければ、その計画では資格不能とする。
- 26価格を同時に使う場合は例えばα=.05/26でunion boundを使える。X/Yの独立性は不要。
  複数回の判定をする設計なら、そのlookにもαを配るか適切な逐次区間を使う。
  科学的な厳密性の選択自体を技術PASSで代行しない。

直接法は測定したroundの価格だけを評価する。m0/m6 pilotで他11roundを埋めない。
最終的に全26 E[C]が必要なら各点の直接証拠、検証済みmodel誤差、または十分な解析上界が必要。
全roundで同じnを強制する必要はないが、未測定roundを0や同一係数で埋めることはしない。

## 5. rare order2を調べる段階

pilotは無条件samplerのまま取得し、強制rareを各roundに追加しない。
P1のorder2未観測は既存観測であり、rare寄与0の証拠ではない。
平均価格を資格する**前**に、対象lawでp_A=P(path内に少なくとも1 order2)と
必要な価格範囲/tail誤差を評価する。Cの既存有限law診断案F0を利用できるが、実数値計算は
今回も上記20 compile案にも自動では含めない。別途、保存確率の有限評価として許可範囲を明示する。

無条件平均はrareも含めて不偏だが、少数標本の標準誤差だけでは未観測tailを抑えられない。
全support CIが十分なら追加rare実行は不要。order0近似を使う場合は
p_A|μ_{D|A}−μ_{D|A^c}|の上界をrare枠と比較し、十分小さければ細かな機構分解を省く。

上界が不十分な場合だけ、真のconditional lawに従う層別samplingを計画し、
(1−p_A)μ_{D|A^c}+p_A μ_{D|A}と層別不確かさを合成する。
固定位置・固定productへの強制一例はconditional分布ではなく、その平均推定に使えない。
複数order2の確率と残余上界も必要。無条件cohortとの二重計上を避ける。
特定eventの機構完全分解ではなく、誤差予算を満たすことを判定条件とする。

## 6. law差と有限cohortの分離

IをCのideal source law、Qを標本化すると仮定する実装lawとする。
`E_I S + mean_Q D`をIの価格へ使うbiasはE_Q D−E_I D。
Qの価格へ使うbiasはE_I S−E_Q S。target lawを先に選ぶ。
このbiasはnを増やしても消えず、cohort CIやK1の数値enclosureと別に保持する。

`d72dc4e6`の方法でweight正規化・CDF丸め・53-bit格子差を調べ、適用条件付きの
trajectory couplingとoscillationから価格影響を評価するのが最小のlaw作業。
eventの周辺TVだけで固定PCG64列のjoint-law距離を保証しない。
seed namespaceの分離もIID証明ではない。実用PRNG/IIDモデルを明示採択するか、厳密な
joint-law保証を要求して未資格を維持するかを区別する。law不明を0にしてBへ供給しない。

## 7. modelと大きなq/R：必要になった場合だけ

直接法が必要roundまで実行可能ならmodelを作らない。m12の保存schedule metadataは
q4096/R8160、33,423,360 event位置であり、今回の人工256 event benchmarkから
full compileのwall/RSSや実行可能性を外挿しない。

modelが必要なら、まず最小1係数の候補例`μ_D=β(qR)`を**仮説**として選び、
少なくとも2つの異なるroundのcalibration平均で1残差自由度を持たせる。
例えばm3/m6各4 pathを校正、fresh m7各4 pathをholdoutとする案は
3 round/12 path/24 compileの診断案になる（今回の20 compileとは別枠・別承認）。
元P1を校正へ流用せず、m0は境界項を無視せず直接扱う。m3のq/RはB元metadataから取得する。

2係数を要するなら最低3 calibration round＋1 fresh holdout roundが必要。
同一点のpath増加で設計の階数やlack-of-fit自由度は増えない。
校正後に係数・誤差metric・絶対許容差・適用domainを凍結してからholdoutを見る。
holdout平均のsampling誤差とmodel誤差を分離し、2%の合計予算内に収める。
holdoutを見て形式を変更した場合、そのholdoutはDEVELOPMENTへ戻す。

m7 holdoutを通してもm8..12を資格しない。大きなq/Rへ使う最小の追加確認は
**最も大きな使用点での独立paired cohort**（初期提案4 path、8 compile）であり、
大きな範囲を跨ぐなら中間点の独立確認も必要。nは精度式とtail条件で再決定する。
最大点が実行不能なら、短いwindowの成功だけで全回路へ外挿せず、有限のcompiler影響範囲や
残余上界の解析が必要になる。得られなければ長roundはUNKNOWNでSTOPする。

S0経路上でqとRがともに増えることから、q効果/R効果を個別に識別したとは言わない。
S0以外のscheduleへ移送しない限り、旧P3のoff-path統制は不要。
N_ZZ-only modelもP1から採択しない。局所機構の追加分解はholdout誤差が予算を超えた際に、
その原因を絞る範囲に限る。

## 8. 判定と次の担当

今回の達成は人工同値性・人工性能のみ。P2の完了や適格E[C]ではない。
将来の価格認定には、同一SOURCE/physical identity/lowering/compiler、全wrapperのpairing、
選択したlaw・cohort・rare・model・S数値区間を含む誤差予算と使用domainの充足が必要。
`compiled_residual.combine/b_price_record`の未資格表示を黙って変更しない。

- **ユーザー**：2%×E[S]の計画精度・95%同時confidence・law/PRNGの扱いを判断し、
  必要なら上記replay2＋fresh8・最大20 compileとその固定SOURCEの科学実行を別途明示許可する。
- **Codex D**：承認された場合のみ専用runner/SOURCEを具体化してその有限範囲を実行。
  現在は成果物commit/pushとremote照合後mandatory STOP。
- **Codex C**：現時点の追加作業なし。law診断F0が必要になれば別途明示範囲で担当する。
- **Codex B**：現時点の追加実行なし。将来の比較精度・target law・必要価格domainを明示し、
  未測定round・未知誤差・旧loweringのidentityを保つ。
