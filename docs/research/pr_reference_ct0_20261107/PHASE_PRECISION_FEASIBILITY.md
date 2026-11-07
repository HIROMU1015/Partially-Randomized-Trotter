# CT-0：全support位相とprecision実行可能性

[REPORT](REPORT.md)の固定H4/STO-3G/DF12・Partial S2 LD3 h=.02 K2、compiler A、canonical finite-bit lawに限定する。以下の数式はSOURCE対応を調べた技術的導出、数値は承認された保存結果の再解析。新gadget・合成法・precision配分の採択ではない。

## B：位相生成規則

### B′のevent・D・scalar・controlled角

`df_rte_qiskit._build` は、最初に全eventの符号位相を加え、次に `(*products,rotation)` 順のapplicationを処理してproductの負係数ごとにπを加える。保存tailは非identityのZ/ZZ support、K2ではevent orderは0/2。1 occurrence内のπ加算は最大r+2r=3r回。

binary64の列を `p_0=0; p_{k+1}=fl(p_k+float(pi))`、k=0…3rで保存した。+0の挿入はこの非負列を変えない。このsupersetは全SOURCE event位相を含むが、全kがcanonical lawで到達すると主張していない。単一乗算k*piやparityへの置換はしない。

| 保存r | 非ゼロ位相keyのsuperset件数 | k*piとの不一致件数 |
|---:|---:|---:|
| 1 | 3 | 0 |
| 2 | 6 | 0 |
| 4 | 12 | 0 |
| 8 | 24 | 11 |
| 16 | 48 | 35 |
| 32 | 96 | 81 |

Pはcompiler Aで実角度を保ったRZへlowerされる。PとRZの差は既にcontrolledなjoint全体の共通phaseであり、制御前の相対phaseを捨てる操作ではない。実numeric keyごとの保存record照合は `PHASE_SUPPORT.json` に別記し、V3 coverage数に混ぜない。

Dのglobal phaseは保存D lowering/SOURCE scalarに従う。B′のdirectional Dは符号を反転した対角とancilla相対scalarを持ち、境界controlled Dの実half-angleも保持する。RTEのZ/ZZ controlled中央は実±theta/2のnative RZ2個、ReferenceはthetaのRZ1個。productのZ/CZ、X/Y wrapper、CSWAP内のbinary π由来角も評価対象で、exact Cliffordへ無料で置換しない。

### Referenceのvacuum位相

`transfer_qualification.assemble` はq1でG=左D+scalar+tail、F=右D。q≥2では既存境界融合済みの前半/後半をG/Fとする。`reference_core` はFの**実adjoint**とGを別registerに配置し、ancillaに次を与える。

`theta_P = fl(vacuum_phase(F) + vacuum_phase(G))`

`vacuum_phase` は、circuit global phaseを初期値にして、**実際の命令順に**RZ/RZZ角の半分を順次減算する。XXPlusYY/Z/P/idは真空作用に寄与しない。各lane global phase自体もQiskitのbinary64 `rem_euclid(float(2*pi))` と合成順序を経る。RTE basis共有、event符号、product順序、rotation sign/order、D境界、scalarが関係する。解析的な真空位相でbasis対が相殺しても、naive float加算列まで相殺するとは限らない。

有限SOURCE alphabet、q/rで定まる有限長、binary64状態・固定remainder・逐次加算/減算のrecurrenceで全supportを表現できる。これは**同じSOURCE計算法の表現**であって、新位相生成法ではない。少数の整数countだけで実keyが一意になるとは証明していない。全trajectory・全reachable floatの直接列挙は行わなかった。

保存 `phase_arithmetic` のexact-prefix境界Pとnaive最終誤差Eから、`abs(theta_P) <= 2P+E` を外向き評価した。保存phase誤差のSOURCE正規化差は、旧raw diskに含めたまま再加算していない。

| round | actual binary vacuum角の絶対値上界 | lane毎の減算数上界 | 旧raw diskに既に含む位相生成誤差 |
|---|---:|---:|---:|
| m0、q1/r1 | 51.056633906468726 | 409 | 2.8431237013036845e-12 |
| m17、q131072/r32 | 32751.887444214277 | 124780568 | 4.537775727688065e-4 |

全18行は [PHASE_SUPPORT.json](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)。この境界は既存の丸めモデル（round-to-nearest、overflow/underflowなし、保存remainder誤差式）に条件付き。

### 資格の区別

- **位相/誤差域：`FULL_SUPPORT_REPRESENTABLE`**。有限recurrenceと有界角度域、後述の一様operator誤差条件で全supportを覆える。
- **全support strict word/T価格：`UNQUALIFIED`**。角度域が有界でbinary値集合が有限でも、その全域のwordが保存済み・有限の実用的catalogueに網羅済みとは言えない。価格の安全な上界も取得していない。
- **Stage B総合：`UNQUALIFIED`**。`FINITE_COHORT_ONLY`の観測を全supportへ格上げしたものではない。保存cohortの代表位相は今回使っていない。

uniform guardを全位相域で満たすword policyが別途資格されれば誤差側は扱えるが、現段階ではそのpolicyの全key実装・価格保証はない。未確認位相を0、平均、固定π、modulo keyに置換しない。新生成法・universal synthesis価格式・gadgetの探索は開始せずSTOPした。

## C：raw信号への接続

U_S、U_N、U_CTを区別し、追加誤差はU_N→U_CTの全joint演算子差とする。old compiler/source/位相算術は既存の `total_raw_disk_upper=d_old` に**一度だけ**含まれる。旧betaからさらにcompilerを引かず、旧diskを再構築して二重計上もしない。

同じreset/初期化とreadoutを保ち、各trajectoryの測定前unitaryのspectral差がepsilon_CT以下なら、各X/Y期待値の差≤2epsilon_CT、複素信号の差≤2sqrt(2)epsilon_CT。固定lawの平均にもtriangle inequalityで同じ上界が適用できる。channel平均一致をcoherent信号一致の代用にしていない。

局所strict Frobenius guardはspectral norm上界で、unitary telescopingにより各trajectoryの `epsilon_CT <= sum_j epsilon_j`。W、actual adjoint、relative phaseを保ち、合成後に新controlを付けないことが条件。resetは両側で同じ参照準備、測定は同じobservableとして扱う。particle-sector投影はしない。

nominal raw半径rho0は保存transferの外向き下端を利用した。PF phase上界からB=.4−beta_PFを外向きに評価し、

`d = d_old + d_syn`

`rho_lower = rho0 - d`

`beta_stat = B - asin(d/rho0)`

とした。d_syn≤2sqrt(2)epsilon_CTで同じ単位のdiskを合算してから**一度だけ**phaseへ変換する。strict guardがbinary targetとの差を評価するため、π残差を別項で二重加算しない。RTE attenuation/normalizationを1へ変更せず、元のrho0を用いる。既存alias/branch/energy保証は同じphase cap .4のまま継承する。

## 正の予算と保存Nのheadroom

正のrho/betaを保証する追加diskの十分条件は、

`0 <= d_syn < rho0*sin(B) - d_old`。

全18round・両armで右辺が正だった。旧Nを維持する場合は、旧alpha=1/720（36軸の総和.05）について、さらに

`4*log(1440) / [(rho0-d_old-d_syn)^2 * sin(B-asin((d_old+d_syn)/rho0))^2] <= N_saved`

を要求する。192-bit区間で単調性を用いた二分評価を行い、旧整数Nを満たす追加disk上限の安全な下端を保存した。**新しいN、alpha配分、precision候補は選んでいない。** 旧N8598/8606自体は不変更。

以下は保存Nを維持できる追加raw diskの安全な下端の概数。厳密な採択値ではなく、正の余地の説明である。採用する際はJSONの下端以下を使う。

| m | q/r | B′の追加disk余地（概数） | Referenceの追加disk余地（概数） |
|---:|---|---:|---:|
| 0 | 1/1 | 9.5073e-5 | 9.5073e-5 |
| 1 | 2/1 | 5.4566e-5 | 5.4566e-5 |
| 2 | 4/1 | 7.4436e-4 | 7.4436e-4 |
| 3 | 8/1 | 5.8233e-4 | 5.8233e-4 |
| 4 | 16/1 | 2.5861e-4 | 2.5861e-4 |
| 5 | 32/1 | 3.7711e-4 | 3.7711e-4 |
| 6 | 64/1 | 6.0076e-4 | 6.0076e-4 |
| 7 | 128/1 | 2.6585e-4 | 2.6585e-4 |
| 8 | 256/2 | 2.5630e-4 | 2.5630e-4 |
| 9 | 512/2 | 1.9523e-4 | 1.9523e-4 |
| 10 | 1024/4 | 1.5802e-4 | 1.5801e-4 |
| 11 | 2048/8 | 8.3663e-5 | 8.3616e-5 |
| 12 | 4096/8 | 2.2030e-5 | 2.1851e-5 |
| 13 | 8192/16 | 3.5921e-4 | 3.5814e-4 |
| 14 | 16384/16 | 2.0480e-4 | 2.0058e-4 |
| 15 | 32768/32 | 1.3085e-4 | 1.0235e-4 |
| 16 | 65536/32 | 2.0628e-4 | 9.2653e-5 |
| 17 | 131072/32 | 2.6883e-5 | 9.9492e-5 |

正betaだけの余地と保存N維持の余地は違う。例えばm17では前者がB′約.21560、Reference約.21515だが、後者は上表まで狭くなる。これは既存Nの整数ceilの余裕が小さいため。予算を超えても直ちに推定不能とは言えず、新Nの再評価が必要になる。今回はその再評価をしていない。

## 全support回転数と一様precisionの条件

worst-case native RZ数Lを、保存contextの最大局所RZ数と固定構造から構成した。

- Dは保存B′境界融合列、Reference正方向Dと**実SOURCE inverse D**の最大RZ数を使用。q1の非対称なF/Gは別に扱う。
- 各eventは最大3 application、basis run数≤3qr。各runのbasisとinverseを保持し、最大pair RZ数を使用する。全basisの保存両方向を含む。
- rotationはB′最大2 RZ/event、Reference1。product数≤2/event、各Z/ZZ support≤2で、最大4qr個のZ/CZ loweringを計上。
- B′ event P≤q、scalar/vacuum PとX/Y wrapperを含める。Referenceの16 CSWAP（各11 RZ）を含める。resetは同じstabilizer準備で追加RZなし。

source/最終相殺はこの上界を増やさない。期待frame数・cohort平均数を全trajectoryの上界にしていない。C2-BのBDB†重みも使用していない。

各round/axisで一様guard epsilonが満たす十分条件は、`2sqrt(2)*L*epsilon <= d_syn_safe`。X/Y両軸を満たす必要がある。両arm・全18round共通のepsilonを使う場合の保守的十分境界は

**`epsilon <= 6.902168833913054e-16`**。

これは**候補precisionの採択ではなく、保存Nと保守的worst-case Lから得た条件**。全keyがこの誤差で合成済みとも、これより緩い精度が不可能とも述べない。最も厳しい条件はB′ m17/Yで、L上界13,770,556,062。Reference m17/YのLは13,714,327,030で、同じ計算の十分境界は2.5648979454057876e-15。詳細な全18round・両軸のL/部品/予算は [PRECISION_FEASIBILITY.json](../../../artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json)。

保存recordのepsilonだけでこの条件に対応させたcoverage判定や、実guard値による誤差配分/候補性能比較はしていない。非一様配分を使う場合は、全support重み・有限precision・actual word・両armの同等機会を固定し直す必要がある。期待重みを使うなら、固定lawに対する平均誤差伝播を別途資格する必要がある。通常加算はその後も強baselineであり、一様が最適とは仮定しない。

## 判定とSTOP

- **予算の存在：`FEASIBLE`**。追加raw diskと一様精度の正の十分条件が全round・両armに存在する。
- **実word/key/価格を含むprecision資格：`CONDITIONALLY_FEASIBLE`**。未取得key、vacuum全support word価格、precision候補の採択が必要。
- **全support価格・総T：`UNQUALIFIED`／未取得**。`INFEASIBLE_IN_TESTED_SCOPE`という結論ではない。

両armに同じtarget、候補precision、guard、word価格取得機会、配分自由度、同じenergy/全失敗率を与えることが後続の条件。旧nativeの有利な指標からT改善を推測しない。**mandatory STOP。次はGPT／ユーザーが全support位相価格と独立pilot scopeを判断する。**
