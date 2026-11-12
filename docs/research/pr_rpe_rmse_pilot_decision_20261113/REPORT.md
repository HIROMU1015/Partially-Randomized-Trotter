# 二時点PR–RPE RMSE pilot：固定条件の選択案（2026-11-13 JST）

**推奨案は、H4 m0/m6・保存P1のR=(2,128)・finite_B_diagnosticによる保存再解析限定pilot。正式採択・実行はしていない。** κは保存meanを再現する記号的決定式、Mは指定QPE予算からの経験的決定式を提示する。εEには既存研究と同じchemical-accuracy規模を候補として示すが、新RMSE目標へ自動継承しない。

成果物Aが本REPORT、Bが[条件付き実行計画](EXECUTION_PLAN.md)、Cが[ユーザーの決定事項](USER_DECISIONS.md)。[既存入力template](../pr_rpe_rmse_pilot_preparation_20261113/PILOT_INPUT_TEMPLATE.json)は**全未定欄をnullのまま、byte不変で再利用**する。今回の成果は文書だけで、既存module/runner/testの再実装・変更はない。

実Hのsignal/N/RMSE、保存数値の再解析、実λによるκ区間の数値計算・探索、sampling、C/G、P2/P3は0。以下の整数・分数の整理は、指定された候補metadataと既存mapperからの**記号的推論**であり、実λ・signal・normalization・保存Nを評価していない。科学的PASSや詳細科学レビューではない。

## SOURCEと証拠の範囲

| SOURCE | 本整理に使うもの |
|---|---|
| A `ee15bbd29ec571dd2526feecdb81ed757f783557` | [準備REPORT](../pr_rpe_rmse_pilot_preparation_20261113/REPORT.md)、HANDOFF、template、[P1候補metadata](../pr_rpe_rmse_pilot_preparation_20261113/P1_MATCH_METADATA.json)、二時点入口 |
| A model `510fbacd01a2d855202de4c045d8ba1972e5ff3d` | `pr_rpe_rmse.py`のκ mapper、M最大index、経験的Cmax/Nm、二つのnormalization rule・二つの合成則 |
| B `f09af82482aad11460c4bb73d3e13d7ada40be6f` | A固定grid/全schedule identity、H4 B0識別、H6初期override、nominalと十分N/boundの分離 |
| P1 `71ba49bb28b295aa28e51101bc3b1bf07d500767`、科学SOURCE `a1a79213a9b5331145e20e8778a3be9d9fc4e497` | [保存REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71ba49bb28b295aa28e51101bc3b1bf07d500767/docs/research/pr_r_detailed_n_p1_results_20261111/REPORT.md)：12候補のmean/reference/数値診断が保存済。旧policyの条件付きPASSを新RMSE資格へ転用しない |
| [PR論文v2](https://arxiv.org/pdf/2503.05647v2) | Appendix B.2/Fig.13の経験的統計尺度、E.1の誤差分離、E.3のκ/B² shot負担。既知理論を再証明しない |

確認した固定A/B/P1成果には、実Hに対して共通κから二時点を選んだRMSE評価結果はない。A/Bのκ実行例は人工入力であり、実κの推薦値として利用しない。この確認範囲外の全研究にκ成果がないとは主張しない。

## A：固定条件の提案表

全項目の**現在値は未定**。表の推奨はユーザーが採択するための候補であり、設定fileを埋めたものではない。

| 項目 | 根拠のある候補・推奨 | 未定の理由／ユーザーの判断 |
|---|---|---|
| 対象系 | **初回H4 DF10 m0/m6を推奨**。H6 m4/m8はsigned/negative time検証を優先する代案 | 最小構成の初回診断を優先するか、H6固有の構造検証を優先するか |
| M | **予算に従うM_emp（後述）を推奨**。H4≥6、H6≥8は包含条件だけ。旧M12は条件付き候補 | εqpeが未選択。最小round数や過去のM12だけでは精度根拠にならない |
| 共通κ | H4のR=(2,128)を再現する許容帯内点。**帯の中点という決定式**を提案（後述） | 実λを代入した数値κは未計算。保存再利用のための選択と総費用最適κは別 |
| εE | **0.00159360010199040 Ha（既存1 kcal/mol規模）**をRMSE目標候補として推奨。論文の丸め値0.0016 Haも比較候補だが同一値とは扱わない | P1/D1/D2では高確率型energy precisionだった。RMSEへの意味変更はユーザー判断 |
| εqpe | 初回の透明な予算案は**εE/2**。quadratureならεE/√2の等配分も選択肢 | 実decoder/finite-Kに対する統計RMSの資格・達成証拠は未取得 |
| εtrot | **εE/2**を対称な設計予算として提案。実bias boundの推定値ではない | 二時点のbiasだけで全scheduleの時間発展RMSE上界を設定できない |
| 合成則 | 初回は既存APIの**minkowski**を設計予算案として推奨。原論文比較を重視するならpaper_quadratureを別policyで選択可能 | 両成分のRMS上界自体が未資格。Minkowskiを選んでも新しい保証は生じない |
| normalization | 初回は**finite_B_diagnostic**を推奨。paper_kappa_envelopeは追加の仮定照合を目的とする別案 | finite B plug-inの経験的校正は未確立。特にH6で単純envelopeは自動成立しない |

外側条件の保持案は、Linear H4/H6・隣接1.00Å/STO-3G、保存tol-only DF10/DF19、保存H/state/snapshot・completion・fragment/signed順・scalar、P=0、無energy shift。H4はS2/L_D1/ΔPF=.1/K2/h=.1、H6 regularはm5_best/L_D7/ΔPF=1.6/K2/h=.1と初期m0..3 p5/R32/qPF1/δ=Tを保つ。prior/decoder/compiler profileも同じSOURCEを指定する。

## 対象系の比較と初回推奨の理由

| 観点 | H4 DF10 m0/m6 | H6 DF19 m4/m8 |
|---|---|---|
| 保存再利用 | P1 m0 R2/4/8、m6 R4/128/256。mean/allowance/独立reference pairのfieldがある | P1 m4 R11/16/32、m8 R16/128/256。T1.6の不足pairはP1で追加済 |
| 記号的κ整合 | 下記2組が元mapper上で整合 | 下記3組が元mapper上で整合 |
| 構造 | S2、一piece、初期例外なし。sector36 | 11 signed piece・負時間・整数配分、sector400。全scheduleに初期例外あり |
| 未解決 | finite-K bias、保存state/有効発展、finite B経験的校正、alias/decoder | 左記に加えsigned norm/envelope、初期prefix5とregular prefix7の異なるlaw/λ |
| 追加計算 | 保存候補を完全一致で使う案なら新mean/referenceなしを目指せる。gate再解析は必要 | 同様。単にH6だから新pairが必要とはしない |
| 得られる情報 | 最も単純な構成で共通κ→保存mean→経験的N→B packetの科学診断を開始できる | negative time/11pieceでも局所有限mean・normの診断が使えるかを調べられる |
| 依存とSTOP | B0/p7 PFをFull-Hへ変換しない。PRだけのpilotとする | 早期p5/R32を変更しない。regular二時点の結果で早期roundを認定しない |

**H4推奨は技術的な構成の単純さに基づく。** 保存P1は両系で旧policyの条件付き検査を通過しており、今回その数値を比較・再解析して優劣を決めていない。sector次元の違いから追加meanの負担はH4で小さそうだという推論はできるが、二時点の実測runtimeやcompiled費用順位を新たに認定しない。PR–PF–純RTEの最終比較への一般化も行わない。

## 共通κ：実λを使わない組合せの整理

保存λ_R>0と元SOURCEの一致を前提とする。既存mapperは `R=max(J,ceil(ceil(r*)/qPF))`、`r*=κλ_R²T²`。整数qPFについてnested ceilは`ceil(r*/qPF)`と同じであり、従来REPORTの許容条件をそのまま使える。R>Jでは `qPF(R−1)<r*≤qPF R`、R=Jでは `0<r*≤qPF J`。Rと同じ元signed weight/公式を渡せば、保存piece allocationも元関数で照合できる。数え方を変更しない。

**H4**：`x=κ λ_1² h²` と置く。qPF0=1、qPF6=64、T6=64hなので、R0=ceil(x)（J=1）、R6=ceil(64x)。保存集合から両立する組は次だけになる。

| (R0,R6) | 無次元xの許容条件 | 用途候補 |
|---|---|---|
| **(2,128)** | **127/64 < x ≤ 2** | 初回推奨。二候補ともP1保存済、下の組よりRが小さい。価格最小という意味ではない |
| (4,256) | 255/64 < x ≤ 4 | 別の保存再利用案。N/C優劣は今回未評価 |

R0=8やR6=4を含む残りの保存組合せは同じxの条件を満たさない。新meanを避けるために、別κをroundごとに使ってこれらをつなぐことはしない。

**H6 regular**：`y=κ λ_7² T4²`、T8=16T4、qPF4=1、qPF8=16、J=11。R4=max(11,ceil(y))、R8=max(11,ceil(16y))。

| (R4,R8) | 無次元yの許容条件 | 注意 |
|---|---|---|
| (11,16) | 15/16 < y ≤ 1 | m4はmin-oneによるR=11 floorに張り付く |
| (11,128) | 127/16 < y ≤ 8 | m4は同じR=11でもκと全B identityは上の組と異なる |
| (16,256) | 255/16 < y ≤ 16 | regular両時点ともR>J |

R4=32を含む組など、残る保存組はこの条件では両立しない。これらは**記号的に空でない帯**であり、実κの数値区間、実signalの精度や最適Rを求めた結果ではない。

初回H4推奨帯には、端点から距離がある中点を使う選択則を提案する：

`x*=(127/64+2)/2=255/128`、`κ_A=(255/128)/(λ_1² h²)`。

これは浮動小数の丸め境界を避けやすくする**候補指定の規則**であり、N/Gの最適化ではない。実λの代入、数値κ、schedule生成は今回行っていない。ユーザーはこの記号式を固定条件として選ぶか、同じ帯内の別κを指定できる。式の採用後もSOURCEのbinary64をexact Fractionで扱い、実装のR=(2,128)/allocation一致を確認する。合わなければ自動修正せずSTOP。

## M・h・RMSE目標と予算

[原論文Fig.13／Appendix B.2](https://arxiv.org/pdf/2503.05647v2#page=30)は統計精度を最大時間で経験的に記述する。現行AはPFのCmax=.08とPRのCmax=.10を別に保持する。PR pilotの経験的判定は

`εqpe,emp ≈ 0.10π/(h 2^M)`、`M_emp=max(m_late,ceil(log2(0.10π/(h εqpe))))`。

ここでhはRPE基準時間であり、H6のPF幅ΔPF=1.6とは別。h=.1では前者は記号的に`π/2^M`。H4の包含下限M6ならπ/64、H6のM8ならπ/256という尺度で、chemical-accuracyのQPE予算に達するという根拠にはならない。旧M12の再利用を主張するのではなく、**π/4096≤εqpe<π/2048ならこのモデルがM12を選ぶ**という条件付き候補を示す。予算確定後に元A判定で確認する。今回Mの数値評価器は呼んでいない。Mは全horizonとnominal Nmを決めるが、二時点以外のsignalを評価する許可にはならない。

εEの0.00159360010199040 HaはP1/D2で既に使われた1 kcal/mol規模のenergy precisionに由来する。ここでRMSE metricとして再採択する必要があり、旧α=.05やβ=.4/.2から新RMSE成功保証を作らない。論文のε=.0016 Haは丸めた比較候補として区別する。

[原論文E.1/E.3](https://arxiv.org/pdf/2503.05647v2#page=39)の二乗和は、有効Hamiltonianのenergyに対する統計推定と時間発展biasを分ける仮定に依存する。有限K2のcorrected meanはexact RTEと一般に異なり、保存stateも有効PFの厳密固有状態とは限らない。B²補償だけでこのbiasやstate/branch誤差が消えるわけではない。二点のphase/Tをεtrotの全schedule上界へ読み替えない。

合成則と予算の選択肢は次の通り。

- **paper_quadrature**：εqpe²+εtrot²≤εE²。例えば両予算εE/√2は中立な等配分案だが、paperにある最適配分や現系での成立の証拠ではない。cross termを無視できる条件などが必要。
- **minkowski（初回推奨案）**：両成分が実際のRMS上界なら、三角不等式によりεqpe+εtrot≤εEと設計できる。統計誤差がunbiasedであることに依存しない予算案。ただし成分の分離・上界自体は未資格なので、これも保証ではない。

εqpe=εtrot=εE/2は、対称で有理数として指定でき、両方の予算条件を満たす明示案である。時間発展biasがこの額以下と測定済みという意味ではなく、費用最適配分とも主張しない。各policyは別identityのまま保持し、初回で両方を自動計算しない。finite-K・state・実lawを含む誤差分解が正当化できない場合は、局所診断とnominalモデル値までに主張を限定する。

## normalization policyの比較

| 観点 | finite_B_diagnostic（初回推奨） | paper_kappa_envelope |
|---|---|---|
| 入力 | 一致candidateのfinite-K logB、mean/raw/corrected/reference、数値allowance/SOURCE、M | 左記に加え同一paper κとenvelope適用証拠。現行AはlogB_used=1/κを使う |
| 意味 | 保存finite Bによる経験的plug-in shot診断 | 論文のB≤exp(1/κ)というenvelopeを使うモデル |
| 仮定・未解決 | finite-Kのbias/mean lawと論文経験式の校正。H4でも未資格 | ideal/exact RTEとfinite K、abstract回転とeventの対応、H6 signed time/丸め・初期例外 |
| 保存再利用 | P1のlogBと独立raw/corrected等を再解析できる | 同じP1保存logBと式の比較に使えるが、新policyでの数値照合は別の科学再解析 |
| 追加検証 | reference/closure/norm・原allowance、予算/horizon、未知誤差を区別 | 左記に加えてfinite logB≤1/κ、数学的envelope移送。局所の不等式成立だけで全round定理を認定しない |
| 結果の主張 | nominal Nであり、十分N/actual minimum/全RMSEではない | envelopeモデルのnominal N。論文の保証を現実装へ自動移植しない |

現行Aはfinite logB>1/κなら`INAPPLICABLE_NORMALIZATION_ENVELOPE`を返し、κや閾値を変えて通さない。H6はwhole `log B=qPF Σ_j r_j log b_K(λ_R δ w_j/r_j)`を保つ。Rを11pieceへ重複して配ったり、κ_Bやabs-weight係数を黙って使わない。初期m0..3はκ_applied=falseなのでprefix7のλ/envelopeを適用したことにしない。今回これらの不等式の実値検査はしていない。[原論文E.3](https://arxiv.org/pdf/2503.05647v2#page=42)の条件を現在の有限Kへ適用したという主張ではない。

## 一括して選べる提案（未採択）

**提案A**：H4 m0/m6、R=(2,128)を再現する上記κ_A中点式、既存外側条件保持、εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski、finite_B_diagnostic、Mは上記M_emp式。M/κの実数値はユーザーがこの決定式を採択した後、明示承認された範囲で具体化する。元templateのnull欄は今回変更しない。

提案Aを推す理由は、保存mean/referenceを活用しやすいこと、S2一pieceで初期例外がないこと、κの端点を避けられること、未証明のsigned envelopeをN算出の前提にしないこと。Minkowskiは誤差保証の追加ではなく、不必要なunbiased仮定を予算上置かない設計案である。科学的な十分N・実RMSE・費用優位は依然未資格。

原論文のquadratureによる比較を優先するならそのruleと予算に変更、H6のnegative timeを優先するなら上のH6適合組の一つに変更できる。いずれもユーザー判断であり、変更後の別policyや別系を今回併走評価しない。

**実行は[保存再解析限定S0](EXECUTION_PLAN.md)を初回推奨**。新mean/referenceを自動fallbackしない。実κの数値具体化を含め科学承認前には実行しない。選択・承認後も、二時点の経験的nominal N以外のroundはNOT_EVALUATED、十分N/global RMSE bound/C/Gはnullとする。

## 保全と確認

既存44人工テストはee15の保存技術証拠であり、今回再実行した新しいPASSとして扱わない。今回の変更は文書だけなので科学・人工数値テストは実施していない。SOURCE/文書link/template不変とGit差分だけを確認する。詳細科学レビュー、旧grant/ledger/marker/KEEP/STOPの変更、他worktreeの変更はなし。公開remote SHA/主要成果物の照合後mandatory STOP。
