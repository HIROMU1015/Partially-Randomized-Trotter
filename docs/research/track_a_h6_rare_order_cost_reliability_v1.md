# H6希少Taylor次数：保存費用の寄与解析と次の判断

2026-10-13。利用者指定の[完了済みGPT科学レビュー](track_a_h6_independent_confirmation_scientific_review_20261013.md)を引継ぐ技術解析であり、科学レビューのやり直しではない。

**保存データは大きなPR/PF費用差の点推定を支持するが、未観測層のcompiler後RZ上界は証明できなかった。** 同じ次数でもcomponent・基底列によって費用は変わる。主な追加情報はq8/r2のnonzero次数を含むfull-wrapper費用で、無条件trajectoryを一律増量する必要はない。追加計算は提案だけで、今回の新trajectory/compile/信号/H8/grantは0。

[一つの証拠入口](../../artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/README.md)に、320本の最小抽出、全832本の次数とRZ、既知確率、逆転感度、component catalog、限定testsと公開監査をまとめる。

## 1. 固定条件と来歴

linear H6/1.00 Å/STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。tol-only DF1e-8/rank19/cutoff0/signed generation order/weighted Hermitian projection。登録prefix5/10/15/full19、q1/2/4/8、delta0.8/0.4/0.2/0.1。選択8構成はprefix5、q1/2/4 r4・q8 r2、各K2/K4。

finite-time complex-signal精度epsilon0.05/0.01/0.005/0.001、経験的u、各軸alpha0.025の保存整数shot予算。symmetric_directionalの測定込みfull Hadamard wrapper、state preparation除外P=0。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layout/routing=None。科学SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12、結果commit321cd8c3066194fe98741cbebbcdba1515d1deaa、確認SOURCE73018cbc（159）/87b6b9f4（97）を保全する。

全PR探索576本（72×8）と確認256本（8×32）を保存JSONだけで抽出した。次数・events digest・元task/seed・両軸費用・compiler・cost/trajectory bytes/SHAを照合する。選択8構成の探索64＋確認256＝320本はcomponent/basis列を含む最小metadataを公開する。元raw/NPZ/cacheは追加しない。旧roundoff bridgeを保持し、旧compilerとのbit完全一致は未検証のまま。

## 2. 事前winnerと確認後の点最小

事前選択8構成と、確認を見てから最小を選んだ点推定を区別する。探索8と確認32を平均へpoolせず、K2/K4の細かな順位の証明を目的にしない。

| epsilon | 探索時winnerを固定して確認したPR/PF比 | 確認後の点最小PR/PF比 | 確認点最小 |
|---|---:|---:|---|
| .05 | .459978 | .459978 | q1 r4 K4 |
| .01 | .399990 | .399630 | q2 r4 K2 |
| .005 | .374969 | .372105 | q4 r4 K2 |
| .001 | .387450 | .387006 | q8 r2 K2 |

事前winnerは全4精度K4で、固定winnerの再評価でも約54–62.5%の点削減が残る。確認後の最小だけで利益が現れたわけではない。K差の記述的2SE留保、選択後推定、経験的uの条件を維持する。全8構成の各精度・各phaseの値はJSONへ保存した。

## 3. 次数とwhole-trajectory層

保存source `rte.py:iter_sample_rte_events` は各occurrenceの次数をfinite lawから抽選し、次数kごとにk+1個のcomponentを元component確率で選ぶ。R=qr個のoccurrenceをまとめたwhole trajectoryの費用を数える。次数0でもcomponent/basisはランダムである。

occurrence確率をp0,p2,p4とし、排反層をs0=全次数0、s2=order4なし・order2あり、s4=order4ありと定義する。

`pi0=p0^R, pi2=(p0+p2)^R-p0^R, pi4=1-(p0+p2)^R`。

これは保存finite lawの全期待値分解で、量子PR分布・B・signal・Nの変更ではない。binary64の小さい確率はlog1p/expm1で計算し、さらに(n2,n4)のmultinomial層を列挙して確率和を照合した。保存p0と補集合算術の丸め差もJSONへ保持する。探索/確認の保存tau差は最大5.56e-17、order確率差は最大5.51e-21で、元lawを別保存した。表の確率算術は固定確認lawを基準とし、既存roundoff受理をbit一致へ格上げしない。

| q/r/K | 1 occurrenceのp2 | p4 | 1 trajectoryのpi2 | pi4 | 確認32本の層数(s0/s2/s4) |
|---|---:|---:|---:|---:|---|
| 1/4/2 | .0163446 | 0 | .0637930 | 0 | 32/0/0 |
| 1/4/4 | .0163439 | 4.58782e-5 | .0637813 | .000183500 | 30/2/0 |
| 2/4/2 | .00418237 | 0 | .0329732 | 0 | 30/2/0 |
| 2/4/4 | .00418235 | 2.93766e-6 | .0329725 | .0000235010 | 29/3/0 |
| 4/4/2 | .00105181 | 0 | .0166969 | 0 | 31/1/0 |
| 4/4/4 | .00105181 | 1.84738e-7 | .0166969 | .00000295581 | 31/1/0 |
| 8/2/2 | .00105181 | 0 | .0166969 | 0 | 32/0/0 |
| 8/2/4 | .00105181 | 1.84738e-7 | .0166969 | .00000295581 | 32/0/0 |

order2 occurrence数とs2 trajectory数は異なる。例えばq1/r4/K4のorder2は4 occurrenceだが、s2は2本だけ。q8/K4の探索8本にはs2が1本あったが、確認32にはなかった。探索標本を確認平均へ補充しない。

## 4. 既存の層別full-wrapper RZ

選択320本ではcosine/sineのRZはすべて同数だった。一般に同数と仮定せず、算術は両軸の整数費用をshotで重み付けしたpaired D=N_cos*C_cos+N_sin*C_sinで行う。

| q/r/K | 確認s0平均RZ/shot | 確認s2平均RZ/shot（本数） |
|---|---:|---:|
| 1/4/2 | 13,341.625 | 未観測 |
| 1/4/4 | 13,260.2333 | 13,424.5（2） |
| 2/4/2 | 26,030.4 | 25,457（2） |
| 2/4/4 | 25,873.9310 | 27,398（3） |
| 4/4/2 | 51,212.7419 | 51,898（1） |
| 4/4/4 | 51,583.2903 | 53,022（1） |
| 8/2/2 | 95,863.7813 | 未観測 |
| 8/2/4 | 95,973.9063 | 未観測 |

未観測平均はnull、n=1のSEもnullとし、0へ補完しない。q2/K2の観測s2平均がs0より低いことからも、次数による部品数増加をcompiler後費用の単調増加や単純加算とみなせない。component/basis選択、隣接基底reuse、回路全体の最適化が関わる。

全72探索構成には **q1/r1/K4 replica1でorder4を含む1本** があり、RZは両軸12,521だった。これは選択8構成外、R=1・異なる回転時間のwhole wrapperである。order4が必ず巨大費用という根拠はないが、q1/r4やq8/r2の層平均・上界として流用できない。個別event列と費用を別の参考artifactへ保存した。

prefix5の凍結準備から1,092 component、14 tail fragment、基底operation数62–78をarray復元なしで抽出した。観測component variantsは旧/新basis hashを分け、旧結果を新catalogのbit identityへ書き換えない。有限component supportの費用をすべて測定したわけではない。

## 5. 上界として成立するもの・成立しないもの

`actual_bounds`、`block_instruction_bound`、`estimate_df_rte_structural_size_upper_bound`をscalar metadataから再会計した。prefix5の選択構成の全次数pretranspile instruction boundは、q1 K2/K4で5,956/7,220、q2で11,907/14,435、q4で23,809/28,865、q8で40,045/45,101。生のinstruction数の構造上界で、RZ数ではない。

sourceは基底gateの逆変換、controlled RZ/RZZ、Pauli product、相対phaseを構成した後に、**whole wrapperをQiskit opt1へ渡す**。原instructionが最終basisの複数gateへ展開され、全体最適化も入る。固定compilerの全passに対するRZ増加上界と全component列の関係は保存契約で証明されていない。

従って、実行guardの5,000,000 compiled instructions、観測最大RZ、部品費用和、AX-1bモデル値を母平均上界へ格上げしない。コンパイル成功枝でRZ<=5,000,000という条件付き事実も、未生成のすべての枝がそのguardを通過するという証明ではない。

参考診断として、もし全branchで `compiled_RZ <= rho * native_instruction_bound` を証明できれば、保存lawによるE[native_bound]から期待RZを抑えられる。4精度の当該比較を保つrhoの閾値は約6.12/6.93/7.34/7.08。しかし、そのrhoの全supportに対する成立は未検証なので、証明や採用済み仮定として扱わない。

## 6. PR/PF差を覆す条件付き費用

まずs0以外をひとまとめにし、**s0平均を保存確認の点推定に固定する仮定**の下で逆転閾値を計算する。

`C_rare_break = [G_PF - pi0 * Ghat_0] / [(1-pi0)*N_total]`。

| epsilon・確認点最良 | nonzero層の逆転RZ/shot | 観測s0平均の何倍か |
|---|---:|---:|
| .05 q1 r4 K4 | 256,989 | 19.38倍 |
| .01 q2 r4 K2 | 1,209,297 | 46.46倍 |
| .005 q4 r4 K2 | 5,230,290 | 102.13倍 |
| .001 q8 r2 K2 | 9,189,927 | 95.86倍 |

通常の数%の費用揺れで逆転する差ではない。既存のs2費用はs0と同程度だった。しかしこの表はzero層の推定誤差、未観測component列、compiler上界を含む保証ではない。

さらに観測した(n2,n4)層の平均を固定し、**未観測histogram層だけ** を未知とする感度も保存した。その確率質量は4精度で.00173420 /.000481651 /.000131461 /.0166969、共通条件付き平均の逆転目安は約9.03e6 /8.11e7 /6.58e8 /9.19e6 RZ/shot。order4だけでなく、複数order2の未観測も含む。

仮に未観測histogram層が各候補のs0平均の10倍ならPR/PF比は.4653 /.4016 /.3725 /.4452で、4精度とも低い。100倍なら.5370 /.4190 /.3769 /1.0267となり、.001では逆転し得る。これは明示的な仮定付き感度で、未観測費用が10倍以下という証拠ではない。

q8/K2のnonzero層確率は約1.67%、確認32で未観測となる確率は約58.34%。保存shot予算で想定するnonzero量子実行の期待件数は約896,400で、32費用標本の欠測と量子shot上の無視可能性を混同しない。

## 7. 必要な最小追加情報と未承認案

**最小の優先対象はq8/r2/K2のs2条件付きfull-wrapper費用**。s2の正しい条件付きorder分布と元component/basis選択を保ち、両軸・同じcompilerで取得する。既存signal、B、Nを変えない。意図的に1個のorder2だけを固定する方法はs2全体の分布と違うため、そのまま代表させない。

条件付きsamplingの設計は、(n2,n4)のmultinomial法則を対象層へ正規化し、occurrence位置を一様な配置で選び、各eventのk+1個のcomponentを元probabilityから引く方式。今回は確率表と人工scalar検査だけで、新eventや条件付きsampler/runner/grantは実装・起動していない。残るcomponent列・複数rareのばらつきと母平均tail保証を別に扱う。

| 未承認の情報取得案 | 新trajectory | 新compile（両軸） | carry/予約込みの必要上限 |
|---|---:|---:|---|
| 最小：q8 K2のs2を8本 | 8 | 16 | compile1736 /trajectory852 |
| q8事前K4も確認：K2/K4のs2各8、K4のs4を4 | 20 | 40 | compile1760 /trajectory864 |
| 選択8構成を一括：s2各8、K4四構成s4各4 | 80 | 160 | compile1880 /trajectory924 |

これらの本数は限定pilotで、条件付き平均の精度保証や正式な受理基準ではない。既存探索・確認は分離して保全し、新取得は補足費用評価の別来歴にする。機械的増量・自動retry・次案へ自動進行は提案しない。

carry known compile1700/trajectory834、予約20/10を保持し、採用済み1720/844から勝手に上限を増やさない。他のcall counterも保存law/sourceに従って事前予算化する。旧失敗分返却・resetなし。

資源案は単独workerで優先対象を取得する設計を第一案とし、既存private venv/内部thread1/AS32GiB/RSS9GiB、既存H6のcoordinator/PSI/OOM/5秒監視、own-run CPUとSTOP/cleanupを再bindingする。CPU20とcoordinator22/23は候補で、fresh共有資源検査と別認可が必要。old H4には作用しない。実high-order compileのwall/RSSは未測定で、完走保証はない。phase/total wall capなしの既定は維持し、output・call・memoryを有限とする。

## 8. q依存とサイズ比較の位置付け

保存q4/r4/K2とq8/r2/K2は同じR=16・prefix5・logB=.03368262897922761。qを倍にするとtotal biasは約4.0106分の1、epsilon.001のshot予算は546,721,412から53,687,969へ（約10.1833分の1）。finite RTE残差は約2.49e-8から2.48e-8でほぼ同じ、outer PF残差が主要に変わる。これは保存signalの機構整理で、signal再計算やrefitではない。

H4旧M1/PMとH6ではDF/control/PF候補/数値会計が違う。[比較対応とH8で固定が必要な条件](track_a_h4_h6_h8_comparison_readiness_20261013.md)を別表にした。H4の進行中geometry系列は変更・停止しない。H8の新しい科学比較契約やgridを今回確定しない。

## 9. 検証・固定・次の判断

10人工scalar testsは、排反確率、多項分布、小さいorder4質量、未観測meanのnull、n1のSE、巨大整数平均、array decode拒否、科学import0を確認した。公開抽出からの別算術検査も保存する。旧101 testsや完了proof、GPT科学レビューは繰り返さない。

source251と元報告/結果byte SHAを保持し、専用branchには解析script/tests・軽量metadata・比較資料だけを明示commitする。既存venv、共有設定、他job、H4、科学runtime変更0。GitHubへの公開後にref/SHAと取得内容を照合してmandatory STOPとする。取得確認はCIや科学的winner認定ではない。

**GPTへ戻す判断点は「既存保存値だけではcompiled平均の保証まで閉じられない」と判明したこと**。主差を覆す条件付き費用は大きく、既存rareの費用はその水準を示していない。次は、上の最小条件付き情報取得を承認するか、明示的な工学的感度を留保した限定benchmarkとして整理するかを判断する。大規模無条件増量やH8先行を自動的に始めない。
