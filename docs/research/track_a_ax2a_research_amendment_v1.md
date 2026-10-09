# Track A AX-2A：研究・比較契約の追補案 v1

作成：2026-10-09 JST。状態：`AX2A_PREPARED_AX2B_NOT_AUTHORIZED`。

利用者が共有した[AX-1b後の科学レビュー](track_a_ax1b_post_scientific_review_2026-10-09.md)と
「次の作業に進めるか」を受け、設計、source接続、synthetic検証、実行前準備を行った。
本書はAX-0/AX-1aを改変せず、以降の研究設計を具体化する追補案である。
研究方針は採用済み、以下の新規実験条件・予算は提案であり、実行認可ではない。
原稿v0.1、M1〜PM-2、AX-1b、旧科学source・契約・manifestは保存する。

## 1. 研究課題と成果の条件

主課題RQ-Rは、同じ有限時間complex signal taskに対する測定込み実コンパイル資源の比較と、
PRの競争力が成立・消失する条件の説明である。補助RQ-P1はone-shot費用予測の適用範囲。
参照bias・shot・eligibilityを知らないoperational RQ-P2は必須成果から外し、現状N/Aとする。
H4 FEWの追加fit探索は行わない。PRが勝つことを完成条件にしない。

原PR論文には部分ランダム化、QPE、化学benchmark、具体的資源評価がある。
新規性候補は、DF-native有限RTEを同一signal task・強い対照・同一wrapperで評価し、
回路費用、normalization、bias margin、表現政策の影響を分けて説明すること。
新アルゴリズム、初のPR資源評価、一般的優位性、3サイズからの漸近法則を主張しない。
文献との対応は旧[AX-0対応表](track_a_ax0_model_correspondence.md)と共有レビューを使う。

最小成果は、公平な対照と数値不確かさを満たす限定resource study。
目標成果は、H6 developmentで固定した説明・予測をH8で独立評価する適用範囲研究。
強い対照を実現できない、比較targetが揃わない、結果を説明する差分が先行研究と区別できない場合は
独立論文の主張を縮小する。PRの敗北だけをSTOP理由にしない。

## 2. 比較可能性matrix

| レイヤー | target / geometry / basis / rank政策 | state・T・precision | 情報区分・現状 |
|---|---|---|---|
| H4 legacy M1/PM | 線形H4、1.00 Å、STO-3G、保存DF rank12 | 保存state identity、T=0.8、登録精度 | 既観測development。元snapshotとphaseを保持 |
| H4 legacy M2 | 線形H4、1.30 Å、STO-3G、保存DF rank12 | 保存state、T=0.8、固定5構成 | 既観測geometry診断。新しい盲検集合ではない |
| H4共通DF政策 | 同じ1.00 Åを接続候補、STO-3G、正の共通tol・実rank | state再利用/新stateの区別を先に固定 | 必要性判断・作成とも未認可。legacyと分離 |
| H6技術pilot | 線形H6、1.00 Å・STO-3Gを提案、tol/rank未固定 | state未固定、T=0.8を提案、数値目標ε=0.001を提案 | 全出力development。実行未認可 |
| H6本検証 AX-3 | pilot後にgeometry/timeとDF政策を登録 | 同一taskで精度を再会計、必要なら独立検証 | 候補quota・予算未固定。未認可 |
| H8 AX-4 | H6までで固定した共通政策、geometry未固定 | state/T/precisionは結果前に固定 | 未観測独立確認。入力・truthへアクセスしない |

各比較cellでは全方式が同じ`H_DF`、state、T、複素誤差・failure配分、compiler、wrapper scopeを使う。
state準備費用を除くことを明記し、必要なら同じ規則の感度として加える。
B0はこの共通targetからtailを捨てる近似であり、別Hamiltonianを正解に変更しない。
B3もone-body/scalarを保持するrandom-dominant実装である。

主要費用は各軸one-shot RZと`G=Σ_a N_a E[C_a]`。
CX、total depth、実際に抽出可能ならRZ layer depthを副指標とする。
任意角RZ数、T count、物理runtime、古典評価wall/RSSは別単位である。
有限時間signal精度ε=0.001をchemical accuracyのenergy estimationと同一視しない。

## 3. 対照・探索自由度

従来B0/B1/B2/B3に、少なくともglobal DF-term四次PFを強い決定論対照として追加する。
`H_D`内部だけの四次化やexact二block法をglobal四次と表示しない。
四次は既存Yoshida係数と対称二次compositionを使い、負時間とscalarの相対位相を残す。
対称controlled構成はB0/B1とB2/B3の決定論backboneに公平に提供する。
四次化したpartial全体はfinite tailとnormalizationの別導出が必要で、本実装には含めない。
その候補を要求する場合、比較開始前に数式・意味論・予算を追加する。

実rank Lからprefix fraction `{0,1/8,1/4,1/2,3/4,1}`の最寄り整数と±1をclip/deduplicateする。
旧AX-0規則を再利用し、B0/B2に同じprefix候補を与える。
prefix 0でもone-bodyは決定論的に残る。endpointの同一回路は重複計数しない。
`q`と`R=qr`を分け、`R mod q=0`の組だけ採用する。
同じRでもouter PFとfinite tailの配置が異なるので同一候補として潰さない。

本検証のdirect集合・方式別quota・q/R/K上限は、予測順位やbias/costを見る前に固定する。
pilotの6 cellは実装負荷を見るための構造選択で、最良候補探索集合ではない。
pilot上限q=8/R=64を本研究の探索上限へ流用しない。
上限で最良になった結果は`boundary_limited`とし、全方式の不可能性や真の最適性を主張しない。

## 4. 誤差・normalization・shotの契約

各軸 `a` のprimaryは対称配分 `ε_a=ε_sig/√2` と登録failure配分。
正の数値headroom `h_a=ε_a-b_a-u_a` を使う。
`b_a`は共通targetからのbias、`u_a`は参照・近似作用・roundoff等の数値不確かさ。
`u_a=0`の旧保存解析を変更せず、新しい高精度科学検証として扱わない。

| 判定 | 条件 | 資源集計での扱い |
|---|---|---|
| ELIGIBLE | b+u < ε_a | 両軸とも成立する場合にshot/workを計算 |
| INELIGIBLE | max(0,b−u) ≥ ε_a | 比較から外し、理由を保存 |
| UNDETERMINED | 上記以外、またはuが未確定 | 有限shot/workを確定せず、coverageに残す |

参考の十分shot式は各軸 `ceil(2 B² log(2/α_a)/h_a²)`。
これはcorrected estimatorの範囲が±Bであるcanonical有限RTEのHoeffding会計で、
実機shotを実行した値ではない。正式runnerでは既存shot helperとの再現を先に確認する。
一軸・ceil無視の資源比は `C_x/C_y × (B_x/B_y)² × (h_y/h_x)²`。
二軸ではそれぞれの寄与を計算する。固定参照Nを掛けるだけでは回路費用の相対誤差は増幅しない。

phaseと複素方向を保持して `Δz=Δz_PF+Δz_RTE`、
B0では `Δz=(z_trunc−z_exact)+(z_B0−z_trunc)` を保存する。
絶対値の和を等式にしない。referenceは同じnormalized stateへの作用を比較する。
真のground stateであることと、指定stateに対するsignalが正しいことは別判定とする。

数値誤差の提案目標は `u_a≤min(0.01 ε_a,0.05 h_a)`。
この比率はengineering規則であり理論保証ではない。境界で達成できない場合は未確定として残す。
solver toleranceだけをuの証明にしない。残差、作用誤差、roundoff、別精度照合を記録する。
軸配分の変更は一回限りの事前固定感度とし、primaryを上書きしない。

## 5. DF政策の静的確認と未解決事項

既存`df_hamiltonian._low_rank_kwargs`は`df_tol`をOpenFermionの`truncation_threshold`へ渡す。
インストール済み`low_rank.py`の実装では、降順weight
`w_l=|λ_l|(Σ_pq |g_lpq|)²`の**捨てたtail和**でrankを選ぶ。
`final_rank`はthresholdより優先する。返却`truncation_value`はconstant補正に加えない。
API proseの添字はretained和に見えるため、実装と区別する。
[OpenFermion公式API](https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition)

二体項の正しい再構成・全one-body補正・同じspin規約を仮定すると、
`||a†_p a_q||≤1`と三角不等式から捨てたweight和はその項のoperator-norm上界となる。
その場合Duhamelにより固定stateのDF表現signal誤差は `≤ |T| Σ_discard w_l`。
これは静的な条件付き導出で、今回の分子pipeline全体の誤差保証を実証した結果ではない。
hermitization、integral対称性、精度、後段cutoff、metadata provenanceを照合して初めて利用する。
採用tol、DF representation予算、本検証の最大Tは未固定。

新DF target間の比較でDF誤差をsignal予算へ二重加算しない。
元分子Hamiltonianへのaccuracyを主張する場合だけ、別の表現誤差層を明示して加える。
legacy rank12と共通tol政策の違いを系サイズ効果へ帰属しない。

## 6. 実装と数値確認の到達点

詳しいsource再利用・計算量は[技術実装とpilot計画](track_a_ax2a_technical_preparation_v1.md)。
追加はTrack A専用namespaceのstate-action、controlled意味論IR、metadata writerとsynthetic tests。
既存DF sector/matrix-free、有限RTE、PF係数、wrapper/compilerを再利用する。
新しい分子生成・状態生成・signal計算・trajectory sampling・compileは0件。

対称controlled法は、forward無制御half、中央ordinary-controlled unitary event、
reverse directional-controlled halfでcontrol=0の相殺とcontrol=1の同じS2を表す。
四次も対称pieceごとにcomposeする。独立scalar/event phaseをordinary controlで保つ。
これは[Simon–Loveの一次資料](https://arxiv.org/html/2511.13855v1)に基づく作用IRで、
DF-native lowering・回路費用削減・full-wrapper compileはまだ未検証である。
有限平均の非unitary多項式を量子回路としてcompileしない。

synthetic検証はlocal、未commitのsourceによる実装証拠。immutable CI、分子追試、科学的結論ではない。
H4 dense/actionの旧snapshot一致、native DF高次・directional wrapperの接続、H6 profileがAX-2Bに残る。
それらのadapter実装は正式pilot開始前の追加技術準備として扱い、未完了のまま科学runnerを起動しない。

## 7. モデル凍結・情報分離

旧H4 full210 FEWはAX-1b `model_fits.json`のhashで固定した対照。今回refit・再解析しない。
既存analytic deterministic RZ、wrapper proxy、fragment metadataは候補特徴であり、
native高次・新controlでの妥当性を未確認のまま主モデルにしない。
解析だけで得る特徴、局所compileで校正するproxy、full-wrapper参照oracleを別ラベルにする。
新構造モデルを作るならI1取得規則、古典費用、H6 development使用範囲を登録する。
conditional-oracle選択をoperational選択と表示しない。

H6 pilotのbias/cost/最悪回路長は全てdevelopmentとして公開扱い。
H8を見る前にsource、モデル、候補生成、比較条件、説明仮説、予測、quotaをhashで固定する。
random cost標本数と量子shot数を分け、paired axis共分散とcross-candidate uncertaintyを区別する。
主要候補の独立confirmation規則は結果前に登録し、安く見えたwinnerだけを事後救済しない。

## 8. 段階とGO/STOP

| 段階 | 入力・目的 | 成果物・判定 | 依存・STOP |
|---|---|---|---|
| AX-0/1 | 既存契約と保存値 | 完了した歴史的証拠として保持 | 再実行・再fitしない |
| AX-2A | 科学レビュー、既存source | 追補案、source、synthetic証拠、pilot草案 | 本書の準備まで。AX-2Bは未認可 |
| AX-2B | 固定したH4 snapshot・H6 task・adapter・割当資源 | 作用/phase/sector/normalization/強いbaselineの一致、wall/RSS/回路長 | 下記launch条件を全て満たし、別の明示認可後だけ起動 |
| AX-3 | AX-2Bの正しさとprofile | H6 development直接資源と説明、frozen旧FEW移送診断 | 科学的条件・main quotaをレビュー。自動開始しない |
| AX-4 | H6までのfreezeと予測 | H8独立評価、coverage、境界と不確かさ | H8予算・認可前はtruthを見ない |
| AX-5 | 直接証拠と独立評価 | 限定/目標claim、先行研究対応、原稿 | データで成立する主張だけ採用 |

AX-2B launch条件：target/state/snapshot hash、DF政策、primitive-sector証明、native callback/wrapper、
数値allowance、seed/task identity、gate/matvec上限、実資源割当、source/plan freeze、別の明示認可。
現時点では未解決項目があるので**launch可能な確定計画ではない**。

失敗、メモリ・wall・compile/sample・disk上限、基底順序不一致、phase不一致、primitive leakage、
数値headroom不足、未登録target/情報流入ではSTOPし、partial出力と理由を保持する。
resume/retry、tol変更、quota増加、代替sectorへの変更は自動実行しない。
単なるprecision点追加、全geometry網羅、H8先行profile、H4再fitはこの段階の情報価値が低く、実施しない。

## 9. 次に固定する事項

AX-2B準備の追加adapterとH4一致の仕様を確認し、DF tol・state/reference・実資源を埋める。
H4 correctnessを通してから6つの構造的H6 cellをprofileする順序を提案する。
得られた新しい科学結果と本検証予算をGPTへ渡す。
今回の終端は準備完了・科学計算STOPであり、AX-2B/AX-3/AX-4のauthorizationを発行しない。
