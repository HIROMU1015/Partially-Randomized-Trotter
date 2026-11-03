# 研究C：精度評価の検証scope比較（未採択・B2公開未確認）

2026-11-03。[統合準備・固定参照・誤差partition](research_c_b1_b2_accuracy_scope_integration_20261103.md)に付属する技術案。
対象は固定新PR、Linear H4/1.00 Å/STO-3G、Nα=Nβ=2/sector36、number CDF8/元順LD2、ordered S2/s1/r32/K0/h=.1/m0..12、
own-DF ground oracle/状態準備除外、保存whole-P0/controlled full512、準備H/X/Y shell込みfenced serial。
IID独立53-bit/event・occurrence・shotモデルを維持する。新しい保証・samplingモデル・科学的主張は採択していない。
B2固有結果に依存する可否判定は保留で、以下は独立に整理できる設計条件である。

## 三案の比較

| 比較項目 | 1：全domain一様評価 | 2：固定有限回路集合 | 3：IID-grid確率的誤差評価 |
|---|---|---|---|
| 研究上の問い | 任意合法wordの最終P0誤差を一様に制限できるか | 固定した境界/代表回路の生成・phase/inverse/信号と評価器が整合するか | 母lawの平均誤差または期待信号biasを、明示confidenceで制限できるか |
| 確率モデル | 全wordを覆うため特定分布に依存しない。RPEへは同じIID-gridを適用 | 固定list。IID lawへ変更したことにはしない | 事前固定したqのfresh IID-grid抽出。実PCG64のIID証明ではない |
| 得られる証拠 | 成立すれば全word/occurrenceのoperator→signal bound | 対象回路だけの数値観測、またはそのinstanceのvalidated bound | 条件付きのmean/tail/round-bias confidence statement |
| 得られない証拠 | Actual CA、実PCG64、reference/DF資格を自動取得しない | 未検証word・全law母平均・actual CA | 全wordの一様保証・実PCG64無条件保証・actual CA |
| 全lawとの関係 | 最も強いcoverageだが現在未成立 | 有限listの最大値を全lawへ外挿不可 | Sampling law下の期待値等。悪い稀なwordを確率とともに扱う別scope |
| Coherent反復 | 各word一様boundをn=2^m回telescoping | 同じ固定回路の再利用はそのinstanceだけ。独立再抽出とは異なる | 各occurrenceのfresh抽出を固定。Kernel-level boundのn倍伝播か、round-level差を直接校正 |
| RPE接続条件 | Native/bridge/reference/categoricalとshellのraw単位bound、own-ground/alias/DF | 単独では元IID-grid RPEを資格付けない。全実行が固定listなら研究モデルの変更判断が必要 | 期待信号biasの有効なbound、独立measurement、校正confidenceと既存failure政策の整合、別qualification |
| 再利用 | B1/C2-A/ZSXX composer、full synthesis、ColumnStream、既存IID accuracy | 既存build_kernel、CAR reference、raw/compiled/inverse/XY、fixed_certificate | 同native/reference/per-instance verifier、既存tail設計、propagate、RoundEvidence/accuracy |
| 必要な新差分 | a/ζ/translation/phaseの全到達domain評価とraw/inverse/shell coverage | 事前word listと評価target・precisionを固定する小runner | 事前holdout、誤差変数の範囲とper-sample誤差処理、confidence集約と資格gate |
| 新科学計算 | 新しい全domain数値/解析評価。有限trajectoryだけでは代替不可 | F個の固定現候補回路のbuild/compile/reference/phase/inverse/信号 | N個のfresh抽出とnative/reference誤差評価。Direct roundではoccurrence数が増える |
| 負担 | 全列挙216^32は既定にしない。専門的な全domain数値モデルが支配的、上限未確定 | Fとmatrix/state-action精度で決まる。Full512とsector-stateを明確に選ぶ | 標本数・誤差margin・confidence・coherent lengthに依存。小sampleが十分とは限らない |
| 実行許可 | 方法・SOURCE・callsが閉じてから別許可。現NOT_READY | Word集合・target・SOURCE・calls・STOP固定後に別許可 | Estimand/抽出/精度/失敗政策をユーザーが採択し、SOURCE・calls固定後に別許可 |

技術的には**有限pilotで評価器とper-sample targetを確認し、fresh IID-gridの期待bias校正へ進める設計**を第一候補とする。
既存compiler全域の内部誤差を解決することだけに依存せず、研究目標である期待信号への情報を集められるためである。
これは未採択の提案。有限pilotが必要か、B2の独立reference検証を先行できるかも、実際のB2公開結果を確認して決める。

## 案1の残存課題と有用性

B1は現在grammarのSOURCE構造上限N≤107522、accepted rewrite≤N、accepted gate visits≤6N、
条件付き到達domainと既存composerへの接続を具体化した。これらは実H4 gate数や数値誤差の測定ではない。
残るのは全到達domainの1Q行列評価/逐次積、Rust ZSXX/全branch、initial translation/phase、
raw Gaussian/relative/diagonal→exact saved-atom、actual inverse/shell、全wordでの正常適用資格である。

既存式`eps_translation + 6Np/(1−Np) + N*zeta(D_N) + eps_phase`を再利用できるが、a/ζ等は未取得。
数値pilotに厳密enclosureを一律要求するものではない。それでも数値誤差モデルの全domainへの適用根拠は必要で、
観測最大ULP・少数回路の残差・compiler seed固定だけでは埋まらない。
有限な形式上のboundが得られても、m12の反復・attenuation・phase roomとの比較で使えない可能性がある。
現在未成立という技術判定を、数学的不可能性やPR-aware DFの科学的FAILへ拡張しない。

## 案2：有限pilotの設計条件

対象wordは結果を見る前に固定する。候補は同一basis連続、異basis境界、sign/phase条件、head-tail境界などだが、
現在の固定F・具体word listは未選択。既存native32を新しい未使用holdoutに再分類しない。
Raw/reference・compiled/reference・actual inverse・準備H/X/Yとcoherent反復のどれを測るか、full512か正当に限定したstate/sectorかを固定する。

数値残差だけなら有限診断。Validated per-instance boundを得るなら既存fixed_certificate/Arb等を利用できるが、
raw→exact-atom/reference評価の誤差と数値simulate誤差が別に必要。Native raw対compiledのcertificateだけでphysical referenceまで保証したことにはならない。
同じ固定回路を全coherent occurrenceへ流用する方法は`E[U^n]`の別モデルになり得るので、元の`(E U)^n`へ代入しない。

必要callsの骨格は、F wordに対しbuild F、whole-P0 compile F、trajectory reference F、raw gate F、compiled gate F、actual-adjoint pair gate F、XY gate F。
共通model/bases/relative transitions/shell/reference作業が別に必要で、Fや評価precision未確定の現状では総数null。
Dense operatorとColumnStreamを両方無目的に再評価しない。StreamingならG個の異なるfull512 mapに既存H6方式で512G logical columnsを計上する。
Reference側もstate-actionならそのmapをGへ含める。Arb pair単位など別instrumentationを選ぶ場合は、その内部column/action数も明記して契約を固定する。

## 案3：確率的scopeの候補とRPEへの接続

### 3a：Kernel誤差の期待値を校正する

候補変数は、同じq-word Wに対するphase-sensitive synthesis差`d(W)=||C(W)−U_exact_atom(W)||2`。
**両mapのexact unitary意味論が正当化される場合に限り**d∈[0,2]を使える。
保存frame/CARの近似unitarityを観測しただけなら範囲2を仮定せず、有効なnorm guardか異なる変数範囲が必要である。
Per-sample observed residualは真のdではない。Referenceと数値評価の未包含誤差を覆う上側量または明示した数値モデルを必要とする。
上側量を2へclipする場合も、真のd≤2という根拠が先に必要。

Fresh IID-grid標本、固定n、妥当なper-sample量という条件で、有界平均に対する
`mu ≤ min(2, d_bar + sqrt(2 log(1/delta_cal)/n))`という片側校正案がある。
根拠は[Hoeffding原論文・Theorem 2](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)（JASA 58, 1963, DOI:10.1080/01621459.1963.10500830）。
これは条件付き統計式で、今回mu・n・CIを実H4について計算していない。

得られたmu上側値を各fresh occurrenceの期待operator差へ適用し、既存のkernel/shell・both-norm-growth伝播を再利用できるか検証する。
α=1・shell誤差0の単純条件でも、既存generic signal boundには2√2×2^mの係数が現れる。
Compiled作用がexactな制御形を保つという追加根拠なしに、raw complex差を単に2^m muと置かない。
この経路は反復で緩くなり得るが、long-round全体を毎回simulateする必要を減らせる可能性がある。実speedupや実H4での十分性は未評価。

### 3b：Tail確率から期待誤差へ接続する

既存`c_pr_aware_df_guarantee_strategy.zero_failure_tail_limit`を再利用候補とする。
事前固定のεに対し、検証不能・interval straddleもbadと数えたfresh IID holdoutでbad=0なら、
confidence 1−δ_calで`p_bad ≤ 1−delta_cal^(1/n)`。
真のd∈[0,2]と妥当な各sample判定があれば、`E d ≤ epsilon+(2−epsilon)*p_bad`へ接続できる。
これも全word保証ではなく、旧sample32やsample最大値からtailを0にする方法ではない。
Rare-event条件を厳しくすると必要nが大きくなる。実H4のnやεは今回算定・選択していない。

### 3c：Roundのraw complex期待biasを直接校正する

各mの2^m occurrenceを独立に再抽出し、同じword列に対するnative/reference X/Y差をpairedに評価する案。
別案として、native期待値の推定とStage Aのgrid平均を比較する方法も考えられるが、その保存平均の不確かさをB2等で覆う必要がある。
各軸の正規化された±1測定期待値の差は[-2,2]として扱え、axis/round confidenceの同時性を別に設計する。
Compiled X/Y shellが異なる実装である場合に、複素量全体の絶対値≤1を無条件に仮定しない。

測るのは期待信号の系統差であり、量子measurement shotの統計ではない。既存`RoundEvidence`へ必要なのはこのraw biasの有効な上側値。
同じcalibrationを複数roundへ利用するなら、依存を無視せず同時confidenceを設計する。
Per-round n_m標本では概ねn_m×2^m回のkernel生成/compile/actionが必要になるため、kernel平均法より重くなり得る。
有限cohortから復元抽出するだけではqのfresh IID lawを検証したことにならない。正確に同じ入力のcache再利用を行う場合も、lawと契約上のidentityを保持する。

### Confidenceと既存RPE failure政策

校正confidence δ_cal、wordのbad-tail確率p_bad、測定shotのαを同じ量として扱わない。
既存政策はα_axis=.01/26、13 round×2軸でα_total=.01、β_total=π/6、β_stat=π/12。
校正の失敗をδ_cal追加する場合、jointな主張は一般に`P(failure)≤alpha_RPE+delta_cal`という別のaccountingになる。
既存α=.01を保ったまま、追加δ_calを隠して総failure=.01とは言えない。

選択肢は「校正eventが成立する条件の下で測定failure≤.01、校正confidenceは別表示」か、
ユーザーが明示的に採択した新しいjoint failure配分である。後者は既存固定α政策を変更するため、今回勝手に実施しない。
β/CA/r/h/Kの変更で結果を救済することもしない。`confidence_schema`はこの主張の技術整合だけを検査し、方針採択の証明や承認機構ではない。

## Accuracyとの互換性・B2との関係

RPEの既存計算はraw期待bias・radius・phase roomと独立shot条件を消費する。全word一様boundはその入力を得る十分な経路だが、
期待biasの証拠を得る唯一の論理経路とは限らない。確率的scopeを使う場合は、高信頼度eventとfailure政策を明示した**別qualification**が必要。
既存の厳しいall-word gateを黙って通したり、actual-law証拠に改名したりしない。
必要なraw boundが科学的に適格になれば、`RoundEvidence/assess_round/assess_energy`と既存IID adapterの算術は再利用し、DFはenergyで一回だけ扱う。

B2のsector36 mean-reference不確かさは、full512 per-word operator差の不確かさを自動的に覆わない。
Kernel-operator案とstate/round-signal案でB2に求める範囲は異なる。B2公開後にtarget・norm・numerical qualification・既包含誤差を確認する。
B1未成立だけを理由にB2の独立した限定reference検証を止める必要はないが、その契約の実際の実行可能性は現在UNKNOWN。

## 次に選ぶ条件と未承認設計

[比較設計JSON](research_c_b1_b2_accuracy_scope_design_unapproved_20261103.json)では各案を分離し、全science callsを未確定のまま保持した。
既存Stage Cの最大16 callsは最後のaccuracy adapterだけで、検証全体の予算ではない。

ユーザー／研究Cが選ぶ必要があるのは、有限scopeのF/word集合、期待operator差かtailかround biasか、
正当な変数範囲と数値精度、校正confidenceの表示/配分、B2との分担である。
CodexはB2固定公開成果の取得後にschemaを照合し、選ばれた案についてSOURCE・新実装差分・科学calls・fresh/partial STOPを閉じる。
その後の新科学計算には別の明示実行許可が必要。今回の推奨は未採択で、新科学0・mandatory STOP。
