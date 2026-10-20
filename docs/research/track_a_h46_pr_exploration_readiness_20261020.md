# H4/H6高次PF：本探索の標本数設計・実行準備

2026-10-20。**準備完了・MANDATORY STOP。本探索の科学実行は未承認。**

Pilot評価結果commit `bc09510b488bcb1e69fb7579b26dcae1459a8810`を再利用した。新しいpilot統計・共分散・MCSE・希少層確率は計算していない。保存216名目標本数を転記し、配分・科学call予算・保存容量の計画算術だけを行った。`n_E`、相対MCSE目標ρ、分散倍率f、最低本数、資源・保存政策は今回採択していない。

## 推奨案と群別本数

**第一候補はρ=2%、f=4、群別配分に、提案としてn=33の下限を付けた案** `MCSE_2PCT_F4_GROUP_MIN33`。各軸で保存名目MCSE条件を満たすための本数を使い、cosine/sineを独立とは仮定しない。確認用の差の検出力設計とは区別する。

| 系 | 公式 | 保存名目n（q1 / q2 / q4 / q8、ρ2%・f4） | 推奨n_E案（q1 / q2 / q4 / q8） |
|---|---|---|---|
| H4 DF10 | S2 | 255 / 143 / 62 / 48 | **255 / 143 / 62 / 48** |
| H4 DF10 | Yoshida4 | 54 / 22 / 12 / 8 | **54 / 33 / 33 / 33** |
| H4 DF10 | m5_best | 4 / 2 / 2 / 2 | **33 / 33 / 33 / 33** |
| H6 DF19 | S2 | 167 / 110 / 31 / 23 | **167 / 110 / 33 / 33** |
| H6 DF19 | Yoshida4 | 23 / 18 / 8 / 5 | **33 / 33 / 33 / 33** |
| H6 DF19 | m5_best | 2 / 2 / 2 / 2 | **33 / 33 / 33 / 33** |

H4 **793 trajectory・1,586軸**、H6 **607 trajectory・1,214軸**、合計 **1,400 trajectory・2,800 compile軸**。全て新規EXPLORATION cohortであり、pilot792本からの差分取得やtop-upではない。

根拠と判断の限界：

- ρ5%案は粗い資源比較向け、ρ1%案はより多い取得を要する。ρ2%を有限標本の経験的比較の折衷案として提案する。母平均順位や僅差の識別力は保証しない。
- f4は保存分散の過小評価に対する計画感度の選択であり、真の分散が4倍以内という保証ではない。ρ1%・f1とρ2%・f4は同じ名目本数になるが、その科学的仮定は同一ではない。
- 下限33は、pilotで用いた「正規IID参照で標本分散の相対SE25%」という設計目安 `sqrt(2/(n−1))` を新cohortにも残す**追加の提案**である。分布の正規性を認定せず、必要十分な最低本数やrare層の検出保証とも呼ばない。名目2本を新探索の十分性と取り違えないための運用上の余裕で、採用にはユーザー判断が必要である。
- 下限なしなら同じρ2%・f4・群別案は1,007本・2,014軸。下限33案はそこから393本多い。pilotと異なる独立標本を得るため、低変動群にも取得を割り当てる。
- 同じ下限33で4qを共通本数にすると2,300本・4,600軸。群別案の方が900本・1,800軸少ない。共通配分はq間で標本数を揃えられるが、低変動群への追加取得が大きい。候補間のpaired構造・同一乱数による比較を仮定しない。

各群の保存共分散・paired MCSE・層出現数と提案本数は[24群表](../../artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/recommended_groups_24.csv)に対応する。共分散の再推定、shot重み付き分散の新推定はしていない。

## 目標・倍率・配分の全比較

表はH4/H6合計trajectory数。compile軸数は2倍。名目のみは保存式の最小値2を保持し、下限33案とは別案として保存した。36案はいずれも未採択。

| 相対MCSE目標 | 分散倍率 | 群別・名目のみ | 4q共通・名目のみ | 群別・下限33案 | 4q共通・下限33案 |
|---|---:|---:|---:|---:|---:|
| 5% | 1 | 71 | 108 | 792 | 792 |
| 5% | 2 | 105 | 184 | 792 | 792 |
| 5% | 4 | 180 | 340 | 800 | 824 |
| 2% | 1 | 268 | 520 | 835 | 952 |
| 2% | 2 | 514 | 1,020 | 999 | 1,376 |
| 2% | 4 | 1,007 | 2,020 | 1,400 | 2,300 |
| 1% | 1 | 1,007 | 2,020 | 1,400 | 2,300 |
| 1% | 2 | 1,995 | 4,028 | 2,301 | 4,248 |
| 1% | 4 | 3,976 | 8,052 | 4,214 | 8,228 |

保存式は `max(2, ceil(f*s_cos²/(ρ*mean_cos)²), ceil(f*s_sin²/(ρ*mean_sin)²))`。本作業ではこの式による統計の再計算はせず、保存nを使用した。群の名目nがnullなら下限で補完しない。共通4q案は系・公式ごとに4群の最大値を採る。5精度のshot、総費用、順位の比較ではない。

[36案の予算](../../artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/option_budgets_36.csv)、[864配分行](../../artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/group_allocation_864.csv)、[固定提案JSON](../../references/h46_pr_exploration_proposal_v1.json)。

## 科学実行予算（推奨案・未承認）

| call | H4 | H6 | 合計 |
|---|---:|---:|---:|
| trajectory / native evolution | 793 | 607 | 1,400 |
| piece request sampling / native piece composition | 8,166 | 7,713 | 15,879 |
| RTE event（32q/trajectory） | 69,888 | 56,736 | 126,624 |
| independent occupation oracle action | 793 | 607 | 1,400 |
| native low-control action | 793 | 607 | 1,400 |
| native high-control action | 793 | 607 | 1,400 |
| uncompiled wrapper action | 1,586 | 1,214 | 2,800 |
| compile / compiled wrapper action | 1,586 | 1,214 | 2,800 |

新signal・reference・SCF/DF/state生成・shot計算・fit・総費用会計・ranking・条件付きsamplingは全て予算0。S2/Yoshida4/m5の各outerのpiece数は1/3/11、32 stepsはouter内全pieceの合計である。

## SOURCE・再利用・契約

科学実行経路SOURCE：`96259cbe0a650970d331806d0963815130617328`。固定pilot評価SOURCEは`9b3f7aeacbe1d4311e9135dc239bb309e693ce37`。本準備のmetadata監査器は後続資料commitで記録する。最終evidence freezeや独立科学レビューは行っていない。

- 入力は線形H4/H6・1 Å・STO-3G、tol-only1e-8の保存DF10/19・保存state、T=.8・P=0・prefix3/5。固定24群、q1/2/4/8、K4、outer32と固定係数・signed scheduleを維持する。
- 既存sampling/native DF/occupation oracle/Qiskit compiler/8検査/終了監査/可逆保存を再利用する。新adapterは契約・cohort bindingだけを分け、native関数の命令bytecodeと数値定数が同一で、厳密一致する文字列`DESIGN_PILOT`だけを`EXPLORATION`へ置換することを人工テストした。既存moduleは変更しない。
- Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。symmetric-directional full Hadamard wrapper、cosine判定後sine、scalar once。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
- compiled state差は厳密に1e-7未満、他のnative/oracle/Hadamard/log_B検査は厳密に1e-9未満。位相合わせなし、単一保存入力状態の受入であり全状態保証ではない。canonical cutoff演算順序の修正を維持する。
- `H4_PR_EXPLORATION`と`H6_PR_EXPLORATION`は別plan/grant/root/output/ledger/marker/STOP。各2,140 sealのdisabled draftを確認した。H4 plan SHA `836ea1ae186a495476ea6f42d7cba0e6218eca97bd378082b96cd7cfd5bec0bd`、H6 `24e6f24041234fde0df406380260ff45cfa174e3bc72a3e9e60cd3ab31823177`。
- 処理の既定はscope表示。無効grantはmolecular inputを開く前に拒否する。有効grant・CONSUMED・科学outputは作っていない。現在の契約は提案本数を結合しているが採択済み本数とは表示しない。
- 将来は採択したoption、各系の実行承認、freshな入力/runtime/seeds/資源・保存計画を再sealしてから一回実行する。現在nullの資源値を埋めるためのfresh admissionは起動前手続きであり、新たなsamplingではない。現DRAFTをそのまま実行できるとは主張しない。

## 新cohort・seed・保存

EXPLORATION専用の決定論的seed namespaceを使用し、既知209,156 seedと提案H4 8,166 / H6 7,713 seedの内部・相互衝突0を確認した。pilot、旧高次PF/H2等の明示的既知registryを除外した有限の検査であり、全研究のseed網羅やPRNG独立性の証明ではない。実行直前に除外registryの更新を確認し、衝突時はSTOP、再hashや自動retryはしない。pilot/本探索/CONFIRMATIONを混合せず、旧pilotや既存13/45/45確認をn_Eへpoolしない。seed一覧はprivateに保持した。

保存codecは既存level1 byte可逆QPY/NPZ保存を再利用する。歴史的codec policy IDに`DESIGN_PILOT`を含むが、適用cohortの判定は別のstage/SOURCE/plan/消費済grantで行い、本探索から旧pilotの原本を解放できない。元byte SHAとarchive SHAを別記録し、durable manifest・完全復元SHA検査の後に新runだけのraw一時fileを解放する。byte PASSと科学native PASSは別である。

## 並列化・資源・STOP

trajectoryをprocess並列化し、両軸は同じtrajectory内で順次処理する。fork前の単一thread親でread-only contextを準備し、worker間で共有する。JSONは各lookupで全byte/SHAを検査した上でparse cacheを使い、fingerprint cacheも維持する。長いq×piece数を先にdispatchし、結果を見た割当・n変更はしない。外側process並列と内側BLAS/Qiskit並列の重複を避ける設定を、自分の子processだけに適用する。

H4/H6を同時実行する場合、freshな空きCPU・cgroup実効RAM・保存scratch・inode・I/Oを観測してjoint allocationをsealし、prewarm後も再確認する。過去のworker数は上限にしない。今回の参考観測ではCPU約61.5 core相当、実効空きRAM約466.1 GiBからH4 8 / H6 28 worker案になったが、実行時の採用値ではない。

旧同入力DETの保守的proxy（圧縮pair最大×4＋16MiB/trajectory、RAMは保存peak×2）を再利用した計画では、final保存約216.17 GB、raw/partial各worker2 bufferと復元領域を加えた同時peak約338.71 GB。同じworker shareでの逐次peak参考約331.69 GB。観測空き約431.95 GBに収まるproxyであるが、PR/rare出力の上界ではない。逐次実行でも最終保存量は減らない。実行時に再取得し、容量不足なら当該契約をSTOPする。既存原本削除、出力切捨て、少数本案への自動変更はしない。

OOM・継続不能なRAM/scratch/inode不足・SHA不一致・seed衝突・native数値FAIL・自プロセス異常・ledger/exit/reap不一致で当該契約をSTOPし、自動retry/refit/top-upをしない。一時的な上位PSI・CPU競合・他worktreeの資料更新だけでは停止しない。共通SOURCE/入力/保存方式の整合問題は影響する系の起動/継続を拒否する。各系の終了後mandatory STOPし、次の統計や会計を開始しない。

## 検証・保全・未確立事項

新規人工16件、既存回帰88件（cutoff/cache/fork16、保存障害54、codec18）を固定Python3.12.3で全PASS。4-mode人工入力×3公式×H4/H6ラベルの6case・12人工compile軸を含み、実分子計算とは区別する。人工poolは成功・失敗とも全worker exit0/reapを確認した。実DRAFT2件のCLI6拒否、metadata改変10拒否、SOURCE/input/compiler/seeds照合もPASS。保存17,867file・670,563,287 bytesの固定SOURCE/runtime/評価入力/旧終了証拠を照合し変更0。既存rawの再decodeや全QPYの再検証はしていない。これはCodex自身の技術検証である。

全24群のS4はpilot未観測、S2_multiはH6 Yoshida4 q1/q2で各1件しかなく条件付き分散はunknown。下限33・f4や小さな観測MCSEでも未観測層を0補完せず、母平均費用・分散の保証は得ない。強制event、棄却、条件付きpilot混入はしない。

## 次に必要なユーザーの判断・承認

1. 目的を有限標本による経験的資源比較とすること、unknown rare tailの限界を受け入れること。
2. ρ、f、群別/4q共通、下限33を採るか、各群のn_Eと上記call予算を正式採択すること。第一候補は推奨24群表だが変更案の選択も可能。
3. 可逆保存・容量/RAM proxyとfresh joint admission政策を採択すること。観測worker数は自動的な固定条件にしない。
4. H4とH6の新sampling・compile取得を**各系独立に一回承認**すること。n_E採択だけを実行承認と解釈しない。

取得後の新標本の統計評価は別承認。正式shot政策にはεの軸配分・α・shot整数式・normalization/bias/allowance/u/margin/guardの採択が必要で、その後にshot込み会計の実行判断を行う。独立確認Pは探索後に候補を固定し、目的と差/分散/希少層を踏まえて新cohortのn_Cを個別設計する。n48を強制せず、Pと探索を混ぜない。m5主比較採用AD03、詳細科学レビュー、最終freezeは保留。

**次の担当：ユーザーが上記設計・取得承認を判断する。Codexは明示承認後のみfresh seal・有効grant・科学実行へ進む。**
