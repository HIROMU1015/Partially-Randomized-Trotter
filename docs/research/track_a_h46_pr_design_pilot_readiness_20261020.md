# H4/H6高次PF：outer32標本設計pilotの準備

2026-10-20。**技術準備のみ。標本数は提案であり未採択、新しいH4/H6 sampling・compile・signal・shot・費用解析は0件。有効grantは発行せずmandatory STOPする。**

基点`489422d38bad4d5892cf98c580d93cc4b602bf0c`の[統合データ](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/README.md)と[付属プロトコル](track_a_h46_higher_pf_supplementary_protocol_v1.md)に基づく。専用branchは`track-a-h46-higher-pf-pr-design-pilot-readiness-20261020`。実装SOURCEはbundleの[summary](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/summary.json)を正とする。

## 1. 目的・対象・固定条件

pilotの目的は、**新しいouter32 lawのpaired単価の変動と未観測部分を把握し、本探索のn_Eを設計すること**である。最良qの選択・PR優位の検定・母平均保証を行う研究ではない。

H4/H6、S2/Yoshida4/m5_best、q=1/2/4/8の全24群を保持する。入力は保存tol-only1e-8のDF10/19、線形1 Å、STO-3G、T=.8、P=0、prefix3/5。K4、`ABS_TIME_PER_OUTER=32`、PF係数版とsigned piece順を変更しない。32 stepは各outer全pieceの合計であり、trajectory全体では32qである。

保存signal88候補・適格性880行は再評価しない。PR24群の保存適格εとpreparationは[候補表](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/pilot_groups_24.csv)へ転記する。各精度のために別trajectoryを作らず、将来の会計で同じpaired単価を使用できるようにする。今回の取得案は、正式shot政策やAD03を採択するものではない。

full Hadamard wrapper両軸、symmetric-directional control、scalar once、Python3.12.3/NumPy1.26.4/Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を維持する。compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。旧ordinary-control n8、piece-r2 pilot、別の13/45/45確認を同law分散や標本数根拠に流用しない。

## 2. 推奨標本数と根拠：未採択の2案

同lawの費用分散・尾部はまだ取得されていない。現段階で群別の分散に基づく最適配分を計算することはできない。そこで、各群で分散を粗く推定するための**共通の名目情報精度**を根拠に2案を作った。

正規IIDという参照モデルなら、標本分散S²の相対SEは `sqrt(2/(n_P−1))`。したがって、目標をηとして `n_P=ceil(2/η²)+1` と置く。この式はNISTの[標本分散とχ²の関係](https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm)から導く設計参照値である。PRの混合分布が正規であると認定したものではなく、名目値を実際の精度保証へ読み替えない。

| 未採択案 | 各群n_P | 名目・標本分散の相対SE | 全24群trajectory | compile上限 |
| --- | ---: | ---: | ---: | ---: |
| COARSE_SCREEN | 9 | 50% | 216 | 432軸 |
| VARIANCE_PLANNING（推奨） | 33 | 25% | 792 | 1,584軸 |

**33本案を推奨する理由**は、n_Eが分散に比例する設計で、名目50%の分散変動を許す9本案より安定した基礎資料を得るためである。33が実費用分布に対して必要十分だという証明はない。9本案は負担を抑えた粗い技術・分散screenとして選べるが、さらに大きい不確実性を残す。9本から33本への自動増量は行わない。

現時点で同じnを提案するのは、群間で分散が等しいと仮定したためではない。同lawの群別情報がなく、各群へ同じ名目分散精度目標を置くためである。根拠のない群別の大小を捏造せず、n_Pとn_Eを区別する。ユーザーが目標・有限標本の限界を判断するまで`n_P_adopted=null`を維持する。本探索n_Eも全24群nullである。

希少層が未観測なら、33本でも真の分散・母平均費用を同定できない。分散を2倍・4倍にする感度案も、未知tailを上から抑える証明ではない。pilotの目的を母平均順位保証へ拡張する場合は別設計が必要になる。

## 3. 計算規模と契約境界

H4とH6を独立した`H4_DESIGN_PILOT`/`H6_DESIGN_PILOT`に分ける。SOURCE・入力bindingは共有しても、承認、grant、registry、ledger、marker、output、外部終了witnessはstage別にする。一方のSTOPで別stageを停止しない。

| 33本案のcall | H4 | H6 | 合計 |
| --- | ---: | ---: | ---: |
| trajectory/native evolution | 396 | 396 | 792 |
| paired compile軸 | 792 | 792 | 1,584 |
| unconditional piece request | 7,425 | 7,425 | 14,850 |
| RTE event | 47,520 | 47,520 | 95,040 |

各trajectoryでnative low/high・独立oracleを各1回、uncompiled/compiled wrapper作用を各2回予約する。native piece合成数はpiece request数と同じ。各pairはcosine検査後にsineへ進む。新しいfull-H reference・mean signal・SCF/DF/state生成・fit・shot・会計・ranking・条件付きsamplingは0。表は登録格子と提案nから得るcall予算で、実測した費用・所要時間ではない。

9本案の系別予算と33本案の全call項目は[proposal](../../references/h46_pr_design_pilot_proposal_v1.json)に機械可読で記録する。実行契約のnとcall範囲はまだnullであり、表の上限を実行承認として扱わない。ユーザーの正式採択後に新しいapproval/config/plan digestへ固定する。

## 4. 取得するデータと希少event

- trajectory ID、候補、SOURCE/input/preparation/compiler/native identity、seed cohort、registry/event digest。
- cosine/sineのRZ/CX count、RZ/CX depth、total depth、circuit size。両軸の同一trajectoryをpairedで保持する。
- pieceごとのsigned schedule・整数step数に対応するRTE order列、N2/N4、normalization・order probabilitiesを持つ元lawの`to_dict()`、request index。
- native・oracle・Hadamard・compiled状態差のgate、wrapper/compiled SHA、結果SHA、実行時間、own peak RSS、exit/reap・台帳との対応。

新adapterは元のrequestを変えず、生成済みlawを保存するだけである。law記録のために追加samplingをしない。privateにはQPY/vector/quarantineも保持し、公開時には科学metadataだけを別途選別する。

| stratum | 排他的定義 | 未観測時の扱い |
| --- | --- | --- |
| S0 | N2=0、N4=0 | 観測件数0、条件付き平均・分散はnull |
| S2_single | N2=1、N4=0 | 同上 |
| S2_multi | N2≥2、N4=0 | 同上 |
| S4 | N4≥1（N2の有無を問わない） | 同上 |

希少eventの強制・棄却、S0だけの受入、未観測費用・分散の0補完は禁止する。層の観測件数0と未知条件付き費用0を混同しない。各層のlaw確率や期待費用への寄与を計算する場合は、保存lawの結合を含む別の数値評価承認が必要である。今回は確率も期待費用も計算していない。

## 5. seed/cohortの非混合

専用namespace `H46_OUTER32_DESIGN_V1` と`DESIGN_PILOT`を使用する。SOURCE、新campaign、cohort、task、outer/pieceからSHA256でseedを導出する。`EXPLORATION`/`CONFIRMATION`とは異なるcohortを必須にする。

現在のknown exclusion 4,603個を新しいprivate資料へコピーした。これは既知registryの閉包で、他研究の全seedを網羅する認定ではない。**今回は実候補の新seedを発行していない。** 実行前に最新除外一覧とfresh registryを照合し、内部/既知seed衝突でSTOPする。自動再hashはしない。seed値一覧・private実行パスは公開しない。

pilot結果は本探索・独立確認へpoolしない。pilotを用いてn_Eを設計した後、本探索は新campaign/EXPLORATION cohort、確認は事前固定候補と新CONFIRMATION cohortで行う。旧pilot、探索n8、確認13/45/45や他研究の標本を加算しない。

## 6. n_Eを決める評価手順（数値評価は別承認）

取得契約はunit-cost取得だけを許す設計で、`statistics_execution_authorized=false`。取得終了後は一度mandatory STOPし、以下のpilot限定の評価を別途承認する。解析moduleは今回人工値でのみテストした。実pilot原結果から解析表を組み立てる際には、固定n・SOURCE・結果SHA・cohort・全gate・exit/reapを結合する受入契約が必要になる。

1. 全12群×採択n_Pの完全性と両軸pairingを確認する。部分成功だけで固定nを縮めたり、失敗分を自動で補充したりしない。
2. 候補ごとに軸平均、2×2標本共分散、軸別MCSE、`C_cos+C_sin`の平均・分散・MCSEを出す。この和は**未加重の単価**で、shot込み総費用ではない。候補間のpaired構造を仮定しない。
3. 各stratumの観測件数・条件付き平均・分散を分離する。1件しかなければ層内分散はnull。未観測層、全層S0、ゼロ標本分散、大きな変動・極端値による不安定性を明示する。ゼロ観測分散から母分散0やn_E=2を決定しない。
4. ユーザーが目標を採択した後、軸別relative MCSE目標ρ（候補案5%/2%/1%）と分散感度f（候補案1/2/4）について、`max_axis ceil(f*s_axis²/(ρ*mean_axis)²)`の名目n_E候補を提示する。ρやfは未採択。分散が0なら自動計算値はnullにする。
5. 各qのnを個別にする案と、同一系・同一公式の4qへ最大提案nを共通割当する案を、科学目的と計算規模からユーザーが選ぶ。大小の費用順位で群を落としたり、自動採択しない。
6. 全ての結果にtail未解決・population保証なしを付ける。目標を有限標本の探索に限定してn_Eを採択できるか、追加の別設計が必要かをユーザーが判断する。本探索を自動起動しない。

将来shot重みmを正式採択した場合の分散は`m_cos²*s_cc + m_sin²*s_ss + 2*m_cos*m_sin*s_cs`に対応するため、軸共分散を捨てない。今回その重み・shot整数式・総費用を計算しない。別々の正の軸精度を使う案は将来の正の重みへの接続に備える名目設計であり、厳密な同時保証ではない。

pilotには検定・winner選択を行わないため、多重検定による優位性判定は設けない。MCSEは有限標本の診断であり、母平均winnerの保証や厳密信頼区間とは呼ばない。共通プロトコルの独立確認をpilotで代替しない。

## 7. 実装・並列化・資源・STOP

既存SOURCE`635ba563766875d688ad6912499b7f90397d0cc5`の費用adapter/launcherを再利用した新ファイルを用意した。

- [設計・seed・人工検証用統計module](../../scripts/h46_pr_design_pilot.py)
- [専用認可契約](../../scripts/h46_pr_pilot_contract.py)、[native adapter](../../scripts/h46_pr_pilot_native.py)、[並列launcher](../../scripts/run_h46_pr_design_pilot.py)
- [提案・無効契約の封印](../../scripts/prepare_h46_pr_design_pilot.py)、[人工テスト](../../tests/test_h46_pr_design_pilot.py)

旧nativeとの差分は契約import、cohort表示、生成済みpiece lawの保存であり、PF合成・乱数法則・compiler・数値gateは変えていない。旧launcherへはpilot cohort・stratum・law receipt完全性の検査を追加した。既存ファイルは変更せず、差分をbundleで監査した。

独立trajectoryをspawn process poolで並列実行する。worker内BLAS/transpilerは1threadとして二重並列を避け、process内の保存入力/prefix context、preparation、fingerprintをcacheする。同一evolutionをcosine/sineへ再利用する。各pair内はcosine判定後にsine、異なるtrajectory・H4/H6は並行可能。

worker数はfresh idle CPU、実効空きRAM、task数からadmissionする。固定CPU数・worker数・runtime capはない。初期RSS proxyは保存同入力DET stageの観測値×2という未採択案で、H4 4,401,127,424 bytes/H6 16,509,009,920 bytes。**PR/rare回路の上界ではなく、実際のピークは未確定。** 別stageの使用量を見て順にadmissionし、同じ空きRAMを二重計上しない。

保存入力byte数だけではQPY/vector全出力容量を確保できないため、**採択nに対応するtrajectory単位の出力容量proxyと全stage scratch予算の採択を起動条件にした**。現在は両方未確定・未採択で、grantをtrueにするだけでは起動できない。将来のfresh gateはこの全予算を確認する。disk/RSS proxyは観測ベースの計画値で、容量保証や恣意的な恒常上限ではない。

[scratch案](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/resource_storage_proposal.json)には、保存DET19/27pairの出力byte数metadataだけを使用した案を追加した。各系の最大保存pair×2をproxyとした33本案の予算はH4 173,674,016,208 bytes、H6 802,562,873,904 bytes。実PR/rare出力の上界ではなく、採択値もまだnullである。QPY/NPZ原本は読み直していない。両系を並行するなら合算容量をfreshに確認し、入力サイズだけのgateで起動しない。[準備時の観測](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/preparation_resource_observation.json)は時点資料であり、実行前のfresh gateの代用ではない。

準備時のscratch空きは456,851,746,816 bytesで、33本案の合計proxy 976,236,890,112 bytesには足りない。**統計設計としての33本推奨と、現在そのまま起動できることは別である。** 9本案は合計proxy 266,246,424,576 bytesで観測空きに収まるが、同じく実容量保証ではない。33本案を選ぶなら別の十分な保存先・保存計画を先に確定する。既存rawの削除・移動、共有容量制御の変更、無断の出力削減は行わない。

compiled状態差はCT01の1e-7未満、その他native/control/oracle/Hadamard/log_Bは1e-9未満、有限値・位相合わせなしを維持する。SOURCE/input/compiler/runtime不一致、seed衝突、数値FAIL、台帳不整合、worker異常・実OOM・継続不能なscratch不足で当該stageをSTOPする。retry・再fit・追加候補・自動top-upはない。全worker回収とnative exit、結果SHAを保存してmandatory STOPする。

一時的なhost/cgroup PSI・CPU競合・監視遅延だけで正常workerをkillしない。無関係なworktreeのHEAD/statusを凍結せず、他ユーザーのprocess・共有設定・環境を変更しない。

## 8. 技術検証・保全

新しい人工/構造テスト20件（nativeの3公式subcaseを含む）を固定runtimeで実施した。cohort、衝突拒否、排他的stratum、paired共分散、未観測null、ゼロ分散のn_E保留、DRAFT拒否、one-shot/非返金台帳、独立process成功/失敗、scratch未採択拒否を確認する。既存費用adapter18件・統合reader31件も回帰対象にした。結果は[validation](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/validation.json)を正とする。

人工native検証は小さな4-mode合成Hamiltonianと人工seedのみであり、H4/H6保存配列を復元していない。stdlib Pythonでの最初のnativeテストは任意依存の不足で失敗したため、既存の封印済みPython3.12.3環境で検証した。環境や依存packageの変更はしていない。

封印検査で原signal configの明示結合と全出力scratch採択gateを追加した。準備中の旧DRAFTは別に保持し、最終SOURCEで新しい無効DRAFTを作成した。科学計算の起動・retryではない。既存のsignal/DET SOURCE、raw、grant、ledger、marker、STOP、共通v2本文、AD03保留を変更していない。

## 9. 次の担当・必要な承認

| 担当 | 判断・作業 |
| --- | --- |
| ユーザー | pilot目的、名目精度とtail限界を受け入れ、33本案/9本案/根拠付き別案を選択する |
| ユーザー・Codex | 実行前にRSS proxy・全出力scratch予算、最新seed除外・fresh registry、採択nのexact callを固定する |
| ユーザー | H4/H6の各pilot取得を独立契約で一回実行するか明示承認する |
| Codex | 承認後だけ新しい有効grant・fresh gateを結合して取得し、正常/異常ともmandatory STOPする |
| ユーザー | 取得後のpilot限定数値評価と目標ρ/f、さらに本探索n_Eの採択・実行を別に判断する |

正式shot政策、m5主比較AD03、母平均保証、詳細科学レビュー、最終evidence freezeは保留のまま。本準備から次の科学Phaseへ自動移行しない。
