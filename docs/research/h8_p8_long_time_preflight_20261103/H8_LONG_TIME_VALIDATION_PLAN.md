# H8長時間・高次PR：最小科学検証計画案

2026-11-03 JST。**未承認の計画案。新しい科学SOURCE・実行契約・実行許可ではなく、今回の科学実行は0。** 固定値の再ランキング、signal/shot再計算、RNG replay、回路build/compile、pipeline実装は行っていない。

## 1. 問いと既存証拠

問いは「同じ保存H8 DF28で時間を延ばすと、高次PRによる必要qの削減が、signed piece・finite-RTE bias・normalization B・shot headroom・compiled単価の負担を上回り、強いPFに対する競争力を変えるか」。S2、高次PR、決定論PFのいずれが勝つかは仮定しない。負結果・不適格・近差も結果として扱う。

[H4/H6長時間研究](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)はT=.8/1.2/1.6/2、S2/Yoshida4/m5、Full/B0/B0I/PRを検査した。厳しいε=.0005ではT=1.6でS2がq32域を必要とする例と、m5がq1で適格な例が保存されている。報告の探索点費用m5/S2比はH4 PR=.3880、H6 PR=.3348。緩いε=.05ではS2が各取得域の点最良。これらは保存結果の引用で、H8への数値外挿ではない。

H8既存結果はT=.8の5精度、追加6精度、強化184のS2/S4決定論対照、旧resource map、固定p8/q2・p7/q16の別n48確認、正式終了済みの希少RTE評価を持つ。高次**決定論**PFがあることから高次**PR**検証済みとは推論しない。後続DBAのB0 p15/q8 S2、B0I p16/q4 S4も既存であり、未実施として計画へ加えない。[既存coverage](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/713c45f84c8d7a41ceaef93381bd083e22189085/docs/research/h8_h46_validation_coverage_gap_20261103/REPORT.md)、[固定主表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)、[希少RTE終了](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)

旧資料の「限定O1に新H8計算は必須ではない」という判断は、その論文scopeのまま保持する。今回はユーザーの新しい指示に沿い、原稿より新しい時間・公式の検証を優先するための提案である。

## 2. 第一候補は一つのstratum

**第一候補：T=1.6、要求精度ε=0.0005、保存linear H8/1.00 Å/STO-3G/tol-only1e-8 DF28。** T=.8からの時間移送を明確にし、H4/H6で少q高次と多qS2の競合が大きかった厳精度を選ぶ。時間だけ・次数だけの因果を自動認定せず、下記の同公式・同q/prefix対照で分ける。

|項目|提案する限定範囲|理由・固定事項|
|---|---|---|
|主PR比較|S2（2nd）対m5_best（4th(m5_best)）|m5は既存binary64公式PF_M5_BEST_20D64C2_BINARY64_V1を使い、新しい係数最適化をしない|
|PR prefix|7、8|既存H8固定候補に接続できる小さい近傍。H4/H6のprefixを移送しない|
|S2 outer q|8、16、32|厳精度の多q側を候補化。q32でも不足ならこの範囲では未判定で、q64へ自動拡張しない|
|m5 outer q|主候補1、2、4。同q診断に限り選定S2のqを1点追加|少q利得とsigned-piece価格を判別。診断追加は高q全探索と区別|
|R_outer候補|S2は4/32、m5は16/32|S2の低R対照を落とさず、共通R32の機構対照も保持|
|K|第一passはK4。必要な候補だけK2を感度対照|paired cutoffの分子最高次数はK+1。K6・全R/K格子は第一案に含めない|
|PF対照|Full28、およびB0/B0I prefix15/16。公式はS2/Yoshida4/m5|元H8で強かったprefix域とFullを含める。PFはq=1/2/4/8/16/32の保存規則でsignal適格性を見て、直接価格候補を絞る|
|共通機構対照|採択前に明記した同q・同prefix・R32/K4のS2/m5|signalが両方適格な場合だけ同条件価格を比較。不適格なら欠測理由を残し、費用0にしない|
|副候補Yoshida4 PR|第一結果でm5のsigned-piece/B負担が支配的な場合のみ同T/εで検討|3 signed piece対m5の11 pieceを区別する後続候補。第一比較へ無条件追加しない|

この表は許可を得るための候補範囲であり、全直積をcompileする契約ではない。m5のR16/32は11pieceの各最低1stepとABS_TIME_PER_OUTER配分に適合する候補。R_outerは1 outerにおける総RTE short-step数、trajectory総event数はq×R_outer。元S2候補のr4はR_outer4に対応する。高次公式でr4を各pieceへ複製して同じ総Rだと表記しない。[signed schedule](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/src/trotterlib/h46_m5_plan.py)

主候補のq域はS2とm5で異なるため、同q対照を行う場合は選定S2のq（8/16/32のいずれか）・同prefix・R32/K4のm5を診断1点として明示的に追加する。両方式が適格な場合だけ直接価格を対応させる。最初は共通R32/K4のsignalを比較し、適格候補近傍にS2のR4/K4とm5のR16/K4、必要ならK2を限定追加する案とする。tuned S2を省略してm5だけ有利になる比較をしない。PFは同じT・ε・input・誤差政策で選び、元T=.8の固定PFだけを新しい強対照としない。「最良」は明示したsignal域・直接価格を取得したsubset内に限定する。

## 3. 実行順と、各段階で分かること

1. **入力と新時間referenceを準備する。** 保存H/state/DF順序/scalar・元T=.8 referenceを保全する。T=1.6では同じH/stateに対するfull-H reference作用・signalを別途求め、新しい結果領域へ保存する必要がある。元T=.8 referenceを書き換えない。T=.8の複素期待値を二乗したり、近似ground-state energyの位相へ置換したりしない。referenceの新時間評価も科学計算であり、将来の許可に含める。
2. **signalとshotを評価する。** 元H8のsparse/spin/direct-tailとfinite polynomialを使い、outer PF bias、finite-RTE bias、B、raw/corrected closure、intermediate norm、headroomと解析的Nを保存する。不適格や過大Bの候補を見分けられるが、compiled資源順位はここでは決まらない。
3. **必要な候補だけ直接価格を取得する。** 明記したwork proxyは選定補助に限り、単価や下界として使わない。各PR公式の上位候補と必須S2/共通対照、強いPFの各系統を保持し、まず主比較のPR候補は合計4件以内を目安に絞る。これは候補数の提案で、全domainの最適性ではない。PFは決定論回路の両軸を直接compile、PRは同じevent列からcos/sinをpairedで取得する。
4. **有限価格の方向を確認する。** PR探索は1候補あたりn=9 paired trajectoryを出発案にする。H46の粗い分散計画の再利用で、必要精度や希少tailを保証するnではない。候補と固定Nを選んだ後、別cohortの確認を設計する。旧H8のn48をpoolせず、同数n48を機械的に採用もしない。

確認nは現時点では未定。将来の探索価格からpaired分散と近差を評価し、取得前にnと判断規則を固定する必要がある。H46の「探索分散×4・名目相対MCSE2%・最低33」は参考条件であり、H8での妥当性は未検証。 実装のconfirmation_plan()は単価和C_cos+C_sinの平均・共分散からnを設計しているため、Nsが異なるH8の固定εにおけるG精度の設計へ、そのまま転用しない。H8では保存N_cos/N_sinを掛けたpaired Gを対象にするか、単価精度を対象にするかを事前に明記する。旧研究の名目基準をGの厳密保証と読み替えない。探索分散からのn設計も新しい数値解析として許可scopeに含める。確認後top-up・候補差替え・探索とのpooling・確認後minによる事前方向の言い換えは行わない。

独立確認が必要なのは、価格を見て選んだPR対強PF、またはm5対S2の点方向を新しい研究結果として述べる場合である。選抜バイアスと価格標本誤差を区別するため、探索で負方向だった比較も、研究上の主比較に固定するなら同じ事前規則で扱う。cost samplingだけで全support母平均を保証しない。近差・未観測次数が残る場合は「点方向未確定／母平均未保証」を結果として残す。

## 4. 結果に応じた次候補

- 第一T=1.6で両方式の適格・直接価格比較が成立した場合：方向が正負どちらでも、次候補はT=2.0・ε=.0005の同じ限定比較。時間移送が持続するかを見る。第一結果から自動実行しない。
- m5の11pieceによるB/N/C負担が主な制約の場合：同T=1.6でYoshida4 PRを次候補とし、高次一般の失敗とm5固有の負担を区別する。
- 最初の範囲で適格候補が得られない場合：最小の緩和候補はT=1.2・ε=.0005。元の精度やallowanceを黙って緩めず、q境界・K不足・数値精度不足を分けて未判定と報告する。

どの分岐も未承認案。新しい科学的選定にsignal/shot/価格解析が必要なら、その値を今回推測で確定しない。

## 5. 既存実装をどう再利用するか

[SOURCE_REUSE.csv](SOURCE_REUSE.csv)に固定commit/file/hash、関数、再利用内容とH8差分を記録する。

- **H8入力・preparation**：[H8 loader](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_ab_native.py)と[Full-block restore](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_b0_anchor_adapter.py)を優先。p8のキー欠落は本調査で意味を解明済みで、保存p8 preparationを誤りとして再生成しない。H46のrestoreはH4/H6専用で、H8を渡すだけではH6分岐へ入る。 PF prefix15/16のpreparation identityは既存DBAの保存receiptから結合可能か先に確認し、元のPR payloadの登録prefix6/7/8へ無検証で追加しない。未取得なら新adapterの明示的な準備・検査対象にする。
- **H8 signal**：[h8_ab_signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c49a88134066177816feca2741f36f6b88d23ba/scripts/h8_ab_signal.py)のH8CoreSignalPort、OccupationCSR/df_operator、spin作用、direct-tail oracleと、既存forward_finiteを再利用候補にする。ただし元scriptのT=.8固定、prefix6/7/8・旧gate・予算を新研究にそのまま適用しない。新時間・signed pieceの薄いadapterが必要。
- **H46 schedule/mean/accounting**：[long_time_signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_signal.py)のbatch・raw/corrected・B・共通適格性と、[core](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_new2_core.py)のpaired polynomial、[schedule](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/src/trotterlib/h46_m5_plan.py)の符号とABS_TIME整数配分を再利用。H8のsector4900にdense spectral法を無条件移植しない。
- **native/価格**：[long_time_native](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_native.py)のcompose/sample_requests、actual adjoint、scalar一回、full-wrapper、fingerprint cache、paired統計と確認分離を再利用。ただしnative_contextはH46 preparation再構築・occupationスペクトルが前提。H8の封印loaderとoracleへ差し替え、過去のprivate path・worker/RAM見積りは移送しない。
- **公式**：[m5固定定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/references/h46_m5_best_formula_v1.json)の係数bytesと順序を保持。新しい最適化や厳密4次代数certificateを主張しない。H46の資料carrier155529e2…と、provenanceに記録された元SOURCE群を区別する。

## 6. 公平性・数値精度・資源

固定すべきものは、Hamiltonian/state/snapshot identity、保存DF28の符号・生成順序・係数cutoff、full-H referenceの定義、T/ε、P=0、通常signed bias、allowance政策、finite paired law、Kの意味、scalar/control/adjoint、compiler/wrapper、cohort・選定/確認規則である。元のu・log_B marginを縮めず、新時間の誤差・intermediate normに必要な増分を見積もる。H46のu floor1e-9をH8全体の誤差保証として転用しない。headroomが数値allowanceで判定できない候補はUNDETERMINEDにする。

compilerは元のQiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし、測定込みfull wrapper、state preparation除外を比較条件案とする。歴史的runtime bit一致を要求するものではないが、異なるcompilerで再取得する場合は科学条件差として明示する。Gは解析的N×実測compiled RZで、FT T数・noise・最終energy/RPE総費用ではない。

H8は16 modes、保存spin sector4900。H46の36/400次元向けに29個のdense行列・全スペクトル・prefixごとのtailスペクトルを持つ構成は、H8で大きいd²記憶とd³処理を生む。既存のsparse/matrix-free作用でDF二乗を明示的にdense化せず、共有prefix/q/公式のraw/corrected/K列をbatch化し、同一入力・preparation・operator・compiled fingerprint cacheを再利用する。計算候補の独立並列と、contextをCOWで共有する方法を検討する。

worker/thread数は、その時点の空きCPU/RAM、実測H8 context・action・compileのメモリ、BLAS/OpenMP・nested compilerの競合から選ぶ。H46の63/64workerや512MiB/2GiB見積りをH8の恒常条件へ転用しない。新しいCPU/RAM上限や共有設定変更は不要。速度倍率・GPU適合・wall時間は今回測定していない。原子task・checkpointを使い、完成結果の再取得や二重標本化を避ける。

## 7. 実行前検証の切り分け

|対象|既存資産の静的確認で分かること|将来追加する人工検査|別許可が必要な新科学検査|
|---|---|---|---|
|入力・p8 flag|固定H/state/export/configとflag意味論。今回7人工検査済み|H8入力schema・candidate重複・正しいloader分岐|新adapterでの実入力復元/preparation作用確認が必要な範囲|
|signed schedule|S2=1/Y4=3/m5=11piece、ABS_TIME配分、scalar/control規則|mock lowererで符号・順序・整数配分・scalar一回|H8の新時間・正負pieceに対する作用・adjoint検証|
|finite RTE|K+1分子、元law/canonical_tau、raw/corrected/B処理|人工小行列等の既存testを隔離し、科学setup無しで実行|H8新Tのreference、signal/bias/B/N、精度・direct-tail照合|
|価格・確認|wrapper/fingerprint/paired会計と探索/確認の分離|人工metric/fixtureでpair、欠測、同cohort拒否|実native build/compile、探索価格、確認n設計と固定別cohort|

既存 `test_h46_long_time_signal.py` / `test_h46_long_time_native.py` はmodule読込時にinstall()を呼ぶ。今回はテスト一式を無条件実行せず、SOURCEを静的に読んだ。必要な人工fixtureだけを将来隔離して流用する。SOURCE中に機能があることは、H8適合検査が済んだ証拠ではない。

## 8. Task Aとの接続・必要な許可

[原因調査](P8_ROOT_CAUSE_REPORT.md)は、このfalseを科学的不一致とする根拠を解消した。したがって、このflagだけを理由とする共通H8入力のHOLDは不要。ただし、将来の復元でidentity不一致、真のpreparation作用差、referenceの不一致が見つかった場合は、その影響範囲の科学実行をHOLDする。p8限定・共通入力・価格限定の問題を分ける。

現在は新科学実行が未承認で、H8高次/time adapterとその検証も未実装である。**EXECUTION_HOLD_NOT_AUTHORIZED_AND_ADAPTER_NOT_VALIDATED**。既存H8の希少RTE終了・元確認のSTOPは維持し、旧grant/ledgerを使わない。

次はユーザーが第一stratum・候補域・判定範囲を採るか判断する。採る場合、Codexが既存codeを最小限拡張して人工検査・新SOURCE固定と具体的call/n設計を準備し、ユーザーがそのSOURCE/input/範囲に対する科学実行を明示許可する。許可は新reference→signal/shot→限定価格→確認設計/固定確認まで必要な範囲をまとめて定め、通常debugや処理段階ごとの不要な再承認は求めない。別の重要科学レビューや条件の実質変更は、その対象について別に判断する。

**この計画作成・公開・照合後mandatory STOP。** 新しいH8科学計算を開始しない。
