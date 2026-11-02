# RPE/QPE energy接続の既存証拠・前提に関する技術調査

2026-11-03 JST、担当LOCAL Codex、主エージェントのみ。既存metadata・固定Git object・SOURCEの静的読取による調査であり、新しい科学計算、保存数値の再解析、科学レビュー、中心claimの採択は行っていない。

**既存証拠は4roundと費用モデルだけではない。長roundの56点行列検証に加え、WP04の108点gridには、後続coherent opt2が選択したPartial18roundの全q/r/Kが含まれ、Fullにも18roundの物理信号・半径診断が保存されている。** 一方、費用モデルが仮定するPF固有状態・単位半径から、同じ元HamiltonianのE₀について最終精度・成功率が成立するという接続は、現存証拠だけでは確定しない。最終配分のbranch検証と長q価格の移送も区別が必要である。[保存grid][ABLGRID]、[Full診断][ABLJSON]、[opt2結果][OPT2JSON]

## 成果物・読取範囲

- [ASSUMPTION_EVIDENCE_MATRIX.csv](ASSUMPTION_EVIDENCE_MATRIX.csv)：33論点の定義・直接証拠・仮定・未解決事項。
- [ROUND_EVIDENCE_MAP.csv](ROUND_EVIDENCE_MAP.csv)：保存選択済みδ=0.02、L_D=3/12各18roundの36行。物理信号、PF/RTE、半径、統計、branch、直接価格、モデル価格を別欄にした。
- [MINIMAL_VALIDATION_PLAN.md](MINIMAL_VALIDATION_PLAN.md)：共通E₀・状態・半径を先に確認する提案。実行条件を新規固定していない。
- [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)：固定commit/path/blob/SHA-256、参照ID、報告済みSOURCE hashとの照合。68件はファイル参照数であり実験数ではない。
- [validate_metadata.py](validate_metadata.py)：標準ライブラリのみのmetadata照合。数値literalは文字列として転記・比較する。

必読の研究目的・方法・評価計画は固定`9a0bc7f17f47823e46c3945734127e9f0d09154c`、RPE4・horizon・delta schedule・M06-F文書／結果／SOURCEは固定`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、Phase3は固定`235e56ee7d5cac4682775a62bc2a8262f15047dd`を使用した。GitHubから必読文書も取得し、ローカル固定objectと対応を確認した。

実際のAGENTS.mdとPROJECT_MAP.md、および正式共通運用指示`research_project_instructions_unified_2026-10-25.md`の§8.1・§8.4・§10を確認した。研究概要・現状、VALIDATION_STATUS、validation_manifestの関連記録を併用し、履歴の「未評価」を後続完了記録より優先していない。SOURCEは関連関数・依存境界・runner・既存テストを静的に調べた。取得した資料をすべて全文精読したとの主張はしない。[現行目的][PURPOSE]、[検証履歴][VSTATUS]、[manifest][VMANIFEST]

読取対象はLOCALで到達できる当該repoの既存objectと保全済み索引である。SERVER1/2へSSH・調査・転送していない。固定公開成果物の記録を読んだことを、その実行環境や非公開rawの現在状態を直接検証したとは扱わない。

## 1. 共通energy、符号、scalarはどう定義されるか

有限信号検証のreferenceは、保存DF Hamiltonianを指定粒子数sectorへ制限した元Hの最低固有状態とE₀である。今回参照するH4 linear 1.0 Å/STO-3G/legacy DF12/8qubit/4電子sectorでは、保存値は`E0=-2.1663874486347616` Ha。Hamiltonian hashは`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、snapshotファイルSHA-256は`13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`である。両hashは違う対象の識別子なので混同しない。これは新しいenergy計算や、未切断の電子Hamiltonianへの一般保証ではない。[保存Hamiltonian metadata][ABLGRID]

既存SOURCEは`exp(-i t H)`、信号は`<psi|U|psi>`を使う。方法文書の`Z_star,m=rho_star,m exp(-i q_m theta_star)`における元energy referenceは`theta_star=E0*delta_time`。RPE4実装が復元する基底位相は信号のargなので、符号は`phi=-theta_star` modulo 2πである。既存の最終energyへの解釈は、同じalias-free branch上でこの符号とδを用いる必要がある。RPE4保存結果は位相復元を返しており、ここで新しいenergy値や最終成功確率を算出していない。[METHOD][METHOD]、[finite SOURCE][FINITESRC]、[reconstructor][RPE4SRC]

Hadamard wrapperは既に制御化された`diag(I,U)`を再制御化せず、ancilla H、controlled evolution、cosineはH、sineはSdg→H、測定という規約で、実部／虚部に対応する。したがってsineの符号を逆に読まない。`paper_cpm_halving`へ置き換えた価格比較でもない。[wrapper SOURCE][WRAPPER]

constantとtailから抽出したidentityは決定論側で含む。step SOURCEはそれぞれ`-delta*constant`、`-delta*extracted_identity`、反復経路は`-q*delta*(constant+extracted_identity)`を用い、制御付きではancilla phase、非制御ではglobal phaseとして保持する。既知scalarを既に含むsignalからもう一度差し引く理由はない。有限RTEのnormalization補正はPF biasの補正ではなく、元E₀を別energyへ変更する処理でもない。[step SOURCE][DFSTEP]、[反復SOURCE][REPEATED]

## 2. 共通位相はどこまで確認されているか

同じL_D・δのideal-tail partial-S2 stepを固定し、qごとにそのべき乗と元Hの`q*delta`発展を比較する実装がある。roundごとのr/K変更はfinite-RTE近似を変えるもので、比較先PFの定義を変えない。ただし各観測信号を厳密な`rho exp(-iq theta_star)`として表せることは、その実装構造だけからは導かれない。[FINITESRC][FINITESRC]

物理H基底状態はHの固有状態であるが、partial-S2の固有状態になるとは限らない。有限信号コードは、物理H基底状態と、その状態とのoverlapが最大のPF固有状態controlを**別々に**構築・保存している。δ=.02/LD3の保存controlのphysical-state overlapは`0.9999999999396312`で、両者のstate fingerprintも別である。これはこの保存H4条件の数値診断であり、実際の状態準備の保証ではない。[grid states][ABLGRID]

一方、resource accountingのassumptionsには`exact_effective_partial_s2_eigenstate_input_assumed`と`unit_unattenuated_survival_signal_radius_assumed`があり、reference radiusは1に固定される。PF固有状態を使えばそのPFの有効位相が対象となり、L_Dを変えた二候補にはそれぞれのPF operatorがある。同じ元E₀への比較は、適用可能なPF bias／誤差の根拠を介する条件付き接続である。物理H基底状態を入力する検証と、このPF固有状態入力モデルを、同じ入力状態と呼ばない。[ACCOUNT][ACCOUNT]、[METHOD][METHOD]

物理PF信号のnear-one radius、単一支配位相の短q診断、長qのPF-vs-exact比較は限定的な支持証拠として存在する。**exact H ground stateだけを理由にrho_star_lb=1を保証せず、数学的に共通の単一位相が成立したと新たに認定しない。** energy windowのalias span診断も、外部から指定すべき絶対energy branchの代わりにはならない。[評価計画][PLAN]、[PF3][PF3]、[PF12][PF12]

## 3. 4round、長round行列、18roundモデルの実証範囲

| 保存段階 | 直接保存された内容 | その保存結果だけでは示さないもの |
|---|---|---|
| RPE4：LD3、δ=.1、r4/K2、q1,2,4,8 | 物理複素信号・半径、厳密二項失敗率、1572 fresh-IID trajectory、解析信号／明示測定1回／10万回周辺分布によるbranch復元 | δ=.02の最終18roundの成功率、最終energy資源保証 |
| Horizon：同旧固定設定 | q8,4096,32768行列診断、目標依存round、旧固定設定の単純長q移送の不成立 | 後続scheduleまで不成立という結論 |
| Delta schedule | δ=.01/.0125/.02の選択全56点、operator・signal・PF/RTE phase・半径検査 | 最終opt2 N/α/β・branch・価格の直接検証 |
| WP04 | δ=.02/LD3の108点gridと、LD12の18round信号／半径、旧配分の長round厳密二項・union診断 | 最適化後の別N/α/βでの成功率、最終energy復元 |
| M06-F coherent opt2 | 51task価格監査、同compilerでのβ/α/整数N/schedule最適化、18roundの保存費用集計 | 長q直接compile、実backend、状態準備、最終energy成功・優位性の保証 |

[RPE4][RPE4DOC]、[horizon][HORIZONDOC]、[delta schedule][SCHEDULEDOC]、[WP04][ABLDOC]、[M06-F][OPT2DOC]

RPE4は固定4roundのlocal end-to-end信号・統計・位相検証である。q8観測半径`0.993219824261`、系統位相差`1.06819e-4` rad、4round統計位相失敗率`2.14873e-6`、10万回の分枝・最終位相失敗0件と片側95%上限`2.99569e-5`が報告されている。1572 trajectoryによる一回の成功と、10万回の周辺分布診断は別の役割であり、独立した科学cohortとして足し合わせない。経験的PF screeningの範囲を保つ。[RPE4保存結果][RPE4JSON]

旧delta scheduleのδ=.02行は初期K0を使い、最終opt2は全round K2でrも異なる。旧56点の同δ scheduleとの完全q/r/K一致は最終q131072だけである。しかし**後続WP04の108点gridにはr1,2,4,8,16,32・K2と全18qがあり、最終opt2選択の18 tupleがすべて存在する**。本対応表はその保存点を転記している。gridの実行時配分`.015/.005/.38`、uniform α、数値許容差`1e-10`は、最適化後の配分とは別のまま記録した。信号を再生成したり、新予算に対するPASSを再判定したりしていない。[保存grid][ABLGRID]、[最終round][OPT2JSON]

FullにもLD12/δ=.02/q1..131072の物理信号・半径があり、最小半径`0.9999999998284864`が保存されている。この診断は半径用で、同じ点ごとのPF-vs-exact phase誤差欄はないため、該当欄を`NOT_RECORDED`／`UNRESOLVED`とした。tail-freeという語はPF演算子が元Hの厳密発展であるという意味ではない。[Full診断][ABLJSON]、[診断SOURCE][ABLATIONSRC]

これにより、Phase3 R01/R02/G10の「最終energy資源実証は未完成」という境界は維持しつつ、G10の長q物理signalを一括して未実施と読む解釈を避けられる。Phase3の固定CSV／REPORTは変更せず、本調査を補足として置く。A〜E分類・中心claim・既存科学statusを更新しない。[Phase3 matrix][P3MATRIX]、[G10][P3GAPS]

## 4. PF・finite RTE・半径・統計・branchの接続

PF係数は、WP01が参照する同snapshotの保存D6係数であり、LD3は`0.013399136433495978`、LD12は`0.013425665145396077`。元full-H基底状態へのstate actionとD6のconditioningを使った経験的fitである。支配PF固有位相係数や旧scheduleの丸め値`.01342567`、後続系サイズ検証のC_useとは別物として追跡した。短qのD6対支配固有位相・単一位相診断は通過が記録される一方、両PF artifactの`qpe_spectral_diagnostics_pass=false`はそのまま保持する。`overall_pass=true`を全QPE/RPE統計診断の通過と読まない。[PF3][PF3]、[PF12][PF12]、[prevalidation runner][PREPRUN]

resource accountingは`C*q*delta^3`をPF位相proxyとして使う。経験係数から厳密上界へ昇格していない。長roundでのPartial PF phase比較は既存gridにあるが、最終の細かい最適化予算との照合を新しく数値実行してはいない。[ACCOUNT][ACCOUNT]、[OPT2JSON][OPT2JSON]

finite-RTEでは、`r`は一つのtail occurrence内のshort step数、q回のpartial stepにはqr short stepがある。`tau=exact_rte_lambda_r*delta/r`、偶数Taylor lawと保存Bを使う。gridのexact lambdaは`.5832704246339503`、ranking proxyは`.37036367910736034`なので置き換えない。補正信号対PFの誤差epsilon_Z、Bの正規化、既知の`A_att=B^(-qr)`、PF対元E₀の誤差は別に扱う。保存modelに直接ない数値をtauから逆算して埋めていない。[METHOD][METHOD]、[RTE][RTE]、[ABLGRID][ABLGRID]

既存のTaylor残差をR_K(tau)とすると、補正後のround誤差は既存の合成式`epsilon_Z=(1+R_K(tau))^(q*r)-1`で扱われる。Bは有限event分布のnormalizationであり、Taylor残差そのものではない。この式の新しい数値評価や、モデルに未記録の値の逆算はしていない。[METHOD][METHOD]、[ACCOUNT][ACCOUNT]

asinによるphase boundには`0<=epsilon_Z<rho_star_lb`、観測半径下界には正の`A_att*(rho_star_lb-epsilon_Z)`が必要である。finite検証は実際の`abs(PF signal)`を使い、accountingは単位reference radiusを使う。near-one差や浮動小数点許容差付きPASSは、厳密な単位下界の証明ではない。保存model半径と物理評価半径は同じ欄に上書きしていない。[FINITESRC][FINITESRC]、[ACCOUNT][ACCOUNT]

gridにはDF truncation value0と、symbolic coefficient thresholdで落とした非ゼロoperator bound`4.8294649659693557e-14`が別々に保存されている。これを0へ変えていない。accountingはzero-threshold条件と、非ゼロthresholdがphase予算へ含まれないことを別に記録しており、今回新しくenergy予算へ再配分していない。[保存partial-S2 metadata][ABLGRID]、[ACCOUNT][ACCOUNT]

統計モデルは正の半径から各軸座標許容差を作り、独立有界Hadamard±1、shotごとのfresh-IID RTE、Hoeffding、round軸α配分、全round union boundを使う。最終N/α/coordinate/βは保存値をそのまま転記した。両最終scenarioの`alpha_total_allocated=.05`も保存値であり、今回再集計していない。union bound自体に軸間独立は不要だが、軸内標本の独立条件と、exact product確率を扱う際の条件は別に必要である。[METHOD][METHOD]、[ACCOUNT][ACCOUNT]、[ABLATIONSRC][ABLATIONSRC]

既存の対称座標予算は`epsilon_coord=rho_obs_lb*sin(beta_stat)/sqrt(2)`、各軸のHoeffding Nは`ceil(2/epsilon_coord^2 * log(2/alpha_axis))`である。必要な全round配分条件は`sum_m(alpha_m,c+alpha_m,s)<=alpha_total`。energy分解能の条件`beta_RPE/(q_max*delta_time)<=epsilon_E`は、共通位相・正しいbranch・系統誤差条件が成立するときの接続として読む。ここでは式を追跡し、保存Nや成功率を再計算していない。[METHOD][METHOD]、[ACCOUNT][ACCOUNT]、[HORIZONSRC][HORIZONSRC]

WP04の長round統計診断は実在するが、旧N/α/βとsignal meanに対する結果である。最終opt2配分に対する新しい厳密二項確率、trajectory監査、branch成功率は本調査で作らない。古典価格標本のn8/n32、量子Hadamard shot N、RPE4の周辺反復数を混同しない。[WP04統計][ABLJSON]、[opt2価格監査][AUDIT]

branch実装は`arg(X+iY)`から候補`(phase+2pi*k)/q`を作り、直前推定位相に円周上で最も近いものを選ぶ。最終energy誤差への変換には、同じ共通位相・alias-free window・適用可能な系統誤差条件と、厳密branch条件`beta_RPE<pi/3`が必要である。RPE4はこれを固定小系の位相実験として調べたが、最終18round opt2の復元結果は調査範囲で未発見である。保存の不在と全環境での未実施を同義にしない。[RPE4SRC][RPE4SRC]、[METHOD][METHOD]

## 5. 直接compiled価格とモデル移送の範囲

| 候補・条件 | 保存直接価格 | 用途と限界 |
|---|---|---|
| Partial LD3、δ=.01/.02、r1,2,4,8,16,32、K2 | 各q1,2,8 | q1,2較正／q8固定holdout。初期n8、一部fresh32でprecision gate回復 |
| Partial LD3、δ=.02、r32、K2 | 旧q16,32も存在 | A0が最新fresh較正へ照合したのはselected-policy RZだけ。他r・全metricへ移送しない |
| Full LD12、δ=.01、tail-free | q1,2,8 | 同opt2 deterministic provider |
| Full LD12、δ=.02、tail-free | 最終直接表はq1,2,16,32 | q8直接値が同表にあるとは記録しない |
| q>32 | 直接値なし | affine modelの外挿。物理信号行列検証とは別 |

[直接価格表][OPT2JSON]、[legacy価格][LEGACY]、[A0][A0]

価格はQiskit1.3.0、basis `rz,sx,x,cx`、opt2、seed17、backend/coupling/layout/routingなし、ordinary controlled、cosine/sine full-wrapper、状態準備除外という条件に限定される。主指標はcompiled RZであり、実機料金やbackend実行時間ではない。[M06-F条件][OPT2DOC]、[wrapper][WRAPPER]

A0はδ=.02/r32/K2で最新n32 q1,2較正を旧n8 q16,32へ固定holdout適用し、selected RZは最大4.911%点誤差で既存5%基準を通過した。q32標準化残差2.012、非gating full-basis q32の5.598%不通過も維持する。これを統計的同一性・全metric/q32保証と呼ばない。[A0保存結果][A0]

最終Partial scheduleは`r=[1x8,2x2,4x1,8x2,16x2,32x3]`、K2である。従ってq16,32ではr1であり、r32のq16,32実測をこのroundの直接価格にしない。q4のような較正点間の予測も直接compile済みとはしない。Full最終r0/K0はtailなしのcanonical表現であり、全候補のTaylor設定K2と矛盾するランダムtailを新設しない。[保存round][OPT2JSON]

全roundの単価は軸別q1,2 affine predictionとして使われ、直接anchorがあるroundでもモデル欄は`EMPIRICAL_PROXY`とした。該当セルの保存holdout域を超えるものは`EXTRAPOLATED`。一方、同tupleの直接価格記録があるかは別欄に保存する。したがってq≤32を一律直接検証とせず、q>32は明確に外挿として残る。保存Nと予測単価の接続は`N_c*g_c+N_s*g_s`で、古典価格nをさらに掛けない。[ABLATIONSRC][ABLATIONSRC]、[DECISIONSRC][DECISIONSRC]

## 6. 費用結果をどこまで強く記述できるか

H4/DF12、LD3対12、δ=.01/.02、同coherent opt2 context、外部仮定を含むCA/10暫定taskで、β/α/整数N/scheduleを最適化した保存モデルは完成している。両候補はδ=.02、18round、最終q131072を選んだ。保存の状態準備なしcompiled-RZ点推定は`1.263314e12`対`1.327822e12`、比`.9514180827942961`で、Partialが約4.858%低い。[最終モデル][OPT2JSON]

local5%区間はPartial`[1.1416,1.3850]e12`、Full`[1.2614,1.3942]e12`で重複する。較正sampling SE、per-r selected discrepancy、full-basis診断、25%transfer scenarioは区別して保存され、全scenarioで区間が重なる。5%や25%を全長q価格の厳密信頼上界とは呼ばない。[M06-F解釈][OPT2DOC]、[保存comparison][OPT2JSON]

旧focused解析のopt1/opt2混在結果は別referenceであり、以前の8.422%点差と現coherent4.858%を混ぜない。51/51価格監査が、初期36task／指定fresh32拡張15taskと初期precision不通過の履歴を保ちながら、現H4測定域のcompiler非対称を解消した。科学的成功率・新しい独立物理再現を51taskから推定しない。[AUDIT][AUDIT]、[OPT2EXTSRC][OPT2EXTSRC]

状態準備は比較scopeから除外されている。saved shotsは13588対11162で異なるので、共通準備単価Pでも相殺しない。点break-evenの保存値26590335 RZ-equivalents/shot、P=0で既に全区間重複という条件はそのまま保持する。実際にPF固有状態を準備する費用・候補ごとの違い・noise/backendは未測定である。[保存preparation_break_even][OPT2JSON]、[METHOD][METHOD]

従って「限定H4の明示仮定付き18round費用モデルでPartial点推定が低く、不確かさ区間は重なる」と記述できる。「同じE₀の最終精度・成功率まで実証して資源優位を保証した」とは記述できない。後続DF10/19・opt1等の固定時間resource comparisonは別科学条件であり、このモデルへの不足証拠を無条件に埋めない。[PURPOSE][PURPOSE]、[Phase3][P3REPORT]

## 7. SOURCE・入力のprovenanceと未解決事項

固定文書commitは公開carrierであり、科学実行SOURCEそのものではない。次の保存git_commitはdirty/untracked状態と組で記録される。

| 保存段階 | 記録git_commit | 今回照合した内容 |
|---|---|---|
| RPE4 / horizon / delta schedule | `2bf31169e53f747601f2012425e525f0d59af988` | 各記録runner/moduleのSHA-256とcarrier byteが一致 |
| WP04 grid / ablation | `efa90e0b14a95c296b305bfeea37867e56c83a0d` | 記録runner/module一致、ABLJSONにあるgridファイルhashも一致 |
| M06-F初期compute | `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c` | 公開run_provenanceの10 SOURCE file hashとcarrier一致 |
| M06-F extension compute / coherent解析 | `7753eb71844e429facdb6f32c2def6e37379541f` | extensionの10 SOURCE hashは初期と一致、解析runner/moduleも一致 |

[provenance一覧][RPE4JSON]、[WP04][ABLJSON]、[初期compute][INITPROV]、[拡張compute][EXTPROV]、[解析][OPT2JSON]

38件の報告済みSOURCE-file hash比較はすべて一致した。同一fileを複数結果の記録に対して比べる件数を含み、38個の独立SOURCE一式という意味ではない。Pythonは旧物理検証3.11.0rc1と後続解析3.12.3等で異なり、Qiskit1.3.0を記録する。全推移的科学依存の完全hash、clean CI、raw/QPYからの新再生成を今回認定していない。51task完了・集計再構築は既存監査の記録を再利用し、private checkpointを復元して監査を繰り返していない。

旧RPE4／scheduleのpreparation hashは`50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc`、WP04/LD3 PF検証は`89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`。HamiltonianとLD3 partitionのhashは一致するが、preparation識別子を一つに正規化しない。モデルには物理H基底状態の実準備fingerprint自体が記録されておらず、これが物理比較signalとモデル仮定の完全な入力同一性を認定しない理由でもある。SOURCE静的定義・共通snapshot・保存tupleとの対応は示すが、全実行条件のbitwise同一性を補わない。

調査範囲で残った主な不明は、最終モデルのinput-state／reference-radius仮定を物理H基底状態へ適用する根拠、同じ最終N/α/βに対する18round branch／energy成功、Fullの同δ長q PF-vs-exact phase誤差の保存比較、q>32価格移送、実state preparationと全費用不確かさである。`NOT_RECORDED`は該当保存欄の不在、`NOT_FOUND`は限定読取範囲で該当証拠未発見であり、全研究で未実施と断定しない。既存nullはJSON nullとして保ち0にしていない。

## 8. 新しい科学計算なしで分かったことと次候補

今回だけで、共通energyと符号、scalar、別々の状態モデル、56点と108点の対応、全36roundの保存証拠、既存長round統計、直接価格の限定domain、混在解析からcoherent解析への更新、SOURCE-file hashの対応を整理できた。既存証拠の再利用によって「長roundを全部再検証」「51taskを再compile」「全qを直接compile」という追加作業を当然とはしない。

最も重要な未確認接続は**共通E₀に対してどの状態・PF信号・半径を使い、どのPF biasを予算化しているか**である。次の限定検証候補は、既存保存信号を最大限使ってこの接続を確かめるV1。その結論に依存して最終配分のbranch／成功率V2、最終資源比較を要求するときだけ価格移送V3を考える。保存signalから新しい位相残差や成功率を算出するだけでも科学的再解析なので、今回のmetadata許可で実行しない。[候補と担当・承認][MINPLAN]

ユーザーが追究する主張範囲と次検証を選ぶ。Codexの科学的再解析・新科学計算・compile、GPT等の詳細科学レビューはそれぞれ必要な別承認後の候補であり、本書では開始・正式採択していない。

## 技術確認・原本保全・Mandatory STOP

標準ライブラリのvalidatorによる検査結果は**PASS**。固定68 objectのcommit/path/blob/size/SHA、38 SOURCE-file比較、33論点、36roundの保存literal・selector・参照、Partial18 tuple／Full18半径、直接価格とモデル状態の分離を確認した。人工metadataのq/r/K不一致・JSON pointer・null保持も確認した。固定snapshot参照、WP04 gridファイルhash、A0の入力hash、初期／拡張computeのSOURCE一覧一致もmetadataとして確認した。scientific moduleやrunnerはimportせず、既存scientific testsも実行していない。今回のPASSはmetadataの技術整合性だけである。

```bash
python3 docs/research/pr_rpe_energy_bridge_readiness_20261103/validate_metadata.py --evidence-repo '<existing-repository-with-fixed-objects>'
```

原本読取には`GIT_OPTIONAL_LOCKS=0`、`GIT_NO_LAZY_FETCH=1`を用いた。出力は独立Git作業ディレクトリ・専用新branchに作成し、既存G01/G02成果物commit`fdc7fe58e7e887786322c6204db3654111873958`を親として、新directoryだけを追加する。公開前の元worktreeのHEAD・既存refs・tracked/untracked statusは開始時と一致した。元indexも保全し、SOURCE/input/raw/QPY/grant/ledger/marker/STOP、Phase0〜3、G01/G02、他worktree／archiveを変更しない。元repoのcheckout/reset/clean/gc/prune/fetch、削除、job停止、共有設定変更は行っていない。

公開するのは匿名化した本調査成果物だけ。出力に内部host/IP・絶対パス・秘密情報・private rawを含めない。既存公開provenanceへの固定リンクは残すが、そのprivate commandやpathを本文へ転載しない。full rawのdecode・hash、archive全展開、無関係directory探索は行っていない。

公開branchのcommit後にremote SHA、変更file範囲、公開各fileの全文byte/blob/sizeを照合する。照合結果は最終チャットで報告し、未公開を公開済みと扱わない。

**Mandatory STOP：公開・照合後、この技術調査を終了する。** SERVER1/2、追加科学計算、数値再解析、RNG replay、compile、18round再最適化、詳細科学レビュー、正式claim採択へ自動移行しない。旧contract／STOPの解除、新契約、grantや承認証明も作成しない。

[MINPLAN]: MINIMAL_VALIDATION_PLAN.md
[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究目的・研究課題.md
[METHOD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究方法・解析手順.md
[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/数値実験・評価計画.md
[SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究概要・現状.md
[RPE4DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rpe_four_round_phase_validation.md
[HORIZONDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rpe_target_round_horizon_validation.md
[SCHEDULEDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rpe_delta_round_schedule_validation.md
[OPT2DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[FINITE-DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/finite_rte_signal_validation.md
[PF-DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/pf_delta_validation.md
[COEFF-DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/pf_c_system_size_validation.md
[P3REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md
[P3MATRIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/COVERAGE_MATRIX.csv
[P3GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/GAP_CANDIDATES.csv
[RPE4JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json
[HORIZONJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rpe_target_round_horizon_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json
[SCHEDULEJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rpe_delta_round_schedule_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[RPE4SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_four_round_phase_validation.py
[HORIZONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_target_round_horizon_validation.py
[SCHEDULESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_delta_round_schedule_validation.py
[OPT2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2.py
[OPT2EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py
[DECISIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_decision_cost.py
[OPT2COMPSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_completion.py
[LINEAGESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_proxy_lineage_reconciliation.py
[FINITESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/finite_rte_signal_validation.py
[PFDELTASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/pf_delta_validation.py
[RPE4RUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_rpe_four_round_phase_validation.py
[RPE4TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_four_round_phase_validation.py
[HORIZONTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_target_round_horizon_validation.py
[SCHEDULETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_delta_round_schedule_validation.py
[OPT2TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_research_direction_full_opt2.py
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[PRESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_prevalidation.py
[DFSTEP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2.py
[REPEATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2_repeated.py
[HADAMARD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rte.py
[HORIZONRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_rpe_target_round_horizon_validation.py
[SCHEDULERUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_rpe_delta_round_schedule_validation.py
[OPT2EXTRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_research_direction_full_opt2_extension_analysis.py
[WP01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json
[INITIALMAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012.manifest.json
[EXTMAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_extension_20260924_222811.manifest.json
[LEGACY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json
[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_resource_accounting.py
[WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_interrogation.py
[ABLDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_ablation.md
[ABLJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json
[ABLGRID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json
[PREPRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_research_direction_prevalidation.py
[PF3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot/h4_sto3g_d100_rank12_ld3_v5.json
[PF12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/pf_delta_same_snapshot/h4_sto3g_d100_rank12_ld12_v5.json
[INITPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012/run_provenance.json
[EXTPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/run_provenance.json
[ABLRUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_research_direction_ablation.py
[COMPSRC0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_research_direction_full_opt2_compute.py
[COMPSRC2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_partial_s2_repeated_cost.py
[COMPSRC3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/parallel_validation_executor.py
[COMPSRC6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope.py
[COMPSRC7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_sequence_policy.py
[COMPSRC9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rte_compiled_cost.py
[PFDELTARUN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/scripts/run_pf_delta_validation.py
[VSTATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/VALIDATION_STATUS.md
[VMANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/validation_manifest.json
