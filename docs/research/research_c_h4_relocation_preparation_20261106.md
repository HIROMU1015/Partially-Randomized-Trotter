# H4線形Z項再配置診断：技術準備（2026-11-06）

**技術準備完了、実H4保存Jの新規数値再解析は未実施。公開照合後MANDATORY STOP。**

H4直鎖・隣接1.00 Å/STO-3G、number-CDF8、Nα=Nβ=2、元frame順LD2を固定する。H4・1.15 Å、H6、既存SOURCE・契約・raw・STOP、J/Uは変更しない。新しいoptimizer、DF fit、Hamiltonian評価、PF/RTE/RPE、shots/W、回路、native価格、総RZ、新規性・GO判断は実装・実行しない。

[未承認科学契約](research_c_h4_relocation_contract_unapproved_20261106.json)／[SOURCE](../../artifacts/research_c_h4_relocation/2026-11-06/technical/SOURCE.json)／[37入力の場所・identity](../../artifacts/research_c_h4_relocation/2026-11-06/technical/INPUTS.json)／[passive preflight](../../artifacts/research_c_h4_relocation/2026-11-06/technical/passive_preflight.json)／[人工検証・保全記録](../../artifacts/research_c_h4_relocation/2026-11-06/technical/verification.json)。

## 参照と確認範囲

科学結果の固定commitは417fa459b823509be5c2010c5836e58955088282。[元のH4報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md)と、固定SOURCE SHA256 8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539ebを参照した。元SOURCE386件は原archiveでbyte hash一致を確認した。新解析の実行依存は今回利用する14ファイルと保存入力・metadataに限定し、無関係なworktreeのHEADや資料更新を拒否条件へ追加していない。

正式運用指示の原本は /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-25.md。指定の§8.1、8.4、10、10.1、10.3を確認した。原本hashはverification.jsonへ記録する。

Claude_研究C_PRawareDF_独立科学レビュー_20261106.mdの原本は、既知の共有位置および指定固定commit内から取得できなかった。本指示に転載された関係式と数学SOURCEの整合性だけを確認した。レビュー全体を読了したとは扱わず、独立科学レビューも新規に実施していない。引用された0.413 Ha等の実データ上界や再配置後の順位は、今回の準備で新規計算していない。元の45.47%という条件付き総RZ結果は従来の辞書・政策に対する履歴である。

## number → I/Z/ZZの規約

spin modeはa=2i+σでinterleaved。各frameで

\[
D_t=\sum_{ab}K_{ab}n_an_b,\qquad K_{ab}=J_{a//2,b//2}/2
\]

となる。既存general_k_coefficientsの定義は

\[
c_I=\frac{\sum_{ab}K_{ab}+\operatorname{tr}K}{4},\quad
c_{Z_a}=-\frac12\sum_bK_{ab},\quad
c_{Z_aZ_b}=\frac12K_{ab}\ (a<b).
\]

c_pr_aware_df.coefficientsはidentityを除く同じ係数写像である。s_i=Σ_j J_ij、S=Σ_i s_iとおくと、各spinのZ係数は−s_i/2。spatial offdiagonal pairの4 spin組合せと、同一spatial軌道のup/down pairを数えるため、

\[
O=\sum_{t\ge LD,i<j}|J^t_{ij}|,\quad
D_{\rm diag}=\sum_{t\ge LD,i}|J^t_{ii}|,\quad
Z=\sum_{t\ge LD,i}|s_i^t|,
\]
\[
M=O+D_{\rm diag}/4+Z,\quad E=3O+5D_{\rm diag}/4,\quad
E-M=\sum_{t\ge LD,i}\left(\sum_j|J^t_{ij}|-|s_i^t|\right)\ge0.
\]

したがって線形Zを除いたmassは

\[
M_{\rm ZZ}=O+D_{\rm diag}/4,\qquad E/5\le M_{\rm ZZ}\le E/3.
\]

下限側の差は2O/5、上限側の差はD_diag/6であり、非負性から直ちに従う。D_diagは対角係数massで、強決定論baseline Dとは別の記号である。signed対称Jの符号は保持し、identity/nuclear scalarはどの非identity massにも加えない。

## 一体項とscalarの帳尻：設計上の整理だけ

Uの列がframe軌道を表すSOURCE規約に従う。tailの線形部分は

\[
L_t=-\frac12\sum_{i\sigma}s_i^t\widetilde Z_{i\sigma}
=\sum_{i\sigma}s_i^t\widetilde n_{i\sigma}-S_t I.
\]

元gからのcorrected κはκ=h−(1/2)Σ_r g_prrqであり、fitされたJから補正を再計算しない。再配置を実施する将来の方式では、一体項に

\[
\Delta\kappa=\sum_{t\ge LD}U_t\operatorname{diag}(s^t)U_t^T
\]

を加える。

既存tail identity c_Iをそのままdeterministic phaseへ保持する帳簿なら、headへΔκのnumber項と追加scalar−Σ_tail S_tを一緒に移す。nuclear scalarは元の値を一回だけ保持し、この追加scalarと混同しない。別の同値な帳簿として、元nuclear scalarを一体blockで維持し、tailのdeterministic identityをΣ(c_I−S_t)へ再割当てしてもよいが、両方で同じ補正を二重計数してはならない。controlled相対位相も含め、phaseを捨てない。

元のhead DF frameは0,1のまま、tail frameの順も変えない。ただし再配置後のhead一体blockとtailの内容は変わる。Hamiltonian総和の代数的同値性は、元のPF順序・交換子誤差・RTE・shots・native費用の同値性を意味しない。今回のcodeはκ、scalar、U、Hamiltonian、回路を書き換えず、固定Jのmassだけを返す。

## 最小実装と再利用

追加はc_h4_z_relocation.py、run_research_c_h4_relocation.py、対応する人工testだけ。既存の以下を呼び出す。

- c_pr_aware_df.coefficients：Z/ZZの係数写像。
- c_near_objective.mass_diagnostics：元のexact M/E、row cancellation gap。内部のsmooth値は本判定に使用しない。
- c1_general_k_adapter.general_k_coefficients：人工testでI/Z/ZZ規約を独立に照合する。
- c1_stage1_execution.sha/read/save/array_record：既存byte/配列identityと保存処理。旧approval/grant dispatcherは呼ばない。
- c_pr_aware_df_stage_b_runner.load：将来の許可後もJだけを読むwhitelist。U/h/gや保存状態をdecodeしない。

追加の数値処理は、既存関数に独立出力がないO、D_diag、Z、M_ZZ、比、恒等式残差だけ。M/E/gapの係数写像や既存診断の再実装はしていない。dtype float64、対称性、finite、LD、保存mass、SOURCEと入力identityを確認する。

## 固定した入力

INPUTS.jsonに全checkpointのcandidate ID、start/arm/strength/budget、元SOURCE、容器path/SHA256、保存J shape/dtype/SHA256、元report path/hash、既存M/E/gapを記録した。

- original系列：17 trajectory × 16/64 checkpoint＝34件。実容器はresearch-c-integrated-server-20261106の artifacts/research_c_integrated/2026-11-06/development/run01/optimization/ 以下。元SOURCEは200418968aa9e60e5ce842425e0373a4d8fafdb4f78c53a22df37b204989afd6。
- picklefix系列：C0/ENTRY/strength .3の16/64、計2件。実容器はresearch-c-integrated-picklefix-20261106の artifacts/research_c_integrated/2026-11-06-picklefix/development/run01/optimization/ 以下。元SOURCEは8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb。
- 中立C0参照1件：original serverの artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/initial.npz。36件のgridに混ぜない。容器hash78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856、J hash9fbac6c4d20cf4bb3cb7f0840c407a28bad366883cc7dc150f80e1ccf7604233。保存Mとの照合を行うが、元E/gapはこの参照のmetadataにないため、架空の参照値を補わない。

以上の37容器はbyte hashを照合し、J配列hashは保存JSONから取得した。**今回の実Jのdecodeは0**であり、J hashと実memberの照合・再配置massの評価は別許可後に初めて行う。rawは元位置に保全し、Gitへ移送・再保存しない。このサーバーでは全入力が利用可能だが、公開repositoryは未収録rawの配布物ではない。

## 比較と数値規則

全36件を同じ定義で処理し、同じstart/strength/更新予算の18組を比較する。加えて元の政策で事前選択済みの主比較を固定した。

| 主比較 | PR | ENTRY |
|---|---|---|
| common政策の候補 | C0/.1/64 | C0/.1/64 |
| fair-retuned政策の候補 | C0/.1/64 | C0/.03/64 |

後者は異なるstrength同士であることを明記する。rはmassの引数に含まれず、この表から再配置後のr最適化や価格を認定しない。結果を見て候補・strength・予算を入れ替えない。

恒等式・保存mass照合のallowanceはatol=1e-12 Ha、rtol=1e-12。差がatol+rtol×max(|PR|,|ENTRY|)以内ならNUMERICALLY_SAME、外側は符号に従いPR_LOWER/ENTRY_LOWER。これは丸め範囲内の同程度という数値判定であり、新しい科学的成功閾値や統計区間ではない。ENTRY分母が同じallowance以下なら比はnullで、値と差を残す。欠損・改変・非finite・不正dtype/対称性・binding違い・保存mass不一致・恒等式違反はUNRESOLVED_INPUT_OR_NUMERICSとしてSTOPし、代替候補や追加実行で埋めない。

出力はcheckpoints.tsv、matched_pairs.tsv、result.json（元値・比・恒等式と保存値の残差・SOURCE/contract hash）、必要時failure.json、必須STOP.json。科学出力先は artifacts/research_c_h4_relocation/2026-11-06/science/run01。Freshのみ、retry/resume=0、追加実行なし、完了/失敗ともmandatory STOP。total_RZ、shots、W、PF/RTE、新規性の判定はnullで保持する。

## 人工検証・passive preflight

新規28件、既存46件＝74 testsと追加29 subtestがPASS。signed対称J、非負/同符号でM=E、相殺、上下限の等号例、tail mask、frame順、二重計数、既存2写像との一致、identity除外を確認した。人工の全occupancyに対するnumber polynomialの式照合も行ったが、実分子Hamiltonianは生成していない。

人工一時archiveの36件E2EとC0参照分離、J-only whitelist（Uを読むと失敗するfixture）、欠損/容器改変、metadata/SOURCE/保護記録不一致、既存mass違いでのSTOP、再開拒否、無関係な歴史資料を新たな依存にしないことを確認した。実保存Jへアクセスする人工testは禁止した。

general-K既存testはprocess内にffsim等が存在しないことも検査するため独立Python processで実施した。最初の一括起動は他suiteがimportしたffsimによりそのteardownが失敗したが、既存testを変更せず、指定の分離条件で全件PASSした。XMLと実行条件をverification.jsonに記録した。

passive preflightはstdlib/JSON/byte hashだけで全37容器・metadata・SOURCEを照合しPASS。既存fit専用環境を読取利用し、package/OS/schedulerや他processは変更していない。軽量な係数縮約であり、vector化した既存写像を再利用する。人工test suiteの独立process以外に管理frameworkや重い並列処理は追加しなかった。

## 実行前に残る事項と一度の許可

技術的な入力欠損はない。Claudeレビュー原本の未取得は明示したとおりで、本指示で定義された診断の準備を妨げない。実Jのshape/finite/対称性・member hash・保存mass一致はまだ数値的に確認しておらず、許可後の診断そのものが検査する。

ユーザーがこの契約・SOURCE・37入力の固定範囲について明示的に実行を指示した後、Codexは次の一回の起動で全範囲を処理できる。未承認というfilename/statusは準備時点の状態であり、承認証明書やgrant、承認原文保存、各checkpointごとの再承認は不要。

現時点で実行してよいのは既定のpassive preflightだけである。

    /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106/.venv-fit/bin/python scripts/run_research_c_h4_relocation.py

別途の実行許可後にだけ同じcommandへ --execute を追加する。今回は --execute を使用していない。

## 最終確認Q1〜Q12

| 問い | 回答 |
|---|---|
| Q1 SOURCE一致 | M/E/gap、ZZ-only式と上下限、I/Z/ZZの係数規約が一致。レビュー原本全体は未取得。 |
| Q2 再利用 | coefficients、mass_diagnostics、general_k_coefficients、既存sha/array_record/J whitelist loader。 |
| Q3 実装 | 固定Jの内訳抽出・照合・記録の最小経路が完成。既定はpassive。 |
| Q4 所在・hash | 36件＝original34＋picklefix2と、別枠C0参照1の容器/保存J hashを確認。実Jは未decode。 |
| Q5 テスト | 新規28＋既存46＝74件、追加29 subtestがPASS。 |
| Q6 恒等式 | O/D_diag/Z/M/E/M_ZZ、gap、上下限、identity除外が人工データでPASS。 |
| Q7 固定比較 | 18同条件組と元政策の主比較2組、36件全表、tie/不適格規則を契約で固定。 |
| Q8 下流との区別 | massの順位だけを扱い、PF/RTE・shots/W・native/総RZ・新規性は対象外/null。 |
| Q9 残る問題 | 技術的欠損なし。レビュー原本未取得と実member内容未検証は明示。科学実行は未許可。 |
| Q10 保全 | 旧SOURCE386件と旧容器/契約/STOPを読取照合。旧J/U・raw・H6環境へ書込みなし。 |
| Q11 一括実行 | このサーバーで一度の明示許可後に全37入力をfresh実行できる状態。まだ開始しない。 |
| Q12 次の判断 | ユーザー／GPTが未承認契約の固定範囲の診断を実行するか判断。重要レビュー・GOは開始しない。 |

公開commit/tree/本文を照合した時点でMANDATORY STOPする。次はユーザーの科学実行判断であり、H4保存J解析、再配置後の下流評価、H6/H4・1.15 Åへ自動進行しない。
