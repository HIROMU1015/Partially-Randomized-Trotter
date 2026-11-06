# 研究C：H4線形Z再配置後の下流比較・技術準備

2026-11-06 JST。**技術準備と人工検証のみ。再配置後の実H4 PF/RTE・RPE・native・shots/W/総RZは未実行。** 重要科学レビューや論文化判断も実施していない。

[未承認科学契約](research_c_h4_relocated_downstream_contract_unapproved_20261106.json)／[SOURCE](../../artifacts/research_c_h4_relocated_downstream/2026-11-06/technical/SOURCE.json)／[verification](../../artifacts/research_c_h4_relocated_downstream/2026-11-06/technical/verification.json)／[人工・回帰検証XML](../../artifacts/research_c_h4_relocated_downstream/2026-11-06/technical/tests.xml)。

独立branch `codex/research-c-h4-relocated-downstream-preparation-20261106`。基点は再配置診断の固定commit `d601e0ef4375c37185211aa75685ae0e582f3e64`。旧科学SOURCE・raw・契約・grant・ledger・marker・STOPや他worktreeを変更していない。共通運用指示2026-10-25の§8.1/8.4/10/10.1/10.3とprojects/AGENTS.mdを確認した。作業は主agent自身による技術確認であり、独立科学レビューではない。

## 1. 問いと固定する入力

Linear H4、隣接1.00 Å、STO-3G、number CDF8、Nα=Nβ=2、sector36、元frame順LD2。元の主比較に対応するC0/64更新の保存3候補だけを使う。

| ID | 目的arm | strength | 保存容器SHA256 |
|---|---|---:|---|
| P | PR-aware | .1 | `1bd7c00ec11ef468993b36edcd267b84f921d66acb8994a4f6a9c751ddb66cfb` |
| E1 | ENTRY | .1 | `97eb7d540d15d13d961237445912f73536fb0c857de88cb1c9f9fcc20370a7a9` |
| E2 | ENTRY | .03 | `0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5` |

J/Uそれぞれのshape/dtype/SHA256、保存DF誤差・guard・予算、元report/SOURCE、candidate固有referenceの容器identityも契約に固定した。16個・662,319 bytesの必要入力を新namespaceへ**byteコピー**した。保存J/U、κ、reference、QPYの数値decodeは今回0。外部worktreeの将来の通常更新を停止条件にせず、今回使う入力コピーだけを実行依存とする。

元κは既存Stage1 `fit_export.npz` の `kappa` をそのまま使う。新たにh/gから補正κを計算せず、保存metadataの配列identityで結ぶ。元h/g/scalar入力の容器SHA256は `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。将来の許可済み実行で、κを用いた元9因子を各候補の保存CAR因子と照合する。κの元入力とh/g subsetのファイルhashが異なることを、同一bytesと誤表記しない。

既存報告の引用：元の辞書での公平retuning総RZは93.007対170.560 billion、45.47%差だった（[固定417fa459の報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md)）。[固定d601e0efの再配置mass診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d601e0ef4375c37185211aa75685ae0e582f3e64/docs/research/research_c_h4_relocation_execution_20261106.md)では主P/E1/E2のZZ-only massは .472232/.380832/.381049 Ha。この順位は新しい下流費用の順位ではなく、元の45.47%を再配置後へ流用しない。

## 2. 再利用と必要な差分

| SOURCE/API | 適合性と利用箇所 |
|---|---|
| `c_pr_aware_df.coefficients`、`c_near_objective.mass_diagnostics`、`c_h4_z_relocation.diagnose` | 元I/Z/ZZ係数規約とZZ-only mass検査を再利用。目的関数・optimizerは起動しない。 |
| `c1_general_k_adapter` | 元のnumber-CDF8/J/U/列規約のimportと各frame primitives。一体κを再補正しない。再配置済みtailをnumber frameと偽装しない。 |
| `c1_stage1`、`c1_molecule_extension.SectorCAR` | raw number-CARの一体項・frame、既存fermion/sector/bit順helper。新しいHamiltonian生成やDF fitは追加しない。 |
| `c_pr_aware_df_stage_b` | Arm、atom operator、configure、partial S2/s1、trajectory、native kernel。元number専用`tail_gate`だけは旧massを検査するため置換が必要。 |
| `c_pr_aware_df_stage_b2`、`stage_b2_compat` | candidate固有CAR/head/event cache、law一致、全行列raw/compiled/実逆回路/return、shell/X-Y、native32標本の校正、Qiskit共有mutex。 |
| `c_pr_integrated_performance.native` | native32、保存・全512入出力検査・副指標の既存本体。再配置済みprepared contextを必ず渡し、旧h/g経路での再構築を避ける。 |
| `c_pr_aware_df_m4`、`c1_stage3_reference`、`c_rte_r_retuning` | h=.1/K0、13round、candidate固有PF、正規化補正RTE、減衰、適格性・RPE誤差配分。新しいshots式なし。 |
| `c1_stage3_accuracy`、`c_pr_aware_df_stage_c`、`c_pr_aware_df_m3` | 既存accuracy targetと3lane、s1のserial会計、Wとshell、shared calibrationの不確かさ。 |
| `c_integrated_retuning_adapter`、`c_integrated_retuning_policy` | selection32/final64とPF cache。既存既定r16/32/64は保持し、明示r集合とtail gateを受け取る任意引数だけ追加。 |
| `c_update064_strong_d`、`c_same_ju_strong_d`、`c_pr_aware_df_m2` | 同じ再配置済みmodel/全CAR因子を渡せる任意引数を追加。全frame S2、Dのnative/逆回路/13round/会計本体は再利用。 |
| `c1_stage2_native`、`c1_h4_strong_baseline`、`c1_h4_native_fair`、`c1_h6_compiler_precision`、`c_full_unitary` | NativeModel、spin-adapted head、精密Gaussian・relative fusion、P0の位相保持、Aer CPU double全行列評価。 |
| `c_near_r32_statistics`、`c_r32_price_statistics` | Welch差区間、個別PR/Dのt診断、分布・影響度診断。新たな統計式なし。 |

新規は `c_h4_relocated_downstream.py` の再配置・raw reference接続と、`c_h4_relocated_execution.py` の3候補固定orchestration。既存科学runnerをresumeせず、新fresh outputへ接続する。CLI `scripts/check_research_c_h4_relocated_downstream.py` はstdlib/JSON/byte hashのpassive preflight専用で、科学実行optionを持たない。

## 3. κ・scalar・identity・phase

各tailについて sᵗᵢ=ΣⱼJᵗᵢⱼ、Sₜ=Σᵢsᵗᵢ とし、Lₜ=Σᵢσsᵗᵢ ñᵢσ−SₜI。次の一つの帳簿に固定した。

1. 一体headをκ+Σtail Uₜdiag(sᵗ)Uₜᵀに変更する。
2. 全体scalarに−Σtail Sₜを一回加える。
3. Tail frameの元identityとZZ係数を維持し、線形Z primitivesだけを除く。
4. Prefix frame0,1、元frame順、spin順、J/Uを保持する。

Tail identityは既存`split_native_model`がexact deterministic phaseとして抽出し、λの外に置く。全体scalarには保存核scalarが既に一回含まれる。Kernel emitterが全体scalarとtail identityを既存規則で一回ずつancilla phaseへ渡す。DはLD2再配置済みの**同一NativeModel**から全frame ordered S2を作り、LD8で再配置し直さない。

元number modelは別の`original`として保持する。再配置Armの`general=None`により、旧number-frame helperを誤使用すると失敗し、旧tail線形Zをreferenceへ混入させない。raw referenceは既存number-CAR frameからLₜを差し引き、native primitivesからreferenceを生成しない。

人工2軌道/3frameと4軌道/8frameで、number/fermionic expansion、sector、full-Fockの全演算子とscalarを照合する。全Hamiltonianのbefore/afterはatol 2e−12、rtol0の数値検査。これは代数的恒等式を有限精度実装で確認する検査であり、丸め誤差の厳密enclosureではない。Hamiltonian等価性からPF/RTE/RPE/nativeの等価性は導かない。

## 4. 下流接続と正しさ資格

PF/RTEはpartial ordered symmetric S2/s1、δ=h=.1、K0、m0..12。新head/tailからPF kernel・13 PF誤差・候補own-DF ground/aliasを作り、rごとに新λ・確率・RTE減衰とRPEを評価する設計。元PF、λ、減衰、shots、W、native価格の再利用は禁止。PF kernel/誤差を共有できるのは新しい**同一候補・同一再配置representation・同一input**のr違いだけ。

旧H4と同じCA=.00159 Ha、DF reserve1/8、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26を契約に固定。`evidence` laneの全law signal誤差は不明のままnull。`ideal_exact_native`主laneと`assumed_raw_signal_budget`の反実条件を明示し、未資格を0で埋めない。

各適格価格標本と各Dについてraw/reference、compiled/raw/reference、実際に生成したraw/compiled両inverse、inverse return、control0/1相対位相、scalar/identity、shell/X-Y、全sector外を含む**512×512**の全入力/全出力を検査する。既存Aer CPU unitary doubleを利用し、component/adjoint1e−10、complex signal2e−10、normalization1e−12、rtol0、global-phase fitなしを保持する。全行列とnumber保存CAR referenceの照合がoff-sector leakageも拘束する。中間Gaussianをsector射影しない。

旧H4の正しさ検査範囲を弱める提案はしていない。人工CDF8のPR/D双方でも全512検査を実施した。ただし、有限本のnative標本の数値検査であり、PR全law synthesis、実装有限bit lawとの信号距離、丸めenclosure、actual CA、FT資源を認定しない。

## 5. 比較・選択・統計の未承認案

**推奨r集合は{8,16,32,64,128}。** 元の16/64端点を各側一octave広げ、同じ5候補を全armへ与える。既存RTE engineは正のrに対応し、任意引数の明示集合で新しい両端も接続できた。計算負担は3rから5rへ増える。端点採用時は「試験格子内の端点解」と記載し、結果を見た拡張・標本追加はしない。

| 比較 | 事前固定する政策 |
|---|---|
| A | P+PR r32対E1+PR r32。strength/J/Uは固定。 |
| B | P+PR対E2+PR。各候補のselection32で同じ5rから選び、独立final64で比較。 |
| 同J/U D | P/E1/E2それぞれのretuned PR対各自D。P/E1の共通r32 finalも該当Dと比較。 |
| 表現ごとの実行政策 | 各候補でselectionのPR最小費用と候補固有Dを比較してPR/Dを固定。P対E1、P対E2を独立finalまたは確定的Dで個別副比較する。 |

候補identity/strengthを再配置後の結果から交換しない。r選択の費用同値時は32,16,64,8,128の順、PR/D同値時はDを選ぶ。Dは同J/Uの登録S2/s1対照であり、決定論PF全体の大域最適ではない。旧XDF8/CDF8価格は別SOURCEの歴史的baselineで、新finalへpoolしない。

主familyは新しいA/Bの2差、family α=.05、各97.5%の既存近似Welch差区間を**提案**する。旧研究の4差/98.75%はその履歴のまま保持する。新family数を2にしたための事前設計であり、結果取得後の変更ではない。PR/Dとbest-policy比較は個別nominal95%の副比較に限定する。これらの水準・family・r gridは契約案に具体化したが、科学的採択も実行承認も未取得。

精度目標は主差のhalfwidth/ENTRY total≤.05、PR/Dは既存.05、skewness2・delete-one shift/halfwidth .25の診断を維持。zero variance・missing・ineligible・精度未達はunresolvedとし、PR勝利や費用0にしない。比は点比だけで、random denominatorからの比区間を新設しない。

Selectionは各適格policy32、finalは各unique(candidate,representation,r)64（既存32 batch×2）。最大final pairはP32/E132と各候補のtuned3件の計5件。重複は一つのfinalを共有し、A/Bを独立反復と数えない。Selectionとfinal、候補/arm間の統計標本は共有しない。Dと確定referenceは全identity一致時だけ共有する。Finalは選択に渡さない。

## 6. 会計と実行規模

既存 `C_total=W×mean_kernel+shell` を保持。s1ではW=Σm 2×shots_per_axis[m]×2^m。rはkernel価格に既に含まれ、Wへ再乗算しない。Native32/64は古典的価格標本数であり、RPE shotsでもkernel実行回数でもない。会計・副指標と正しさ資格を分離して保存する。

| 論理単位 | 最大数・定義 |
|---|---|
| 候補/PR policy/D policy | 3 / 3×5=15 / 3 |
| Selection native | 15×32=480 |
| Final native | ≤5×64=320 |
| 完全kernelの価格標本 | PR≤800＋D≤3 |
| Kernel全行列decode | raw/compiled/両実inverseの4×803=3212。shell/basis等の検査は別途。 |
| PR PF/RTE round reference | 3×5×13=195 |
| D PF round reference | 3×13=39 |
| 新PR PF kernel/RTE kernel | 3 / 15。候補内PF cacheのみ共有。 |
| PR exact evolution | 3×13=39。Dは同じ候補H/eigenpairから既存PF reference処理。 |
| round lane assessment | 15×13×3＋3×13×2=663 |
| PR event数 | selection 3×32×(8+16+32+64+128)=23808、final最大64×(32+32+3×128)=28672 |

全512 complex128行列一つは4 MiB、既存8×8 full-CAR bilinearは64 MiB。既存resource plannerのdense作業量見積もりはnative worker当たり256 MiBであり、CPU/RAMのquotaではない。32 sample task batchを実際のaffinity/quota/MemAvailableに応じて並列化する。3候補の上位driverは逐次実行する。理由は再利用した`SharedQiskitCalls.adjoint_builds`がprocess-global inverse helperを一時変更するためで、複数候補を同一processで同時に実行しない。候補内のCAR reference・decode・行列検査は並列。

人工回路の過去実測（元H4報告より）は4行列decodeでr16/32/64が7.82/11.47/18.30秒。今回の再配置実H4は測定していない。これを基にした**予測**はdecode3〜8 core-hours、全工程wall20〜90分、peak RAM10〜24 GiB、結果3〜8 GiB、作業領域10〜20 GiB。選択r・compiler負担・共有host負荷で変わる見通しであり、測定値・保証・強制上限ではない。

準備時の読取観測はaffinity64 CPU、MemAvailable約452 GiB、disk free約319 GiB。恒常的な予約や制限とはしない。科学実行前に実際の空き資源/quota/共有競合を再確認する。他ユーザーのprocessやOS/cgroup/scheduler/共有packageは変更しない。無関係worktreeのHEADや一時的PSIだけを強制停止の理由にせず、実OOM・入力破損・科学的不整合・実行異常は保護する。

## 7. 検証・保存・STOP

最終suiteは新規25件＋既存93件＝118件がPASS（212.69秒）。最終のsigned-atom明示検査追加後は関連8件を再確認してPASSした。重複8件を独立testとして水増ししない。詳細の件数・時間は[verification](../../artifacts/research_c_h4_relocated_downstream/2026-11-06/technical/verification.json)に記録する。新人工testは保存配列loaderを拒否し、invented J/U/Hamiltonianだけを使う。検査範囲は以下。

- Number/fermionic/sector/full-FockのHamiltonian、scalar、元tail identity、ZZ保存、線形Z除去、mass。
- PR/ENTRY共通変換、read-only J/U、元frame順・spin規約、再配置済みreferenceの13round binding、五つのr。
- 人工CDF8の全512 raw/compiled/実inverse/return/X-Yと相対位相故障、sector外の故障検出。
- 同じ再配置済みmodelを渡すDのnative・13round・RPE・会計全経路。
- 実native batchが用いるseed keyとSelection/Final/batchの分離、selectionのみの固定、missing/ineligible/端点処理。
- s1のshots/W/shell会計、fresh output、途中結果保存、例外STOP、retry/resume拒否。
- 既存retuning/policy/native/full-unitary回帰。旧既定APIの動作を維持。

人工testの初期追加時にseed検査用mockの`m3`不足で1件失敗し、mockを補完した。実科学の失敗・再実行ではない。検査を弱めず最終suiteを再実行した。元ファイル由来のSyntax/Deprecation warningは検査資格や許容差を変更しない。

新科学出力先は `artifacts/research_c_h4_relocated_downstream/2026-11-06/science/run01`、現時点では未作成。Retry/resume=0。科学実行時はSOURCE/contract snapshotと完了済みrawを残し、完了・異常の双方でSTOPを保存する。入力・数学・位相・correctnessの不整合はMANDATORY STOP。不適格は別候補で埋めない。旧STOPや消費済みgrantを再利用・再発行せず、新たな承認証明書・認証・grant発行も要求しない。

## 8. 解釈と次の判断

保存J/Uは元辞書の目的で最適化されたもので、再配置目的で再最適化した結果ではない。同一候補のbefore/afterはscalar込みの全Hを保持するが、P/E1/E2間の近似Hが完全に同一とは限らない。Tail massの小ささはPF誤差・RPE shots・総RZの小ささを保証しない。ENTRY有利でも全PR-aware目的の一般的不利を示さず、PR-aware有利でも標準辞書での再最適化や強い外部baselineへの優位は示さない。

既存のideal-law・exact-native等に条件付けた資源モデルの範囲で比較する。実装全law誤差、actual CA、FT資源、状態準備込み費用、他geometry/分子への一般性は未資格のまま。

**Q1〜Q12への回答**：既存科学エンジンを再利用し（§2）、head/tail/scalar/phaseを実装した（§3）。人工全演算子の数値等価性とPF/RTE/RPE/native/同JU D接続を確認した（§3/4/7）。PR/ENTRYは同じ変換。推奨格子は8/16/32/64/128で端点拡張なし。Selection32/final64、D、2差family97.5%を未承認案に固定した（§5）。Native資格は旧H4と同じ全512/実inverseの数値検査であり全law保証ではない（§4）。検証の件数と未確認事項はverification/§7、論理規模と資源予測は§6、SOURCE/入力identity/fresh outputは契約と§1/7に記載した。

次の担当は**ユーザー**：五つのr、A/Bの主family97.5%、個別副比較、全512・selection32/final64、固定3候補の契約案を採択し、このSOURCE/入力/範囲の科学実行を明示許可するか判断する。承認証明の発行は不要。**Codex**：その許可がある場合のみ、一回のfresh科学実行と結果報告を行う。今回はcommit/push/公開照合後にMANDATORY STOPし、実H4/H2/H6/H4 1.15 Åや重要科学レビューへ進まない。
