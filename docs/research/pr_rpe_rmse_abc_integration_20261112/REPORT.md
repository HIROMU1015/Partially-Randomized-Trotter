# PR–RPE RMSE型評価：A/B/C最終版の技術統合（2026-11-12）

**TECHNICAL_INTEGRATION_PASS**：固定Aのschedule/nominal-N packet、固定Cの受動lookup、Bのgenerate→assess→cost→selectを人工入力で接続し、135技術testが全PASSした。**SCIENTIFIC_QUALIFICATION_PENDING**：実RMSE・十分N・平均compiled価格の資格は未確定。**RESOURCE_OPTIMIZATION_NOT_RUN**：実N/C/G・実M/κ最適化は未実行。人工例だけで条件付き費用と全schedule選択を検証した。

## 固定SOURCEと再利用

| 担当 | 固定commit | 統合内容 |
|---|---|---|
| A | `510fbacd01a2d855202de4c045d8ba1972e5ff3d` | `pr_rpe_rmse.py`、人工demo/test、必要な旧P1/phase0/screen依存をbyte一致で取込み |
| B | `82bc774a92b65cac06b88ebcb2c6979c2726253b` | 最新controllerを継続。既存generateのidentity組立を小関数へ抽出し、source-bound A scheduleの明示入口を追加 |
| C | 公開`2a882fc9d17ed5e32f5581d0f220514daad0235a`、実装`09489eedbbd61c7396138634a896b8d8ecc75161` | `mean_compiled_price.py`と人工testをbyte一致で取込み |

A/CのREPORT/HANDOFF、実装とテストをGit固定成果から確認した。[SOURCE_MAP](SOURCE_MAP.json)に取込み元とbyte照合を記録する。A/Cのbranch/worktree、実行環境、cache/rawを変更していない。旧P1等の依存moduleはAの既存`existing_finite_mean`が参照する`checked_reference`をそのまま使うために必要であり、execute/preflightや科学runnerを起動するために取り込んだものではない。

新しいB側adapterは[`pr_rpe_rmse_integration.py`](../../../src/trottertracks/pr_rpe_rmse_integration.py)。既存のN/C会計・最適化・HP pipelineは重複実装していない。詳細APIと対応表は[HANDOFF](HANDOFF.md)。

## A→B：grid・κ・identity

統合経路はAの`KAPPA_DEFINITION`と`A.schedule`を直接使う。Aのr_target/R_outer/11piece配分を別式で再計算・置換しない。Bのnative ceil generatorはそのまま残し、A接続specには別tag `A_FIXED_GRID_510FBACD_V1`を付けた。

- Aの`EXACT_FIXED_DELTA_WITH_EXPLICIT_SAVED_EARLY`は非整数T/Δを拒否する。例えば人工T=.1/Δ=.15をBの一般ceilならqPF1にできるが、A経路は拒否した。実PF幅を便宜的に変更しない。
- H6初期m0..3のqPF1/δ=T/prefix5/R32/K2とκ_applied=falseを保全した。regularはprefix7/δ1.6、11 signed piece・負時間・元の整数allocationを保つ。
- 全schedule/M/h/κ・元A入力・policy/law/物理candidateをB identityへ含める。`source_candidate_id`と`physical_method`を残し、新しいB configurationへ古い価格を付け替えない。M/κが変わればPRの全identityが変わる。
- PFにκの探索依存を持たせない。A再構成時の内部中立値1はPF未使用であり、PF specのκはnull、mapping.requested_kappaもnull。元A handoffは別に無変更保存し、PF packetのκ非依存性を人工テストで確認した。

Aのraw policyはBのerror_contract内にそのまま保持し、目標RMSEのfield表現だけをexact rationalへ合わせた。数学的な誤差契約を新しく採択していない。AのH4 PF B0等の元方式は保持し、新Bのfull-H PF scopeへ自動昇格しない。対応scopeの科学的確認がなければ`UNQUALIFIED_A_PF_BASELINE_SCOPE`を残す。人工full-H PFとPRも別identityである。

## Aのnominal Nは十分Nへ昇格しない

`n_from_a`はAの最終`b_handoff` spec/configuration/各round identity/law/source_request_idを再構成したmetadataへ照合する。Aの`evaluate`をmean_providerなしで使うのはschedule検査だけで、signal/normalizer/nominal Nの数値leafを呼ばない。元packetは`A_original_packet`にcopyする。

`nominal_N_X/Y`とmodel_statusを保持する一方、N_X/Yとglobal RMSE boundはnull、finite-K/signed mapping/error contractは未資格のまま。Aが現在持たない十分Nやboundを混入したpacketは拒否する。技術的なidentity一致を`technical_identity_verified=true`とするが、科学的RMSE保証を意味しない。旧HP policyやA0/P1 Nを再ラベルしない。

後段testではA packetとは別に、SOURCEを`GIVEN_MOCK_SUFFICIENT_N_NOT_NOMINAL`とした人工Nを与えた。nominalから値を導出せず、元A packetのnullを変更しない。人工RMSE資格はSYNTHETIC_ONLYで、実分子の十分NやRMSE保証ではない。

## C→B：受動lookupと表現adapter

`MeanPriceLookup`はCの`MeanPriceProvider`を使う。Cの旧APIは物理candidateのfloat時刻/δ・旧methodを参照するため、Aが生成した物理rowをlookup用viewとして渡す。API façadeのidentityは常にBの全schedule identityを返す。

Cが必要とするR/K/qPFのlaw fieldとMODEL domainの物理表現はcopy上だけで整形する。入力は先にB full identity/lawへexact一致することを検査し、返却時に元law/model domainを復元する。価格値、SOURCE、cohort、classification/expectation/n_pathsは元と同値でなければ拒否する。元C lawの違いを無断aliasしたり、C1を新identityへ再ラベルする処理ではない。

技術的不具合として、C固定constructorは渡されたpipeline引数を使わずmoduleのpを保持する。C SOURCEを変更せず、Bが所有するprovider instanceのpipeline属性だけをfaçadeに設定した。global関数や他のinstanceはpatchしない。

Cの元domain制限も維持する。PF・H6 prefix5早期round・他domainはOUTSIDE_DOMAIN/nullで、mean/fit/compileへfallbackしない。PF価格は将来の適合したPF providerから別途供給する必要がある。CのMODEL falseは返却後もfalseであり、default expectation policyは有限cohortを適格な全law期待価格にしない。

κ/M/configuration IDは`external schedule_context`としてechoするだけで、Cにscheduleの科学資格を付与させない。未知CI/共分散・R/q systematicはnullを保つ。C閉状態kernel、local residual、combine/fitはprovider内部から呼ばれないことを例外化したtestで確認した。

## end-to-end人工検証と回帰

[TESTS.txt](TESTS.txt)、[SYNTHETIC_SUMMARY.json](SYNTHETIC_SUMMARY.json)参照。新12件はA schedule→B identity→A nominal packetの保全→別mock N→C lookup→B assess→exact G→全schedule選択を確認した。人工C recordの値はCの一状態closed reward kernelから作り、mock compiled価格としてプロトコルを試す。実compileや物理closureの証拠ではない。binding/state/compiler/SOURCEは人工値で、H4/H6というrecipe labelだけで実分子の結果とみなさない。

人工κ二候補ではG=202/20、合法最小は20であり、異なるκのroundを混ぜた4を選ばない。M変更、SOURCE/candidate/law/compiler不一致、A nominalの十分Nへの混入、欠測・MODEL false、巨大mock N、normalization/古典n_pathsの二重乗算禁止、PF/PR/全H純RTEラベルの非同一性も確認した。A current packetのままではG=nullである。

関連回帰はA17、C15、B17、R可変N12、provider8、旧HP/会計15、PF/PR pipeline12、旧R選択12、C1受入15の計123件を選び、新12件と合わせて135件が全PASSした。実分子計算や全repository回帰は実施していない。これは実装者自身の技術確認で、独立科学レビューではない。

## 未接続・科学的未資格と次の範囲

- Aのnominal経験的モデル→科学的に適格なN_X/Y/global RMSE boundは未接続。finite-K・signed negative time・state/effective evolution・alias/decoder/全round・実samplerに対する成立証拠が必要。
- Cの実H4/H6閉状態表現、平均構造価格、compiler残差の校正/独立検証、長q/大R/全law/共分散/CIは未取得・未資格。C1のfalseや旧roleは変更していない。
- AのFULL_H_RTE経路は現固定実装にない。Bの既存方式識別は維持するが、B3を純RTEへ偽装して接続しない。PF/早期H6のC価格と、A supplied PF baselineの科学scopeも別途必要。
- 実G、実資源優位、最適M/κ/L_D/ΔPFは未評価。今回の人工最小はこれらの採択ではない。

次は**ユーザー/A**がRMSE/κ・finite-K/早期override/対象state・誤差契約と有限(M,κ,L_D,ΔPF)の範囲を決める。**C**は同じphysical identityでのkernel closure/価格model・校正/holdout/cohort/不確かさを設計する。**ユーザー**が固定SOURCE/input/precisionと対象を明示して科学実行を許可した後だけ、**A/C**がその範囲の成果を取得する。**B**の実N/C/G評価や実最適化にも別の明示許可された有限menuが必要。

P2/P3、C2 40/16 wrapper、S0、保存科学値再解析、新compile/fit、旧STOP解除は行っていない。既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPとA/C環境を保全した。commit/push・remote/主要成果物照合後mandatory STOPする。
