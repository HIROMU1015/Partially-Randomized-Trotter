# S1：lowering domainと必要な整合性修正

価格は一shotの**full controlled measured X/Y Hadamard wrapperのcompiled RZ数、P0**。ancilla H・X/Y basis・control phase・測定は含める。P0を理由にscalarやwrapperを削除しない。状態準備費は除外する。今回その定義、科学的scalar、compiler、既存recordを書き換えていない。

## SOURCEから判別できる実装

| method / 証拠 | RTE rotation lowering | 再利用できる範囲・不足 |
|---|---|---|
| PR P1/P2・現D adapter | `CONTROLLED_ZZ_CX_CRZ_CX_V1`、D `efficient_controlled_zz.py` | 元H4 DF10/LD1/S2/.1/K2、元exactκ・q/R/law・cohort/axis/source限定。新LD2や全law平均へ付け替えない |
| 旧Full-H light / T2.0 | `CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`、provider c31、Full-H9b83/7a8b | 元H4 DF10/H6 DF19、T=.8又は2.0、取得したq/R/K・one-body込みfull tail・cohortの有限価格。現C scheduleとは別domain |
| 旧PR C1等 | 旧汎用control、元cohort/model資格 | 新ZZ係数/価格の代用不可。旧同domainの開発参照のみ |
| Full PF | RTE loweringは**非該当**。元deterministic native policyを記録 | e7d確定receiptを同formula/actual幅/T/q/元H・state/scalar/control/wrapperで再利用。新たなq-affine資格はない |
| Truncated PF B0/B0I | RTE loweringは非該当。元retained-H/native/scalarを記録 | Full PF receiptを流用しない。B0/B0Iのscalar規則をそれぞれ保持し、retained targetと元energy targetを区別 |

旧DF emitterの二wire回転は `RZZGate(2*unsigned_angle).control(1)`。P0で確認した汎用controlの展開（CCX経由）に対し、新subclassは `CX(s0,s1), CRZ(2u)(anc,s1), CX(s0,s1)`を使う。controlled/rotation/support2以外は旧static methodへ委譲する。回転のunitary/control phase同値は既存P0/P1証拠であり、価格同値ではない。全wrapperの価格差を `N_ZZ×17～19` とする証拠もない。

旧Full-H `setup/cost_job`とT2.0の`setup/worker`はこのnew context managerを呼んでいない。Full-Hの`prepare_full_h`はone-body ZをDF tail末尾へ追加しheadを空にする。抽出scalarは `one_scalar=math.fsum(one.diagonal_eigenvalues)/2`、`extracted=math.fsum((tail_identity,one_scalar))`、native cost側は `ham.constant+extracted`。LD2 descriptorのtail-only s_Rや期待totalをFull-Hへ移植しない。

旧Full-H lightはT=.8の固定H4 q16/R16/K4とH6 q32/R32/K6、各n9。T2.0は保存10候補の探索n9と2候補の独立確認で、取得候補/roleだけの価格証拠。元SOURCEの作用検査・normalization・既存性能は再利用できるが、未取得候補や新loweringのC/平均/CIの証拠にはならない。保存lawのs=qR signal縮約は価格等価の根拠ではない。

現C Full-HはK2/q=2^m/R448、m0 T=.1/q1、m12 T409.6/q4096。旧T2.0のH4 q8/R448/K2とR/Kが同じでもT/q・physical ID・seed/lawは異なる。lightのT=.8候補もq/R/Kが異なる。旧価格を現Cの同名methodへ再ラベルしない。

## C/Bへ渡す最小修正案（今回未実装）

1. **SOURCEに結び付いたrecord側lowering**：PR/Full-HなどRTEを持つrecordは、IDだけでなくnative/emitter/切替SOURCEとfull-wrapper/measurement/compilerを必須にする。保存recordでID欠測なら、固定SOURCEから分かった旧IDを**別sidecarのderived metadata**として添付し、元recordとproducer sourceを保全する。欠測を現在のdefault新IDで補完しない。
2. **comparison側policy**：候補間で共通のcompiler/control/wrapper/P0と「RTE rotationは新ZZ」等の実装条件を明示する。RTE recordが旧/未知ならそのscenarioへ不受入とする。self-declared extensionだけでSOURCE一致を証明したことにしない。未知はGの入力欠測であり0価格ではない。
3. **決定論方式の適用区分**：Full/Truncated PFは `rte_rotation_lowering=NOT_APPLICABLE_NO_RTE` という意味上の区分と元native SOURCE/policyを保つ。固定composeのrequests無し分岐はRTE builderを作らないことを確認した。新tail lowering切替だけならdeterministic回路を変更しない。ただし別native emitter/compiler/wrapperへの変更まで同値とはしない。
4. **ID・cache・quantity**：既存canonical configを無断で再hashしない。C/Bがcanonical profileへloweringを加えるか、consumerのcomparison policy＋source-bound sidecarで扱うかを統一する。新context/consumer keyが必要なら新生成し、旧ID/結果は不変。cacheは実wrapper fingerprintとcompiler・物理/sourceを照合し、価格recordの平均資格やlowering識別をcache filenameだけで作らない。

これは既存profile_extension、physical_context、DIRECT/MODEL資格、price sidecarの小さい整合修正で足りる。新しい認証・承認証明・汎用registry・別compile pipelineを作る必要はない。C/BのbranchをDが修正しない。

## 技術検査と科学契約の境界

最低限のC人工ケースは、同compiler hashの旧Full-H＋新PRが共通新scenarioで拒否されること、旧ID/欠測/source矛盾の区別、同domain新recordの通過、決定論PFの非該当区分、異なるscalar/wrapper/axisの拒否。単にmissing extension拒否が通っただけでは混在防止を完了としない。

metadata上のtruthfulな識別・拒否は価格の物理定義を変えない技術整合修正。**旧Full-Hを新ZZで取り直すこと、正式比較の実装scenarioを選ぶこと、旧数値を新consumerへ受入/再解析すること**はそれぞれ明示された科学契約/ユーザー判断を要する。旧SOURCEのone-shot・STOPを解除しない。現在のD成果はSOURCEの確認と提案までで、family比較gateを実装・実行していない。

旧Full-Hは旧実装scenarioの資料として残せる。共通新scenarioから除外したままならFull-H対PRの正式順位はUNKNOWNであり、「Full-Hが劣る」とはしない。旧Full-H価格を新価格の厳密上界とする変換証明もない。
