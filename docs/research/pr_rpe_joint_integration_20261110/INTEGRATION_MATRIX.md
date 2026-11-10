# 改善系列・SOURCE・接続条件

[総括](REPORT.md)／[出典・API](SOURCE_INDEX.md)／[次回案](NEXT_EXECUTION_PROPOSAL.md)。名称ではなく以下のdomainとSOURCEで区別する。表は既存報告・コードの静的な対応付けで、新しい数値比較・再資格ではない。

## 1. 同名研究と科学条件の区別

| ID | 系列・対象 | 固定条件の要点 | 接続上の境界 |
|---|---|---|---|
| Original | 元のPR＋RPE資源研究 | fixed DF/prefix、partial S2、finite paired RTE、round別r/K、誤差/α配分、compiled provider | q≤4直接optimizerと長round proxyを区別。初期normativeな未統合記述を全系列の現在地としない |
| Legacy A | 旧Track A・DF/Pauli資源地図 | PM/geometry/H4/H6/H8・高次PF・有限T/長時間B2/B3等、条件別SOURCE | 論文用有限T比較と全energy-RPEを区別。改善系列Aと同名でも別 |
| A-O1 | 元Track A（2026-11-06〜08）の改善 | O1-M1：DF/Pauli辞書、M2：core＋residual、B2-P1：元H ground／DF12／sector36／18round | C0のrank10/LD0/7round、M2 operator-only、B2元H条件を混ぜない |
| New A | PR Reference／V6／Q3 | H4/1Å/STO-3G、DF12/LD3、δ=.02、h=.02、K2、18round、元H ground、compiler A | canonical B′/Reference-B1、m17 q131072/r32/bits88。D-CDF8ともClosed最適policyとも別 |
| B/O2 | Track B／sampling・return・推定器 | G6–G10、Closed P3/P5、Parent、IS/CV/LCU。toy/H4/価格scopeを各SOURCEで指定 | 研究B/B2-DのD内部PFとは別。現在O2担当と過去Track Bは一対一でない |
| C-O3 | 改善系列Track C | H4/DF12/LD3、218成分、sector70、δ0=.02、18round、ε=1.5936e−4 Ha、failure=.05、compiler A | ground BCH・joint basis・Closed P3・可変h/rate・α、強Full/Truncated。modular価格・serial depth上界 |
| RC | 研究C・PR-aware/ENTRY DF | H4/number-CDF8等／LD2／h=.1／K0／13round、own-DF ground・DF reserve | 近似表現ごとのstate/target資格。C-O3とは別、Dの元H groundへ自動移送しない |
| D8 | Track D／Reference/control | H4/1Å/STO-3G/Nα=Nβ2、元H、number-CDF8/元順LD2、S2/s1/r32/K0、h=.1、13round、P0 | 強symmetric対照、whole-path一gadget、元H ground・準備費除外、固定十分N＋A/F/S条件付き価格 |

研究B／B2-DはD内部・outer分割の別研究。Bという文字だけでB/O2のreturn法と統合しない。元Hの保存packet、近似DF、state、sector、compilerのidentityは分子名だけでは一致しない。

## 2. 改善機構ごとの接続状況

| 改善・変更部分 | 既存SOURCE／pipelineと証拠 | 接続済み／未接続・組合せ条件 | 追加で必要なもの |
|---|---|---|---|
| 分割・r/K・誤差配分 | Original：df_partial_s2、rte、rpe_resource_accounting、rpe_short_round_optimization | short-round実装あり。D8の固定r32/K0と一般gridのtargetは別 | 外側LD/h変更では新PF/RTE/半径・単価・coverage。古いproviderの全round化は禁止 |
| O1-M1辞書 | A-O1 C0/r8・既存matched/selector系列 | 単価低下を長round attenuation/shotが相殺する負結果あり。固定LD0等の結果 | 新LD/lawで元H signal・N・費用を評価。価格差だけで採択しない |
| O1-M2 core/residual | Hermitian-B source68ef4fee、core-only c75d3cfd | C_H＋R_H恒等式のoperator-only PASS、物理model適格性UNASSESSED | target H規約の適格性、PF/RTE、補償range、元H N、full-wrapper価格。残差が小さいだけでは不足 |
| A-B2/P1 元H接続 | 元H DF12/sector36、HP結果3e316e07・e162ef70 | 18roundのshot/branch接続あり。旧強PF5 STOPは後続HPで条件付き再資格、全8 arm | 強PFを除外しない。D8やC-O3とstate/sector/閾値/compilerを照合、価格不足は別取得 |
| 研究C 表現・Z再配置 | 417fa459／3015d4fb／91417fb3のCDF/ENTRY/XDF、既存c1 accuracy | own-DF＋DF reserveの13round結果あり。表現・PF分割変更はλ/誤差/Nも変える | 元H groundへのoperator十分条件又は別の状態移送。D-B1のDF信号会計とWeyl項を二重加算しない |
| Return／Closed P3 | B/O2→C-O3、track_c_closed_p3、G7 proposal/weight | **C-O3へ接続済み**。πw、全path weight、finite-bit/root/native disk、rangeと十分Nを会計 | D8 K0へ同Nで移せない。canonical→Closedでも同じ平均だけでrange/価格/shot一致を仮定しない |
| P5・Parent・IS/CV/LCU | O2統合d28164b5、既存ReturnKernel／G9/G10／Parent | toyで局所利益、強対照・学習費込みで負結果/未確定もある。Closed P5をK2=P3と同じtargetにしない | mean・range/二次moment・online取得費・Born noiseを含む同task検証。安価oracleを仮定しない |
| D内部BCH/PF調整 | C-O3旧統合100377da→c5a3c102 | 元H groundの主要bias調整とRTE/shot/価格を接続済み | 一般作用素4次ではない。別state/H/LD/hで誤差証拠を再利用できるか判定 |
| joint basis／単一Pauli control／対角融合 | C-O3 2745d57b／94b6499e、625 catalogue等 | 強Fullにも同じ機会。Closed最終価格に既に含まれる | Dのrelative GaussianやC1節約を追加控除しない。Reference cutが共有機会を変える場合は実際の構成で比較 |
| h/resolution/α共同選択 | C-O3 c5a3c102 extended policy、track_c_bottleneck.allocate | **全18roundで実施済み**、trainingと独立holdoutを分離、有限menu。大域最適性なし | 同menu再探索を新機構としない。A3のH4 LD4–7等は別候補で未実行・別承認 |
| Full／Truncated PF | C-O3 da8b2c3f／92495a04／b3c81c26 | 同じ元H ground・18round、combinedと保守的個別誤差評価を区別 | dropped項誤差、相殺、state移送を保持。PRだけにQ/融合/α選択を与えない |
| D8 Reference/control | B6b 7ffe8b28→total8828b298、固定SOURCE b09b7e7a | 理想測定同値性→保存N共通→A/F/S 13round条件付き会計まで**接続済み** | actualのlaw/native/χ/leakage誤差、長q期待価格/不確かさは未資格。新条件では三接続を個別に再確認 |
| D8 cut／head-tail C1 | 保存q4 metadata16、C1実H41 path | 中央cut選択と追加融合の局所結果あり。A/F/SはC1 OFFの価格 | C1 ON節約を旧modelへ加算しない。別cut・全law・長qへの費用移送証拠が必要 |
| Native Streaming＋部分Bridge | c002153a、保存q4/q8各sample00、X/Y×2方式 | 価格検査器として保全。70人工回帰・保存54case・固定56資源一致は既存証拠 | 一般P0/全law/長qを認定しない。今回開発を止め、必要な有限検査だけ将来scopeで使う |
| 新A Reference-B1／Q readout | 08f9cbe7、pr_reference_feedback_q1/q3b/q3cd、V6-Q producer/consumer | canonicalのB′/Reference-B1で固定モデル保証・18round native費用へ**接続済み** | Closed、CDF8/P0、強Full/Truncatedへの移送は未資格。branch/radius/virtual-state、価格contextを再確認。物理shot consumerは巨大buildを呼ぶので今回実行しない |

## 3. 四段階pipelineの再利用区分

| 接続 | 区分 | 具体的な判断 |
|---|---|---|
| 固定D8 PR構成→保存operator誤差/ε/N | 既存コードで直接利用可能 | B1/B4の保存13行。own-DF信号を使わない。今回はschema/来歴を読むだけ |
| 固定D8→保存Strong/Reference単価→固定N総資源 | 既存コードで直接利用可能 | bc_total.price_records/model_value→b6b.aggregate_registered→c1.aggregate_totals。新再集計は未実施 |
| D8のαのみ変更→新N→凍結単価総費用 | 限定接続・拡張＋新計算が必要 | 保存εと既存Hoeffding算術を使えるが、元H adapterはuniform α、B4/total wrapperは固定Nを検査する。新allocation用namespace/検査が必要 |
| Original short-round provider→D8 whole-path価格 | 科学条件の照合が必要 | ordinary controlled semantics、q≤4、state/λ/scope/P0差があり、既存providerを全条件の同義語にしない |
| Closed→C-O3 PF/weighted N/価格/総計 | 既存接続済み、SOURCE別に再利用可能 | 同じC-O3 policyと誤差/price役割を守る。固定CDF8 A/F/Sを使わない |
| Q readout→canonical B′/Reference-B1の成功・価格 | 既存接続済み、固定条件限定 | 18round、m17 Q、native model内。Reference-B1と旧naive Referenceを混ぜない |
| Q又はReference→最適Closed／強Full／Truncated | 新しい条件照合・証明/計算が必要 | 同じfair readout機会、whole-vacuum/actual inverse、weighted range、prefix interval、compiler context。成功と費用を分離 |
| O1-M2／RC own-DF→共通元H最適化 | 現時点で証拠不足 | model適格性又はstate/誤差gapを閉じる前に数値順位やNを移送できない |

「直接利用可能」は既存SOURCEで処理済み又はAPIが適合するという意味であり、今回の再実行許可ではない。各STOP・one-shot・SOURCE lockを維持する。

## 4. 二重計上と非互換の点検項目

1. 同じscalar/identityとDF誤差は一回。raw signal補正、最終energy shift、native global phaseを混同しない。
2. 同一meanでもlaw/weight/rangeが違えばshot条件を再評価する。価格MCのcouplingとquantum trialのIIDは別。
3. D8のP0最適化済み全wrapperに、Cの共有・局所CX節約を足さない。Cのserial上界をDのDAG価格として置換しない。
4. QでX/YをQ一軸へ変えた場合、26/36軸の古いΣ式をそのまま適用しない。interrogation型とprefix/terminalの実shot表を使う。
5. 新しいαを両方式が別々に選ぶ場合、同じ法則でも**採択後のNは一致しないことがある**。同じαなら共通Nという同値性と、最適配分後のN同一性を区別する。
6. 全体covariance/CIや外挿biasを未知のまま0にしない。A/F/S感度は独立実験・確認的CIではない。
