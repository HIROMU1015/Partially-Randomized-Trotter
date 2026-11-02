# Track B W2c-B3 初回実科学correctness：結果と引き継ぎ

固定SOURCE `f8aad18d385c9dd787c5be8c794cf3c339335919` から一回実行し、S256件・P1256件、計512件を完了した。分類は `LIVE_TECHNICAL_COMPLETE_AWAITING_REVIEW`。全32 condition/cellでaccepted>=1、INCONCLUSIVE_COVERAGEなし。失敗・UNKNOWN・retryは0件、resumeなし。実行状態COMPLETED、claimはSTOPPED/spent=1、mandatory STOPを維持する。

実科学Evidenceの保存値検査はPASS。ただし最終reportの継承フィールド `artificial_only=True` は実行内容と食い違う。全512件の各trialでは `artificial_only=False`、runtime_origin=PINNED_G7_F3、実Core6 moduleのloadが記録されている。SimpleStore snapshotの `real_authority=False` も無条件の継承定数であり、実承認経路の判定には使用できない。実承認はapproved manifest/activationとdual SPENT intentsで照合する。原本・SOURCEを修正せず、[metadata残件](observed_metadata_issues.json)として保全した。観測された互換性と、未確立の一般保証を区別する。

| 質問 | 実証・回答 |
|---|---|
| Q1 正式採択・固定 | 完了。[採択](adoption.md)、[承認原文](user_approval.json)、[approved manifest](approved_manifest.json)、[preflight](preflight.json)。承認原文SHA256 `9a9ba20e8f5fd41a0e3d8456e8bb43f786f528351540e6e4ec996f831869d3a0`、manifest digest `80570f8d89d23793bf35652ac08fb492b7c76910defb84a65f00599ec4b25c08`、contract digest `f47dd643c4ea3f8d7bb5355824be39825e8e2d2ffa366e1889e4d3bff4e0a098`。内部activation content IDは外部署名ではない。 |
| Q2 実Core/API | 親はCore未loadのまま、spawn worker16個で6 Coreすべてをload。resolverの実callable/location/hash/signature照合を通過し、Canonical構成・snapshot・Full F3 prepare/session・Original Evidenceの接続を完了。各trialのsource/approval/runtime bindingも保存照合した。 |
| Q3 実call | S256/P1256 completed。Primary16、G7 make14、F3 prepare2（内包Full constructorは16の内数）、AuditParity8、snapshot6。B sampling audit256、B angle/requery各250、Canonical angle192、Full angle58。Full sample packet124、Full angle packet58。driver correctness event_witness500、nested B witness250、helper総750。F3 session.sample64はFull sample128の内包であり追加samplingではない。 |
| Q4 科学的invariant | 全512 full Evidenceの原文digest・Exact Codec・Result projection・乱数消費位置、accepted500件のweight/proposal/coefficient・phase・word/reduction・circuit order等の採択invariantを照合。A/B・S/P1対応prefixも一致。独立したlaw/atanの数学的証明ではない。 |
| Q5 各cell/coverage | [32セル表](cell_counts.csv)、[JSON](result_summary.json)。下表のとおり全cell accepted>=13。ZERO_R4件、ZERO_C/coin reject8件、accepted500件。追加samplingなし。 |
| Q6 接続・保存 | F3 angle/requery58件、Canonical angle192件、Exact Codec、Global Bundle、Storage V2、receipt→delivery→caller、状態/trace SHAの保存値照合PASS。実F01失敗は発生していないため実科学失敗時のF01動作を認定しない。原経路保持と今回の人工失敗2ケースを証拠とする。メタデータ残件は冒頭参照。 |
| Q7 費用 | Parent CPU6.495904145秒、reaped child hierarchy CPU77.030382秒（workerとGit/helperを含む）、Total83.526286145秒、global wall52.418725915秒。粗・細partitionは別々に整合し、加算しない。[費用原記録](resource_summary.json)、[細stage抜粋](selected_stage_costs.json)。 |
| Q8 状態 | COMPLETED、claim STOPPED/spent=1、receipt link保持、retry0/resume false。未知trial0。二重実行・追加seed・追加samplingを行わずSTOP。 |
| Q9 保全・公開 | [SOURCE inventory](source_inventory.json)、[raw inventory](raw_inventory.json)、[旧成果保全](preservation_after.json)。公開SOURCE snapshotは `e150965d486050579956058d9e833377bcc4da61`（測定SOURCEとGit tree完全一致）。元commitは[Git bundle](measured_SOURCE.bundle)にも保存。公開最終HEADと全対象bytes検査はpublication receiptと最終チャットに実値を記録する。 |
| Q10 限界・次判断 | 固定入力/精度/seedの512件について初回実API・保存・採択invariantが通ったという技術的結果。一般的正しさ、IID、atan theorem、物理RZ合成、性能優位性、科学的GO/NO-GOは認定しない。metadata矛盾の将来修正・追加検証の要否と範囲はユーザー・GPTが判断する。新レビューは開始していない。 |

科学条件はP5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma+1、root256/H160、eta/rho1e-12、epsilon_axis1/200、primitive1e-6、alpha_axis1/320、familywise1/20、A6 tau1e-6/Kphi32/J32の固定契約。研究用の局所生成・数学的enclosureであり、新Hamiltonian・分子/DFへの一般化は行わない。

各行はS/P1およびA/Bで共通の**各16 trials当たり**。32cellの完全表はCSVを参照。

| Arm | Axis | accepted | ZERO_R (RAW_NONREDUCED) | ZERO_C (COIN_REJECT) |
|---|---|---:|---:|---:|
| ordinary | X | 16 | 0 | 0 |
| ordinary | Y | 16 | 0 | 0 |
| partial_return_tail | X | 16 | 0 | 0 |
| partial_return_tail | Y | 16 | 0 | 0 |
| closed_P3_tail | X | 16 | 0 | 0 |
| closed_P3_tail | Y | 16 | 0 | 0 |
| full_return | X | 16 | 0 | 0 |
| full_return | Y | 13 | 1 | 2 |

科学samplingはCanonical.sample384＋FullReturnGenerator.sample128＝512。後者の64件がF3 session.sampleから呼ばれる。Canonical実constructor12＋Full実constructor4＝16。B AuditParity.invoke448＝sampling audit256＋Canonical angle192。Full angleは元F3 sessionの経路を保持する。

乱数は1,288 draws、delivered206,080 bits、discard123,648 bits。2/3/4/5 drawsのtrial数は328/124/40/20。対応A/B・S/P1で記録済みdraw値と位置が一致し、audit/angle/requeryによる追加drawはない。新しいbitstreamは検査時に生成していない。元文書の `real_variable_consumption=NOT_ESTABLISHED` という準備ラベルは保持し、今回の有限観測結果を別記した。

| 粗stage（worker合計・exclusive） | CPU秒 |
|---|---:|
| Source check | 1.937817147 |
| Constructor + preflight | 11.556525322 |
| Sampling + audit | 19.575784687 |
| Angle + requery | 15.594817019 |
| Correctness event invariants | 0.158533196 |
| Serialize | 0.783408050 |
| Write | 0.151352003 |

Fineの選択frameだけでは、科学bit provider0.010450秒、law/root/coefficient共有frame0.894891秒、angle enclosure core0.846274秒、Exact Codec/adapter11.694191秒、IPC client project frame0.156127秒。これらを上の粗stageやTotalへ再加算しない。Observer固有費用、IPCの全stdlib費用、STOP/close固有費用は単独には分離できず、実測Totalとresidualに含まれる。通常完了のためstop-to-join値はnullであり、0秒とみなさない。計測は同じ固定correctness pass内で行い、追加instrumented passは0。

最大worker VmHWM40,124,416 bytes。Coordinatorのsampled RSS合計ピーク356,012,032 bytesは共有pageの重複を含み、unique memory上界ではない。親の終端VmHWMは40,304KiB。worker I/O合計read_bytes0（page cacheへの論理readは4,833,637,933 bytes）、write_bytes59,969,536、wchar84,737,586。I/O、粗/細stage、親CPUのwindow差と未分離observer費用をresource summaryに明記した。CPU/RAM/wallの人工上限は追加せず、採択S直列/P1並列8のscheduleを維持した。

最終物理出力はResult38,884,103 bytes、Log1,731,946 bytesで共通cap内。全512 full Evidenceを保持し、間引き0。原 `final.output` は終端slot export前のcheckpointなので未使用creditが残る表示だが、最終物理file inventory、host_outer直前ledger＋後続12,288 bytes、COMPLETED状態とjoinして照合した。これはUNKNOWN科学trialを表さない。SimpleStore DBとdual intentsは制御metadataとして別に保全した。

原本は `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/g6-w2c-b3-correctness-512-authorized-run-20261020/artifacts/track_b_g6_w2c_b3_correctness_512/2026-10-20/b103af049c7db5d9` に保全。公開用完全raw archiveは [raw_scientific.tar.gz](raw_scientific.tar.gz)（4,621,543 bytes、SHA256 `608fc40b10e47d9439b95329fe9c24b84c879be42bf486a14c2fdfeda80b7a94`）。全fileのarchive往復SHA256と原本再照合を実施した。取得に追加権限や秘密鍵は不要。測定SOURCEのGit bundleは公開基点bf86をprerequisiteとする標準形式で、bundle verify済み。

今回の検査は同じprimary Codexによる保存値検査であり、別reviewerの独立再現ではない。六つの科学Core、既存adapter/driver/Global Bundle/F01/Storage V2、旧raw・grant・claim・STOP、他worktreeは書き換えていない。実行開始後のsource修正、再実行、pilot、Full-N、合成、再ランキング、新科学レビューはいずれも0。

次の担当：**ユーザー・GPT**。今回の有限correctness結果とmetadata残件をもとに、将来の技術修正・研究判断・追加検証の必要性と範囲を決める。Codexはこのscopeでmandatory STOPを維持し、自動で次の計算へ進まない。

CPUの厳密な集計窓：原 `all_worker_cpu_ns` はRUSAGE_CHILDREN差分であり、joined workerおよびwait済み子孫Git/helperと親の直接helperを含む。worker interpreterだけの完全な終端CPUとは分離できない。16 workerのEND直前process_time_ns合計は54.359214673秒、worker内計測窓の合計は51.386673494秒。これらをchild hierarchy値へ加算しない。Total83.526286145秒は_run内窓の実測であり、その前のrunner activation/preflight/store生成と末尾の観測export・mark_completed・summary書込み、および窓内で未waitのresource tracker固有CPUは範囲外。全作業のend-to-end CPUと同一視しない。
