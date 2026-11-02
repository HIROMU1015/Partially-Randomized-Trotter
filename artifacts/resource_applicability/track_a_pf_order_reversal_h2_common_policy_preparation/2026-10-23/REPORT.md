# PF次数逆転C：H2共通条件検証の準備報告

2026-10-23 JST、Codex一次担当のみ。**技術調査・政策3案・最小実行前計画を作成した。共通政策は未採択、比較Cは未認定。mandatory STOP。**

推奨は、reference/path/closure/roundoff/normalizationの意味を揃える**共通の適応的allowance生成規則（案1）**である。新PRの入力不足は残るため、現在これを適用できたとはしていない。ユーザーの「一旦なしで進めて後で渡す」に従い、private資料は未取得として完了し、待機中の科学実行や代替計算は行わない。

## 1. 対象と変更しなかった範囲

前段固定監査 `366b2fd24f7e86aa82b5b518b6e3774520f65596` から独立したdetached worktreeで作業した。[9必須資料の継承hash](inherited_audit_files.json)を保存し、75行監査・shot関数のAST同値・元ランキングは再実行せず再利用した。SOURCEの不足分だけ固定Git blobから取得し、実取得file/hash/locatorを[索引](SOURCE_INDEX.md)に残した。

対象はLinear H2、隣接1.00 Å、STO-3G、tol-only1e-8、保存DF rank3、T=.8、ε_sig=.0005、P=0。Full PFは全DF因子prefix3、PRはprefix1/K4/ABS_TIME_PER_OUTER32。symmetric_directional control、cos/sinのfull controlled Hadamard wrapperのcompiled RZ、軸精度ε/√2、α_axis=.025。δ=T/qはS2で1/5、Y4で2/5であり、別のδ窓や次数係数fitは扱っていない。

以下は元保存行の転記である。[4行CSV](candidate_records_4.csv)に元locatorを保持し、新しい大小判定・会計・再選択はしていない。

| 比較 | 保存候補ID | q | 保存C_cos=C_sin | 保存N_cos / N_sin | 元保存G |
|---|---|---:|---:|---|---|
| A Full S2 | B1_p3_q4_2nd | 4 | 1367 | 117378315 / 106246467 | 305695076994 |
| A Full Y4 | B1_p3_q2_4th | 2 | 2033 | 76747409 / 75520789 | 309561246534 |
| B PR S2 | H2O32_c7ec590a32a859bc | 4 | 3914.5151515151515 | 117441887 / 106313566 | 9634835221029/11 |
| B PR Y4 | H2O32_8a2ecdd0294ffb74 | 2 | 2931.029411764706 | 79644174 / 78371520 | 7873526992785/17 |

BのC/Gは探索n33/n34の標本平均・そのshot重み付き値であり、母平均保証ではない。元監査での「AはS2低費用、BはY4低費用」という保存方向を引用するだけで、共通政策適用後の順位やCは未評価。全探索bestや全体系の最終費用評価を認定しない。

今回の科学call、保存数値の新科学解析、u評価、shot再計算、費用再会計、sampling、signal、simulation、build/compile、新ranking、PF最適化、非自明証明、科学GO、政策採択、詳細科学レビュー、commit/pushはいずれも0。

## 2. Task A：native/runtime証拠の追加照合

| 項目 | 今回確認した内容 | 状態・残存不足 |
|---|---|---|
| 入力・shot目的 | H2のHamiltonian/state/snapshot/compiler設定hashとshot3関数同値は固定監査を継承 | `INHERITED_METADATA_MATCH`。再監査しない |
| 旧Full PF preparation | D公開resultは両候補のpreparation hash `8e32e931047087a4a9c1f613d406b6fe353c0aa4601b7e5328750ab6ab2c9a42` と各軸fingerprintを保存。旧recoveryは保存preparationと再構築hash一致を要求 | `OLD_ACCEPTED_METADATA_AVAILABLE`。private原本3既知locatorは本環境に存在しない。前監査の実読取結果を歴史的証拠として保持 |
| 旧native構成 | 実provider217c971dのbuild_deterministic_native→deterministic_control_plan→三段S2。全blocksに対称方向制御、最終scalar一回 | `STATIC_SOURCE_MAPPED`。新PRとの全word一致は要求せず、実ロードbyteの対応と方式差を追う |
| 新PR preparation/native | H2 contextは保存npz/receiptとH/stateを照合しfullprep3/cut1、保持fragment index(0,)、tail scalar/lambdaを要求。composeは各signed pieceでprefix S2＋RTEを合成し最後scalar一回 | `STATIC_SOURCE_MAPPED`。full/cut preparation hash、block列、実provider/import closureの原receiptは未取得。prefix3とprefix1のhashを同一にする要求ではない |
| providerのGit照合 | 前監査の旧sealed12 fileから、新取得SOURCE4ec26921の同じpathだけ確認。4 library blobは旧sealと一致、8 native/provider pathは新取得Git treeに存在しない | [byte照合](provider_git_comparison.json)。外部providerを使う設計と整合するが、**新runで読んだbyteの証拠ではない**。実providerをSOURCE4ecのGit treeから推定しない |
| 新runtime | installはnumpy1.26.4/Qiskit1.3.0、ロードpathのseal掲載、trotterlib/trottertracksのprovider_root配下を要求。prepareはtemplate runtime/package/distribution sealを継承 | `EXPECTED_CONTRACT_ONLY`。実config/sealed inventory/実行binary/import manifest未取得。version/設定hash一致だけでclosure完了にしない |
| compiler guardの範囲 | H2専用process-local BasisTranslator wrapper。num_processes1・main thread、live zero symbolic scalar、translator各passでgate/angle/wire stream同一を要求。scalar global_phase帳簿を保持し、state fit・postcompile phase補正なし | SOURCE上の作用は特定。全runtimeの同一性・全compiler後段の不変性・旧回路との等価性をこのdigestだけで保証しない |
| guardの実受入 | 公開a05f4ed1文書はpilot792軸、exploration1050軸のPASSを記載。runtime.workerは各task2receiptを要求しextra_auditで集約 | `PUBLIC_SUMMARY_ONLY`。`compiler_phase_guard_0/1.json` と `compiler_policy_end_audit.json` の実byte/hash・対象67taskへの結合は未取得。SIGNALや旧Full PFの全軸へ適用済みと拡張しない |
| compiled correctnessの証拠水準 | 新PR workerは各taskのnative/occupation/Hadamard `<1e-9`、compiled vector `<1e-7`・phase alignmentなしと保存SHAを要求。旧compile_oneはnative controls、compile fingerprint/settings、整数metricsを保存 | 取得した旧経路に同等のcompiled-vector gateは見当たらない。旧原receiptの有無を先に調べ、新compileを直ちに要求しない |
| Cへの影響範囲 | preparation/order、signed binary64 time、lowerer、basis/global scalar、compiler実装/pass、数値最適化がRZへ影響し得る。guardは当該translatorのstream不変を要求する | runtime差が実際のRZ差の原因だとは未認定。通常のphase帳簿だけの差ならRZ不変が期待されても、実receiptなしで全差を無影響としない |

旧実行bindingのruntime manifest SHAは `432e5f360030f30571e8ee8758a25f7972a25f964787cb82b124a805e0acb258`、provider SOURCEは217c971d、runtime12202file/provider262fileという保存metadataである。新Eのsealed4187fileは**会計plan全体の件数**で、新取得runのruntime全件数とは読まない。旧manifest実体を今回再取得したとはしていない。[既知locator確認](exact_locator_checks.json)。

新公開EXPLORATIONのplan/end_audit hashを既存E provenanceと結べることは確認した。実auditの内容やguard receiptまで取得したことにはならない。科学SOURCE4ec26921、保存会計SOURCE9a77d2b0、結果commit a05f4ed1/8613b3ecは役割が異なる。

必要資料は [missing_evidence.json](missing_evidence.json) に、原SOURCEから導けるrelative locatorと既知SHAを記載した。新root/hostは不明で、ユーザーが後日提供する。ここでは全projects/tmp探索や別サーバーへの推測接続をしていない。

## 3. Task B：政策差と推奨

詳細は [ALLOWANCE_OPTIONS.md](ALLOWANCE_OPTIONS.md)。重要な追加知見は次の3点である。

1. Aはreference誤差＋path signal/state差＋補正済signal closureとwork/norm依存roundoff proxyから経験的uを作り、独立logB差をmarginへ入れる。H2 referenceの生成は汎用H6 setupをoverrideしている。
2. Bの保存 `u=1e-9, margin=1e-12` は下限付きclosure/roundoff規則の結果。H2独立forward signal差は別gateで測るが、u生成には入らない。reference絶対誤差もnullで条件付きである。両政策はfloorの数値だけの違いではない。
3. 両者とも経験的で、厳密bias certificateはない。native/compileの許容差をshot allowanceと同一視しない。PF/finite-RTE bias、数値u、normalization margin、量子shot誤差、MC単価の不確実性を別会計にする。

案1＝共通適応的生成規則を推奨する。Aの既存式を出発点として各誤差成分の意味を共通化し、方式別に必要なuの値は認める。新PR側で不足するreference/path vector/work/full-stage norm等が保存資料で充足するかは未確認。足りなければ入力schemaを曖昧にして採択・実行しない。

案2＝共通固定u/marginは、全4候補の同じ誤差範囲を覆う根拠、又は条件付き仮定をユーザーが採用する場合の選択肢。現保存の1e-9/1e-12を無条件にAへ移す理由は未取得。案3＝方式固有allowanceを共通の被覆原則・校正/欠測基準に対応付ける代替案。元政策をそのまま併用して公正とするだけでは不十分である。

順位を見て政策を選ばず、ユーザーが評価前に採択する。今回、どの案で逆転が残るか・政策差が順位を何%変えるかは一切評価していない。

## 4. Task C：再利用と最小の将来処理

[実行計画草案](EXECUTION_PLAN_DRAFT.md)にgate・callの種類・入力・停止条件を整理した。

| 再利用候補 | 今回の取得状態 | 次の最小処理 |
|---|---|---|
| Aの保存reference/signals/診断/2軸C/fingerprint | 公開resultから4点に関係する部分を転記済み | 原受入metadataへ結合後、共通schemaと照合。Aの不要な再signalを避ける |
| Bの保存bias/logB/u/N/G・paired単価統計 | Eから2候補の対象行・統計を転記済み | 新政策のNは使い回さず、C/cohortは証拠結合後に維持 |
| B原signals/forward診断/reference/preparation | private未取得 | まず既存ファイルを取得。足りない数値成分だけ将来診断を設計 |
| B実task/guard/runtime、A compiled受入 | 原本の結合未完 | receipt照合を先行。不足を自動的に新compileで埋めない |

入力が揃えば、将来必要なのは4候補の共通u/margin生成、paired shot評価4件（8軸）、保存CでのG4件、A/B各1比較と方向関係の判定である。これは数値再解析なので**別の明示承認が必要**。既存回路/law/compiler条件を維持するなら、Nの変更だけで再compileする必要はない。

保存入力不足なら、B2候補だけの不足した独立経路/closure/norm/work診断、必要性が確認された場合だけH2 reference診断等を候補とする。native/compiled証拠はまず保存receipt/vector/回路から埋める。どうしても要求基準を満たさず新build/compileが必要と判明した場合だけ旧PF2候補×2軸を具体化する。現時点では診断・compileの必要性/件数を断定せず、枠も予約していない。

共通政策下の固定4点の比較は、各方式の全探索best、PR母平均優位、因果機構、他系/精度への一般化を意味しない。以前のH2独立確認cohortもB内の別証拠であり、この固定4点の探索標本へpoolしない。

## 5. Task D：原因候補の整理

| 要因 | 既存SOURCE・記録から切り分けられる範囲 | 追加しない結論・将来の最小証拠 |
|---|---|---|
| q4→q2 | A/Bとも保存点は同じ低下 | q低下だけを逆転原因としない。固定qの反事実実験は未承認 |
| Y4三段化 | Full PFは全DF blocks、PRはprefixに作用する構造を静的に特定 | 三段化単独のRZ寄与は未分離。既存gate-stream/segment count metadataがあれば後日照合 |
| outer32のtail | 1 outer全pieceで32 short stepsを配分、qとsigned piece別時間を保持 | event分布・RTE order・負時間・tail lowererとcompilerの寄与は混在。step数だけでRZを決めない |
| biasとnormalization | PFとfinite-RTEのsigned構成、各piece logB累積のSOURCEを確認 | 物理biasとuの差を混同しない。負時間でlogBが相殺されるとしない |
| CとN | 元保存Cの変化方向と元N/Gは前監査の引用として保持 | 新しい政策別分解・因果効果は未評価。まず共通政策下の固定点比較が必要 |
| compiler/runtime | 設定一致、guardの目的と静的範囲、実closureの不足を特定 | runtime差の費用寄与は未識別。原pass/stream/fingerprint receiptが最小の追加証拠 |

回路単価とshot数、piece時間、normalization、event/cohortを独立に操作していないため、原因の認定はしない。SOURCE上の構造仮説として保持し、非自明な新証明・最適化・因果実験・詳細科学レビューは始めない。

## 6. 技術検証・保全・次の担当

今回の検証はJSON/CSVの構造、指定4行の文字列一致、固定Git blob/SHA、AST parseと定義・呼出しの静的確認、no-grant/STOP境界、tracked原本不変の確認である。科学関数をimport/実行せず、pytest等で科学計算を誘発していない。一次担当の自己検証であり独立科学レビューではない。[技術検証](technical_verification.json)と[file inventory](file_inventory.json)に結果を保存した。

変更はこの新規準備ディレクトリ内だけ。科学SOURCE、旧結果/raw/grant/ledger/marker/STOP、既存研究文書・validation manifestは変更していない。他worktreeのdirty状態を取り込まず、他jobや共有環境に変更していない。全private原本や他人のworktreeを横断検証したとは主張しない。

次の担当はユーザー：必要private資料を後日提示し、共通政策と固定4点の数値評価範囲を採択・承認する。Codexが資料を照合した後、不足する科学callがあれば具体的scope/SOURCE/input/数値許容差を示して別承認へ戻す。locatorの提供だけで計算承認とみなさない。新SOURCEを固定する前にgrantを有効化せず、既存grant/失敗run/STOPを再使用しない。

今回の成果物は未commit・未公開の準備資料。**mandatory STOP。科学的GO・正式政策採択・数値再解析・commit/push・詳細科学レビュー・次campaignへ進まない。**
