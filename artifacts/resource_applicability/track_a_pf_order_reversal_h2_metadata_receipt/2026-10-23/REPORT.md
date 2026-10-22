# H2共通条件検証：共有metadata受領後の静的照合

2026-10-23 JST、Codex一次担当のみ。**受領・静的照合を完了した。新PRのtask/guard/signalとprovider出典の不足は大きく縮小したが、共通適応的allowanceを直ちに適用できる状態ではない。科学的比較Cは未認定、mandatory STOP。**

## 1. 共有と受領の確認

GitHub APIから、branch `codex/h2-outer32-metadata-export-20261023` のrefが `e3531956af6a6d359e2f1fb320e9a5179f41eb98` を指すことと、同commitの取得可能性を確認した。ユーザー提示の「push失敗」は送付側のその時点の報告であり、受領時点では共有済みだった。受領側からpushやremote変更はしていない。

[固定公開README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/resource_applicability/track_a_h2_outer32_metadata_export/2026-10-23/README.md)を入口に、16個の本文/集約metadataと134 guard、計150ファイルを取得した。大きいruntime/provider・origin manifestはGitHubの固定blob APIで取得し、他の本文は固定commitのraw、guardは同commitの完全な追加file patchからbyteを復元した。**全150件のGit blob SHAが一致**し、送付側file inventoryの掲載対象ともSHA256/bytesが一致した。[受領manifest](acquisition_manifest.json)。送付bundleの全ファイルを取得したとはしていない。

原本private資料は元環境に置かれている。今回読んだのは公開projectionであり、原本byteそのものではない。origin ID/JSON pointer/原本SHAと公開projection SHAを分離して追跡した。送付側の「人工8件・静的11,494件・既知原本6,000件不変」は送付資料の記録として保持し、受領側で再実行・独立再現したとはしていない。

受領側作業は、既存の独立detached worktree、HEAD `366b2fd24f7e86aa82b5b518b6e3774520f65596` で行った。前の共通政策準備資料を上書きせず、新規namespaceに追記した。前段75行科学監査も再実行していない。

## 2. 対象と照合できた証拠

対象はlinear H2、隣接1.00 Å、STO-3G、保存tol-only1e-8・DF3、T=.8、ε_sig=.0005、P=0。Full PFは全DF因子prefix3のS2 q4/Yoshida4 q2、PRはprefix1/K4/ABS_TIME_PER_OUTER32の同じformula/q。δ=T/qは1/5と2/5。symmetric_directional、cos/sin full Hadamard wrapperのcompiled RZ。追加δ窓、q探索、係数fitは扱わない。

| 証拠 | 更新した状態 | 判定の限界 |
|---|---|---|
| 新PR task受入 | S2 `H2O32_c7ec590a32a859bc` n33、Y4 `H2O32_8a2ecdd0294ffb74` n34、計67 task/134軸を元end auditのaccepted result SHAへ結合 | 保存metadataの受入記録。native/compiled計算そのものの再実行ではない |
| plan/end audit | plan `4dda5e68815f8359b3890dc7a9e301956a631d6cea500e8a90703224db719373`、audit `38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723` と元provenanceが対応 | 元stage全525 taskのauditと、今回取得した対象67 taskの範囲を区別 |
| guard実receipt | 134件をtask/軸locatorとstage guard auditへ結合。policy・保存stream不変・state fitなし・postcompile補正なしを照合 | 元accepted resultはguard file自体のSHAを含まない。今回の原本SHA採録を、当時のbyte封印へ格上げしない |
| native/compiled gate | 各taskの保存8 gateのPASS、phase alignmentなし、cost/event/preparation/compiler/fingerprintを結合 | saved-input-stateの検証。全入力に対するoperator保証ではない。誤差値の再評価なし |
| signed piece/RTE記録 | 対象67 taskの336片を取得し、candidate/task/cohortを照合。pilot・旧PR・独立確認cohortは混入なし | signed timeや有限lawの数値を再計算していない |
| 元signalとE会計 | corrected/raw/exact-tail、signed bias成分、logB、u/margin、forward signal差、norm/closureを取得。既存Eの対象2候補のbias/logB/u/marginと保存値が一致 | common policyへの再評価ではない。保存時点の「政策未採択」headerと後続Eのseries限定採択を混同しない |
| preparation | 新fullprep3 `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9`、cut1 `85b8243d752d3390e214b429a80e3899f0cc8b0166e0958fc1d73a930a77597c` を取得。taskはcut1へ結合 | 旧Full PFのfullprep3 `8e32e931047087a4a9c1f613d406b6fe353c0aa4601b7e5328750ab6ab2c9a42` とも異なる。full/cut serialized block/basis/partition全体の照合は未完了 |
| 旧PF2候補 | `B1_p3_q4_2nd` と `B1_p3_q2_4th` のcontrol/4軸measurement/fingerprint/compilerを旧公開resultへ結合 | 正確なworker出力でcompiled-vector/phase receiptや保存QPY/vectorは見つからないという送付記録を維持。新PRのPASSを移送しない |

受領した主要資料は [task_acceptance_67.json](received/task_acceptance_67.json)、[stage_end_metadata.json](received/stage_end_metadata.json)、[signal_records_2.json](received/signal_records_2.json)、[old_pf_receipts.json](received/old_pf_receipts.json)。原科学結果のN/C/Gと順位を新しく評価していない。

## 3. runtime/provider：解消した出典と残る差

新実行configが示すproviderは **`c31ef424175c02379eafebe8fc6c7548c1d9aabe`**。科学取得SOURCE `4ec26921c6872ed6e8667f167074e1e405cf1a3f` と別である。封印278 provider fileのSHAが同provider固定Gitと一致することを、受領側でもGit blobから照合した。科学SOURCE側の封印70 fileも固定Gitへ照合した。実importの完全記録を得たという意味ではない。

新providerの同278 pathを旧provider `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` と比べると、260一致、2差異、16 path未存在という送付記録と固定Gitが対応した。差異のある既存2 fileは `src/trotterlib/README.md` と `h48_preflight_contract_v1.py`。後者はH8_STATE等の追加があり、H2旧経路が継承する `h4_plan` 本体のASTは一致した。これをもって全runtimeや全科学経路が同一とはしない。[provider差一覧](provider_scope_update.json)。

runtime封印2,012件のうち送付時点の現在byteは2,011一致、liblzma1件不一致と記録されている。この「現在」は送付元環境の観測であり、受領側環境のruntimeを再検査した値ではない。保存SHA `696e868dd0700a19a6d65fc01608ec2d70d3cb91f65710e89180cd2e688f30cb`、送付時点現在SHA `26b37d25f7ca2a2d348de8ecabacda12e921cd933b21fa069a179760db77219d` を区別する。原因・時期・過去科学結果への影響は未確認。不一致だけで過去結果を無効又は無影響と断定しない。

新Python保存SHA `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` と旧 `ebc2fae9fa0845b8c1fc09ed10196c0e93c26cb8fbd2a84d64548326ef93209b` は異なる。設定hashやQiskit1.3.0等のversionが同じでも同一runtimeではない。runtimeが異なるだけであらゆる比較を禁止するという基準も採択していない。比較する実装と証拠の強さを明記する必要がある。[runtime/provider manifest](received/runtime_provider_manifest.json)。

### 基底変換に関する具体的なSOURCE差

共有preparation hashを起点にnative準備の呼出しを追加で追った。新H2 contextは取得SOURCE4ec26921の `h46_new2_native.prepare(ham,prefix)` を呼ぶ。同関数59–68行は、process内で `ops.U_to_qiskit_ops_jw` を明示的 `gaussian_operations` に置き換えて `_prepare` を呼び、finallyで戻す。同adapterは複素QRによる隣接二modeのPhase/XXPlusYY操作列を生成する。

旧Full PFの取得経路はproviderの通常 `_prepare` を使う。通常 `U_to_qiskit_ops_jw` はBogoliubovTransform経路、import不可時にはdense Gaussian経路を選ぶSOURCEであり、今回得たmetadataから旧実行の全blockの実操作列を復元できたわけではない。旧を一律にdense経路だったと断定しない。

`df_partial_s2._block_hash_payload` はbasis hashとbasis_change_operationsをpreparation hashへ含めるため、同じfull Hamiltonian/prefix3でも、基底変換の実装が変わればpreparation hashが異なり得る。今回のfullprep hash差を説明し得る具体的経路を特定したが、差の全原因を実objectで実証したとはしない。関数名にnew2を含むadapterを使用していても、今回のPF公式は保存S2/Yoshida4のままであり、new_2公式へ読み替えない。

主要native lowererのprovider byteが一致していても、渡す基底操作列・準備adapterが同じとは限らない。この差はcompiled RZに影響し得る。**共通allowanceだけを揃えた固定点比較から、PR化だけが逆転を生んだ因果効果を認定できない。** 新しい共通native実験を今回実行・採択することもしない。

追加読取SOURCEは [supplemental_source_manifest.json](supplemental_source_manifest.json) に固定commit/path/SHAを保存した。

## 4. 共通政策の入力充足と次の最小処理

前の[政策3案](../../track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/ALLOWANCE_OPTIONS.md)と[実行前計画](../../track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/EXECUTION_PLAN_DRAFT.md)は未採択のまま。案1（共通適応的生成規則）の推奨も維持するが、次のように入力状況を更新する。

| 成分 | 共有で取得できたもの | なお不足するもの |
|---|---|---|
| reference | 元npz保存SHA/所在、各signalのreference scalar、旧FP reference診断 | 新旧referenceの保存scalarは文字列/値が完全一致ではない。同一基準への対応と数値経路の受入方針。新referenceの厳密誤差上界はない |
| candidate独立経路 | 対象2候補の独立forward signal差、normalization_error | A規則に含む独立path-vector差の名前付き保存値 |
| closure/norm | raw-corrected vector closureとmaximum_intermediate_norm | Aの補正済signal単位closureとの対応、同じ観測範囲の全stage norm。保存scalarから未記録vectorを再構成しない |
| roundoff proxy | B規則のroundoff_observation、policyとfloor/係数 | A規則の両経路work/action count。Bのstage proxyを同じ量として代入しない |
| C | Aの4軸実測receipt、B67 paired taskのcostと保存E単価統計 | 旧PF compiled correctnessの証拠水準。共通nativeを要求するかという比較範囲の判断 |

案1の必要入力を欠測0として処理しない。受領した小さいforward差からreference絶対誤差上界やcombined uncertaintyを新たに推定しない。保存政策下のN/Gは共通政策下の結果として再使用しない。

今後の最小処理は、ユーザーが求める比較の強さで分かれる。

1. **保存実装同士の、宣言した政策下での固定4点比較**を目指す場合：基底変換/runtime差と旧compiled証拠の限界を明記して受け入れるか、ユーザーが判断する。必要な共通政策を採択した後、固定4点のu/margin・N/G・方向判定だけを別承認で行う。回路/lawを維持しC受入を認めるなら、N変更だけを理由に再compileしない。
2. **推奨の共通適応的規則**を使い、原保存資料にも不足成分がない場合：PRのS2 q4/Y4 q2だけを起点に、必要な独立path-vector/work/full-stage norm等の限定診断を具体化する。reference基準の補足が必要ならその範囲も示す。SOURCE/input/数値許容差/call内訳/one-shotを固定して追加承認を得る。現在の科学call予約・消費は0。
3. **同じnative基底変換・同じcompiled受入基準で次数差を比べたい場合**：これは保存metadataを埋めるだけの作業を超える。旧PF2候補×2軸の新しい共通native対照等が本当に必要かを先に判断し、実験変更として別設計・別承認にする。新Cを旧Cに黙って置換しない。現在この実験を開始・採択していない。

案2の共通固定u/marginを選ぶ場合にも、同じ数値を置いた条件付き比較であることと誤差範囲の根拠・仮定をユーザーが採択する必要がある。新政策の再会計は別承認。案3の方式固有allowanceを使うなら被覆対象・reference/欠測基準を対応付ける。いずれの案も逆転を維持するために選ばず、数値順位を見る前に固定する。

固定4点の結果から、全候補最適・PR母平均優位・逆転の因果機構・他系/精度への一般化を認定しない。旧STOP・grantを保持し、詳細科学レビューや次campaignへ進まない。

## 5. 検証・保全・終了

[受領側検証](technical_verification.json)は固定blob/projection hash、origin参照、67 task/134軸の受入とcost/guardの対応、336片のcohort、既存E保存入力との同値、provider/source Git hashとAST、元tracked文書・前準備資料不変を検査した。3,692項目PASS、科学関数のimport/実行0、shot計算0、費用評価0。これは一次担当による静的自己確認であり独立科学レビューではない。元環境の原private6,000件を受領側で再hashしたとはしない。

変更は新しい受領資料ディレクトリ内に限る。旧科学SOURCE・結果・raw・grant・ledger・marker・STOP、前の準備資料、既存tracked文書・index/validation manifestは不変。科学的結果/採択や新validation pathの更新はしていない。commit/push0、新規grant・call予約0、科学計算・数値再解析・再会計・再ランキング・compile0。

次の担当はユーザー：比較を保存実装間の条件付き固定点比較とするか、共通nativeまで要求するか、共通allowance案と不足診断の範囲を決める。Codexは未承認計算を起動せず、具体的な実行SOURCE/call/許容差を固定する段階へは別指示で進む。**mandatory STOP。**
