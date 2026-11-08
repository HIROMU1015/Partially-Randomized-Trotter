# 次の固定有限診断：43 frame候補のS1 impact

2026-11-08 JST、SERVER 2。基準 **340670387dbe58bdda5606fdf222408774be1932**。本書と[準備済みSOURCE](../../../scripts/diagnose_pr_reference_frame_impact.py)の科学数値entrypointは**未実行・別途承認待ち**。今回の静的作業はmandatory STOPする。追加の科学条件・候補選択をせず実行できるscopeを以下に固定する。

## 1. 目的・固定入力・SOURCE

保存recordの再選択が、SOURCE RTEのframe最大上界と両armの非中央十分上界を下げるか、local離散Tとともに診断する。新しいword policyの採択やRPE成功の判定ではない。

REPORT §1の物理条件・law・compiler・構成・m17 q/r/bits・626/630 shots・alpha/failure/phase/energyを維持する。Inputは全て固定baseのGit objectから読む。

| 保存入力 | 使用箇所 |
|---|---|
| `artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/RESULTS.json` | 訂正済みSOURCE限定baseline4行、template-max scalar、joint予算の引用 |
| `artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json` | 保存375 frontierのうち対象43のoffer／非劣IDだけ。再計算しない |
| `artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz` | 現C0／85 compositeのguard、Q＋G、保存T、元record metadata |
| `artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz` | 固定43 literal、SOURCE7 basis、SOURCE/template/contextと候補IDの対応 |
| `artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz` | 現C0の保存guardとT。Backend／guard実行なし |
| `artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz` | B′のSOURCE必須20 template、保存4 context |
| `artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz` | ReferenceのSOURCE必須21 template、保存4 context |
| `artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz` | actual SOURCE inverse Dの5 template／保存context |
| `artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json` | 保存m17条件とjoint capの一致確認だけ。RPE再計算しない |

算術は旧S1 coverage訂正runnerの`load_legacy`、`required_templates`、`scoped_parts`と、その中で不変更AST抽出する`weighted_template_max`／`template_weights`／arm別式を再利用する。旧S1 `main/evaluate`、S2 `pareto_ids`、offer再構築、B1 fixtureは実行しない。旧SOURCE参照commitは既存runnerの固定値を維持し、科学入力のresolverだけを上記baseに固定する。元のhash／path、算術AST来歴を結果へ記録する。

## 2. Target・有限選択規則

対象keyはSTATIC_FINDINGSの`candidates_43[].literal`を辞書順に固定した43 positive binary64 hex key。完全な表は[STATIC_CROSSWALK](../../../artifacts/pr_reference_frame_impact/2026-11-08/STATIC_CROSSWALK.json.xz)。近い角度、modulo、decimalによるkey統合はしない。43以外のguard／word／Tは現policyのまま。

各keyのpoolは保存S2 `nondominated_ids`に対応する既存offerだけとする。NTのfallbackに限り保存`fixed_current`も許す。Candidate IDは保存archive/memberへ対応し、新record／precision追加をしない。

- **G**：`(exact Fraction guard, integer T, record ID)`の昇順最初。
- **T**：`(integer T, exact Fraction guard, record ID)`の昇順最初。
- **NT**：現T以下の保存非劣offer＋現wordから、Gと同じ順序で最初。

3つを全て評価し、Gだけを勝者としない。両armに同一record mapを適用し、arm別最適化／追加情報を与えない。同一mapならaliasを記録し、同じ計算を繰り返さない。有限per-keyルールであり、LP/MILP、全体T-count最適化、探索gridではない。

## 3. 計算するものと科学call範囲

1. **Baseline**：旧S1のSOURCE限定4行・frame/D/other parts・保存joint予算・残余は引用し、旧aggregate runnerを再実行しない。
2. **現在のwitness**：SOURCE必須templateの保存native列からkey別物理多重度を読み、現guardでcontext和を評価する。保存template-maxとの一致を確認し、forward／inverseを別々にcontext-maxにした上で7 SOURCE basisのpair-maxと全tieを特定する。これは新しい科学再解析であり、今回未実行。
3. **3有限map**：選ばれた保存guardだけを置換し、同じ旧weighted-template式・同じcoverageでtemplate/context上界、frame最大、両arm各X/Yの非中央上界、保存joint capとの差を評価する。
4. **Tradeoff**：keyごとの保存guard／T差と、各固定local template/contextの実出現数による保存Tの和を出す。Context毎のTとguard-maxのwitnessを別に表示する。3qr／shotsをTへ掛ける総価格計算はしない。

変更できるtemplateはmetadataに対応する3 SOURCE forward basisのみ。Inverse全7 basis、他4 forward basis、D／products／CSWAPの上界が保存値と不変であることを確認する。Contextの結合、SOURCE alphabet拡張、期待値による全support最大の置換はしない。

評価範囲は、現在の固定template/context表1組＋G/T/NTの最大3 map。同じmapはdeduplicateする。SOURCE必須mappingはb20／ref21／inv5で、各々保存contextのみ。Native guardは旧S1と同じlogical word単位で、C0は保存strict guard、compositeは保存Q＋Gを使う。Emitter／shell等を0にする処理はなく、これらは未資格の別項として残す。

新しいoperator行列、norm／eta、strict guard、word取得／生成、provider／consumer起動、回路build/compile/simulation、sampling、RPE式／半径／shot、全18round総Tはcall範囲に含めない。B1のruntime試験と旧S1/S2 one-shotは再実行しない。

## 4. 出力・検算・判定基準

実行する場合だけ新しい`artifacts/pr_reference_frame_impact/2026-11-08/numeric/`に、INPUTS.json／RESULTS.json／STOP.jsonを作る。失敗時だけFAILURE.jsonを残す。今回はこのdirectoryは存在しない。

INPUTSには実行前SOURCE hash、固定Git入力hash、再利用式の来歴、43 target、3規則、既存条件を保存する。RESULTSには以下を含める。

- 保存baseline4行と現在のtemplate/key寄与、最大basis／context全tie。
- 各規則のrecord ID・sequence hash・保存guard/Tと差。
- Template/context別guardとlocal T、forward/inverse別witness、frame pair-max。
- 両arm各X/Yのparts、非中央上界、保存joint予算、残余、frame変化。
- Missing／domain／central／emitter／shell／terminalはUNQUALIFIED_NOT_ZERO、full RPEはUNQUALIFIED、global TはNOT_EVALUATED。

検算は、保存current offerと現guard/Tの一致、保存frontier ID一致、SOURCE coverage、現context-maxと保存scalar一致、未変更template不変、部分和と残余のexact Fraction整合、固定SOURCE不変とする。新しいfrontier最適性や真のoperator誤差を証明する検算ではない。

判定は次を区別する。

- **Frame上界改善あり**：同じS1式のframe値が下がる。Local Tの増減を併記し、全体T削減とはしない。
- **局所だけ改善／frame最大不変**：他basis、inverse又はmax-contextが残る可能性をwitnessで示す。
- **上界悪化又はtradeoff**：T優先等で保証を緩めた結果も保存し、candidateを隠さない。
- **非中央残余が負**：この保存加法十分証明では閉じない。実誤差下界やRPE失敗ではない。
- **UNQUALIFIED**：必要key／guard／T／context／仕様が不足又は不一致。0／推定値／別recordで補わない。

## 5. STOPと次の判断

固定比較を完了、又は入力／SOURCE／coverage不一致に到達したらmandatory STOP。科学条件やmapを追加し、成果が出るまで選択規則を拡張しない。失敗出力を保全し、別word／precision／guardで埋めない。

今回の承認待ちは**保存guard/Tを使う上記有限選択、current argmax／weighted-template再解析、局所T集計、非中央残余計算**である。Static準備と人工8 testは実施済みだが、この科学callは実行していない。実行承認には新しい承認証明書／grantは不要。

次担当はGPT／ユーザー。診断を承認するか保留するかを判断する。その結果を得てから、43-key runtime拡張、D保証改善、S3／A1の必要性を判断する。D単独でも旧joint予算を超える課題、B′ scalar／Reference vacuum／event domain、central／emitter／shell／terminalと強B′への公平性は維持する。A1、追加合成、RPE、総Tへ自動的に進まない。
