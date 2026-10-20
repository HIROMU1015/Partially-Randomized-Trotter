# H4/H6 PR pilot：保存容量最適化の調査・実装

2026-10-20。**33本/群案を優先したまま、全証拠をbyte単位で復元できる圧縮保存案を作成した。科学実行への採用は未承認。既存raw・契約は変更せず、新しいH4/H6 sampling・compile・signal・shot・数値再解析は0件。mandatory STOP。**

基点：`3e7365a606e18efa585252eef23f7b13264b0f38`。専用branch：`track-a-h46-pr-pilot-storage-optimization-20261020`。[調査bundle](../../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/README.md)と[保存policy DRAFT](../../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/storage_policy_DRAFT.json)を併読する。これは旧契約の変更・pilot実行承認・n_Pの正式採択ではない。

## 1. 結果と推奨

**推奨は、新しいpilot出力のQPY/NPZをlossless圧縮し、元byteのSHAとarchiveのSHAを別々に保持する方法。** 先に非圧縮の1 pairを保存してから圧縮・復元SHA検証を行う方式でも、33本案の容量proxyは現在の空きに収まる。direct streamを必須にしなくても実行可能性がある。

保存済みDET QPY/NPZ全276fileを**別のprivate領域へコピー圧縮**した。10,356,702,848 bytesが635,361,139 bytes（約6.13%、削減約93.87%）となり、全276fileで展開したbyteのSHA・サイズが原本と一致した。H4/H6各最大ファイルは新しい一時fileへ実際に復元し、一致を再確認した。QPY回路やNPZ配列は復元・シミュレーションしておらず、圧縮器がopaque byte streamだけを処理した。

この観測から、各系で最大だった圧縮済みpairの**4倍**と、trajectory当たり**16 MiBのmetadata余裕**を置いた計画案では、33本/群の最終保存増分は約132.5 GB。raw一時領域と復元用1 pairを含むピークproxyは、各系1 workerの並行で約137.0 GB、合計64 workerをH6へ全て割り当てる容量計算上のenvelopeでも約264.2 GBになる。調査時の空き約455.2 GBに収まる。

**これはPR/rare出力サイズの上界ではない。** 4倍・16 MiBは未採択の計画余裕で、実出力を切り詰める上限ではない。異なる係数・qのPRやrare eventで圧縮率・rawピークが変わるリスクを残す。実行前に新保存契約、archive対応のadapter/終了監査・復元経路、資源予算を固定する必要がある。現在の旧pilot runnerは非圧縮方式のままである。

## 2. 既存結果の容量内訳

H4 DET38軸・H6 DET54軸の固定[raw inventory](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/private_raw_inventory.json)3482fileのサイズ・SHAを使用した。既存原本は読取のみ。対象はこの既知directoryだけで、他worktreeやagent履歴の広域検索をしていない。

| 系 | compiled QPY | wrapper QPY | vector NPZ | その他metadata＋ledger |
| --- | ---: | ---: | ---: | ---: |
| H4 | 1,231,672,712 B | 540,408,825 B | 962,236 B | 1,574,225 B |
| H6 | 6,154,480,044 B | 2,407,905,083 B | 21,273,948 B | 3,126,062 B |

QPYが支配的である。状態vectorは各NPZに`low.npy / before.npy / after.npy`を保持し、現方式はZIP_STORED（非圧縮）だった。NPZを数値配列として読み込まず、ZIP headerだけで確認した。ledgerはH4 1012/H6 1430fileで、全て保全する。

完全一致のfileを共有するだけの論理削減量は27,760 bytesだった。cosine/sineの回路に共通部分があっても、file全体のSHAが違うため同一物として扱わない。共通の初期状態だけを取り出す方法も主な容量問題を解決しない。vector比較や値の近似による同一化は行っていない。

## 3. 可逆圧縮の実測

Python標準zlib/DEFLATE level1を使用し、1 MiB単位でstream圧縮、元SHA計算、archive SHA計算、展開SHA確認を行った。全fileをmemoryへ読み込まず、QPY/NPZの科学objectを生成しない。圧縮作業は開始時idle CPUに応じた61 threadで行い、約14.4秒で終了した。これはbyte圧縮・検証時間であり、将来のPR計算時間見積りではない。

| 系・種別 | file数 | 原サイズ | 圧縮後 |
| --- | ---: | ---: | ---: |
| H4 compiled QPY | 38 | 1,231,672,712 B | 67,687,975 B |
| H4 wrapper QPY | 38 | 540,408,825 B | 39,731,072 B |
| H4 NPZ | 38 | 962,236 B | 431,435 B |
| H6 compiled QPY | 54 | 6,154,480,044 B | 336,574,817 B |
| H6 wrapper QPY | 54 | 2,407,905,083 B | 181,873,553 B |
| H6 NPZ | 54 | 21,273,948 B | 9,062,287 B |

[276fileの圧縮証拠](../../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/compression_evidence_276.csv)には候補ID・軸・種別・元/archive SHAとbyte数を公開する。原本やarchive本体、内部絶対パスは公開しない。

最大圧縮pairはH4 13,343,802 B、H6 61,925,972 B。これにmetadataの未取得部分を含む別余裕を加えて計画する。level6も両系の各最大wrapper/compiledの4fileで確認し、level1より小さくなった。ただし全276fileのlevel6結果へ外挿せず、主容量案は軽量なlevel1の全測定を用いる。追加の重いcodec導入は不要と判断した。

## 4. 証拠・再現性を維持する方式

新しい保存方式は、**今後承認されるpilotの新規出力だけ**に適用する案である。既存rawをarchiveに置換したり削除したりしない。

1. SOURCE、入力、compiler/runtime、PF版、seed/cohort、events/law、native gate、結果、grant、ledger、marker、STOPを保持する。native検証の閾値・順序・PASS条件を変更しない。
2. QPY/NPZのbyte全体をlossless圧縮する。状態vectorの丸め、部分省略、量子回路の構造だけからの代用をしない。失敗quarantineもUNVALIDATEDのまま保存する。
3. 原byte SHA/sizeとarchive SHA/size/codec/versionを別に記録する。`compiled_SHA`は引き続き非圧縮QPYのbyte identityであり、archive SHAへ黙って置換しない。
4. 原SHAを期待値に展開streamを検証し、成功後にarchiveをatomic・no-overwriteで確定する。途中失敗の.partialは診断用に残す。重複実行・自動retry・上書きしない。
5. 新しいloader/auditは元logical pathとarchive/原SHAの対応を解決し、必要時に新しい復元先へbyte完全一致で戻す。再compile・再samplingで代用しない。保存artifactに関するPASSと科学native PASSは別である。

実装した[standalone prototype](../../scripts/h46_pilot_lossless_storage.py)は元fileを削除しない。将来の「新規raw一時fileを圧縮・検証後に解放する」操作は、新しい保存契約として別途採択する必要がある。この原則を適用しないまま元と圧縮copyを両方残すだけでは、旧約976 GBを解消できない。

prototypeはarchive file本体のfsyncとno-overwriteの確定を行うが、科学実行のcrash-recovery journalはまだ実装していない。将来の一時file解放前には、archiveとlogical mapping manifestの確定・directoryの永続化・終了監査の整合を接続し、電源断や親process異常でも復元可能な状態を確認する必要がある。今回の試作に元file解放機能を入れていないのは、この採択・監査経路が未確定だからである。

direct-stream writerも試作した。小さな人工QPYについて、通常dumpと圧縮sinkへのdumpから同じ非圧縮byte SHAを得て、完全復元を確認した。seekを要求するserializerは拒否する。NPZを別のZIP規約へ無断変換せず、現NPZをそのまま外側圧縮できる。direct streamはraw disk temporaryを減らせるが、本容量案はそれを仮定せず、workerごとにraw pair全体のscratchを含めた。

## 5. 最終保存・scratch・ピーク・逐次/並列

以下はGB=10⁹ bytes。既存成果は残したまま、**現在の空き容量に対する新規増分**を評価する。既存使用量を再び必要容量へ足したり、既存rawの削除で空きを捻出したりしない。

| n_P/群・方式案 | 最終保存の増分 | 各系1worker並行のピーク | 逐次実行のピーク |
| --- | ---: | ---: | ---: |
| 33・旧非圧縮proxy | 約976.2 GB | finalだけで不足 | finalだけで不足 |
| 33・圧縮4倍＋16 MiB/trajectory | 約132.5 GB | 約137.0 GB | 約136.6 GB |
| 9・旧非圧縮proxy | 約266.2 GB | 別途raw一時領域が必要 | 同左 |
| 9・圧縮4倍＋16 MiB/trajectory | 約36.1 GB | 約40.6 GB | 約40.2 GB |

H4/H6を逐次にしても、両系の完了archiveを残す以上、最終保存増分は同じである。逐次化が減らすのは同時raw scratchであり、旧976 GBの問題を逐次化だけで解消することはできない。

具体的には、Fを両系の最終archive＋metadata予算、W4/W6を同時worker数、R4/R6をraw pair proxyとすると、並列peakは`F + W4*R4 + W6*R6 + restore_reserve`。R4=438,570,748 B、R6=2,026,673,924 Bを旧案から保持し、復元用は最大raw pair1個分を別枠にした。rawが別名で二重に生成される実装へ変更する場合、その分をさらに加える。

worker数は固定しない。観測affinity64 CPUを全てH6へ配る容量計算上のenvelopeでは、33本・4倍案のpeakは約264.2 GB、圧縮8倍の感度案でも約383.5 GBで、調査時空き約455.2 GBに収まる。これを64 worker実行の推奨や承認とは扱わない。実際はfresh idle CPU・実効RAM・stageごとの使用量・scratch予約によりadmissionする。4倍/8倍とも出力サイズの上界ではない。

[全18容量case](../../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/capacity_cases.json)に33/9本、圧縮余裕2/4/8倍、worker配置を区別した。metadata16 MiBは未採択の計画仮定で、PRのevent/law JSON、台帳fileのallocation/inode overhead、異常時のpartialが増えれば再見積りする。現在空きは今後のfresh gateを代替しない。

## 6. 他の方法の評価

| 方法 | 評価 |
| --- | --- |
| byte可逆圧縮level1 | 推奨。全276fileの実測と原SHA復元を確認。科学値を変えない |
| codec level6 | 4最大QPYで追加削減を確認。主案の実行成立に必須ではない |
| exact-file共有/CAS | 元SHA一致なら安全だが、今回の重複は小さい。実装負担に比べ主効果なし |
| cos/sinの共通部分共有・再構成 | opaque chunk dedupは別技術として可能だが未実装。単一部品からの再生成を原本の代用にしない |
| vector NPZの外側可逆圧縮 | 安全な補助策。全low/before/afterの元byteを保持する |
| vectorの低精度化・一部削除 | 採用しない。証拠を失い、今回承認されていない |
| raw QPYを捨ててseedから再compile | 採用しない。compiler依存・実回路同一性・元SHAの証拠を失う |
| ledger/metadataの省略 | 採用しない。費用が小さく監査に必要。packしてもlogical record/SHAを残す別契約が必要 |
| 逐次実行・pairごとの圧縮 | finalを変えずscratchを抑える。33本を9本へ減らす必要性を下げる |
| 別の保存先 | 圧縮率が悪い場合の代替。専用領域と容量を確認し、他ユーザー領域は変更しない |

## 7. 不確実性・不足時のSTOP

- 実測対象はm5のDET46pairであり、S2/Yoshida4/m5のPR24群や未観測rare回路のサイズ・圧縮率は未取得。今回新しいsamplingでその値を埋めていない。
- 同じ元byteでもcodec/versionでarchive byteは変わり得る。元SHAが同じことを正とし、codec実体とarchive SHAを封印する。
- 今回のprototypeは旧科学runnerの保存hook・認可・end audit・復元resolverへ未接続。旧runnerのまま圧縮案の予算だけを採用して起動してはいけない。
- 新契約では全stageのarchive見込み、worker raw temporary、codec partial、失敗quarantine、復元予約、metadata/inode余裕を合算する。同じ空きRAM/diskを2stageで二重計上しない。
- 出力が予算proxyを超えそうなら新taskのadmissionを止める。既に予約したworkerを完了させる容量が足りない、実ENOSPC/I/O異常、原/復元SHA不一致、実OOM、native異常なら当該stageをSTOPする。未確認結果をPASSにしない。
- 成功済みの証拠や既存rawを削除して続行せず、自動top-up・retry・圧縮失敗時の無断省略をしない。再計画には新しい承認が必要。host PSIや無関係な資料更新だけでは正常計算を止めない。

## 8. 検証・保全・次の担当

新しい人工テスト18件：原SHA完全復元、stream serializer、truncation/corruption/trailing-byte拒否、decoded-size guard、既存出力の上書き拒否、空file、disk不足、writer途中例外、旧raw不変、raw/scratchを含むpeak計算、人工QPYのbyte一致を検証した。新しい科学compile・simulationはない。作成者自身の技術検査で、独立科学レビューではない。

原3482fileのSHA・サイズを照合し、既存SOURCE/raw/grant/ledger/marker/STOPを保全する。今回作った圧縮copyと復元確認fileは別private領域に保持し、既存fileには書込していない。公開資料に原本・QPY/NPZ・archive本体・認証情報・不要な内部絶対パスは含めない。詳細は[検証記録](../../artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/validation.json)を正とする。

次は**ユーザー**が33本案・可逆圧縮方式・容量余裕と限界を採択するか判断する。**Codex**は採択に基づき、新SOURCEの保存hook・archive-aware監査/復元・scratch予約・系別無効契約を準備し、人工検証する。その後、**ユーザー**のH4/H6各一回実行の明示承認が必要である。現在の標本数、保存方式、pilot実行はいずれも未承認。

n_E、正式shot政策、AD03は保留。本資料作成から科学実行や数値評価へ自動移行せず、mandatory STOPする。
