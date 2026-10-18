# R1_PARALLEL_8PAIR_CANDIDATE

状態：**DRAFT / NOT_ADOPTED / ARTIFICIAL_ONLY / RUN_READY_FALSE**。正式R1を置換しない。機械可読版は [r1_parallel_contract_candidate.json](r1_parallel_contract_candidate.json)。契約名やJSONの存在は実行承認・正式digestの採択を意味しない。

ユーザーが別途判断すべき意味論の変更は、最小限**6項目**に整理する。各項目の実装/pin更新は別途技術作業。

| # | 現R1 → 候補 | 採択が必要な内容 |
|---|---|---|
| 1 Dispatch | phase内の固定逐次 → 最大8 fresh pair process、完了順は可変 | worker数、固定dispatch identity、固定集約順を選ぶ。3 barrier・X→Y共有・24 constructorを維持 |
| 2 Start/call管理 | 逐次implicit start → 親による一意reservation/ENTER | P1上限workers、またはP2のcapを選ぶ。予約不明枠は返却せずcall上限へ保守課金 |
| 3 STOP/prefix | 逐次最初の失敗prefix → 親の最初の到着通知＋全in-flight drain | 最大workers/cap件の未確定処理、全failure保存、喪失結果不明、全join、時刻順序の限界を受け入れる |
| 4 CPU費用のestimand | 非競合逐次pair → 同時競合下のpair CPUとbundle資源費用 | concurrent費用を主指標にするか選ぶ。旧逐次値と同条件と扱わず、管理/observerを含む費用範囲を固定 |
| 5 保存/予算管理 | 完了pair集計 → 全worker共有の事前byte予約とbounded statefile | 64GiB/256MiBは不変。約30.24MiBのtrial状態file、失敗credit返却なし、保存不完全STOPの意味を採択 |
| 6 SOURCE/認証binding | 逐次host v2 → 新parallel host/schema/roles/pins | 新entrypoint/runtime/import location/manifest/正式契約digestを固定し、1 global approval/claimの全workerへのscopeを束縛 |

これらにより指定の16検討対象を包含する。worker/dispatch、barrier、X/Yは#1、atomic startとin-flightは#2、failure順序/prefix/worker crash/STOPは#3、estimand/競合は#4、outputは#5、global claim/admission/host/SOURCE/schema/manifestは#6。全call予算は#2と#6で固定する。

維持条件：P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、σ=+1、4方式×A/B×X/Y、固定G7 input/policy/law/N/error/confidence、seed/bitstream、event/phase/weight/word順序、R/C/A分類、math-only angle τ=10^-6 rad・Kφ=32・J=32、再query、cache禁止、unbuffered保存、correctness256・full-N両pass各15,855,296、合計31,710,848、retry0/resumeなし/mandatory STOP。物理RZ代表値を適用しない。

人工prototypeは科学moduleをimportせず、実LegacyEndpoint/actual Nの実行入口はない。正式parallel schema/host/実契約はまだ作っていない。現v2に人工policy input pinを足したtestは、v2の正式scheduleを緩めたものではない。将来のversion名・正式digest・scope/claim/approvalの値は未定であり、実値を発行しない。

正式採択に先立ち、ユーザーが上記6項目、特にconcurrent CPU比較とP1/P2のin-flight許容量を決める必要がある。その後Codexが別依頼で新versionを実装し、全pin/host/admission/Boundary/loaded path、元275＋並列人工回帰、F01・budget・guard・SOURCE全domain条件を再検証する。実入力pilotが必要なら独立の小さい固定予算/one-shotの提案と明示的な科学実行承認が必要。本番claimを先に消費しない。

今回実装から技術候補として優先するのはP1/8 worker。P2はin-flight枠を減らす選択肢で、速度とCPUの利益は測定表で個別に判断する。最初の失敗の瞬間に他workerの実関数呼出しを完全阻止する保証はどちらにもない。正式採択は行わない。
