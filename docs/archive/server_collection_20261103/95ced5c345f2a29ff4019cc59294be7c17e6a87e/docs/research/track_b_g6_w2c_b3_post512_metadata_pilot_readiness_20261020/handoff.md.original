# Q1–Q10：技術修正完了、Pilot実行閉鎖

基点は固定公開HEAD `22ef92cef31f98e3e31f27a209405f81f5c4a0e4`。旧512測定SOURCE `f8aad18d385c9dd787c5be8c794cf3c339335919` の原本と成功結果を保持した。新規実科学計算・科学再解析・native compileは0、新実authorization/permit/claimも0。

| 質問 | 回答 |
|---|---|
| Q1 原因・修正 | 人工v3 Coordinator.reportの固定Trueを継承し、integrationがLiveで上書きしなかった。versioned reportでvalidated manifestからmodeとapproval provenanceを生成し、全終端artifactとoffline検査へ伝播。False固定ではない。SOURCE修正完了。 |
| Q2 意味論 | real_authority=Falseは外部認可不使用のlegacy aliasとして残し、execution_type、approval_mode、external_authority_used、user_approved_live_execution、claim_spent、run_statusを分離。一般live_API_compatibilityはNOT_ESTABLISHEDのまま、固定manifestの完了prefixだけを別観測metadataへ。実F01未発生/範囲外入力も別記。 |
| Q3 原本保全 | 旧raw35filesとarchive完全SHA、旧SOURCE/input144pins、executable＋16libraries、receipt/delivery/caller、immutable read-only DBのCOMPLETED/STOPPED/spent1を照合。旧worktree/claim/STOP変更なし。訂正は対象SHA付き別sidecar。 |
| Q4 回帰 | 最終80tests PASS（FAIL/ERROR/SKIP0）、正常/P1 angle障害/ACK不明/receipt close障害の4integrationが期待分類、semantic mutation20全拒否。旧人工正常束512のevent・trace hash・bitsと意味論は一致。初回78件中の1ERRORは診断snapshotを失う追加例外が原因で、保全＋verified=Falseへ直し拒否testを追加。実Live metadataは保存承認によるpure fixtureで、Live科学処理は実行しない。 |
| Q5 保存512の整合 | 限定検査で新規の非metadata不整合は未検出。既存full Codec/digest/invariant/A-B/S-P1検査はSHA固定のまま再利用し、再集計しない。既知CPUのhelper混在/窓除外、終端slot export前credit、実F01未観測は限界として残し、完全な一般保証とはしない。 |
| Q6 旧Pilot照合 | 旧inventory46files/protectedと必須12資料を完全SHA照合。実resolver、Canonical/Full constructor、ScientificBits recorder、Exact Codec/Result、A/B・S/P1、F01、Global Bundle、SimpleStore、Storage V2、receipt、監視primitiveを再利用。旧fake v3 768件、人工Port16896件、実科学512件は別証拠。 |
| Q7 真の残件 | 6phase/48pair/16896のversioned manifest/loop、profile index/N/budget、採択seed guard、observer baseline off/instrumented on、profile保存/チェック、plannedをbindしたterminal state検査、pre-admission outer計測窓が必要。科学CoreとGlobal Bundle等の再開発は不要。旧coverage STOPと512 INCONCLUSIVEの契約差も採択前に確定する。 |
| Q8 準備度・採択 | metadata修正と閉じた候補/guard/静的budget/tasksの準備は完了。実行可能な正式Pilot releaseは未作成。n256/二条件/反復、総16896、seed、coverage遷移、retention/byte budgets、CPU/block/observer estimandの科学的/契約的採択が必要。512の成功だけでn256十分性や性能順位を決めない。 |
| Q9 最小次段階 | ユーザーが設計を採択→Codexが表の限定差分だけの新release・影響人工test・実値SOURCE/runtime/manifestを固定→固定対象へ別の16896 one-shot実行承認→fresh独立scopeで一回実行、保存・公開・STOP。旧512を差し引かず、旧claim再利用なし。巨大consumer/148件/Core/Codec/Global Bundle/F01/Storageの全面再開発・全面再試験、Full-N/native/外部authority構築は不要。 |
| Q10 次担当・承認 | 次はユーザー・GPTが上記候補の採否と具体的契約を決める。続いてCodexへの正式release差分指示、さらに固定SOURCE/manifestへの新規科学実行承認が必要。現段階でsamplingは起動しない。SOURCE/入力/runtime破損、invariant/worker/storage/予算/UNKNOWNはSTOPしprefixとspentを保持。どの終了後もmandatory STOP、retry0/resumeなし。 |

[原因と限定修正](metadata_fix_and_semantics.md)、[旧512照合](saved_512_evidence_verification.json)、[人工結果](artificial/results.json)、[Pilot再利用表](pilot_16896_existing_evidence_crosswalk.md)、[最小SOURCE差分](pilot_contract_and_source_gaps.md)、[未採択契約](pilot_execution_contract_candidate.json)を参照。

Pilot候補はcorrectness512＋baseline8192＋instrumented8192＝16896。固定P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m5、sigma+1、root256/H160、eta/rho1e-12、epsilon_axis1/200、primitive1e-6、alpha_axis1/320、familywise1/20、A6 math-onlyを参照する。新Hamiltonian/分子DF/新アルゴリズム・物理合成へ拡張しない。

現実のruntime資源とclockを観測し、恣意的CPU/RAM/wall上限や共有環境変更を加えていない。旧512の83.526286145秒はworker/helperを含む計測窓内値で、現workerはcorrectness diagnosticsのCachedObserverが常時on。Pilotのuninstrumented baselineやFull-Nへ線形外挿しない。observer固有費用やCPU分散を0と補完しない。

新準備SOURCEのactivation Noneは、消費済み旧承認のhelperを新SOURCE/new scopeへ移して追加512件を可能にしないための閉鎖で、旧SOURCEや旧STOPの変更ではない。候補とmetadata formatterは実行権限ではない。一般的正しさ・IID・性能優位性・科学的GO/NO-GOは認定せず、重要科学レビューも開始していない。同一primary Codexの技術検証で、別reviewerの独立再現とは呼ばない。

SOURCEの実値と完全SHAはsource_inventory.json、公開commit/remote HEAD/bytes照合はpublication_receipt.jsonと最終チャットに記録する。**今回の作業後はmandatory STOP。次の担当：ユーザー・GPT。**
