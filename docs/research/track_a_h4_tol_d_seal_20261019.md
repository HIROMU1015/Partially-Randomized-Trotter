# H4 tol-only DF10：D承認待ち最終seal（2026-10-19）

**D_TECHNICAL_SEAL_COMPLETE_PENDING_USER_APPROVAL。D科学実行0、valid grantなし、mandatory STOP。** ユーザーの今回の指示2にある「Dの数値実行についてユーザーの明示承認を受けた後」に従い、承認待ちの技術sealまで実施した。今回の実行計画の提示を、別途要求されているD科学実行承認として読み替えていない。Aの最終sealは「Dの正式受入＋次段階の指示」という条件が未成立のため実施していない。

## 固定条件と照合結果

Phase 0は `80652ab6b42cf0c11ee97485214027f507b1bf0a`、科学SOURCEは変更せず `84db3909639b34ec521004c3d8f7ef9b5e6ad851`。新しいmetadata-only sealツールのcommitは `a49fe026db2cbefeac70095ad5dd2ef8e0d12231`。既存SOURCEを書き換えず、別ファイルのsealとINVALID grantを保存した。

H4 linear・隣接1.00 Å・STO-3G、保存済みtol-only threshold1e-8のDF rank10、sector36、Nα=Nβ=2、T=.8、P=0、通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005。q=1/2/4/8/16、δ=.8/.4/.2/.1/.05。B0/B0I prefix1〜9、full B1 prefix10、二次/四次の**190候補**を維持する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional control、元constantとtail identityを一度だけ、reference無回転も変更していない。

保存Hamiltonian/state/snapshot/compiler hashは[Phase 0報告](track_a_h4_tol_main_phase0_20261019.md)の固定値と同一。H4 dense eighとH6/H8 eigsh、正規化・残差gateの実装差は保持した。今回のユーザー指示1に従って保存状態を再利用し、SCF・integral・DF・保存stateの再生成をしない。

|照合対象|結果|
|---|---|
|科学SOURCE closure|17ファイルの現在bytesと固定commit blobが一致|
|入力・保全記録|774レコードの原本/copyとSHA一致|
|provider・A kernel|251ファイルのSHAと完全ファイル集合が一致|
|runtime・Python|34,273ファイルとPython実体SHAが一致|
|保存F/FP|固定Phase 0 artifactsのSHAとGit blob一致、D用6 signal・4候補8軸のidentityを再確認|
|科学契約|candidate/full/rank/binding/compiler/call予算はPhase 0と同一|
|grant・output|INVALID grant、科学output・ONE_SHOT・driver未作成|
|人工検証|新seal8件＋既存rank10 adapter32件＝40件PASS|
|実sealの拒否検査|direct/witness 2経路とも科学import・output・marker作成前に拒否|

保存FPのPR n32・64軸をDの新規測定に含めず、rank12 D/Pの科学数値も移していない。FPのD4候補8軸を再利用して重複compileを避ける。保存数値の再集計・新shot・順位計算は行っていない。

## 承認するDの具体的範囲

新規base signal96、解析B0I overlay88、main shot endpoint920、独立forward最大96、打切りexact reference最大88、native evolution最大186、**新規wrapper compile最大372軸**、control最大744、preparation最大10、primitive1095、endpoint accounting1 pass。full-H referenceは保存値を再利用する。SCF・integral・DF・保存state solver・新PR samplingは0。

承認後はsignal・bias・適格性を確定し、適格候補のfull wrapper compiled RZ×保存shotを会計して、精度ごとの登録内最小PFをexact費用・同値ならcell_id辞書順で判定する。不明/欠測を不適格として除外せず、全登録が完結していない結果をPASSにしない。global optimum・厳密bias・母平均優位の保証は付けない。

この範囲は科学callの最大数であり、CPU・RAM・worker・wallの恣意的上限ではない。現在の許可affinityは64 CPUと観測した。CPU topology、cgroup/quota、メモリ、load、PSIは私有receiptに保存したが、**実行workerの割当・資源admissionはまだ行っていない**。承認後、既存OS制御を尊重してfreshに再確認する。歴史的worker数を転用せず、構成間並列、fork/COW、preparation cache、paired evolution、fingerprint再利用、1 worker 1 threadを使う。監視値や一時的競合だけを理由に正常workerを中断しない。

## 保存物と承認時の操作

[承認待ちD契約](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/D_contract_SEALED_PENDING_APPROVAL.json)、[INVALID grant](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/D_grant_INVALID.json)、[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/completion_audit.json)、[人工報告](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/artificial_test_report.json)、[拒否検査](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/refusal_report.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19/artifact_manifest.json)。今回の監査はCodexによる著者チェックで、別reviewerによる独立レビューではない。

私有sealは `.h4_tol_main_private/D_seal01_20261019/`。Phase 0契約・grant・manifest・過去のledger/marker/STOPは保持した。seal時点の契約はstatus=TECHNICALLY_SEALED_PENDING_EXPLICIT_D_APPROVALで、pendingはDの明示承認とfresh資源admissionだけである。既存runnerの承認gateを迂回する機能は追加していない。

明示承認後は、現在の承認待ちsealを残したまま、新しい実行契約にfresh worker割当・receiptを固定し、ユーザーの承認receiptとそのSHA、SOURCE、実行契約digestを結び付けたDだけのone-shot grantを作る。科学plan・入力・SOURCE・call予算は変えない。SOURCE/input/runtimeを再照合し、未使用attemptで既存witness/runnerを一回だけ実行する。失敗時もSTOP・予約・終了記録を保存し、**自動retryしない**。科学条件の変更が必要なら実行せず報告する。

## D結果・A50準備・remote・次の承認

- **D結果**：未実行。signal・bias・適格性・新compiled RZ・最良PFの新結果はない。
- **A50**：Phase 0の50候補案のみ。FP1候補n32と新規49候補n8を区別した状態を維持。新seedの割当・最終衝突検査・最終seal・科学実行はすべて未実施。
- **remote**：今回 `git ls-remote` でPhase 0 branchが `80652ab6b42cf0c11ee97485214027f507b1bf0a` と一致することを確認した。この観測は新sealやD結果のpush証拠ではない。D結果のcommit・push・remote SHA確認は、指示2に従いDの承認後実行・監査を終えた時に行う。
- **未解決事項**：D科学実行の明示承認、承認時のfresh資源割当。科学条件を変更すべき新しい問題は今回のmetadata照合では見つかっていない。実分子Dでの動作・実負荷の速度/RSSは未検証。

次の作業：**ユーザー**が「保存DF10状態・科学SOURCE84db3909…、D190候補・最大372新規軸、one-shot・自動retryなし」のD数値実行を明示承認する。**Codex**は承認後にDだけを実行・監査し、commit・push・remote SHA確認を行ってmandatory STOPする。

その後、**ユーザー**によるD結果の正式受入と次段階の指示があった場合に限り、**Codex**がA50のseed割当・衝突検査・SOURCE/入力/call/契約・人工/拒否検査を最終固定し、commit・pushしてSTOPする。A50の実科学計算は、さらに別の明示承認を必要とする。
