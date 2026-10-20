# H6 noncancellation未解決30件：原因調査と保存証拠の回収

2026-10-20。基点`85433c48fd4c37d4cb4594391da422558ac3ce7f`。
今回の承認は原因調査・既存証拠の照合・文書整理と構造検証のみ。新しいsignal差・非相殺envelope・適格性・shot・費用・順位を計算していない。

**原因は旧B0I overlayの保存schema不足であり、12候補すべてで総biasだけが保存され、V1非相殺診断が要求するsigned成分が欠けていた。元のsignal値は消失していない。**
保存済み非相殺適格性をそのまま回収できたendpointは0、今回解決したscope条件も0で、未解決30件を保持する。
ただし12候補すべてで再構成に必要な保存値とSHAが揃い、1候補のsigned成分は既存公開資料から文字通り回収できた。新規compile・sampling・signal evolutionを必要とする欠測は、今回調べた固定範囲では確認されなかった。

## 対象・数え方

Linear H6、隣接1 Å、STO-3G、tol-only DF19、12 modes/sector400、T=.8、P=0、symmetric_directional full cosine/sine Hadamard wrapper。既存Hamiltonian/state/fragment順・compiler・profile区分・empirical allowanceを維持する。
対象候補はB0I、S2、prefix5/10/15 × q1/2/4/8の12候補。各候補にε=.05/.01/.005/.001/.0005 × allowance倍率1/10の10 endpointがあり、**候補別未確定120行**である。
その12候補を含むold32 / fine S2 96 / S2+truncated S4 136の3scope × 10 endpointが**未解決30件**。uniqueなsystem/bias/ε/factorは10であり、30個の独立実験ではない。
候補を除外せず、`eligible=null`を`false`へ変えず、候補集合の全domain最良・費用比を未解決のまま扱う。

## SOURCEで確認した原因

1. [旧pilot SOURCE 2e733d8aのoverlay](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e733d8a683d415792ce92a56c7d149b78f42716/scripts/h6_native_b0i_bridge_v1.py#L117)と[family SOURCE ac6d28e3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac6d28e35cf01398f169ce59577182a38da9b959/scripts/h6_b0i_family_bridge_v1.py#L117)は、B0のcorrected/exact-truncatedをtail identityで回転し、full-H referenceを回転せず保存する。返す`error_decomposition`は`total_signed`だけで、`signed_parts`はない。
2. [V1 materializer SOURCE 2261a3ad](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2261a3ad1d6c23521474b31204572a6c3a51258b/scripts/prepare_track_a_v1_reaccount.py#L48)は12件の`doc['signal']`をそのままcopyした。新しい位相回転や成分追加は行っていない。
3. [V1 signal_envelopes](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2261a3ad1d6c23521474b31204572a6c3a51258b/src/trotterlib/track_a_v1_contract.py#L109)は`signed_parts`欠測なら非相殺envelopeを`None`にする。[会計SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2261a3ad1d6c23521474b31204572a6c3a51258b/scripts/track_a_v1_reaccount.py#L187)は`accounting=None`、`eligible=None`、`UNRESOLVED_SIGNED_DECOMPOSITION`を保存し、全domain minimumを拒否する。
4. 後続Phase Aの104 signalではB0I signed成分を保存する実装があるが、登録追加集合は元の12候補と重ならない。同名prefixのS4や別qの値をS2へ移さない。

これは非相殺条件での科学的不適格判定やcompile失敗を意味しない。元familyの「12構成の適格性取得完了」は**通常signed bias**の120行についての記録であり、noncancellationの適格性を完成させた記録ではない。

## 保存証拠の照合・回収

[調査packet](../../artifacts/resource_applicability/track_a_h6_noncancellation_investigation/2026-10-20/README.md)の`candidate12_evidence.json/csv`に全12候補を記録した。

| 項目 | 確認結果 |
| --- | --- |
| 旧pilot3＋family9の元signalファイル | 12/12存在、公開input receiptのbytes/SHA完全一致 |
| 各元signalの保存値 | corrected、exact_truncated、full-H reference、allowance、log_Bが12/12存在 |
| input identity | Hamiltonian/DF/state/input/compiler/providerのmetadataが固定比較bindingと12/12一致 |
| 元B0Iのsigned成分field | 12/12欠測、total_signedのみ |
| 別の公開資料からsigned成分をliteral回収 | 1候補：B0I_p15_q4_2nd、T=.8 |
| 保存済みnoncancellation適格性のliteral回収 | 0/120 endpoint |
| 今回の新たな適格性判定・解決条件 | 0/120、0/30 |

`literal_decomposition_recovery.json`は[固定δ結果のsignal_summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/85433c48fd4c37d4cb4594391da422558ac3ce7f/artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/signal_summary_v1.json)にあるp15/q4/T=.8の`decomposition.parts_signed`を転記した。corrected/exact-truncated、reference、allowance、log_Bは元screenの値とすべて完全一致し、差や許容差は計算していない。closureの新しい数値検査も行っていない。T=.4/q2とT=1.6/q8の同名候補は対象外。
この公開表の会計10行は通常signedの保存値で、非相殺判定へ読み替えられない。したがって成分1件の回収だけで10 endpointや全domainを解決したとは扱わない。

旧signal・input metadataの一致は、旧compiler profile/runtime・compiled bytes完全同一性の新しい認定ではない。full-H referenceは不回転のまま扱い、B0の成分を単純にB0Iの成分として流用しない。
private rawのsignal値・grant・ledger・runtime・NPZは新たに公開packetへcopyしていない。private観測は既存公開receiptへの一致、field存在、hashと候補対応のmetadataに限定した。回収した成分値は元から公開されている1候補に限る。

## 後続D要約のunresolved空配列は解決証拠ではない

後続2026-10-18 H6 D公開要約にはnoncancellationの20比較で`unresolved=[]`と費用比較が保存されている。しかし元SOURCEを確認すると、[pr_cost_batch_science.py SOURCE31ef62a1の103–107行](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ef62a12407010e9c1a07f9b37facedb8dff5fe/scripts/pr_cost_batch_science.py#L103)は、利用可能候補と未解決一覧の両方を`r['eligible']`の真偽値条件で絞る。`None`は両方から落ちる。

既知のsealed H6_D_ACCOUNTING payload/resultを読取確認した。両方に同じ120行の`eligible=null`が残り、結果ではshot・費用もnull。terminalのSOURCE・result SHAが一致し、SOURCE file hashも元technical sealと一致した。**未確定候補を解決したのではなく、後続比較の欠測集計から抜けている。**
`later_D_evidence_gap.json`にSOURCE、原本SHA、保存状態と静的な原因を記録した。後続Dの非相殺20比較は元12候補を含む登録全域の完成証拠として採用できない。既存値を訂正・再計算したり、新しい最小費用を作ったりしていない。
元Dのraw/SOURCE/grant/STOPおよび公開要約は不変。本調査はその限定的な注意資料であり、詳細科学レビューではない。

既存70件比較のH6側はSOURCE2261a3adの**signed**会計、H8側はsix-axis会計を基準としている。後続D要約は基準ではなく、この問題による70件の再実行・数値修正は行わない。

## 保存データだけで解決できる範囲と最小の次段階

**記録回収だけでは非相殺適格性を確定できない。一方、別途承認された保存値の数値再解析なら、12候補すべてについて判定入力が揃っている。**
対象signalの再取得が必要だという証拠はなく、新規SCF/DF/state/reference/signal evolution、位相回転、compile、sampling、model評価は提案しない。未知の数値closureや境界判定のPASSをここで先取りしない。

[未承認の最小実行案](../../artifacts/resource_applicability/track_a_h6_noncancellation_investigation/2026-10-20/future_saved_analysis_DRAFT.json)を保存した。実行runner・有効grant・起動commandは用意せず、scientific_execution_authorized=false。

- **段階A（ユーザーの別承認が必要）**：11候補の保存corrected/exact-truncated/referenceからdiscard/PF signed差を新規構成し、p15q4の1候補は回収済み成分をcopyする。12候補で成分closure、既存SOURCEの軸別絶対値和envelope、5ε×2倍率＝120 endpointの非相殺適格性と、適格なら対応するanalytic shotを新しく計算・保存する。元allowance/log_B/α/演算順を変更せず、旧signed結果・PR・費用モデル・候補集合は不変。新しい12候補の成分・envelopeと120回以内の会計だけをsealし、異常/不一致はSTOP、自動retryなし。
- **段階B（段階Aとは別承認）**：30scopeのPF/PR費用比較を行う場合は、段階Aで確定した新会計と旧保存会計を結合する新契約を準備する。元未確定120行には保存shotがないため、旧70件の「保存shot不変」契約をそのまま30件へ拡張できない。既存確定行の再現を確認し、追加会計を新来歴に分離する。単価が欠ける適格候補が残ればSTOPし、新規予測/compileを自動追加しない。現在は再会計・最良候補・PR/PF比・順位を一切計算していない。

段階Aで数値的未解決や不整合が見つかった場合の追加signal計算等は、この案でも未承認。今回も、次段階Aの承認も、そのfallbackの許可にはならない。

## 共通実験プロトコルv2の補足証拠

[独立した補足資料](track_a_protocol_v2_pf_cost_supplement_20261020.md)とpacketの`protocol_v2_supplement_manifest.json`/`paper_claims.csv`を作成した。採用済みv2本文、過去のprotocol manifest、論文主表・Figure2・CEM・元70件packetは変更していない。
70件の保存summaryと図2組を参照する資料で、新しい数値集計や研究採択を行ったものではない。70件はH6 signed30/H8両bias40、unique30条件、PRは旧確認union8/union6各n32に限定する。
主張できるのは、当該保存条件の単価置換で最良PFが2/70件変わり、q/PF次数とPR/PF方向が変わらなかったこと。これはprofile/runtime差等を含む条件付き感度診断で、真の母平均優位・全登録条件の頑健性・純モデル誤差・H6非相殺全域の完成を主張しない。独立P主表の再検証・差替えではない。

## 構造検証・保全・停止

GitHub read-onlyで94個の固定file/blob/size記録をlocal固定Git bytesへ照合した。元12signalの公開receipt一致、候補/時刻/入力identity、元120未確定行と30除外条件、1成分のliteral回収、後続D120未確定行、70件summaryの無変更転記を確認した。
再現入口は`python3 -B scripts/investigate_h6_noncancellation.py --check`。科学moduleをimportせず、成分差・envelope・適格性・shot・費用・順位のAPIを呼ばない。検査はprimary agent自身による構造・記録照合であり、別担当の独立レビューではない。
旧SOURCE/raw/科学契約/grant/ledger/marker/STOP・元70件・protocol v2本文・論文主表/Figure2が基点bytesから不変であること、追加文書とCSV/JSONの一致・参照・manifest SHAを確認する。共有環境・他process・OS/venvは変更しない。CPU/RAM/workerの独自上限は設けず、既知資料だけを読む単一のmetadata処理とした。

新しい公開範囲は未承認、push0。次担当**ユーザー**：段階Aの限定保存値解析と新規公開範囲を別々に判断する。次担当**Codex**：承認された範囲だけで新SOURCE/入力/算術runtime/one-shot契約を固定して実行する。今回の科学実行は0、詳細科学レビュー・最終evidence freezeは未開始。**mandatory STOP**。
