# H8 Phase C：B2完了結果の統合と5精度費用再会計の実行準備

**既存Phase C snapshotの実装を統合し、B2最終受入契約（登録S4全4軸PASS）に再結合した。実行契約（未承認・unsealed）と無効grantを固定した。人工37件はPASS（実launcher→driverの人工end-to-endを含む）。実H8保存値による費用再会計・最良候補・PR削減率の計算は0で、mandatory STOP。**
Phase Cは、ユーザーが下記SOURCE・plan digestに対する一回実行を明示承認すれば実行できる状態である。科学的な前提で不足しているものはない。

- 統合SOURCE：`29ed76544441d534ad99e5c708a2ae17eec78c18`（branch `track-a-h8-phase-c-integration-20261017`）
- 起点：B2完了結果 `fef2db7a79886ab682b652da7266b463c98c7d3a`（Phase C snapshotの起点 `3a450af0…` を祖先に含む）
- snapshot取込commit：`c38ec2cf…`（patch SHA `055848b6…` のcode・tests・referencesをbyte同一で取込み）
- Plan digest：`9e42fc76078568c9aaad4945539ffc46293214f71d4d36db1f0f53beea20d132`
- [準備bundle](../../artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/README.md)／[会計module](../../src/trotterlib/h8_phase_c_accounting.py)／[入力adapter](../../scripts/h8_phase_c_inputs.py)／[runtime](../../scripts/h8_phase_c_runtime.py)／[runner・launcher](../../scripts/run_h8_phase_c.py)／[prepare](../../scripts/prepare_h8_phase_c.py)／[tests](../../tests/test_h8_phase_c.py)／[入力契約v2](../../references/h8_phase_c_input_contract_v2.json)

H8直線鎖1.00 Å/STO-3G、保存tol-only DF28、16 modes/sector4900、T=.8。ε=.05/.01/.005/.001/.0005、allowance倍率1と10。比較対象は旧32決定論候補、強化184候補、PR確認union6（各n32）。費用はstate preparationを除くshot-weighted full-wrapper RZ、P=0。

## 1. 既存Phase C snapshotの確認と再利用

`/home/AbeHiromu/projects/h8-phase-c-preparation/phase-c-20261017-01/` を確認した。統合patchのSHAは記録値 `055848b6…` と一致した。会計module・入力adapter・runtime・runner・prepare・tests（27件）・references を取り出し、byte同一であることを照合して取込commitに固定した。取込直後の起点 `fef2db7a` 上で元の27件は無修正でPASSした。文書のhunkは、B2側の索引更新と衝突するため取り込まず、改めて記載した。snapshot本体は変更していない。

再利用した部分：正確な有理数の費用 `G = N_real*RZ_cosine + N_imag*RZ_sine`、実測優先の単価順位、保存通常会計の三状態適格性、旧32と184の別判定と削減率の変化（percentage points）、確認n32限定、cell IDによる同値処理、`UNRESOLVED_FULL_DOMAIN`、call上限（1,900行・20比較・model推定352軸）、sealed config・grant・fresh gate・one-shot path。

## 2. B2最終受入契約への再結合

旧Phase Cのgateは単一SOURCE `244c45e7…` のB2 terminalを要求していた。実際のB2完了は、保存3軸と新規1軸を最終契約で結合したものである。このため、`b2_full_pass_binding_v1.json` とその終了receipt projectionへ明示的に再結合した（SOURCE固定の黙示的緩和はしていない）。

|対象|軸|実行SOURCE|実測RZ|B1固定予測|判定|受入区分|
|---|---|---|---:|---|---|---|
|B0 p15/q1 S4|cosine|`244c45e7…`|168,353|1175933/7|誤差2538/1178471＜1/50|保存軸を最終契約で受入|
|B0 p15/q1 S4|sine|`244c45e7…`|168,353|1175933/7|同上|保存軸を最終契約で受入|
|B0+I p21/q8 S4|cosine|`2e604e4a…`|1,852,640|1852640|誤差0|保存軸を最終契約で受入|
|B0+I p21/q8 S4|sine|`f58da959…`|1,852,640|1852640|誤差0|新規完了軸|

gateが要求する条件：
- binding statusが `PASS_REGISTERED_FOUR_AXES`、保存3・新規1、軸ごとのSOURCEが登録順に一致すること。
- model・compiler・profileのSHA、厳密な誤差＜1/50（再計算して一致）、PASS、native exit0・reap、fingerprint一致、control誤差＜1e-8。
- 最終gate PASS、`single_original_attempt_completed=false`、過去STOP・未settle予約の保全、`Phase_C_authorized=false`。
- driver terminalが4軸（sine・保存B0 2軸・保存cosine）をbindingとdict完全一致で持つこと。launcher・witness native wait・admission terminalのexit0、追加compile0。

実bindingはこのgateをPASSした。private原本のterminal・final gateのhashはbinding記載値と一致し、B2 bundleのchecker（読取専用）も `PASS_SAVED_B2_FOUR_AXES_AND_COMPLETION_RECEIPTS` を返した。古いB2 runが一度で成功したとは扱わず、過去のSTOP・grant・ledger・markerは変更していない。

単価の優先順位：(1) B2登録S4軸の実測、(2) B1登録anchorの実測24軸、(3) 旧full PFの保存N=1実測（元runtime profileはunknownのまま）、(4) B1固定modelの推定。B0 p15/q1 S4は168,353 RZの実測を使い、予測値167,990.43で置き換えない。旧B0の来歴未確立の費用は使わない。未測定のS4候補は `model_formula=S4` とし、登録4軸だけで仮説が通過した旨を付ける。S2は `model_formula=S2` とし、prefix14〜21外を外挿と明示する。いずれも全格子の誤差保証はない。

## 3. 混入防止と完全性

- 会計行は保存通常signed bias（`bias_kind=USUAL_SIGNED_SAVED`）だけである。非相殺診断の行や欄（`bias_kind`、`usual_saved`、`shot_ratio_to_usual` など）を含む入力は拒否する。非相殺shot診断 `3f902229…` は参照のみで、入力roleに含めない。
- 決定論184（旧32＋追加152）とunion6の各5精度×2倍率＝1,900行が揃わない場合、重複・倍率違い・n8混入はSTOP。未解決の候補がある集合は最良候補とPR比をnullにし、候補を除外しない。旧32の結果は追加候補の状態に依存しない。
- 入力12 role（D0会計・D0 terminal・B0I会計・B0I file receipt・Phase A・B1 model/terminal/native exit/anchor・B2 binding/receipt・registry）のbytes/SHAとGit objectを固定した。D0 terminal→D0会計、B0I receipt→B0I会計、B1 native exitのjoinも照合する。

## 4. 統合時の変更と理由

|変更|理由|
|---|---|
|B2 gateを最終4軸契約へ|SOURCEが軸ごとに異なるため|
|入力契約v2・Phase C用registry file|B2 roleが変わった。`references/registration_v1.json` は別系統で別内容のfileとして存在する|
|非相殺行の拒否・lineage join・S2/S4表示|混入防止と来歴照合のため|
|AS 2 GiB・CPU 600秒・wall 900秒・2 idle coreを廃止|現行規約によりエージェント独自の資源上限を設けないため|
|旧Phase A observerを使わず、wait4 launcherへ|旧observerはcgroup PSIが0を超えるとSTOPし、B2を複数回止めた。B2最終SOURCEもPSIを観測のみに変更している。PSI・OOMは記録し、自job以外の負荷で止めない|
|draftはunsealed・未承認|sealing（最新の保護一覧の取得）は承認後の実行手順に属するため|

科学条件・精度・倍率・shot算術・適格性規則・選択規則・費用式は変えていない。

## 5. 人工検証と実行拒否

[限定検証](../../artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/limited_validation_v1.json)。人工37件（snapshot由来27件を新B2形式へ適合、新規10件）が、固定Python3.11.0rc1とPython3.12.3の両方でPASSした。新規の検査：cosine/sine別単価、B2の軸別SOURCEと単一SOURCE前提の拒否、S2/S4表示、非相殺行の拒否、旧32の独立性、lineage join、資源上限なし、入力pin、B2 job検出の誤検知防止、end-to-end。
end-to-endでは、SOURCEを一時git repositoryへ複製し、人工のbytesを入力にして実launcherから実driverまで動かした。COMPLETE、wait4 native exit0、最終gate PASSを確認した。同じoutputの再起動は `ONE_SHOT_ALREADY_SPENT`、SOURCE変更後の起動は `C_SOURCE_CLOSURE_CHANGED` で止まった。
実入力でのprepareは、pinned入力のbytes/SHAとGit object、B2 gateの照合だけを行った（科学行は解析していない）。生成したunsealed configと無効grantを実launcherへ渡すと、副作用の前に `NEW_SEALED_C_USER_GRANT_REQUIRED` で拒否された。
これは私自身のlocal検証であり、独立review・CI・外部再現ではない。subagentは0。

## 6. 実行条件と承認事項

[実行契約（未承認）](../../artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/execution_contract_UNAPPROVED_v2.json)、[無効grant](../../artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/grant_INVALID_DRAFT_v2.json)、[承認条件](../../artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/approval_conditions_v1.json)。

ユーザーが承認する対象は、SOURCE `29ed7654…`・plan `9e42fc76…`・B2 binding `fef2db7a` に基づく、保存値5精度の通常signed bias RZ費用再会計（P=0）の一回実行である。出力は次の通り：各ε・倍率での旧32と184の決定論最良、union6の有限標本費用、PR削減率とその変化、実測/model単価の区分、未解決・不適格・欠測の影響。非相殺条件の費用比較、B2再実行、再fitは含まない。
承認後の手順は次の通り。
1. SOURCEを正確にcheckoutする（runtimeはHEAD＝SOURCEを要求する）。
2. `prepare` をsealで再実行し、最新の全worktree・保護証拠を保護一覧に取り込む。
3. 承認文言を含む新しい一回限りgrantを発行する。
4. launcherを起動する（fresh gateは30秒以内、wait4、native exit receipt）。
5. 結果を保存し、mandatory STOPする。

## 7. 保全・非干渉と未検証事項

作業は独立したclone（sparse checkout）で行い、登録worktree・共有ref・B2/M1-Sは変更していない。保護snapshot8件の保護file変更は0。worktreeの差分は、既に記録済みの16:43以前の非相殺準備worktreeの件だけである。作業開始時に実行中のjobはなく、広域検索はしていない。自cgroupのPSI累積増分は0。

未検証事項：
- 実データでのsealing・launchは未実行。新しいlauncherが実サーバー上で動くことは、人工end-to-endでしか確認していない。
- S4 modelは登録4軸だけで通過しており、全格子の保証ではない。
- full PFの旧profileはunknownのまま。
- PR費用はn32の有限標本平均と標本SEであり、未観測のTaylor層は上界を与えない。
- 経験的allowanceであり、厳密bias保証はない。

## 8. 公開と次の担当

結果は独立cloneのbranch `track-a-h8-phase-c-integration-20261017` にcommitした。サーバーにはGitHubの認証情報がないため、pushとremote SHAの照合は引継ぎ用bundleで行う。有効grant・private入力・config原本は公開しない。

**ユーザー**：(1) 引継ぎbundleを公開し、remote SHAを照合する。(2) 上記SOURCE・planに対するPhase C一回実行を承認するか判断する。
**Claude Code**：承認があれば、上記の手順で一回だけ実行し、mandatory STOPする。今回はここでmandatory STOP。
