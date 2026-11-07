# C-S同位置residual：V0技術SOURCEの準備結果

2026-11-08 JST／SERVER1 GPUA100-gleap-01。

**技術状態：SOURCE_READY_WITH_SCIENCE_PREFLIGHT。科学条件：SCIENTIFIC_DECISION_REQUIRED（候補一組、未採択）。実H4のscience callは0。公開照合後mandatory STOP。**

## 完了した接続

- 保存N1/selected policyと旧認証のD metadataからfirst D/R joint局所specをbindingする遅延APIを追加。実H4では未呼出。元H4/1Å/STO-3G/legacyDF12/8system/全4電子参照/L_D3/m0,q1/δ.02,r1,K2/threshold0、actual Q/selected fullを維持する。
- exact π+radian／scaled atan、positive key、実adjoint、位相保持fast path、独立strict interval guard、既存native局所会計を接続。旧projective T shortcutをstrictと誤認しない。
- 一つの共通coarse案、r_res1/K_res2、最大8positive keys、局所finite/native/error cap、attempt耐久記録・STOP仕様を具体化。精度・energy/全量子費用を保証したものではない。
- 新規人工18件PASS（0.136秒）。旧9件はSOURCE不変で再利用し、再実行なし。14→18件への拡張後、FIXED partnerのrole明示について対応1件だけ追加確認。計3 invocation、失敗・science retryなし。
- 固定13入力のbytesと20 SOURCEファイル、実runtime/tool依存を読むstatic-only preflight PASS。合成用packageは専用/tmp overlayへ配置し、共有venv/OS/cacheは無変更。provider import確認のみ、backend synthesis0。

新コードはcs_v0.py、cs_v0_native.py、静的preflight、追加test。元same_position_residual.py、Angle、CanonicalGenerator、Parent operator/phase/weight/guard core、DF/RTE sourceは無変更。F1/F2/P3/Parent/S4/SP/PS48/SCV/C1の再実験や重複suiteを実行していない。

## 一組の推奨条件と未解決事項

| 項目 | 次回の提案／残る確認 |
|---|---|
| 選択 | 保存canonical first D、first positive R/order0、それぞれlower済joint factor0のみ。最大2spec。適格性・実angle取得はV0待ち |
| coarse | π保持、nearest 1/256rad/tie-away、全arm/D/R共通。実H4への適用なし |
| residual | exact差、ordinary r_res1/K_res2/root256/proposal160。最大2×2event、sample0 |
| strict | pygridsynth2.0.0/seed0/dps100/up_to_phase=false、ε_native=10^-6、positive≤8key、負角adjoint |
| 局所予算 | finite≤10^-12、各strict≤10^-6、合成local bound≤3×10^-6を候補cap。full budgetはUNKNOWN |
| 残る科学取得 | 1binding、2局所比較、2finite mean/error、≤8strict key、2局所literal価格。今回全0 |
| 保証外 | 旧engine上界のC-Sへの継承、full controlled列/measurement、joint sampling独立性、N/G・資源削減・一般的優位性 |

技術接口は完成したが、実H4の適格性とstrict acquisitionは初回V0でしか確認できない。V0を実行するには上記政策・局所error cap・有限scopeをユーザーが採択し、本公開SOURCEへの明示的な科学実行許可を与える必要がある。許可の証明書/grant発行は不要。次担当はSERVER1 Codex。今は入力binding/数値取得を開始しない。

## 成果物と保全

- [V0_TECHNICAL_BINDING.md](V0_TECHNICAL_BINDING.md)：保存schema・role・slot・phase・再利用。
- [V0_COARSE_RESIDUAL_NATIVE_POLICY.md](V0_COARSE_RESIDUAL_NATIVE_POLICY.md)：一つの政策・誤差・8keyの条件。
- [V0_ARTIFICIAL_CORRECTNESS.md](V0_ARTIFICIAL_CORRECTNESS.md)：追加18件・旧9件の再利用・静的preflight。
- [V0_SCIENCE_EXECUTION_DRAFT.md](V0_SCIENCE_EXECUTION_DRAFT.md)：Stage A–D・最大call・承認範囲・STOP。
- SOURCE_AND_CHECKS、NATIVE_PROFILE、TOOL_IDENTITY、test log、STATIC_PREFLIGHT、STOPを同directoryに保存。

base=c077de79713f793939f7e33c18ca81f4c819588c、branch=codex/cs-v0-technical-preparation-20261108、worktree=/tmp/cs-v0-technical-preparation-20261108。固定入力/既存研究に書込みなし。main HEAD/status/3,087既存fileを前後照合し、旧SOURCE/result/STOP/grant/ledger/marker、SCV raw、C1凍結、他process/cacheを保全する。古いmanifest entryを置換せず技術entryだけ追加。

固定参考24件の公開blobとlocal Git objectを照合済み。公開はGitHub APIで専用branchへ行い、最終commit/parent/tree・全変更ファイルの本文/blobを照合する。最終SHAは公開commitと最終チャットを正本にし、自己参照用の再commitを増やさない。

**Mandatory STOP。新H4演算子/角度/coarse/residual/finite mean/strict acquisition/実compile/価格、state/Born/RNG/shot/N/G、V1以降、旧研究再開、重要科学レビューは実施しない。**
