# PR改善：M1重複排除・C-S同位置residualの最小接続

2026-11-08 JST／SERVER1（GPUA100-gleap-01）。**M1の既存接続を作り直さず、指示された分岐でC-Sの最小接口へ進んだ。人工9件PASS、新科学計算0。科学入力投入前・公開照合後mandatory STOP。**

## 到達点

1. **M1の新しい操作的差分は今回特定できなかった。** S/P辞書・global collect・selector・合法共有は既存。さらにH4 matched Stage1–4/finite-bit診断も存在するため、旧preflightだけから「未実装」と判断してadapterを重複作成しなかった。M1全般の無効性や正式NO-GOではない。
2. **既存証拠をその条件内で再利用した。** F1/F2のcorrectness、F2のPP低単価とN/weighted_costs=null、P3の条件付き会計・実装/価格tailのUNKNOWN、PS48・Parent・研究Cの停止を保持した。
3. **9作業単位の重複実施を回避した。** [D01–D08/D13](EXISTING_EVIDENCE_AND_DUPLICATION_CHECK.md)は元結果/実装を参照し、再代数・再回路・再価格・再会計・旧test全面実行をしなかった。実験や旧testの件数を推定した数ではない。
4. **C-Sの小さい新接口だけ実装した。** [same_position_residual.py](../../../src/trottertracks/algorithm_codesign/same_position_residual.py)は、既存exact Angleとcoherent providerの間で、coarse＋finite residualを元slotへ挿入する。新law/optimizer/DF/合成器/価格器/会計frameworkはない。
5. **新規人工9testが初回PASS。** 非可換隣接順序、負角/π/identity/controlled相対phase、finite meanとexact回転の区別、S0/S1/S2、異常bindingとRNG/合成の非起動を確認した。旧SOURCEは無変更で、旧suiteの大量再実行はしない。

## 成果物

| 文書 | 内容 |
|---|---|
| [EXISTING_EVIDENCE_AND_DUPLICATION_CHECK.md](EXISTING_EVIDENCE_AND_DUPLICATION_CHECK.md) | 5分類の重複表、既存PASS/負結果/STOP、P3後続結果、F2版差 |
| [ALGORITHM_IMPROVEMENT_DELTA.md](ALGORITHM_IMPROVEMENT_DELTA.md) | C-S同位置接口、既知法との関係、利益/相殺と強baseline |
| [IMPLEMENTATION_AND_ARTIFICIAL_VALIDATION.md](IMPLEMENTATION_AND_ARTIFICIAL_VALIDATION.md) | 再利用SOURCE/blob、新規9test・実行範囲・未保証事項 |
| [NEXT_MINIMAL_SCIENCE_CONTRACT_DRAFT.md](NEXT_MINIMAL_SCIENCE_CONTRACT_DRAFT.md) | V0最大2局所slotのstrict技術資格案、未確定条件と後続N×full価格比較の境界 |
| [SOURCE_AND_CHECKS.json](SOURCE_AND_CHECKS.json)、[ARTIFICIAL_TESTS.log](ARTIFICIAL_TESTS.log)、[STOP.json](STOP.json) | 固定参照・技術保全・人工結果・mandatory STOP |

M1の既存科学条件は異なる：F2は人工4mode/2粒子/occupied(0,1)/LD1/残差2fragment/T=1/5/q1/r2/K0、分子geometry/basis/rank非該当。P3はlinear H4/1Å/STO-3G/tol-only1e-8 rank10/T_eff-V_square/T=.8/S2/q8/r4/K4。A1 N1は保存linear H4/1Å/STO-3G/legacy DF12/LD3/m0,q1,δ=.02,r1,K2/threshold0・全4電子sector。これらの価格・N・state・splitを相互移植しない。

## 次の判断を変える証拠

C-Sの操作は既存のcoarse/residual恒等分解の接続で、独立新規性や量子資源優位は未実証。**次に必要なのは、実H4のeligible role/slot・exact角度を同じcoarse政策とstrict providerへ接続でき、追加normalization/rangeを払っても局所tradeoffを検討できるか**という限定資格である。

草案V0はD/R各1・最大2局所specのmetadata/strict局所検証に絞り、実state・Born・RNG・shot・full compile・N/Gを含めない。実slot/role、coarse政策、residual r/K、strict合成precision/toolとerror予算、provider adapterは実行前に確定が必要。K_res2を選ぶなら最大8positive synthesis keysという有限案を示したが、正式採択していない。

V0が成立し、その情報に検証価値がある場合だけ、別許可でq1のS0/S1/S2×X/Y、same target/精度/成功条件・必要N×full native価格へ進む案を残した。全round RPEの正当性・energy bridgeを作り直さず、既存接続を必要範囲で利用する。C-N/C-PやMultilevelを同時実装しない。

## SOURCE・公開・保全

- base=`62c450a6c22fc6cf228bd2fbaabda3c36a7cf169`、branch=`codex/pr-o1-m1-delta-cs-preparation-20261108`、worktree=`/tmp/pr-o1-m1-delta-cs-preparation-20261108`。公開commitが今回の技術SOURCEで、科学実行SOURCEではない。
- 旧src/scripts/testsは無変更、新module/testと資料・索引・技術status登録だけを追加する。元科学result/status・過去の判断履歴は書き換えない。今回指示のSCV一区切り・Product保留と改善着手順位は新しい追記として区別する。
- 元main HEAD/statusと3,087既存ファイルを開始前後で照合。旧SOURCE/result/raw/grant/ledger/marker/STOPを変更せず、並行他研究のworktree/process/cacheへ操作しない。保全の実測範囲はSOURCE_AND_CHECKSを正本とする。
- 新科学計算、保存値の科学的再解析、実H4 operator/state/Born、sampling/RNG、実回路/compile、N/G、旧run再開、重要科学レビュー、C1凍結解除は全0。共有OS/venv/設定への変更0。
- GitHub APIを用いて専用branchへ公開し、remote SHA/parent/tree、変更全ファイルの本文/blob、固定資料リンクを照合する。最終SHAは公開commitと最終チャットを正本とし、自己参照SHAのために旧SOURCEを再変更しない。

**Mandatory STOP：今回の最小技術接続と人工correctnessは完了。次の科学scope・未確定条件をユーザーが採択し、固定科学SOURCEへの明示実行許可を与えるまで、実入力・新合成・科学再解析を開始しない。**
