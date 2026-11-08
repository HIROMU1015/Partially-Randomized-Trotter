# 新A N5-B2：m17 joint scalar・logical policy・全consumer経路の静的接続

2026-11-08 JST、SERVER 2。固定基準 **f250bdb99a15804adfe2cfebc0ce16d5d8be6c1e**。Branch `codex/pr-reference-n5b2-static-20261108`。

**Stage A〜Eの静的調査を完了しmandatory STOP。候補Gは許容native入力クラスに対してSOURCE条件を明示できるが、現policyをそのままm17全supportへ使うことはできない。Logical precision・域・本番terminal経路が未資格で、新規合成の必然性も未確定である。**

新しい保証方式・precision・b・辞書・emitter・providerを正式採択していない。m17 RPE成功、保存N十分性、T優越は認定していない。

## 結論と今回明確になった接続

| 論点 | 静的結論 |
|---|---|
| 候補Gの開始点 | 全control＋compiler Aの後のjoint native。同じword／named gate／wire／順序を保つconsumerなら共通scalar因子分離を支持 |
| Compiler条件との関係 | Gのnative→wordの代数自体にはD固定性・frame共通性は不要。元SOURCEの信号・literal・旧diskへの移送にはCが残る |
| 最終測定 | 同じreset／readout／clbitとterminal sink、後置control／compilerなしを保つ必要がある。固定src/scriptsではm17本番のterminal経路は確認できない |
| 現policy | 有限停止は再利用可能。しかしdomain checkがC0 lookupより先で、m17 B′scalarの実literalをrejectする |
| Logical誤差 | Gはlogical wordのRZ近似差・quantizationを消さない。保存C0／43／85をm17の目標保証へ接続する証拠が不足 |
| Recordの所在 | r32中央はC0／85未対応。旧CT-0 mappingもrecord配列は空で、candidateというstatusだけをword取得済みとみなせない |
| 最小数値候補 | 保存r32中央4 positiveの現b36 quantization＋選択guard成分だけを評価する有限scopeを具体化。今回は未実行、別途明示承認が必要 |

## 固定科学条件

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event位相・X/Yを維持。

M17 **q131072／r32／88 bits、B′626／Reference630 shots/axis**。Energy1.5936e-4 Ha、failure.05、phase cap.4、alpha1/720は不変更。保存gate cap、worst-case RZ数、旧raw disk等は[N5-B0 M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)の引用で、新しい誤差和／RPE評価はしていない。

M7のb36・strict10^-11・L=82.49247654978845をm17へ採択していない。N4-Cのm7結果、N5-B1の160 fixtureは保存結果として再利用し、再実行しなかった。

## 文書と保存証拠

- [JOINT_SCALAR_FULL_PATH_CONDITIONS](JOINT_SCALAR_FULL_PATH_CONDITIONS.md)：SOURCE→control／native→word／emitter→最終測定の型付き条件、Cとの役割分担。
- [M17_LOGICAL_POLICY_REQUIREMENTS](M17_LOGICAL_POLICY_REQUIREMENTS.md)：C0優先、dyadic provider、actual guard、量子化、strict値と信号保証の区別。
- [M17_ANGLE_AND_ROLE_GAPS](M17_ANGLE_AND_ROLE_GAPS.md)：vacuum／scalar／event／中央／frame／D／inverseの保存coverageと7種の不足ラベル。
- [NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)：中央4 positiveの未実行数値scope、7作業単位の再利用・判断・承認・STOP。

[SAVED_METADATA](../../../artifacts/pr_reference_n5b2_static/2026-11-08/SAVED_METADATA.json)はm17条件、N5-A role分類、N5-B0 guard照合summary、保存CT-0の中央／scalar record欄、N5-B1判定の**引用・metadata照合**。[NEXT_NUMERIC_INPUTS](../../../artifacts/pr_reference_n5b2_static/2026-11-08/NEXT_NUMERIC_INPUTS.json)は次の候補入力を保存literalだけで固定したもので、到達集合や新policyの生成ではない。

## 保持すべき科学的限界

Gの条件下で `E_omega=exp(i*chi_omega)L_omega` と書ければ、各trajectory・axisの同じjoint測定は不変である。Chiをtrajectory非依存にする必要はない。しかしstrict operator距離は一般に0でなく、SOURCE vacuum／scalar／eventのrelative angle、logical guard、量子化、compiler／sampling／normalization等の旧誤差は残る。既存certificateや旧diskを減額していない。

固定providerはL内の有限binary64入力に対して停止・既存10^-11 certificateを返す証拠を持つ。M17のB′ scalar `-0x1.86e032ee41d40p+12` は実際に域外であり、現interfaceの適用限界は静的に確定する。一方、vacuum包絡32751.887444214277>Lやevent域外supersetは、各域外角のcanonical到達証明ではない。

C0の保存word、43 basic、85 certificate、emitter／consumerは再利用資産である。Guard上端超過は実誤差の下界ではなく、最適precision・新規合成の必要性を自動決定しない。Policyが完成しない段階でm17 wholeやRPEを実行しても、その欠損の代用にはならない。

## 実施内容と保全

[読取専用helper](../../../scripts/audit_pr_reference_n5b2_static.py)はN5-A/B0のfixed-Git抽出パターンと保存分類を再利用した。25 SOURCE関数blockをASTで記録し、m17の19 role行と既存metadataを抽出。既存runnerは科学処理やより広い抽出を含むため起動せず、この小さいstdlib helperで必要な引用だけを行った。関数bodyのexec／import、angle演算、guard・誤差和・価格の再計算はない。

[SOURCE_INPUTS](../../../artifacts/pr_reference_n5b2_static/2026-11-08/SOURCE_INPUTS.json)、[CHECKS](../../../artifacts/pr_reference_n5b2_static/2026-11-08/CHECKS.json)、[STOP](../../../artifacts/pr_reference_n5b2_static/2026-11-08/STOP.json)を保存。SOURCE→native／RPE関連関数は静的確認のみ。Metadata抽出は約0.59秒の小さい読取処理で、CPU/RAM cap・共有設定変更・他ユーザーへの操作はない。

新規合成、word取得、strict guard、provider／emitter／consumer実行、runtime fixture、native build/compile、simulation／sampling、precision探索、m17誤差・shot再会計、T価格評価はすべて0。正式共通運用指示2026-10-25、親／worktree AGENTS、PROJECT_MAPと既存研究資料を適用した。著者Codexの静的技術調査であり、GPT／Claudeの詳細科学レビューや正式な保証採択ではない。

旧SOURCE／raw／result／grant／ledger／marker／STOPと、統括兼Dのnative／q64／13-round adapter研究は変更・再実施していない。未公開adapterの意味論は推測せず、今回の条件へ無断移送していない。

**Mandatory STOP。次担当はGPT／ユーザー**：Gの今後の使用条件とlogical policy方針を判断する。必要なら固定中央sliceの数値診断を明示承認する。新b／precision／域／policyの採択、新規合成、m17再会計・価格評価へ自動進行しない。
