# 新A N5-A：18round Clifford+T移送の静的監査

2026-11-08 JST、SERVER 2。固定基準 **0af90ca199361810f4efbf0ae86422738f4d7f12**。Branch `codex/pr-reference-n5a-static-20261108`。

**静的監査を完了しmandatory STOP。m7は既存の条件付き資格を維持するが、全18roundのCT実行・RPE成功・総T優越は未資格。** 新しい数値誤差、shot、precision、価格、期待値、改善率を計算していない。旧STOPや科学条件を解除していない。

主な不足は次の4点。

1. **域**：m0〜7の保存vacuum包絡は既存辞書のL以内。m8以降の保存包絡はLを超える。これは域内証明の不足であって、到達角の域外証明とは区別する。一方m11以降にはB′の保存scalar literal自体が域外で、m13以降にはevent位相の安全supersetにも域外keyがある。
2. **精度**：10^-11はm0〜9の保存gate十分条件には収まるが、m10以降はそのまま適用できない。CT-0の全18round共通一様精度の保存十分条件は **6.902168833913054e-16**。これは必要精度・最適精度でも新たな採択値でもない。
3. **Emitter**：m16 Reference、m17両armでは、保存C0 emitter項だけでも保存gate十分条件を上回る。新しい基本wordを高精度で合成するだけで全問題が解消するとは言えない。辞書の量子化、基本guard、emitter、shellを分けた再資格が必要。
4. **総T**：m7の価格anchorと6072<173790は他roundへ移せない。さらにm17の保存shotsはB′626／Reference630で、1-shotの共通frame価格取消しだけからshot重み付き総T差は得られない。

## 成果物と固定条件

- [ROUND_QUALIFICATION_MATRIX](ROUND_QUALIFICATION_MATRIX.md)：全18roundのq/r/bits、保存shot、RZ数、旧disk、追加誤差余地、保存precision条件、vacuum域、資格一覧。
- [CROSS_ROUND_REUSE](CROSS_ROUND_REUSE.md)：C0、43基本word、85 certificate、consumer、C_phi、SOURCE構造の再利用範囲。
- [RPE_AND_T_GAPS](RPE_AND_T_GAPS.md)：RPEとT-countを閉じるための不足証拠、両軸max、shot重みとframe取消しの注意。
- [NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)：小さい次scope候補、担当・承認・STOP条件。提案のみ。

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAPを維持した。Energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720を変更せず、q/r/bits/shotは各roundの保存契約を引用した。m7の73 bits・199 shots・10^-11を他roundへ採択していない。

## 既存結果の保持と新しい解釈の範囲

[N4-C](../pr_reference_n4c_accounting_20261108/REPORT.md)のm7全support・199/axis十分条件と、N1／N3／N4に基づくm7 pointwise T十分条件は、compiler C・共通policy・C_phi/runtime・保存raw信号モデル等を条件に維持する。今回これらを再計算せず、無条件保証へ昇格もしていない。

43基本wordの取得済みという事実、L以内・10^-11での有限停止／strict保証と、より厳しいroundの要求に適合することは別である。C0がliteral一致しても、guard＋emitterが新しい十分条件を満たすかは別に照合する。旧CT-0に残るlogical guard候補も、実consumer・C0優先・新価格anchorまで資格済みとは扱わない。

今回の比較は保存field／literal／域／guard／capの静的照合だけ。新しいerror和、sqrt、disk伝播、phase/radius、shot整数判定、T価格和・極値・期待値は評価していない。m7以外のCT RPEのPASS／FAILを新たに判定していない。保存条件に入らないことを実RPEの必然的失敗や新規合成の必然性へ読み替えない。

## 統括兼Dとの分離

Native CX／RZ／depthの新比較、q64等のnative検証、回路再設計、13-round native adapter、depthモデル拡張、研究全体の優先順位判断は実施していない。RZ worst-case数はCT-0保存値を引用しただけで、新しいnative資源評価ではない。

今回参照できた固定SOURCEはN4-Cに至る既存系列。13-round native adapterについては、本依頼に固定commit/pathがなく、このbaseにも直接の証拠を特定できなかった。GitHubの関連branch名も限定照会したが、branch名検索は他チャットの未公開成果や資料の不存在証明ではない。不可欠でないため取得待ちにせず監査を完了した。そのadapterへ今回のconsumer／誤差保証を勝手に移送していない。

## 検証・再利用・保全

[読取helper](../../../scripts/audit_pr_reference_n5a_static.py)はN4-Aのmetadata処理を再利用し、CT-0 positive関数だけを変更していないASTから実行した。標準ライブラリのみで保存JSON／圧縮metadataを読み、18roundのfieldを抽出し、literalの同一性、域、保存certificateと保存capの大小を照合した。新しい科学上界の算出や科学moduleのimportはない。

[ROUND_METADATA](../../../artifacts/pr_reference_n5a_static/2026-11-08/ROUND_METADATA.json)と[ROLE_METADATA](../../../artifacts/pr_reference_n5a_static/2026-11-08/ROLE_METADATA.json)が抽出値とmetadata分類の正本。Role key数は保存supersetの件数で、物理RZ数／canonical到達数ではない。Inverse-D supplementとsigned境界台帳の重複を誤差数へ足していない。Vacuumは有限key全列挙をせず、保存域のまま扱う。

[SOURCE_REFERENCES](../../../artifacts/pr_reference_n5a_static/2026-11-08/SOURCE_REFERENCES.json)、[CHECKS](../../../artifacts/pr_reference_n5a_static/2026-11-08/CHECKS.json)、[STOP](../../../artifacts/pr_reference_n5a_static/2026-11-08/STOP.json)に出典・技術検査・終了を保存した。正式共通運用指示2026-10-25、親／worktree AGENTS、PROJECT_MAP、既存研究概要と関連固定資料を適用した。著者Codexの静的技術監査で、独立詳細科学レビューや正式な新規性判定ではない。

SERVER 2の通常権限で小さい読取処理を行い、独自CPU/RAM cap、共有設定変更、他ユーザー・他チャットのprocess／ファイル操作を行っていない。旧SOURCE・raw・result・grant・ledger・marker・STOPは不変更。**新規科学数値計算・合成・guard・provider・build/compile・simulation・sampling・価格計算は0。**

**Mandatory STOP。次担当はGPT／ユーザー**：長roundのprecision／emitter／域の不足をどの限定scopeで解消するか判断・承認する。Codexは[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)の候補を自動実行しない。
