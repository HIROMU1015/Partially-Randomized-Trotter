# N4-B終了後：必要最小限のRPE誤差資格scope案

2026-11-08 JST。[REPORT](REPORT.md)。**提案のみ。未実行で、今回の実行承認へ含めない。N4-B完了でmandatory STOP。**

## 1. 次に判断すること

**GPT／ユーザー**：共通policyとconsumerの条件付き保証、および独立したshell項を採用して、固定m7/q128/r1・B′／Reference×X/Yの保存入力再会計を別途承認するか判断する。科学レビューの開始は自動ではない。

**Codex**：その明示承認があった場合だけ、同じ科学条件・precision・compiler・SOURCE・既存199 shots/axisの範囲で実行する。全18round、T総計、T-depth、sampling、合成、guard、native compile、最適化へ拡張しない。

## 2. 使う固定入力

- 本N4-B公開commitのconsumer SOURCE、85 certificate、C0優先検証、局所／保存whole検証、phase会計。
- [CT-0 PRECISION_FEASIBILITY](../../../artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json)のm7のみ：両armのX/Y worst-case native RZ数、物理component上界、元信号半径、追加diskと保存Nの条件。
- [CT-4B ERROR_AND_SHOTS](../../../artifacts/pr_reference_ct4/2026-11-07/ERROR_AND_SHOTS.json)のq128／uniform10^-11条件式。旧finite-pilot価格／誤差を全supportへ取り違えない。
- N4-A ROLE_COVERAGEとSOURCE閉包、CT-5B compiler条件C、N3の固定43基本word／C0全299／C_phiの既存資格。
- 同じ元H ground、normalization、測定、failure.05、phase.4、alpha1/720、199 shots/axis、energy1.5936e-4 Ha。

## 3. 残った計算・証拠

1. 全必要roleが固定C0または辞書L内に入るSOURCE閉包と、compiler条件Cの適用範囲を再利用する。85は保存numeric supersetで、vacuum reachable keyの全列挙ではない。新たなreachable-key生成は不要。
2. 各物理RZのstrict最終上限10^-11を使う保守的joint spectral和、または既に固定された証拠だけでより正確に組める和を選ぶ。新precision／価格取得機会／候補探索を導入しない。V3期待多重度とsigned境界補正をworst-case数へ代用しない。
3. [PHASE_ACCOUNTING](PHASE_ACCOUNTING.md)の初期phase設定とsegment joinを、各RZ certificateとは別のshell項として1回追加する。SOURCE上の全supportでもprefix reset＋1 unitary body＋末尾readoutとなる条件を確認する。新しいcopy/QPY/to_instruction等が要るなら先にSTOP。
4. Full joint spectral→raw synthesis diskへ移し、既存compiler／vacuum diskを同じ単位で1回だけ足す。古いπ差やvacuum算術を新emitter側で再計上しない。
5. 合算diskから位相・半径へ変換し、保存199、alpha1/720、phase.4、failure.05を検算する。十分なら「この固定m7条件で199を保つ」と限定し、不足なら不足として保存する。Shot／alpha最適化を自動開始しない。

今回未実行の主な不足は**shellを含む全位置のjoint synthesis diskと、旧diskへ加えた後の199-shot数値検算**。Local correctnessとword存在をRPE成功資格へ読み替えない。強い一般的新規性、全support実価格、18round総T優越の認定も対象外。

## 4. STOP

域外／C不適合／phase経路の不足、既存SOURCE誤差の欠損、予算不足、未承認の合成・guard・compile・新key・条件変更が必要ならその差分を保存してSTOP。固定再会計を完了した場合もmandatory STOPし、GPT／ユーザーへ返す。N2／SK／長round取得／18round／他Trackへ自動進行しない。
