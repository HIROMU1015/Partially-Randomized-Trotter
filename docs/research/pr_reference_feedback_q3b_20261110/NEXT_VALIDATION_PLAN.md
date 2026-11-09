# Q3-B後：native保証と費用接続の最小判断

2026-11-10 JST。**未実行の後続案。** B0/B1/B2の固定モデル内PASSは[REPORT](REPORT.md)、証明条件は[BRANCH_AND_NUMERICAL_GUARANTEE](BRANCH_AND_NUMERICAL_GUARANTEE.md)。Q0・Q2・count-domain監査を理由なく再実行しない。

## 1. 残る一つ目の判断：native全contextへの保証移送

再利用SOURCEはQ1／新Q3-B adapter、Compiler A／Qiskit1.3、Q2の全16 pass trace、今回の孤立symbolic readout結果。物理SOURCEはf2638a78のB′／Reference catalogue・誤差会計と、固定V6-Q producer／schedule／consumerを保つ。

まず静的に、次を全targetについて証明できるか判断する。

1. historyごとに一度生成した同じbinary feedback parameterを、全trajectoryへ渡す。
2. P→U1→RZの符号・parameterがwhole jointの全contextで保持される。lower後にcontrolを追加しない。
3. 新readoutによってbodyのloweringや旧局所誤差chargeが未資格のものへ変わらない。旧H／Sdg等のbudgetの再利用・除去可能部分を正しく区別する。
4. 追加誤差をtrajectory共通の角度差として扱える。一般のoperator／signal誤差が残る場合はraw diskの単位で再接続する。
5. Referenceのreset prefix、actual inverse、vacuum／scalar、canonical lawとfresh bitsを保つ。旧X/Y consumerのaxis名だけをQへ差し替えない。

孤立symbolic testは2のreadout単体を支持するが、3・4を含む全m17 contextのcertificateではない。静的な証明で閉じる場合は、不要な回路再計算を要求しない。新しい重要保証の正式な採否はGPT／ユーザーへ戻す。

## 2. 新しい科学callが必要な場合のscope固定

静的移送が閉じない部分に限り、旧B′ `Bprime_local_catalogue.tar.xz`とReference `local_catalogue_boundary.tar.xz`の末尾・wrapper境界templateを優先して使う。追加するtemplate identity、context、feedback angle family、compile／operator checkの回数、既存diskへのchargeを**事前に列挙し、別途の明示実行承認を得る**。

今回まだ固定できていないのは、どのbody-lowering seamが未資格となるかという点。その判断を飛ばして恣意的なtrajectory／角度gridやm17 wholeを採用しない。有限角の数値PASSだけで全angle・全supportを保証しない。新compiler・precision・word policy・sampling lawへ変更が必要ならSTOPする。

## 3. 340 shotsの採用と費用へ進む条件

native条件が閉じた後に、Q3-Bの全prefix受理・δarith bound・B2余裕、Q0の未知ρ／bias、historyで条件付けたfresh trialsを接続する。追加の受理／実装失敗確率を無料で除かず、phase cap／failure .05／energy精度を維持する。現時点では正式な実m17成功保証を採用しない。

次に価格が必要となる場合は、f2638a78の`pr_reference_bprime_accounting.price_bounds`、Reference `pr_reference_transfer_accounting.expected_rz_bounds`／cost bound、depthの既存frontierとV6のphase-provider条件から再利用範囲を決める。新Q wrapperと全lawへ保証が移らない旧価格をそのまま使わない。

資源jの判断式は`340 μQ,j+Aj < 626 μX,j+626 μY,j`。Q2のq1/r1有限価格をm17へ外挿せず、AjとμQ,jを未知のまま0にしない。B′にも同じQ機会を与え、強Full／Truncatedとの公平な比較を後続判断に残す。T・depth・RZ・CX・qubits・classical費を合算しない。

## 4. 次担当・STOP

**GPT／ユーザー：** 今回の数値保証を科学的に採用するか、native移送のどの未資格を次に扱うか判断する。必要な科学callだけを固定して明示承認する。

**Codex：** 現Q3-Bの成果公開・remote照合でmandatory STOP。旧SOURCE・raw・結果・STOPを維持し、追加compile／sampling／RPE／m17価格／全18round／Clifford+T／正式科学レビューを自動開始しない。
