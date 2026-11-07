# N5-A：N3／N4成果の再利用範囲

2026-11-08 JST。[REPORT](REPORT.md)、[round表](ROUND_QUALIFICATION_MATRIX.md)。以下は既存SOURCE／結果からの静的整理で、拡張policyの採択ではない。

## 1. Wordと停止性

| 資産 | 再利用できること | 自動移送できないこと |
|---|---|---|
| C0全299 word | Literal一致、同じstrict要求・保存guard＋emitterを満たす位置で元wordを優先。旧word／価格／hashを保全 | 他precisionへの適合、別recordへの無断置換、新policyでの旧価格anchor |
| 固定43基本word | 完了済み取得、保存guard、exact W、全文字列actual adjoint。域内・既存10^-11構成に新規基本合成不要 | より広いL、異なるb/j範囲、より厳しい最終精度への完成policy資格 |
| N4-Bの85 certificate | 固定frame84＋B′ b2。Literal一致するroleへ同じpositive wordを返す根拠 | 全vacuumの列挙、r>1の全event位相、round別scalar／中央角の未実行certificate |
| 全vacuum域のprovider証明 | **L=82.49247654978845、strict10^-11、固定C_phi**なら有限bit走査・連結でbackendなしに停止 | 域外の入力。既存provideはdomain判定で拒否し、C0照合より先に制限を確認する |
| 旧CT-0 logical record候補 | 保存guard／方式／literalが一致する候補の所在 | Emitter込みの資格、C0優先との整合、採用価格、全support成功 |

オンラインproviderは新規pygridsynthを呼ばず、固定43-bit走査と有限文字列処理で終わる。**有限にrejectすることと、有効なstrict wordを返して回路を実行できることは別**。新しい基本角やprecisionを必要とする方式へ変える場合、保存43件の成功を新しい合成入力の成功／停止保証へ転用しない。CT-2／CT-3Aの一般fallback未資格・STOPも維持する。

Angleはcompiler後のsigned binary64 literalが正本。理想π snap、近接hex統合、無条件2π同一視、SOURCE理想角からの再生成、合成後のcontrol追加を行わない。C0の旧価格を保つために不適合precisionのwordを使うことも認めない。

## 2. 域とprecisionを分ける

保存vacuum包絡はm0〜7でL以内、m8以降はLを超える。域を超えた包絡は「この証拠ではL内を保証できない」という結果であり、実到達角が必ず域外という証拠ではない。新しいreachable-key列挙や包絡の数値再評価は行わない。

さらにB′ scalarにはm11以降、固定域外の保存numeric literalがある。ScalarはSOURCEのq依存の固定項で、random eventの到達性だけで除外できない。B′ event位相もr16/r32の安全supersetに域外keyがあるが、各keyのcanonical到達性は別途未確定。Referenceのsymbolic vacuumとこれらを一緒に0／平均値へ補完しない。

CT-0の各roundのuniform_gate_error_strict_upperは、**shellを今回加え直す前の、保存Nを保つ一様gate誤差の十分条件**。要求precisionを既に採択した記録でも必要条件でもない。10^-11との比較ではm0〜9が保存cap内、m10〜17は外となるが、それだけでRPE FAILや「必ず新規合成」と判定しない。

全18round共通の保存十分条件6.902168833913054e-16へ既存辞書を移すことは未資格。N3-B0保存Q_upperは約5.1448789686e-12、基本割当は各10^-13、辞書emitter項は45/140737488355328、C0 emitter項は1/140737488355328で固定されている。基本wordのguardだけを小さくしても、量子化とemitterを消したことにはならない。

保存C0 emitter項だけとの大小照合では、m16 Reference・m17両armのcapを上回る。これは**保存された保守的certificateが、その一様十分条件へ直接入らない**という意味。実float誤差の下界や、機械精度による不可能性を証明したものではない。より正確な保存certificate／allocation／emitter保証の選択は次の科学判断へ返す。

## 3. Frame・中央・scalar・event

| Role | 移送の根拠 | 残る条件 |
|---|---|---|
| Dとactual SOURCE inverse | h=.02固定のSOURCEと保存template。qで多重度・境界構成が変わる | C下のtemplate／literal閉包。Round別価格は未評価 |
| RTE basis frame | 固定DF registryとbasis pair、N4-Bの84 miss解消 | 新しいtarget精度に既存certificateが適合するか。価格取消しと誤差を分離 |
| Central RTE | 同じrなら同じh/rに対するliteral・同じwordが候補 | rが変わる境界でhalf/fullの実literalと価格を確認。r1のd0/d2を流用しない |
| B′ scalar | q依存の保存literalを読む | m0〜6でもm7とは異なる未call keyがある。m11以降は現providerの域外 |
| B′ event | r1のb2は既存85の内数 | r2以降のsupersetへの接続、r16/r32の域外候補。価格非負をword資格の省略へ使わない |
| Reference vacuum | 保存SOURCE recurrenceと各roundの包絡 | m8以降の域、目標精度、取得／停止・価格上界の不足 |

m0はq1の特別なF/G分割、m1以降はq>=2のsplitである。SOURCE inverseと負native RZのword inverseは引き続き別物。同じbasis pairの価格が共通になるにはcompiler Cと同一literal policyが必要で、forward/inverseの別compile価格が同じという仮定は置かない。

## 4. Consumerとshell

既存consumerは固定q/rに依存するword選択分岐を持たず、native rz/sx/x/cxと元reset／measure／barrier、qubit/clbit、numeric phaseを保持する。通常IEEE・固定runtime、inplace=True、actual whole-word adjoint、exact Wを維持する限り、ループ不変条件は再利用候補となる。

固定SOURCEのwrapperは末尾に1 measureを置く。Referenceのresetはunitary bodyの前だけで、B′にはそのprefix resetがない。q1の別F/G分割とr変更も内部unitary列を変えるだけでbody fenceを加えない。したがって**同じSOURCE・compiler C下のshell構造S<=1**は各roundへ記号的に移送できる。保存N4-Cのshell式 `(16+64S)2^-53` はその条件付き根拠として再利用できる。

これは各roundのRPE diskへその数値を既に加えたという意味ではない。新L／b／chunk数／別runtime／copy／QPY／to_instruction／別compilerを持つ経路へ、per-RZ emitterの保存係数まで自動移送しない。既存13-round native adapterが異なるconsumerやphase経路を使う場合、その接続は統括兼Dの公開証拠との対応が別途必要。今回はadapterを実装・再検証していない。
