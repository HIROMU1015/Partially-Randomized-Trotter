# 次に提案する最小の技術判別

2026-11-08。**案Eの2-label poolに限る技術確認を提案する。** 今回の実装開始・人工量子simulation・実科学実行の指示ではない。M/Dは保留、Gaussian fusionをO3へ展開する案も推奨しない。

## 確認すべき新しい性質

主文書の[案E](NEW_MECHANISM_CANDIDATES.md)は、正のconditional pをcoherentに実装し、flag失敗のX/Y寄与が0という構造をsuffix省略に使う。最初に検証すべきことは、元complex平均を保存するinstrumentと、必要な有料制御・分岐を**具体的な一つのgate列**へ落とせるかである。旧A2をそのまま再実行したり、Mの実H4群を再分割したりしない。

## 次回許可を得た場合の有限scope案

人工の1/2 system qubit、二つの異なる明示W_j、共通A/B、exact rational pからなるpair一つのadapterを対象とする。basis共通や対角化を成立条件にしない。prefixにH_a＋controlled B、pool PREP、a=1かつk=jのW_j、逆PREP、kの測定、成功時だけcontrolled AとX/Y、失敗時0というinstruction/decoderを指定する。反対符号で同じunitary・同じgateだけのpairを主な成功例にしない。

既存A2のgate検査とphase/control、Mのbit/人工参照隔離、A1のlaw/記録/校正を適合確認の上再利用し、新規frameworkを作らない。新しいstatevector/Bornは次回の人工instrument検証の別の許可範囲とし、今回の設計作業から自動起動しない。

次回の人工確認は以下に絞る。

1. suffix前の各失敗flagでancillaが|1⟩のみとなり、Aを実行した参照との差がX/Y平均に現れないこと。
2. 元label別controlled平均と新三値平均が一致し、M2がp_sになること。phase、非可換A/B、X/Y符号を保つこと。
3. 全attempt分母・失敗prefix費・成功suffix費を記録し、成功数正規化や自動retryがないこと。
4. 同一Wや小角度では利益が小さい/ゼロという対照と、既知direct/grouping/標準LCUでも同じ処理が可能な対照。
5. PREP実効p誤差・inactive branchへの漏れがあるとtargetにbiasが出る境界を特定すること。旧engine保証を移植しない。

gate数の形式的な比較は可能だが、人工例のM2や省略gate数だけで実H4費用改善を認定しない。実build/compile/価格取得はこの後の別scopeとする。特に動的feedbackを実装しただけでhardwareが対応済みとはしない。

## 技術判別で変わる判断

平均保存/失敗zeroが成立しなければEを修正・保留し、targetの変更で救済しない。実装できても、標準LCUに早期flagを入れたものと同じなら**既知法の応用**として記録する。独立新規性を得るためにbaselineを弱めない。

全共有suffixを二重に実行するSELECTしか作れない場合、M2が高々2倍までしか下がらない利益を追加費が相殺し得る。共通A/Bを一度だけ実行できるかが最初の重要な価格上の判別点である。dynamic prefix/flagが通常Hadamardより明らかに重いなら、実H4の大きなbenchmarkを先に組まない。

## 科学入力へ進む境界

実N1へのpair/pool割当、実probability amplitudeやnew gate列生成、保存結果の再集計、実state/Born/shot、flag校正、native/compiled価格、variance/N/期待費用は新科学scopeであり、今回0。必要になった段階で対象input/SOURCE、有限pool、共通context、必要call、誤差/価格/STOPを固定し、明示実行許可を受ける。旧Mの232群を新pairにまとめ直すだけの実行を自動開始しない。

次回科学条件のε/α、seed、pilot/production数は今回採択・提案数値化しない。まず人工の明示アクセスと全attempt会計を判別する。実科学の量子資源改善を目指すなら、その後に強い既知直接測定/grouping・層別と標準LCU早期終了を含む最小比較案を別途設計する。

## 今回のSTOP

今回はsymbolic設計と一次資料/SOURCE静的確認で十分なため、新しい代数testも不要と判断した。アルゴリズムcode、人工量子simulation、実H4、旧raw再解析、重要科学レビューは開始しない。公開照合後mandatory STOPし、案Eの限定技術作業を行うかを次の判断として残す。
