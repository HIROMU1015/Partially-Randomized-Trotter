# N3-B1b-0：次段階の最小scopeとSTOP

2026-11-08 JST。[REPORT](REPORT.md)。**提案のみ。今回の検証はmandatory STOP。N3-B1b-1の実行許可ではない。**

## 今回解決したこと

固定runtimeの通常IEEE意味論と宣言したinplace位相経路について、既存lower_nativeを維持してN3-B1aの残余条件を満たす上界を適用できる。新しいinteger-W emitterを先に作る必要は認めなかった。

一方、copying compose／to_instructionにも再正規化がある。QPYは今回不要で未試験。任意の後続処理へ資格を移さず、次のproviderもまずメモリ内のcomplete wordと既存inplace経路を優先する。

## 43 key取得前に固定すること

**GPT／ユーザー**が、取得＋最小provider検証へ進むか判断する。実行指示では次を固定する。

- N3-B0の43 literal positive key（j=-36…6）、b36、各基本Frobenius10^-13。対象・precisionを追加選択しない。
- 既存strict backend2.0.0／seed17／up_to_phase=False、exact binary64 RZ targetと既存guard。実SOURCE・出力領域を独立させる。
- 正確な整数丸め／ties-to-even／bit分解、指数順、written productと時系列の対応。未実装providerの順序を今回勝手に採択したものではない。
- complete positive wordのactual adjointでnegativeを作る。Wを失わず、新control／phase fit／pi snapを追加しない。
- local 0-phase emission、inplace join、whole inverse、必要な保存経路。今回のSOURCE証明と操作回数を維持する。
- 旧CT-4BのC0採用wordを優先する規約、旧価格・旧SOURCE・STOPの保全。内部の厳しい基本wordでC0を自動置換しない。

保存coverageは0/43なので、今回の2保存wordテストを基本keyの再利用成功と数えない。一般的な全入力停止保証が得られたわけでもない。

## 明示承認が必要な最小実行

**Codex**が将来実施するなら、別途承認された固定範囲内で次を行う。

1. 43基本keyのstrict取得・guardと、実sequence／exact matrix／guard上界／W指数／T,Tdag metadata／hashの保存。missing追加・precision変更を自動化しない。
2. 固定recordを読む辞書と、整数丸め・固定bit順のcomplete-word providerだけを最小実装する。旧lower_nativeは維持し、composite証明を既存backend recordへ偽装しない。
3. zero、符号、tie／端点、43-chunk上端、全体actual adjointと実recordの局所correctnessを確認する。人工例と実support到達の観測を区別する。
4. 量子化＋基本guard＋emitterの予算をN3-B0保存Fractionへ対応させる。Operatorは診断であり、SOURCE上界を代用しない。

取得recordのT/Tdag metadata保存と、科学的な価格和／最大値／N1比較は別である。後者まで行うなら明示的な範囲に含める。全support期待T、199-shot積、18round総T、T-depthは既定の次scopeに含めない。

新native PF build/compile、sampling、物理系simulation、F/G構成変更はこの最小取得の前提ではない。QPY等が必要なら、保存／読込経路と追加phase正規化を先に固定して資格する。

## 残る証拠と停止条件

実43 recordの取得成功・有限終了、全基本guard、固定順providerとの対応、他C0 wordのemitter余裕、全supportでのCT RPE成功・価格は未資格のまま。N2 missing key／SK／全18roundとは分離する。

固定取得・検証scopeの完了、取得不能、strict phase／順序／adjointの重大な不一致、C_phi不適合、残余条件不成立、固定precision／key／compiler変更、対象外callの必要性があれば結果を保存してSTOP。余裕を得るための再最適化・別fallbackへ自動移行しない。

次担当は**GPT／ユーザー**：このemitter資格と残る取得／provider条件を評価し、必要な限定実行を指示・承認する。**Codex**は今回STOPし、取得・実装を開始しない。
