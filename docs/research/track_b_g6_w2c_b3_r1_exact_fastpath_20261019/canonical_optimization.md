# Fused canonical：formatを変えない融合

対象は`audit.canonical(driver.exact(value))`。整数はhex int、Fractionはrat配列、tuple/listは同じ配列、dict keyはstr化してsort、ensure_ascii=True。
**F3独自の$rat/$tuple/$map codecとは別**であり、F3 Evidence.createや科学SOURCEを置換していない。
通常のnative treeではexactの再帰copyと_packの中間copy/事前sortをまとめ、標準json.dumpsのsortを一度使う。
JSON encoder自体は元のPython標準実装なのでUnicode/surrogate/DEL/controlのnative-library互換fallbackは不要。

## 適応時に追加した境界

回収したserver packed_onceは、キーstr化後に衝突し、古い値がunsupported objectの場合、上書き前にpackして元処理と異なる拒否を起こし得た。
例：`{1: object(), '1': Fraction(2,3)}`は元exact→packでは最後の値だけが残る。
今回の候補はstr化衝突、container subclass、特殊key/値を元のexact→canonicalへ丸ごとfallbackする。
上書きされたnested keyの__str__失敗も元の正規化順で発生する。key順序・last-winsを高速化のために変えていない。
Fraction/intはtypeの一致を維持し、float/非exact値を近似受理しない。

36件の新canonical/driver/guard検査でnative tree1000個、巨大整数・Fraction、phase/weight/word、8 Unicode/control系、衝突、subclass/拒否を照合。
fake clockのStream全bytes/chain/file SHA、6種のF01 failure全文/metadata/coarseが固定baselineと一致した。
旧275件もこのdriver上でPASS。成功bytesだけでなくtyped failure・partial write・flush・STOPを確認した。

## 計測とguard

save.serialize/save.hash等のcoarse spanと外側CPU/wallの関係は不変。
fine call treeはdriver.exact/audit._packから新canonical/_packedへ移るため、同じ関数名の費用として差引き比較しない。
新moduleを完全SOURCE pinとFunctionObserver対象pathに含め、関数費用を落とさないことを人工検査した。
追加moduleのSOURCE照合費用も無料ではない。今回の性能endpointは実F3の全source guard費用を再現していない。
分類Aは人工範囲の技術互換性であり、正式SOURCE採択や実G7全domain認定ではない。
