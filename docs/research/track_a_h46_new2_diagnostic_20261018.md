# H4/H6：4th(new_2)の限定数値検証

利用者の「こちらを使って、H4とH6についてまずできる範囲でよいので、検証を行って。できるだけ並行して行って」を実行根拠とする独立診断。既存の科学契約・SOURCE・grant・ledger・STOPを書き換えない。

対象は保存済みlinear H4/H6、1Å、STO-3G、rank12/19、T=.8、保存状態とsigned generation順を保持する。H4 prefix3/9、H6 prefix5/15、q1/2/4/8。2次・Yoshida4次・new_2の全体PFを比較し、各outer stepのRTE総short-step数32を共通にする。signed係数の絶対値に比例配分し、有限paired Taylor cutoff K2/4を使う。new_2は既存8桁係数の実用4次候補であり、記号的な厳密4次証明とはしない。

既存のprefix preparation・full-Fock scalar抽出・有限RTE分布を再利用し、実際のordered DF fragmentを含む全体合成の期待signalを評価する。exact H_D/H_Rの2-blockだけの検証とは異なる。ランダム抽選なしで有限分布の期待演算子を計算するため、希少event欠測による標本誤差は入らない。ただし回路のnative意味論・compiled費用・量子shot実測は検証しない。

H4とH6を独立した2 processで同時実行し、各process内BLAS4 threads。同一systemの行列・固有分解・符号付きstep係数をcacheし、exact-tail／K2／K4／rawの5列をまとめて伝播する。q/cutoffごとのprocess再初期化や同じ固有分解の反復を避ける。現行共有資源を直前観測し、共有環境・他process・venvは変更しない。過去のwall/AS/RSS上限は新規診断に持ち込まない。

SOURCE固定前の人工5検査はPASS。合成係数、共通RTE予算、負時間、実用次数、scalar位相、raw/corrected normalizationを対象とし、新科学計算の前に実施した。原SOURCEの既存SyntaxWarningは改変しない。

[計画・入力binding・環境・provider closure](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/README.md)。実行結果は別commitで追補する。実装author自身の検査であり、独立科学レビューではない。新しい科学的最終採択や原稿改訂は行わず、診断終了後mandatory STOP。
