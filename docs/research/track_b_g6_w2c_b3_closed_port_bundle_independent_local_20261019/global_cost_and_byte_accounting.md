# Global byte・CPU/wall会計

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

候補の16,896 calls/result64GiB/log256MiBはS/P1共通の一つのbudgetであり、正式採択していない。条件別へ二重付与しない。

公開n256：result reserved=observed=physical **18,345,700 bytes**、log **43,185,015 bytes**、unknown0。trace自己課金42,848,896 bytesをlog内数として照合。journal/state/pair/F01/summary/receipt/diagnostic/marker/finalization/callerをphysical inventoryから照合した。未ACK分はunknownとして非返却。

[独立byte集計](global_byte_accounting_verification.json)は原trace/ACK/physical joinを再計算し、取得済み公開大JSONとも比較する。reserved/observed/physical/unknownを混同せず、condition別attributionの総和が一度だけglobal値になることを確認。

公開n256の人工観測：parent CPU9.274588948 s、全worker43.331549000 s、total52.606137948 s、global outer wall31.218042745 s。pair/IPC/observer/stageは内数でありtotalへ再加算しない。coarse/fineは代替partition。global wallをpair wall和にしない。

setup/online、dispatch/generator-audit相当、angle-requery相当、serialization/hash/storage、IPC、RSS/HWM、own I/OをSOURCEで可能な観測範囲として分ける。worker preflight/setupのtimed region外を0や推定で埋めない。Fraction内部bit演算、純粋observer単独費用、連続unique global RAM peakは識別不能。実G7/F3費用/path分布は未測定null。

新規人工3caseは並行実行で、公開server実行とは環境/負荷が異なる。S→P1固定順も交絡する。CPU母平均CI、方式一般speedup、実G7/F3/Full-N予測、科学的総cost改善を主張しない。
