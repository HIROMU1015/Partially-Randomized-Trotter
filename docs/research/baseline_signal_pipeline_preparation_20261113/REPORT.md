# ベースライン全RPE signal pipelineの技術準備

2026-11-13。Full PF、Truncated PF B0/B0I、Full-H pure RTEの**必要な候補だけを全roundで評価する接続**を実装した。固定20セルrunnerは変更せず、元の数値部品とB V2のidentity生成を再利用する。今回は人工行列・metadataによる技術確認までで、実H4 signal、保存科学値の再解析、残15セル、sampling、compile、N/RMSE/G、候補選定は実施していない。

[HANDOFF](HANDOFF.md)／[次の有限科学案](NEXT_SCIENCE_PLAN.md)／[SOURCE索引](SOURCE_INDEX.json)／[テスト](TESTS.txt)。branchは既存Cの `codex/r-variable-compiled-price-design-20261111` を継続。基準は `331a6ffebe551577eb9574aca354e76175697743`。A/B/Dのbranch・実行領域へ変更しない。

## 実装と再利用

|入口・部品|担当する処理|再利用したSOURCE|
|---|---|---|
|`baseline_signal_pipeline.configure/prepare`|B canonical全round、physical candidate、target/reference/source/lawの接続|B `437ba4ac` controller、D `e7dca5f7` Full PF/Full-H物理schedule|
|`baseline_signal_numerics.Context`|共通保存matrix/state、half step・macro-step・spectrum・多項式・normalizer cache|既存saved-input loaderと `h6_df19_rpe_d2.make_context`。Aの同じcontextも受取可能|
|Full/Truncated PF|signed pieceごとの直接expm、元順のmacro-step、matrix_power、norm/unitarity gate|`dd24dd32` の `expm_step/power_diagnostics`、既存S2/Y4/M5 weights|
|B0I|prefixに応じたdiscarded identity scalarを総Tで一回追加|保存prefix metadata又は元 `matrix_only_tail_coefficients`。B0は元constantのみ|
|Full-H pure RTE|one-body込み全非scalar H、head0、有限K polynomial、grouped raw/corrected独立伝播|固定20-cell meanの演算順、`finite_rte_distribution`、既存spectrum/polynomial|
|`execute`|明示した有限manifestだけ評価、fresh出力、保存copy、部分失敗保存|新しい探索器・decoder・N/価格/G計算器なし|
|`prepare_baseline_signals.py`|B configurationから未評価requestを出力|metadataのみ。実行オプションなし|

C checkoutには新しいB/A module一式がないため、B/Dの固定Git objectをprivate namespaceで読んで使う。Bのglobal moduleを置換しない。既存AST readerを使う数値関数もSOURCEを固定する。standalone RTE moduleはpackageの一括importを避けて読み、不要なcircuit部品を起動しない。固定参照objectがない場合は取得/入力を要求するエラーとし、別版へfallbackしない。

## 対応範囲

- 初期対象は保存H4 DF10。Bの合法なglobal `h/M`、formula、正のΔ、保持prefixから全 `m=0..M` を生成する。13roundに専用化していない。
- Full PFは全10 fragment＋one-body、S2 `2nd`、標準Yoshida4 `4th`、M5 best `4th(m5_best)`。負signed time・fragment順と元constantを維持する。
- Truncated PFはprefix 0〜9、明示したB0/B0I、同3公式。target/referenceは**元full DF-H**で、保持Hのgroundへ差し替えない。PRと同prefixでも別method/physical ID。
- Full-H pure RTEは元preparation binding、one-body randomize、全fragment、deterministic head0、非負偶数global K、各roundの正整数q/Rを明示する。K0/2/4/6を人工確認した。`dt=(T/q)/R`、polynomialのR乗→q乗、rawのgroup-level normalizer、抽出identity＋元constant一回を保つ。同総qRでも価格同値は主張しない。
- 未対応はPR signal dispatch（A担当）、H6/別snapshot、round別formula/prefix/K、Bの正式spec外の任意PF q schedule、新8次PF。これらを探索上不利な科学的除外条件にはせず、必要になれば共通API担当と拡張を決める。

数値ライブラリ・実snapshotはplanningで読まない。`execute` に明示reference readerと共有contextを渡せる。default loaderは元snapshot/receipt・H/state/hash・sectorを検査し、元行列構成関数を呼ぶ入口を用意したが、**実H4入力で今回動作検証したわけではない**。

## 数値結果・欠測の扱い

出力はraw/corrected/reference、signed bias、logB/B/attenuation、局所phase/radius、数値診断・SOURCEを持つ。数値gateは既存1e-9、reference pairは既存1e-10、state normは1e-12。投影、再正規化、clamp、閾値緩和、別q/Kへのretryはしない。pure RTEにunitarity gateや未観測のexact-tail normを捏造しない。normalizerがbinary64で1へ丸まる場合、under/overflow・closure不一致はNUMERICAL_UNRESOLVED。

gate不成立でも得られたsignal・norm/unitarity等を保存し、後続roundを未開始にする。Bへはraw欠測として渡す。入力不足、未対応、数値失敗を科学的FAILに置き換えない。source/input例外でもPLANと途中record/RESULTを残す。既存出力を上書きしない。

保存signalは同じphysical構成、H/state/snapshot、split/source、profile、law、元full-H target、reference、数値policyを持つ成功recordだけcopyする。新旧IDやbackendを自動付け替えしない。旧screenのspectral meanを新direct-expm recordとして再ラベルしない。旧値の受入再資格・数値比較が必要なら別の有限科学scopeとする。

数値gate PASSは厳密数値誤差boundではない。経験的allowance・state誤差・sampler距離はnullのまま。B packetは `UNQUALIFIED` で、実shot/N、全energy RMSE、compiled C、G、最良候補はnull。理想finite-KとPCG64/CDF、固定seedとIIDを区別する。

## 検証・実行範囲

新12件＋既存M5/signed/scalar回帰12件、計24件PASS。新テストは2×2人工行列、明示ARTIFICIAL binding、固定SOURCEのmetadata/APIを使う。3公式・prefix・負時間、B0/B0I scalar/cache、Full-H K/q/Rと独立raw、元mean leafとの人工parity、B既存statistics metadata受入、全13round、保存値誤流用拒否、失敗値保存、欠測/CLIを確認した。metadata importはNumPy/SciPy/Qiskit/PySCFを起動しない。

開発中の接続エラー（private namespaceに旧connectionを渡したこと、AST抽出先の数値gate定数欠落）を修正し、固定B connectionと元SOURCEの定数を使用した。科学条件・閾値変更はない。テストは作者自身の技術確認であり、独立科学レビューではない。実分子性能・全round数値安定性・高速化率は未評価。

全event列やCartesian gridを生成しない。PFは共有matrix指数とmacro-step cache、roundごとの行列累乗、pureは共有固有分解と各lawの有限多項式/normalizerを使用する。要求候補だけ評価し、context単位でcacheを共有する。多候補に際しては同contextを使える範囲をまとめ、独立context間の並列化は呼出側に委ねる。CPU/RAM/workerの恒常上限は追加しない。

既存科学SOURCE/input/raw/結果/cohort/grant/ledger/marker/KEEP/STOPは保全。今回の技術成果公開・remote SHA/tree/主要成果物照合後、**mandatory STOP**。次はユーザーが有限科学manifestを承認し、C/Aがbaseline mean、Bが統計/N、Dがcompiled価格を各承認scope内で接続する。
