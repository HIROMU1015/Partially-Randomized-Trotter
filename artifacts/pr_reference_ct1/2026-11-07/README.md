# CT-1 evidence（2026-11-07 JST、SERVER 2）

[REPORT](../../../docs/research/pr_reference_ct1_20261107/REPORT.md)、[事前scope](../../../docs/research/pr_reference_ct1_20261107/SCOPE.md)、[SOURCE](../../../src/trottertracks/pr_reference_ct1.py)。現在はmandatory STOP済み。再実行しない。

| 保存物 | 内容 |
|---|---|
| `PREDECLARED.json` | 合成前の2 trajectory identity/event hash、268 key、role、e=-9、既存97/新規171区分、CT-0の正確な4条件端点、予定検査 |
| `INPUTS.json` | 固定Git blob/保存common pickleのhash。新SCF/DF/ground solve/samplingなし |
| `NATIVE_INPUT_CHECKS.json` | 6新native compile＋2保存Reference X QPY、保存価格一致 |
| `native_qpy.tar.xz` | 全8 native QPY。2保存QPYは元byte、6 compile結果も実byteのまま保存 |
| `strict_records.tar.xz` | reused/97＋synthesis/171の全268 raw record・word・guard・spec・hash・T/T†・W |
| `KEY_REUSE_AND_SYNTHESIS.json` | key/role/record provenance・precision・既存/新規区分 |
| `WORD_CORRECTNESS.json` | 全268の再guard・exact adjoint、±wordの既存emitterとexact matrixの比較 |
| `LOCAL_TARGETS.json` / `LOCAL_CORRECTNESS.json` | 合成前に固定した21 native local構造、full-space operator結果・telescoping上界 |
| `JOINT_CORRECTNESS.json` | 8有限wholeの保存ground/vacuumとfull-joint人工probe、状態・信号・全joint normの根拠 |
| `ERROR_AND_SHOTS.json` | 旧diskを一度だけ加えたq1誤差予算・保存192/axis条件。全support CT policyの条件式とfinite pilotを分離 |
| `PILOT_T_COUNTS.json` | 8有限回路の実word T/T†とsigned出現数。平均・期待値・改善率・18round総Tなし |
| `TECHNICAL_TESTS.json` | 人工3qubitのnative数値評価器対Qiskit Operator。既存word emitter検査はWORD_CORRECTNESSに別記 |
| `SOURCE.json` / `STAGE_*SOURCE.py` | 最終SOURCE hashと各実行版。A→B/Cの追加・技術修正を区別 |
| `STAGE_A.log`, `STAGE_B.log`, `STAGE_C.log`, `STAGE_C_RESUME.log` | 元logを保全。Aのstdout誤記、Cの任意依存欠如、未完了部分の継続を含む |
| `ENVIRONMENT.json` / `BUNDLES.json` | 専用venv/実affinity/空き資源/worker数の観測、2 archiveのhash |
| `STOP.json` | CT-1完了、全support価格未資格、18round総T未取得、次担当GPT／ユーザー |

実行は `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` と既存専用venv `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python` を使用し、順に `stage_a()`、`stage_b()`、`stage_c()`。SOURCE `src/trottertracks/pr_reference_ct1.py` の明示入口で、旧runnerは呼ばない。Cの技術修正後の継続は `OPENBLAS_NUM_THREADS=3` を専用processに設定した。数はfloor(affinity64/local21task)で、共有BLAS設定を変更していない。

Stage Aは全入力・compile照合・PREDECLARED保存の後にstdoutだけTypeErrorとなった。Stage A再実行なし。Stage C初回はword/人工検査PASSの後、threadpoolctl未導入でlocal診断前に終了。未完了のlocal/whole/予算検査だけを継続し、通過済みword/人工検査と合成を重複実行しなかった。scope/key/precision不変更、科学的guard失敗・correctness不整合なし。

native/、reused/、synthesis/は同じrawを上記archiveで公開。common.pklは元の固定Git blobをINPUTSのhashで参照し、再配布しない。Numba cache・空lockは数値証拠ではなく未公開。展開したH/T巨大whole QPYはなく、native位置＋word表＋SOURCEが合成回路を定義する。同一wordの2×2行列でのwhole simulationは評価高速化であり物理回路変更ではない。

全作用空間の誤差証明はstrict単一RZ区間guard＋unitary telescoping。浮動小数点emitter/state/operatorの一致はその実装診断で、独立レビュー・外部再現とは称さない。有限word集合の資格は全support CT policyやRPE成功保証ではない。全support vacuum価格・全18round総T・T-depth未資格／未取得。

**mandatory STOP**。旧SOURCE/raw/result/grant/ledger/marker/STOP不変更。共有設定・他process/fileへ変更なし。次担当GPT／ユーザー。

raw stderr logの末尾空白も元byteのまま保存した。公開前の空白検査はSOURCE/文書を対象にし、raw logを除外した。
