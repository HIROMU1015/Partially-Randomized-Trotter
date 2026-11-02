# Track A 独立担当の技術監査所見（2026-10-15）

この担当は実装著者の TECHNICAL_READY 判定を引き継がず、数値・bridge の新しい失敗経路を調べた。独立監査は今回の一回で終了する。**確認した6件の不具合は最小修正され、確認済みの未修正不具合は今回の静的・人工範囲では残っていない。これは実行認可、科学的採択、実データによる成立証明を意味しない。**

## 担当と証拠の範囲

- 独立担当：`/root/audit_independent_h8_h6`。親とは別の Codex reviewer として、数値・identity・phase・error・SOURCE pairing を主に担当した。親は認可・監視・exit・quota を担当し、その再現証拠と修正差分も本担当が最後に読取確認した。
- 起点 HEAD：`2ec66aae5e17c9329fcea8ad962be27130980d31`。監査した実装 SOURCE：`60eeb51c148007359cf6e8bcf6288e6837a39701`。
- 修正 SOURCE：`20f89bc32b161ee22e4a1ed709f9e3414c19633a`。この SOURCE と対象3 code files・独立数値 test の disk diff が空であることを確認した。最小修正は親が実装し、本担当は source code を変更していない。
- GitHub 固定資料14件の取得・bytes/SHA照合は親が担当した。本担当が同じ取得を外部で再実行したとは主張しない。親の `github_reference_receipt_v1.json` と、対象 worktree の固定 source・契約を参照した。
- 読取対象は利用者の添付全文、`AGENTS.md`、`PROJECT_MAP.md`、現状の研究資料、共通指示 2026-10-12、両最終実装/handoff 文書、両 contract draft、H8 driver/adapter、凍結 D0/shot rule、H6 bridge/prototype、既存人工 fixtures/tests、H4 `h4_b0i/adapter.py`。固定 H6 provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` の native builder・`_prepare_discard`・`h6_matched_accounting_v1.empirical_allowance` は Git source の文字列として読み、実 import していない。

## 新たに確認した不具合と修正評価

### F1 — CONFIRMED_DEFECT：allowlist 内の異 SOURCE を paired axes に使用できた

人工 H8 の cosine は元 SOURCE、sine は別の許可 SOURCE とし、seed/event/compiler を同じに保つと、元 adapter は探索・確認の両方で受理した。旧 D0 の pair identity 検査にも SOURCE は入っていなかった。

修正は adapter に `(phase, cell_id, replica, control)` ごとの SOURCE equality を追加するだけで、凍結 D0 を変更していない。別 SOURCE を pair 全体へ同時に指定する正しい allowlist 利用は維持された。実保存データに resume 境界で SOURCE が分かれた pair があるかは未確認。その場合は新 guard が旧値照合より前に STOP する。同じ科学的意味だという alias 規約を本監査では新たに採択していない。

### F2 — CONFIRMED_DEFECT：H6 request のコピー・built reset による再 build

元実装の `built` は request digest から除外されており、build 前の deepcopy 又は `built=False` の復元で同じ grant/case に再び native API を呼べた。

修正は grant が保持する `(prefix,q)` build reservation を provider 呼出前に消費する。request のコピー・状態 reset を超えて消費を保持し、provider 例外後の再試行も拒否する。scalar・PF・error の科学的意味は変わらない。

### F3 — CONFIRMED_DEFECT：provider の module 名だけで実 branch の build に到達

新認可は作らず、`synthetic=False` の mock context を用いた。未検証 object に固定 module 名を設定するだけで元 build は API 呼出を許した。

修正は `native_provider` による disk/Git SHA・origin 検査成功後に限って grant に provider object を登録し、build が同一 object を要求する。すべての source bytes/Git/import を mock した正規 success 経路は build を許し、同名の別 object を拒否した。実 native import/build の証明ではない。

### F4 — CONFIRMED_DEFECT：欠測 OOM counter と非有限 PSI が観測証拠として通った

親の baseline 人工 log では、cgroup の必須 `oom`/`oom_kill` 欠測が旧 `get(k,0)` により0として扱われ、NaN PSI も比較を通った。admission と continuation の両方の再現証拠を読取確認した。

新 guard は凍結 D0 の前で必須 counter を非負整数、PSI を有限の0〜100に限定する。旧 kernel で任意の `oom_group_kill` を許す一方、admission との有無の変化は拒否する。既存 pressure/OOM policy・閾値を緩めていない。

### F5 — CONFIRMED_DEFECT：quota receipt の NaN/Infinity availability

親の人工 log では `available_bytes`/`available_inodes` の NaN 又は Infinity が十分な余裕として条件を通った。修正は filesystem/quota の availability を非負整数、観測時刻を有限に限定する。

quota-enabled で receipt 自体がない経路は baseline でも loader 前に STOP し、one-shot marker を消費した。今回発見したのは receipt 欠測の無条件 bypass ではなく、非有限 receipt の schema による fail-open である。

### F6 — CONFIRMED_DEFECT：monitor interval が前回 I/O を除外

親の人工 clock では observation I/O 4秒と次 checkpoint まで4秒を与えると、元実装は開始間の8秒を4秒として扱った。

修正は前回観測開始時刻を保存して start-to-start で間隔を測る。既存5秒 stale 閾値、I/O duration 制限を保持する。資源上限や科学条件の変更を含まない。

## 人工検証と新たな正の証拠

本担当が新規作成した `tests/test_independent_track_a_numeric_audit_v1.py` は13件。既存人工 fixture は入力 scaffolding として利用したが、新しい拒否条件と独立算術を検査した。

- baseline 10 tests は、F1/F2/F3 の4 test methods・5 subfail を再現し、他6 methods は PASS。`numeric_reproduction_red.log` を本担当が確認した。
- 親の execution baseline 9 tests は、F4/F5/F6 の4 methods・11 subfail を再現。他の認可・missing quota・partial output・native wait tests は PASS。`execution_reproduction_red.log` を本担当が確認した。
- 修正後は独立22件（数値13＋実行9）、既存41件、変更 adapter の旧12件、計 **75/75 PASS、failure/error 0**。`regression_green.log` と `regression_result_v1.json` を本担当が読取確認した。
- 実行は親による foreground 1 process/thread1、`/usr/bin/python3.12 -P -B`、AS512MiB/CPU120秒/wall180秒、log1MiB/attempt、home scratch512MiB/5000inode。wall16.08984176442027秒、peak RSS81,489,920B。独立担当が別 process で test campaign を追加実行したことはない。

特に、最終 confirmation map 行の旧値不一致、旧 map 行の欠落、ineligible candidate の旧 precision 不一致が、`.0005` の accounting 呼出より前に STOP することを独立に確認した。部分的旧値 PASS だけで Phase B に進む経路は今回の人工変異では見つからない。反相関した cosine/sine 費用で paired totals の SE が0になる例も確認し、covariance が落ちていない。

two-mode occupation basis の `dGamma(G)^2` 全 Fock trace を独立に構成し、complex Hermitian G・signed lambda の identity と一致した。H4 の式、固定 H6 provider の `P(-T*scalar)` 一度、sine の Sdg、元 `_prepare_discard` の retained prefix も静的に整合した。full-H reference は回転されず、旧 B0/exact-truncated の回転・empirical label と新 certificate 非作成を確認した。

## 残る分類と実行制約

**SCIENTIFIC_ASSUMPTION**：`64*epsilon*(1+abs(.8*C))` と floor1e-12 を含む libm/rotation allowance は経験的診断であり、真の誤差上界の証明ではない。旧 combined u の signal/reference 各寄与への利用も経験的な診断政策である。コードは `certified=False`・`new_scientific_certificate=False` を保持し、rigorous certificate への昇格経路は今回見つけていない。`stored_model_coefficient_error_upper=0` は保存 binary64 モデルに対する追加 input coefficient error が0という定義であり、元化学 Hamiltonian・DF近似・未知の真の数値誤差が0という意味ではない。この政策の科学的採択は GPT へ残す。

**RUNTIME_ONLY / TEST_GAP**：実 H8 の944 precision/3576 map完全一致、実 H6 snapshot/preparation・retained phase・control・native numeric fingerprint・実 compiled Δmetric は未実施。H4 の compiled Δ=0 を H6 の保証へ使っていない。blueprint は compiled fingerprint ではない。実 profile/source/input の fresh binding、実 SOURCE pair の存在、actual wall/RSS/storage は未確認である。

**既知の監視限界**：同じ Python process の timer/checkpoint は長い C処理中に即時実行を保証しない。SIGKILL 中の terminal 保存も保証しない。修正した start-to-start 計測は Python へ戻った際の stale STOP を改善するが、即時 wall/RSS停止を新たに保証しない。kernel AS/CPU 制限と同じ親 shell の foreground wait receipt は別の証拠であり、mock wait137/terminal欠測を成功と補完していない。正確な exit timestamp は未取得で `None`。これを全環境での bounded 実行保証と表示してはいけない。

**将来の接続条件**：CLI default monitor は quota receipt を供給しない。quota-enabled mount では現在も STOP し、実行前の fresh effective receipt 接続が必要である。H6 bridge は完成した bounded pilot runner/compile orchestration ではない。独立した monitor、artifact/carry/resource/one-shot binding、固定 provider/profile の展開、new user execution authorization 等が将来の許可された段階で必要で、今回追加実装・認可していない。

## 最終判定と STOP

本監査で再現した6 findings について、修正と回帰証拠を独立に確認した。**CONFIRMED_DEFECT の未修正残件は今回の範囲で0。上記以外の数値意味について NO_MATERIAL_ISSUE_FOUND は静的・人工範囲に限る。** 数学的 allowance の採択を技術 PASS で代行せず、実行成立・科学的採択の未確認事項を残した。技術的修正のために新しい科学的定義を採択しなければ直せない blocking defect は今回見つからなかった。

新科学計算、実データ再会計、実 Hamiltonian/NPZ array 読込、real native build/compile、trajectory、grant発行はいずれも0。共有 venv/compiler・既存科学 SOURCE/raw/checkpoint/ledger/grant/worktree への変更0。許可された独立 branch の新 guard/bridge/adapter 最小修正と人工 test だけを扱った。

**SOURCE ready ≠ execution authorization。実行・追加科学判断を開始せず、独立担当はこの報告をもって mandatory STOP。** 親が監査資料の公開・remote確認を完了した後も、今回の依頼から次の科学計算へ自動移行しない。
