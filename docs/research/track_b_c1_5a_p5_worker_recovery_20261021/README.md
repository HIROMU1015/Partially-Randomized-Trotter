# Track B C1-5A-P5R：P5 worker修正・別one-shot

旧C1-5AのP5突然終了を静的調査し、科学法則を維持した技術修正・人工テストの後、**Closed P5だけ1024 whole-path×4 microstepsを1回、retry 0**で実行する承認済み作業。旧Ordinary/Fullは保存summary引用のみ。旧SOURCE・結果・grant・STOP、非公開レビュー原本を変更しない。

このSOURCE公開時点では科学計算は未開始。実行後の状態は `result/terminal.json`、完走時の比較は `comparison.md`、最終引渡しは `handoff.md` に追記する。開始後に科学SOURCEを編集しない。

- [旧worker障害の証拠と未確定事項](WORKER_FAILURE_ANALYSIS.md)
- [技術修正・人工検証](TECHNICAL_REPAIR.md)、[テスト結果](technical_tests.json)
- [新実行契約](execution_contract.json)、実行前SOURCE/input/runtime manifests
- [run.py](run.py)、[supervisor.py](supervisor.py)、[job.py](job.py)、[selector.py](selector.py)
- [開始前保全](preservation_before.json)、[共有環境の観測](resource_preflight.json)

固定モデル：Linear H4 1.00 Å / STO-3G / DF12 / R1新partition LD3 / 8qubit / 13 bases・324 labels。T=4/5, q=1, r=4, m=5, sigma=+1, Δ=1/5。H=2048, root bits=4096, eta/rho/angle tau=10^-12, angle K=256,J=64, seed=20261021、旧C1_5A_BITS_V1そのまま。model SHA256 `e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49`。科学参照 `243eea8e840f3c472753f7935e510391626fd325`。

取得範囲は `UNCERTIFIED_EXPLORATORY / TAIL_ONLY_PLUS_SCALAR`。native・deterministic halves・state preparation/readout・十分NはUNKNOWN。shot/coherent estimatorは実行しない。人工テストPASSや完走をcanonical certificate・研究優位・投稿GOへ格上げしない。

新規SOURCE・contractをcommitし、新authorizationのみ直接子commitとして公開した後、新markerを消費する。各childのexit/signal、CPU/RSS/PSS/private、原stdout/stderr、partial、batch、Evidence、terminalを保存する。first failureで再投入せずSTOP。共有OS・他job・他worktreeには操作しない。

Phase D公開・byte/SHA256/HEAD照合後mandatory STOP。次の担当はユーザー・GPT：C1継続/縮小/終了と、必要なら次の最小検証の範囲を判断する。Codexの追加科学実行・新証明・native取得は別承認。
