# 2026-10-15：Track B M1-S G0実施前の研究契約整理

Status: `M1S_G0_PRECONTRACT_PREPARED_NOT_ADOPTED_MANDATORY_STOP`。

ユーザーが指定したGPT完成レビュー原本と作業指示に基づき、[Decision Packet](../../tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md)を作成した。研究上の採否を変更する作業ではなく、何を固定し何が未証明かを次の判断前に揃えるための文書準備である。

GPT原本は46,341 bytes、SHA256 `d8e40225b3b2ddbbe79ed31a5eabf397803689d36f4c5b3021081a990d2ebdb5`。原本保存commit `4f5114e137563bb871c9f881f9eadab0c180bdd4`からGitHub再取得しbyte一致を確認。前回GPT原本と区別し、Claude完成原本は今回直接受領していないことも明記した。

完全乱択の厳密恒等式を長時間保証や部分乱択rhoへ格上げしないため、Q1–Q5の証明義務、controlled ordered law・時間・角度・乱数の未採択欄、H4/H6保存入力と未取得科学量、rhoのdomain・proxy境界、案A/Bを整理した。0.7／0.85等は参考値のみで、rho単独のGO/STOPは作らない。H4 held-out NPZ内部は開かず、保存stateを無料の実用predictorとして使わない。

P0/P0.5/G0-preの調査を再実施せず、元資料を固定commitで照合した。Parent allocation D＋E、M1-A/M2、旧科学分類・source/result/freeze/auth/markerを保持。専用worktreeで新資料と必要索引のみ公開し、親HEAD/index/既存dirty bytesを保護する。

M1-S正式採択、候補契約の正式採択、G0-T・Gate 0・新証明・科学計算・sampling/trajectory/operator guard・回路生成/compile/synthesis・資源forecast/最適化はすべて未実施。次は対象law・実施順序・理論scopeのユーザー/GPT判断で、計算実行承認は独立。mandatory STOPを維持する。
