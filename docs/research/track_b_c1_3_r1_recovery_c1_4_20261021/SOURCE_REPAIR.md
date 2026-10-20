# C1-3-R1：state hash修正と実行前境界

旧C1-3はstate hash規約を取り違えて停止した。旧SOURCE `f3a6aa1108bbc8359e585c470e23033458c57466`、authorization `c7d444d5c59acfc1d57d6b4ed348a86451522b07`、失敗結果、STOPをそのまま保持する。基点は公開HEAD `e72867b87e12d93d4fc1361f45f105febc6a94c6`。

修正は新namespaceのrunner/helperだけ。`state_identity.py`が固定SOURCE `243eea8e840f3c472753f7935e510391626fd325`の`pr2_s0_s1_validation::_array_hash`と同じdtype.str→canonical shape JSON→C-order bytesをSHA256へ入力する。stateの期待hash `c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`、NPZ SHA、値・shape・dtypeは置換しない。local matrixのraw-bytes hashと混用しない。

Phase Aでは既存SOURCEからpure hash関数2個だけをAST抽出し、人工配列で互換性を確認する。whole科学moduleや旧runnerはimport/実行しない。固定人工expectedはpayload `<f8` + `[1,3]` + 明示little-endian double bytesから与え、dtype/shape変更、非連続/Fortran配列、C-order正規化、bytes-only hashとの差を検証する。mismatch failureとterminalを共有Journalで保存し、payload名衝突・stdout例外・二次failureでも一次failure bytesが変わらないことを確認する。mock記録の科学実行回数は0。

既存C1-2のadapter/reference/testは原ファイルを再利用する。新supportが新namespaceへ出力を向け、13 basis/324 labelのH4入力はPhase Aで開かない。人工2-qubit/3-labelの10 controlled比較と63 event first moment、zero/technical failure・全trial分母を確認する。主agent自身の技術確認であり、別reviewerによる独立reviewではない。

H4の科学条件、固定6 event、4096/2048bit、eta/rho/tau=10^-12、angle 256bit/64項、1e-10 operator比較、X/Y診断、reference定義は旧契約のまま。`contract_comparison.json`で旧科学field全ての同一性を機械照合する。新run ID・classification・schema・source/helper/manifestだけを新しくする。巨大有理数を扱う既存runtimeも変更しない。

F1/F3 P0–P5は新one-shotの内部で一度確認する。指定精度が不足したらSTOPし、bit数やlabelを変更しない。root/angle/proposalは既存G6/F3 queryとEvidenceを使い、samplerやproviderアルゴリズムは変更しない。全event列挙、trajectory、shots、native compile/transpile/synthesis、性能benchmark、N/資源順位は行わない。

失敗分類は`journal.classify_failure`に固定する。環境/SOURCE等の例外を科学的不一致へ格上げせず、明示operator gateの不一致はoperator STOPとする。必須GateのPASSだけを`C1_3_CORRECTNESS_PASS`とし、その場合のみC1-4文書・静的監査へ進む。X/Y値は診断であり新rigorous閾値は設定しない。

新SOURCE/契約をcommitした直接子に新authorization receiptだけを作る。実行時に両commit、manifest、runtime、input bytesを照合し、exclusive result directoryで一回を強制する。新runは旧失敗のresume/retryではなく、明示承認された独立one-shot、retry 0。SOURCE固定後の改修・再試行は行わない。

計算資源の人工的上限は設けず、自分のcache/outputだけを使用する。6 eventは最初のfailureで即STOPする依存Gateであり、basis/label行列をcacheして順に確認する。既存NumPy/Qiskitの演算を利用し、共有venv/設定/他者processを変更しない。C1-4終了または失敗後に公開・保全確認しmandatory STOP。C1-5は別の明示承認が必要。
