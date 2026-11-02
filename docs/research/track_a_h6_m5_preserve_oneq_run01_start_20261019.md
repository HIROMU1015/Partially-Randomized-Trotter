# H6 m5：1qubit再合成を保持する対照の一回実検証・起動記録

2026-10-19。利用者の「実際にH6で改善されるか試してみて」に基づき、**同じH6 cosine一軸を、1qubit再合成を通さない新policyで一回再compile・作用比較する実検証を起動した。** これは起動時点の記録であり、結果・native exit・全worker回収は未確認。完了後の最終報告が優先する。local commit、push0。

SOURCE：`68ee03a7cef019daa0e6cf917b6104bb645884f7`。branch：`track-a-h6-m5-preserve-oneq-run01-20261019`。

plan SHA256：`7ab2b57e6a70a138de66ae29acfd3e84d148518474d89c857cebd5b6b3447c3c`。pipeline identity：`8745dcd7ab19e852f13fb67cc2423f87836b566f957ff4624128b6873ae24174`。

[前回H6のraw基準超過STOP](track_a_h6_m5_phase_batch_run01_results_20261019.md)、[直前の人工切分け](track_a_m5_local_resynthesis_diagnostic_20261019.md)、[今回のscope](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01_start/2026-10-19/scope.json)、[起動時点summary](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01_start/2026-10-19/summary.json)。

## 1. 変更と解釈

installed optimizerは小角度簡約を内部で有効にし、pass APIにその切替引数がない。今回はproject内のpreset managerで`Optimize1qGatesDecomposition`一箇所を、**元のDAGをそのまま返すpass**へ置換する。元のflow条件と、他のopt1 taskは維持する。

compiler policyは`M5_OPT1_PRESERVE_ONEQ_RUNS_CONTROL_V1`。単なるatol変更ではなく、1qubit再合成全体を無効にする保守的な対照である。Euler生成・局所簡約・そこでのphase生成を行わない。成功しても小角度簡約だけがH6残差の原因とは認定しない。また、標準opt1・前報のexact phase batchと同一の費用系列にはしない。共有Qiskit・SymEngineは変更していない。

## 2. 固定対象と実行量

H6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8・observed DF19、full `PF_M5_BEST_20D64C2_BINARY64_V1` p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。Qiskit1.3.0、rz/sx/x/cx、preset opt1、seed17、num_processes1と、新policyを明示して使用する。

前回SOURCE02cb8b12の変換後QPY/vector/最初のPASSを、元runのSTOPを保持した独立受入経路でSHA結合して再利用する。変換後の4,076,368gateを再生成・再生せず、今回の新compile一回・compiled回路の作用一回に限定した。

raw検査は以下の順で、全てfiniteかつ厳密に1e-9未満を要求する。

1. 変換後 vs 元wrapper保存vector：既存のPASS（2.326360070803798e-11）を再利用する。
2. 新compiled vs 保存変換後vector：今回評価する。
3. 新compiled vs 元wrapper保存vector：先行PASSの場合だけ今回評価する。

各新判定の前にQPY/vector/SHAを保存する。不一致時は後続検査を抑止し、未実施値を0やPASSで埋めない。位相合わせ・正規化・事後補正・基準緩和・自動retryはしない。sine・sampling・signal・SCF/DF/state/reference生成・shot/費用会計・fit・追加候補・費用受理は0。

## 3. 実装と起動前gate

新SOURCEには専用compiler、native adapter、launcher、prepare、固定scope/認可、人工テストを含む。旧SOURCEには書き込まない。人工12件PASS（compiler5・契約7）：小角度gate/phaseの保持、raw全演算・制御付き演算、他のopt1 taskの実行、profile拒否・上書き拒否、grant範囲、stage順序、数値不一致、spawn/exit/reapを検査した。これは作者自身の確認で、独立AIレビューではない。

起動前22項目と改変拒否12種がPASS。SOURCE/input/runtime計2,011fileをsealし、固定SOURCE・compiler・保存入力・旧STOP受入・新grant・未使用出力・保全・資源を確認した。新grantだけを一回使用し、旧grant/ledger/markerは保持する。

19:37:58 JSTにsupervisorを起動した。fresh gateで実効空きRAMは約469.1 GiB、測定済み旧peakから作るRSS admission proxyは約10.29 GiBだった。これはruntime cap・厳密上界ではない。1compile→1作用の依存関係のためproductive worker一つと、独立監視・外部wait4を使う。重複変換を避けることと、不要な再合成を除くことが今回の主な高速化となる。上位PSI・一時競合・無関係worktree更新では停止しない。

## 4. 保全と次の担当

起動前保護2,658fileには、過去の人工診断・実H6結果とSOURCE/runtime・grant/ledger/marker/STOPの固定証拠を含む。未受理H6の旧STOP、58受理軸、m5_best主比較のAD03保留は維持する。新policyの単価や結果を旧費用へ混合しない。

**Codex**：進行を監視し、終了後にraw結果・policy journal・SOURCE/入力/保全・native exit/reapを監査して報告する。成功・失敗のいずれでもmandatory STOP。新H6再compile・sine・会計へ自動進行しない。

**ユーザー**：最終報告後、この対照policyの採否や、残る必要な検証を判断する。現在の起動記録をPASSや修正完了とはみなさない。
