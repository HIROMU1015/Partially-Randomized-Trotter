# H6 m5：1qubit再合成を無効にする対照の実検証結果

2026-10-19。**H6でraw誤差の改善を確認し、登録した3検査を全て通過した。** 1qubit再合成を無効にする対照では改善したが、原設定の修正完了・小角度簡約だけへの原因認定・旧費用互換性は確立していない。 一回の実検証を終了し、local commit後mandatory STOP。旧STOPと58受理軸は保持し、自動retry・sine・費用受理・会計・pushは0。

SOURCE：`68ee03a7cef019daa0e6cf917b6104bb645884f7`。branch：`track-a-h6-m5-preserve-oneq-run01-20261019`。plan SHA256：`7ab2b57e6a70a138de66ae29acfd3e84d148518474d89c857cebd5b6b3447c3c`。pipeline identity：`8745dcd7ab19e852f13fb67cc2423f87836b566f957ff4624128b6873ae24174`。

[起動・承認・固定scope](track_a_h6_m5_preserve_oneq_run01_start_20261019.md)、[前報のH6 STOP](track_a_h6_m5_phase_batch_run01_results_20261019.md)、[今回summary](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/summary.json)、[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/end_audit.json)。値は実行時に保存された出力の転記であり、資料作成時にQPY/NPZを復元・再解析していない。

## 1. 固定対象と変更

H6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8・observed DF19、full `PF_M5_BEST_20D64C2_BINARY64_V1` p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。Qiskit1.3.0・rz/sx/x/cx・preset opt1・seed17・num_processes1を用いるが、**compiler policyは新しい`M5_OPT1_PRESERVE_ONEQ_RUNS_CONTROL_V1`である。**

小角度の簡約をpass APIで切り替えられないため、今回は1qubit再合成全体を通さず元runを保持する保守的対照とした。他のpreset opt1 taskとflow条件は維持する。共有Qiskit・SymEngine、PF係数、入力・状態・reference・許容誤差を変更していない。前報の外側phase batchだけを使う修正とは別policyであり、同一費用系列にしない。

前回SOURCE02cb8b12の変換後QPY/vector/最初のPASSを独立した受入契約とSHAで結合し、元runのSTOPを保持した。新しい変換後作用を省き、今回の新compile一回・compiled作用一回だけを実行した。新しいsine、sampling、signal、SCF/DF/state/reference生成は0。

## 2. 固定raw判定

global phaseを含め、finiteかつ厳密に1e-9未満を要求した。位相合わせ・正規化・事後phase補正・基準緩和を使っていない。

| 照合 | raw差 | 判定・由来 |
| --- | ---: | --- |
| lowered_vs_saved_wrapper_vector | 2.326360070803798e-11 | 既存PASSをSHA結合しreuse |
| compiled_vs_lowered | 0.0 | PASS |
| compiled_vs_saved_wrapper_vector | 2.326360070803798e-11 | PASS |

直前のexact phase batchではcompiled vs変換後が1.4322891926554257e-9でFAILだった。今回の値は[raw gate出力](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/raw_gates.json)で確認できる。H6保存状態一つに対する結果であり、全状態・全演算・sine・他の候補の等価性を証明するものではない。前報のSTOPを遡及的にPASSへ変更しない。

## 3. 実行されたpolicyと費用上の制限

[policy journal](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/oneq_policy_passes.jsonl)は1回の保持passを記録し、いずれも同じDAGを返し、1qubit再合成・局所簡約・生成scalarの加算を実行していない。保存compiler callbackは17件で、元のOptimize1qGatesDecompositionは呼ばれていない。

compiled gate countsは[receipt](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/compiled_receipt.json)に記録した。RZ 2,855,001、SX 696,962、X 116,160、CX 408,244、measure1。これらは診断値であり、新規に受理した費用行は0。単価・shot込み費用の再会計・順位付けは行わない。

1qubit最適化を外した対照は回路規模が増える。従来の標準opt1によるS2/S4等との公平な資源比較やm5_best主比較への採用は、今回のraw検証だけでは確定しない。AD03の保留を維持する。QPY保存時点のreceiptは`UNVALIDATED_DIAGNOSTIC_NOT_FOR_ACCOUNTING`のまま残し、後段のraw検査と完了証拠を別に示す。QPY/NPZはprivate、bytes/SHAのみ資料へ収録した。

この対照の変更範囲はEuler生成・scalar算術・角度簡約を含む再合成全体である。1qubit再合成を無効にする対照では改善したが、原設定の修正完了・小角度簡約だけへの原因認定・旧費用互換性は確立していない。 新しい標準compiler修正や全系への自動適用は行わない。

## 4. 完了・監査・保全

19:37:58 JST起動。外部wait4のwallは497.631秒、ru_maxrssは4,434,228 KiB。1compile→1作用の依存関係のためproductive worker一つと独立監視・外部wait4を使用した。重複lowering/変換後作用を再利用し、不要な1qubit再合成を省いた。

[外部native exit](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/external_exit.json)は0、[worker exit](../../artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/completion.json)は0で全回収した。signal転送0、transient PSIによるkill0。worker exit単独を科学PASSの根拠にせず、固定raw検査・launcher終了・全回収を結合した。

終了監査39項目がPASS。SOURCE/input/runtime計2,011seal、起動前保護2,658file、旧STOP・58軸・grant/ledger/marker、起動snapshotのSHAを照合した。資料作成・終了監査でのQPY/NPZ復元・vector算術は0。人工12件・preflight22項目・改変拒否12種は固定SOURCEの記録を再利用し、同じ検査を重複実行しない。これは作者自身の検査で、独立AIレビューではない。

旧SOURCE・raw・grant・ledger・marker、他worktree、V1/V2等を変更していない。全実行量は固定契約内に留まり、retry・追加compile・次Phaseは0。

## 5. 次の担当

**ユーザー**：この保守的対照を踏まえ、1qubit最適化の縮減効果を残す修正の設計へ進むか判断する。m5_best主比較のAD03、compiler条件の正式変更、追加軸の実行は今回の結果で自動採択しない。

**Codex（次の承認がある場合）**：上記判断に沿って最小の追加技術作業を具体化する。科学的受入基準を維持し、同じ全H6 compileを修正なしに繰り返さない。

local commit後**MANDATORY STOP**。公開push・sine・費用再会計・追加候補へ自動進行しない。
