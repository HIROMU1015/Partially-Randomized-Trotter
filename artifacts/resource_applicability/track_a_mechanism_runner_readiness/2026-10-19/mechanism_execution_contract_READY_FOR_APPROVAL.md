# Mechanism execution contract — ready for approval

状態：**READY_FOR_USER_AUTHORIZATION**。`approved=false`、`runtime_authorization=false`。今回の作業では数値解析未実行・有効grant発行/消費0。

SOURCE：`5c256067db29a326bcaa0a0e244b38f6b87bd138`。基準は0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce。正確な32対象は[public入力projection](mechanism_input_binding_final_DRAFT.json)に固定し、同名のprivate companionに完全なinput path/SHAとraw provenanceを固定した。private contract/hash、runtime profile、無効grant案は内部に保全する。

## 範囲

H4/H6/H8×5ε（.05/.01/.005/.001/.0005）×旧PR／新Aの30比較、通常signed・factor1・固定強化PF。H4はH4_PRE_D_P_VALIDATION、DF12・subset16、旧n32はfixed anchor。H6 DF19/PF136、H8 DF28/PF184。H8通常n48はp8q2対B0I p11q1 S2（ε=.05）とp7q16対B0I p15q4 S4（ε=.0005）。合計32。

残り212はmetadataのみ。H4 D/P、非相殺、factor10、旧対照scope、H6高次PF独立確認を自動追加しない。候補・cohort・保存shot・比較対象・会計定義を変更しない。

## 承認後だけ行う算術

G_m=Σ_a N_m,a C_m,a、R_G=G_PR/G_PF、R_N=Ntot_PR/Ntot_PF、Ceff_m=G_m/Ntot_m、R_Ceff=Ceff_PR/Ceff_PF。
R_G=R_N×R_Ceff=Σ_a w_PF,a R_N,a R_C,a、w_PF,a=N_PF,a C_PF,a/G_PF。

整数shot・測定RZは整数、保存fraction/modelと軸単価平均はFractionを用いる。同じ固定cohortのpaired軌道から単価平均とGを再構成する範囲のみで、新cohort平均や別のestimandは作らない。両軸を独立の標本数として数えない。再利用されたn48 replicaは一度だけ。

Gは保存exact Gと有理数で完全一致、保存削減率がある場合は100(1−R_G)とも完全一致を要求する。両分解の恒等式も有理数で完全一致。丸め表示しかなければDISPLAY_ONLY_SOURCE。数値を近似して不一致を救済しない。

説明入力が揃う場合だけh=ε/√2−abs(saved signed bias)−allowance、N*=2 B² log(2/α)/h²、κ=N_saved/N*を説明用に評価。N_savedは再最適化・再丸め・置換しない。元binary64演算順序・saved B/log_Bのbindingを保持し、headroom/log_Nの許容abs差は1e−12。ceil境界に整合しない保存整数、nonfinite、欠測、overflow/underflowは説明側UNRESOLVEDまたはLOG_DOMAIN_ONLY。極端な比はlog表現を残し、表現不能を0としない。

連続shot比は(B_PR/B_PF)²×log(2/α_PR)/log(2/α_PF)×(h_PF/h_PR)²、整数丸め因子比を別に保持する。単純平均回路比×単一headroom比を正確なG比としない。Ceffのshot構成依存、q/r schedule、モデル条件付き、finite sample、選択効果を明示し、因果効果・母平均保証・FT資源へ拡張しない。

保存SEは文字列転記のみ。SE/CI/p値/bootstrap/選択補正の計算は契約に含めない。

## 状態と受入

CORE_COMPLETE、CORE_INPUT_MISSING、EXPLANATION_COMPLETE、EXPLANATORY_INPUT_MISSING、INELIGIBLE、UNRESOLVED、DISPLAY_ONLY_SOURCEを区別する。SOURCE_OR_BINDING_MISMATCHは拒否／STOP。eligible:nullをFalseにしない。欠測を0で埋めない。

初回32はcore32件のexact完了を受入条件とし、説明4件の未結合は許容する。説明の数値不一致はcoreと別statusで記録し、説明PASSにしない。core不一致・入力破損・SOURCE/runtime不一致・I/O失敗ではbatchをfail-closed STOPし、完了terminalを出さない。

入力・SOURCE・Python/runtime・grant・ledger・markerは処理前後にbyte照合。private出力の新規専用root、ONE_SHOT、WORKER_CLAIM、ledgerはexclusive create。旧SOURCE・結果・grant・ledger・STOPを出力先にできず、symlink・存在済みrun rootも拒否する。親launcherがnative exit0、wait/reap、固定32 IDs、出力schema、結果SHAを確認した後だけCOMPLETE_MANDATORY_STOPを保存する。

実際に使うSOURCE/input/runtime/結果/認可記録をgate対象にする。無関係worktreeのHEAD/status、資料更新、一時的PSI/CPU競合は停止条件にしない。実OOMや継続不能、own process異常は保護する。科学worker/既存jobへの操作は0。

## call・資源・出力

将来callは保存値解析1batch/pass・32 contrast。新sampling/native quantum build/compile/signal/SCF/DF/state/fit/量子shot/新統計/rare Tier R/非Clifford/ランキング=各0。

小さいscalar処理を1解析workerで重複読込なく行う。これは実装上の仕事の単位でありCPU/RAMの恒常capではない。起動時の実効資源・I/Oを確認し、共有OS/cgroup・ライブラリ・既存venv・他ユーザーを変更しない。

出力は[output_schema.json](output_schema.json)。mechanism_axes.csv、mechanism_totals.csv、results.json、worker_result.json、native_wait.json、terminal.json。失敗はSTOP.jsonと消費済みmarker/ledgerを保全。retry/返却/自動拡張/次stageなし。今回これらの実科学成果物は生成していない。

## ユーザーが次に判断する承認案

> SOURCE `5c256067db29a326bcaa0a0e244b38f6b87bd138` とprivate実行契約・input bindingの固定SHAに結合した主要32比較について、保存cohortの軸単価平均・G再構成／exact一致・回路単価比・shot比・正確な費用分解・取得済み説明入力のnormalization/headroom/丸め因子評価を、専用one-shotで1回承認する。保存SEは転記だけ。説明4件の欠測を保持し、core32件完了を受入とする。212補足・H4 D/P・新科学入力・新統計・ランキング・公開push・retry・次stageは承認に含めない。

これは**承認文案**であり有効grantではない。実際のユーザー承認receiptを受け取るまで、起動command・有効grantを完成／発行／消費しない。準備完了後mandatory STOP。
