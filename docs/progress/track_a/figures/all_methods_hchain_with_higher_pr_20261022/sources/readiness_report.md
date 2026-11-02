# H2/H4/H6高次PF PR：独立確認の設計・実行準備

2026-10-21。**実行準備完了・科学実行前MANDATORY STOP**。基準commit `8613b3ecda2030a677b1690fe7aa3da23dec4873`。ユーザーが承認した保存統計による標本数・予算の設計算術、実装、人工検証、読取照合だけを実施した。新しい実分子sampling・signal・simulation・回路生成・compileは0、有効grant発行0、push0である。

実行SOURCEは **`8825391ff73b531be5834bf852b9bf2479d1799b`**。資料commitはこのSOURCEの子孫として別に保存する。SOURCEに含む既存H2 adapter 3 fileは元科学SOURCE `4ec26921c6872ed6e8667f167074e1e405cf1a3f` のbyteをそのまま再利用した。既存科学SOURCE・入力・過去STOPは変更していない。

## 固定比較と推奨案

目的は、探索で選んだ2候補の費用改善が、新しい独立cohortでも有限標本の点順位として再現するかを確認することである。確認標本での候補・q・標本数の再選択、早期成功停止、自動top-upは行わない。

|契約|ε|左：基準S2|右：高次PF|提案n_C（左/右）|trajectory|compile軸|
|---|---|---|---|---|---|---|
|H2_CONFIRMATION|.0005|S2/q4|Yoshida4/q2|24 / 28|52|104|
|H4_CONFIRMATION|.0005|S2/q8|m5_best/q1|25 / 9|34|68|
|H6_CONFIRMATION|.0005|S2/q8|m5_best/q1|23 / 9|32|64|
|主要合計|||||118|236|

各行のnは**実行採択待ちの推奨値**。探索改善47.12%・53.30%・63.12%は元報告の値で、今回新たにランキングを作成していない。[固定候補・探索統計の転記](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/fixed_candidates_8.csv)、[凍結shot 6行](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/fixed_shots_6.csv)を正とする。

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only 1e-8の保存DF rank3/10/19、prefix1/3/5、T=.8/P=0、K4、ABS_TIME_PER_OUTER=32、元のPF係数・signed schedule・無条件finite-RTE lawを維持する。full Hadamard wrapperのcosine/sineを同一trajectoryで対として取得する。native controlはsymmetric-directional、scalarは元政策どおり一度だけ。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17と既存native実装を固定する。H4 legacy DF12、別788軸研究、旧独立確認を混合しない。

## 標本数の根拠と感度

各候補の保存したshot加重費用 `G_i=N_cos C_cos,i+N_sin C_sin,i` の不偏分散を使う。cosine/sineの共分散を含む保存分散であり、両軸を独立標本として2nに数えない。探索と確認は混合しない。候補間は新しい異なるseed系列を用い、差SEで候補間独立を仮定する。seed衝突0を統計的独立性の証明とはしない。

推奨案は、保存分散 `s_j²` に倍率f=4、候補別の相対MCSE目標ρ=.02を当てた `ceil(f s_j²/(ρ Gbar_j)²)` を基本とする。目的は今回の大きな改善幅の再現であり、f=4は分散推定・分布移送の**設計感度**であって母分散上界ではない。

下限9は、正規IID参照モデルにおける分散推定量の相対SE50%の式 `ceil(2/.5²)+1=9` を使った有限標本の提案である。普遍的な最低n・正規性の認定・希少tail保証ではない。下限33も感度案として併記し、任意の下限を共通正式基準へ昇格させない。

さらに、固定した探索差Δに対し、主要3比較family α=.05の両側Bonferroni正規近似zを使い、(a)観測差を真の差と仮定した名目power90%、(b)近似区間半幅≤探索差の20%、の両設計条件を課す。必要差分分散を `min((Δ/(z+z_.9))²,(.2Δ/z)²)` とし、同等のtrajectory取得単価を仮定したNeyman配分を候補別MCSE案と下限へ重ねる。取得時間の最適化を証明する配分ではない。主要3比較では候補別MCSEと下限が支配する。

ρ=5/2/1%、f=1/2/4、下限9/33の18案を各比較で比較し、[全72感度行](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/sample_design_sensitivity_72.csv)に保存した。二進浮動小数の設計を、保存された数値をDecimalへ写した70桁の別算術経路でも全72案で照合した。正規分位点は共通の設計定数を用いる。この照合は算術・実装の確認であり、実際のpowerや母平均の保証を検証したものではない。

補助H4 ε=.005のS2/q4対Yoshida4/q2（探索差約1.36%）は、同じρ2%・f4・差半幅20%条件なら **23,647 / 10,091本、計33,738本・67,476軸**。小差の精密確認は主要3比較より大幅に重くなる。これは「点符号の再現に必ずこの本数が必要」という下界ではない。今回は主要grantから外し、独立した研究判断に保留する。緩い差精度案や他の感度行は公開表から確認できる。主要実行後に自動追加しない。

## shot・費用・結果判定

[保存値評価の契約](track_a_h246_saved_cost_execution_20261021.md)と[結果](track_a_h246_saved_cost_results_20261021.md)のUSUAL_SIGNED・allowance倍率1、経験的bias・保存normalizationを継承する。対象候補のε=.0005に対応する **保存済みN_cos/N_sinをそのまま固定**する。新しいshot算術やbias計算は行わない。元shot政策の経験的・reference条件付きという制限、binary64整数診断区分も維持する。

将来の取得後は、固定2候補ごとに新確認標本だけの平均・不偏分散・SE、cos/sin共分散、S0/S2_single/S2_multi/S4の観測数を保存する。整数RZと保存shotによるtrajectory別Gの平均・分散を有理数で計算し、別経路 `N_cos²Var(C_cos)+N_sin²Var(C_sin)+2N_cos N_sin Cov` とexact一致させる。

固定差は常に `S2 − 高次PF`。差>0をREPRODUCED、<0をREVERSED、=0をTIEDとする。技術PASSとは別の科学的方向分類であり、逆転や未確定をretryの理由にしない。標本差SEと主要3比較family α=.05の両側Bonferroni正規近似区間を併記する。区間が0を含めば近似方向はUNRESOLVED。符号再現だけで有意な優位性・母平均順位確定とはしない。改善率は点推定を示し、比の保証区間は算出しない。

保存全36群でS4は未観測である。今回の6群でも未知層の費用・母分散上界はnullのまま。実取得でS4等が現れれば、強制・棄却・0補完をせず同じlawの標本として保存する。標本SE0・小MCSE、f4の設計や正規近似区間からrare-event tailや母平均保証は得られない。m5主比較AD03は保留する。

## 実行契約・call予算

H2/H4/H6を別stage、別承認receipt、plan、grant、ledger、marker、run root、STOPとして封印した。private `proposal02` が現行の無効pack。各契約は取得call scopeと、**取得後の固定比較1件だけ**のanalysis scopeを分離して持つ。取得runnerのcost_accounting=0は取得中の会計禁止であり、明示したanalysis scopeの不存在を意味しない。後段評価も将来のユーザー承認receiptに正確に結合しない限り起動できない。

|予約する科学call|H2|H4|H6|計|
|---|---:|---:|---:|---:|
|trajectory|52|34|32|118|
|piece sampling / native piece合成（それぞれ）|264|299|283|846|
|RTE event step|4,864|6,688|6,176|17,728|
|native evolution / low作用 / high作用 / occupation oracle（それぞれ）|52|34|32|118|
|uncompiled wrapper作用 / compiled wrapper作用 / compile（それぞれ）|104|68|64|236|
|新signal / reference / SCF・DF・state生成 / shot再計算 / fit / reranking|0|0|0|0|

TASKと各callの非返金ledger予約を別に記録する。RTE stepはpiece内の固定event総数であり、piece呼出し回数と足し合わせて同じcallの総数として扱わない。[stage予算・plan digest](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/stage_readiness.json)が機械可読な正本である。

## 実装・高速化・資源

- `h246_confirmation_contract.py`：固定候補・n・shot・SOURCE・実入力・runtime closure、既知seed除外、analysis scope、無効grant、将来の一回承認発行。承認receiptごとのGRANT_ISSUEDファイルで二重発行を拒否する。SOURCEや無効packを書き換えて有効化せず、新しい承認済みfileだけを作る。
- `h246_confirmation_runtime.py`：既存native/runner/auditorを再結合し、cohortの文字列だけDESIGN_PILOT→CONFIRMATIONへ変更。科学命令・数値定数・PF係数・閾値は不変。H2の保存input loaderとprocess内のscalar guardを維持し、gate/角度/wire列が不変のreceiptを両軸で要求する。compiler raw CT01は1e-7未満、他native/oracle/Hadamard/log_B検査は1e-9未満、phase alignmentなし。
- `run_h246_confirmation.py`：既定scope表示。将来の明示認可後にsupervise→execute、別audit、固定evaluateを受け付ける。全draftはnative import前にexit2で拒否。取得supervisorはnative exit/reap、pool全worker回収、call完全一致を保存する。
- `launch_h246_confirmation.py`：起動時の実空きCPU/RAM/scratch/inodeに基づき自分のstage間を配分。資源が不足すれば実行前STOPまたは未起動stageを逐次実行。定数のworker上限は置かない。取得後の数値評価を自動起動しない。
- 読取専用contextを一度復元してfork共有、JSONは毎回byte SHAを確認しparseをcache。immutable Git blobを一括読取してcacheし、CURRENT file/input SHAは毎verifyで照合する。独立trajectoryをprocess並列、長いq×pieceから投入する。nested BLAS/compilerは自分のprocess内だけ1 thread/processとしてoversubscriptionを避ける。
- 既存のlevel1 zlib/DEFLATE byte可逆保存、原SHAとarchive SHAの分離、atomic manifest、archive-aware監査を再利用する。新しい自分の一時rawだけを永続化・復元SHA後に解放し、旧rawは触らない。

保存済み対象trajectoryの実行時間metadataをSHA照合した技術負荷proxyは、H2約373、H4約2,861、H6約12,942 worker秒（それぞれ約6.2/47.7/215.7 worker分）。これは同時稼働条件に依存するtask経過時間の加重値であり、厳密なCPU時間・将来wall timeの上界ではない。

読取観測時にはidle約61.9 CPU相当、実効空きRAM約499 GB、disk約419 GB。資源previewはH2/H4/H6=5/14/25 worker、計44。最終保存growth proxyは合計約14.50 GB、raw/partial・復元を含むpeak proxy約134.52 GB。既存の4倍圧縮余裕＋16 MiB/trajectoryとraw/RSS proxyをそのまま使う設計値であり、PR/rare出力の上界ではない。実際の出力を切り捨てない。

上記worker配分での処理部分の単純proxyは約75/204/518秒。ただしcontext復元・seal照合・archive終端監査・競合・希少eventの時間を保証しないため、「全体が9分で終了する」とは見積もらない。実行承認後、起動直前とprewarm後にfresh gateを再度通す。previewの配分を恒常的なcapへ変えない。

## 技術検証・保全・限界

新しい人工31件、既存回帰66件が固定Python3.12.3で全PASS。新人工にはmock processpoolの成功/故障と全reap、3公式の人工Hamiltonian native/compile/archive監査、H2 S2/Yoshidaのscalar guard両軸、無効認可、二重承認発行拒否、未知層null、paired共分散、固定候補逆転、予算・seed衝突を含む。人工系でのsampling/compileを実分子の新取得と混同しない。

構造・設計算術・SOURCE・終了control metadataの181項目PASS、全72設計の別算術照合、実draft12 CLI拒否、in-memory改変27件拒否、SOURCE68 fileのGit byte一致、H2 frozen3 file一致を確認した。[技術記録](../../artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/technical_validation.json)。資料完成後にも表の行数・SOURCE SHA・リンク・公開内容・旧control fileを130項目で確認した。これは作成者Codex自身の検査であり、別担当の独立科学レビュー・外部再現・immutable CIではない。

既知の過去registry・exclusionをunionし、他の新stageのseedも互いに除外した。新piece seedはH2 264/H4 299/H6 283、計846個で衝突0。既知除外はstageごと約25.65万個。privateにseed一覧を保存し、公開候補資料には載せない。これは明示されたregistry範囲の被覆であり、全研究・未来seedの完全探索ではない。実行前に新しく追加された既知cohortがあれば、明示した除外資料との照合を更新する。

実行で拒否する対象は自分が使うSOURCE/input/runtime、保存signal/shot、registry、grant/control/resultに限定する。無関係worktreeの更新や一時host PSI・CPU競合は観測扱い。他jobへのsignal・共有設定変更は行わない。別stageの局所的な故障で正常稼働中stageへsignalを送らない。

現物旧資料の保全は、明示したSOURCE/input/runtimeのSHA、元取得config・plan・marker・completion・audit・native exitなどの限定照合である。全projectsや旧raw archive全件の再走査をしたとは主張しない。過去のSTOP/partial/grantはそのまま保持した。最初の無効proposal01は短縮SOURCE表記を使った技術準備だけの記録としてprivateに残し、完全40桁SOURCEのproposal02で封印し直した。科学実行・grant消費・retryはどちらにもない。

## 実行前STOPと次の担当

入力/SOURCE/compiler SHA不一致、seed衝突、数値FAIL、破損archive、実容量不足、異常exit、未回収worker、call予算違反、結果SHA不一致は当該契約をSTOP。共通SOURCEや入力の問題は依存stageにも波及するため、未起動stageを開始せず原因を調べる。既存STOPをPASSに書き換えず、同grantのretry/resume、無断top-upや候補追加を行わない。

**ユーザー**：主要3比較のn_C=24/28、25/9、23/9と、候補別MCSE2%・分散倍率4・下限9・主要family3の近似区間の設計、希少tailの限界、独立3契約の一回取得236軸および取得後固定3比較の評価範囲を採択・明示承認するか判断する。補助1.36%差は今回保留を推奨する。公開pushは別指示待ち。

**Codex（その承認後）**：新しい明示receiptにSOURCE/候補/n/call/固定analysis scopeを結合し、既知seed・fresh資源・preservation gateを照合して独立one-shotを実行、終了監査と指定された固定比較だけを保存してmandatory STOPする。

本作業はlocal commit後に停止し、上記の科学実行・AD03採用・追加詳細レビュー・本文作成へ自動移行しない。
