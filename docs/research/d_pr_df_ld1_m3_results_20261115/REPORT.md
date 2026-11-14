# PR-DF LD1 m3：fresh4 path・8 compileのS/C取得

2026-11-15 JST、Codex D2が今回のD取得を担当した。**m3 (q8/R16/T.8) のfresh4 path・X/Y計8実compile、8行の同path S/Cを取得し、元receiptとのidentity照合を完了**した。予定8、再利用0、新規8、有効8、保留/欠測0、科学retry0。今回の終了点はm3公開とCへの引渡しである。残差集計・fit・m4予測/取得・E[C]/CI/N/Gは行っていない。

branch `codex/d2-pr-df-ld1-m3-acquisition-20261115`、独立worktree `d2-pr-df-ld1-m3-acquisition-20261115`。元P2 `39ddd59edf54a3a901e5302207b2765ddbd03fbb` から分離した。Pauli LD2 holdout/STOPとDのm9 branch/worktree/jobは変更していない。[HANDOFF](HANDOFF_C.md)／[8行machine input](C_HANDOFF.json)／[PATHS](PATHS.csv)／[最終静的照合](FINAL_CHECK.json)。

## 1. 固定SOURCE・条件と取得前照合

- 実行SOURCE：**`2f423befbba5075bad4ede3057f89edd788d2661`**。新しい薄い入口 [run_pr_df_ld1_m3](../../../scripts/run_pr_df_ld1_m3.py) と固定INPUTS、後続m4規則、人工testを含む取得前commit。
- 元raw公開SHA：**`4031dbab5871d8b51513dae5e97b7a4e45704071`**。実行code SHAと公開commitを分け、元receipt bytesを `run01/receipts/p0..p3/` へ保存した。
- C計画：`8eef0ad7a9b04e04d766f5e05b3b32eee50a794e` の `docs/research/pr_df_ld1_q3_price_admission_20261115/NEXT_VALIDATION_PLAN.md` / HANDOFF。
- C受入索引：`c99d511e2422b8f250490b11054f754a81e536c5` の `docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json`。保存P2 fresh m0/m6各4 pathを、元SOURCE/cohort/seed/roleのまま後続C較正入力として参照する。P1/replayはpoolしない。
- 元P1/P2取得SOURCE・結果：`39ddd59edf54a3a901e5302207b2765ddbd03fbb`。p1_efficient_zz、compiled_price_fast、efficient_controlled_zz、fast fingerprint、旧P2 code/raw/STOPは変更していない。

[INPUTS](INPUTS.json)／[取得前計画](ACQUISITION_PLAN.md)／[差分照合](PREFLIGHT_METADATA.json)を実H4起動前に固定した。元Cの索引・別ID照合を再利用し、その後の関連公開ref・C1/価格coverage・同候補local job・出力先のmetadata差分を確認した。m3有限pathは元C索引に0、保存ideal構造E_I[S]はcompiled Cではない。旧C1はR4/8や旧ZZ条件で今回の点を代替しない。確認範囲に受入可能な同等m3 compiled receiptなし、同じ実取得jobなし、出力先未使用。再利用0、出所未解決で保留した候補0。外部の非公開jobまで確認したとは主張しない。m4価格の開封0。

保存H4 linear1.00Å/STO-3G/tol-only DF10、LD1、S2/Δ.1/K2、元exactκ・H/state/snapshot・fragment/basis順、physical **`045843b2429bece6f29b`** を維持した。元価格configurationは `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。Cの新configuration/total decoder IDへ元receiptのIDを付け替えない。旧B正本のexactκ/law bindingとCの同物理点対応はINPUTSに保持している。

input snapshot `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`、H `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`。preparation `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、fragment順[0]、scalar c+s_R=2.7108187593826942（hex `0x1.5afc1bee774aap+1`）を総Tで一回適用した。元normal PCG64/NumPy CDF sampler Q、新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional/P0/測定付きfull controlled Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を使用した。正規化Bを価格へ掛けていない。

## 2. 同path/axisの8行S/Cと元receipt

cohortは全行 `D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1`。seed表selectorは実行SOURCE `2f423befbba5075bad4ede3057f89edd788d2661` の本dir `INPUTS.json` に対するもの。全axis receiptの公開commitは `4031dbab5871d8b51513dae5e97b7a4e45704071`、selectorは `[]`。元path receipt/result、structural receipt、request・law・gateの完全locatorと元private path/hashはC_HANDOFFにある。

| path | axis | S (構造RZ) | C (compiled RZ) | seed表selector | 元axis receipt |
|---|---|---|---|---|---|
| 0 | X | 36370 | 37973 | `cells[0].seeds[0]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p0/cosine.json` |
| 0 | Y | 36370 | 37973 | `cells[0].seeds[0]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p0/sine.json` |
| 1 | X | 33795 | 35455 | `cells[0].seeds[1]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p1/cosine.json` |
| 1 | Y | 33795 | 35455 | `cells[0].seeds[1]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p1/sine.json` |
| 2 | X | 35333 | 37267 | `cells[0].seeds[2]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p2/cosine.json` |
| 2 | Y | 35333 | 37267 | `cells[0].seeds[2]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p2/sine.json` |
| 3 | X | 34116 | 35834 | `cells[0].seeds[3]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p3/cosine.json` |
| 3 | Y | 34116 | 35834 | `cells[0].seeds[3]` | `docs/research/d_pr_df_ld1_m3_results_20261115/run01/receipts/p3/sine.json` |

| path | 固定8 step seed |
|---|---|
| 0 | `15862913925240177422, 15488494183284887498, 1564618460637046632, 15951578195070489213, 11877523266972387155, 4998606641673919988, 8253659132168221718, 40379074505873190` |
| 1 | `10409054108943232250, 11486726811224441181, 11374354341832797247, 8291415789898524037, 2019907091040116393, 15682791137492008947, 11712921529814513929, 16714130235662496306` |
| 2 | `15324599187504318185, 13079556660754730992, 12100886403759053449, 17918824316954827411, 12078721924440473102, 13144854910264968149, 4007335094648087618, 7307631667800312627` |
| 3 | `883157306594615801, 16296768479466403408, 16353480948952342901, 10822101084101100194, 1306560166932882043, 1359644337304596467, 14367493619153835072, 15931023610610172649` |

X/Yは同じpathの別測定軸であり、独立8 pathではない。今回両軸の観測S/Cは同値だったが、各wrapperを別にcompileし、異なるfingerprint/元axis receiptへ個別に対応付けている。SとCの差や平均・分散・残差fitを今回新計算していない。全値は元Backendの同じrequestsからの単一path/axis直接記録である。

## 3. 実際の取得経路とgateの範囲

元P2 runner全体にはreplay・m0/m6取得・残差summaryが含まれるため起動しなかった。新入口は元 `P.load_c1_backend`（固定C1 `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`）→ `F.make_backend_class` → `Backend.context` / carrier `155529e20a8fac39e00786c40b57830b71203c4f` の `sample_requests` → 同じS Accounting / compose / wrapper / compilerを使用した。新ZZを実際のprice_requestsに対して有効化し、F backend内のlowering ID検査を通した。

固定caller内と[RECEIPT_CHECK](run01/RECEIPT_CHECK.json)で確認したのは、保存入力hash/preparation/H・fragment順/scalar、normal seed、signed event allocation/time、同event digest、軸別compiler/整数S・C/fingerprint、wrapper測定とcompile前後の入力mutationである。元RESULT、STRUCTURAL、cosine/sine、M3_PATHの対応を照合し、元bytesをそのまま公開copyにした。元event列は専用privateに保持し、public索引に元path/hash/digestを保存した。

**新しいfull-wrapper statevector/数値作用検証は実施していない（元P2 freshと同じfalse）。** 元P1の人工unitary/parity、元P2 replayの記録はそのSOURCEの証拠として再利用し、m3に全入力/全lawの新保証を付けない。取得・identity gateの確認と、モデル/全law資格を区別する。C自身による今回8行の受入・残差集計/fitは未実施である。

## 4. 実event、資源と技術検証

| path | event数 | order0 | order2 | ZZ rotation | Z rotation |
|---|---|---|---|---|---|
| 0 | 128 | 128 | 0 | 123 | 5 |
| 1 | 128 | 128 | 0 | 115 | 13 |
| 2 | 128 | 128 | 0 | 114 | 14 |
| 3 | 128 | 128 | 0 | 117 | 11 |

4 path計512 eventはすべてorder0、order2/product application未観測。rare寄与0を認定せず未評価を維持する。追加seed/path、forced event、別軸補完0。ideal law Iと元sampler Qの距離、少標本のばらつき・CIは未評価である。

[EXECUTION_RESOURCES](EXECUTION_RESOURCES.txt)のGNU time：wall **21.40秒**、user79.81秒、system6.93秒、maximum RSS592264 KiB、exit0、swap0。runner内部のcontext準備から4 path完了までは **20.473143秒**。単一の保存contextを確認後、4 pathをforkして並行処理し、元C1 compilerの内部pool1を維持した。共有設定・他processを変更していない。RSSは各processのhigh-waterで、並行全体の同時peakではない。

| path | wall秒 | S秒 | compose秒 | X compile秒 | Y compile秒 | worker peak RSS bytes |
|---|---|---|---|---|---|---|
| 0 | 18.134029 | 1.512681 | 4.265565 | 3.173438 | 3.004405 | 421867520 |
| 1 | 18.098282 | 1.412303 | 4.274547 | 2.983443 | 2.950543 | 416194560 |
| 2 | 17.432844 | 1.467004 | 4.176518 | 2.930968 | 2.900992 | 418648064 |
| 3 | 16.949584 | 1.414125 | 4.073467 | 2.872921 | 2.837644 | 416886784 |

開始前からcontext、sampler、S、compose、軸compile/saveの開始/終了を記録した。publicのPARENT_PROGRESS/PROGRESS_p0..p3、各TECHNICAL_TIMINGで前進と所要時間を確認できる。60秒stack観測はworker別の専用fileに準備したが、今回その前に完了しstack fileは空。CPU生存だけを完了判定にせず、全4 pathのCOMPLETEと元receipt保存を確認した。

[人工test](TECHNICAL_TESTS.txt)は **5 PASS**。実際のF.price_requestsを人工event/fake compilerで通し、同path digest・X/Y別receipt、seed不一致、片軸失敗の保存/無retry、fingerprint不一致拒否、m3専用/m4非実行を確認した。元pipelineの科学/回帰suiteは再起動していない。初回は人工fixtureにlaw_bindingがなく1件失敗したためfixtureへ必要metadataを補い、5件を再確認した。[初回log](TECHNICAL_TESTS_INITIAL.txt)は元bytesを保全した。5件の既存SyntaxWarningは未変更libraryのescape表記である。metadata確認の初回process-name検索はshellが引用したcommandも拾ったため、実際のpython executable/script引数を区別して確認した。これらの確認中に実H4は起動しておらず、科学retryは0。

## 5. 後続m4の固定規則と今回のSTOP

m3取得前に、較正m0/m3/m6、後続m4 (q16/R32、physical `3df0cfd3356c358aeda8`) fresh4 pathのcohort/index/seed、軸別式 `saved E_I[S] + θ0 + θq q + θL qR` をINPUTSへ固定した。m4 cohortは `D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`、較正seedとは区別し、今回の実行menuには含めていない。m4取得/開封・C fit/予測凍結は0。

後続の別指示・C凍結公開後だけ、m4は軸ごとに `O=sum(Z0..Z3)/4` と凍結Pを用い、P−O、abs(P−O)、(P−O)/O、abs(P−O)/Oをexactに保存する。正は過大、負は過小予測。O=0なら相対はnull/理由ZERO_OBSERVED_PRICE、欠測時は該当軸O/誤差nullでn4を変更しない。数値許容差を設けずモデルPASS/FAILを付けない。事後閾値や自動refitも行わない。これはm4実行許可ではない。

## 6. ロードマップへの影響と引渡し

「新m3入力待ち」は今回の8行S/C取得・公開・C引渡しまで進んだ。次の依存は **Cへの別指示による保存P2 m0/m6＋今回m3の残差集計/fit・係数とm4予測の凍結 → Dへの別指示によるm4独立holdout → 固定した記述差の報告**。Cが利用する入口はC_HANDOFF.jsonで、元C受入索引の保存2点のlocatorも含む。ロードマップ本体は統合担当が更新するため変更していない。

今回のS/Cはsampler Qの指定有限cohort直接観測であり、構造期待値は保存ideal law Iの別証拠である。I/Q差・rare・全law E[C]・CI・長round移送・N/G/winnerは未資格のまま。保存m0/m6を再compileせず、P1/replayとのpoolも行っていない。旧raw/receipt/STOPを保全し、独立branchへcommit/push・remote照合後、**m3公開とCへの引渡しでmandatory STOP**する。
