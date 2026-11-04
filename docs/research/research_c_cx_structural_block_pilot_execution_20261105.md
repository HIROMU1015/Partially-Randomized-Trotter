# 研究C：内部block C/R包囲・固定6-call実行結果

2026-11-05 JST。**6/6 logical calls COMPLETE、FAIL/retry/resume 0、1回の起動で正常終了した。**
Full512・Arb192の新しいC/R operator上界は **2**、登録比較は **VALID_BUT_LOOSER**。
局所block方式はこの保存pairの包囲を生成できたが、保存済み約1.586e-11の上界より大幅に粗く、後続期待bias校正に使える鋭さを得られなかった。

## 固定SOURCE・実行条件

- 科学SOURCE：`7f70ca0af877621d6e4abb5db94945819ee0826e`、tree `c50ece9f3baad95e2a6ef2f441bb2d86fc31c600`。
- 実行branch：`codex/research-c-cx-structural-block-pilot-execution-20261105`。
- [固定契約](research_c_cx_structural_block_contract_unapproved_20261105.json)、[技術準備と導出](research_c_cx_structural_block_readiness_20261105.md)。チャット上の明示許可に基づく実行で、歴史的unapproved名・metadataは変更していない。
- Linear H4・1.00 Å/STO-3G、number CDF8/元順LD2、partial S2/s1/r32/K0、h=δ=.1、controlled full512。
- 候補SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- 保存 `CDF_proposed:F6:0` と `CDF_proposed:F7:6,7` を16回交互に置いた固定32-event wordの、同じcompiled/raw数学的gate operatorだけを比較した。
- P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase、Gaussian/relative fusion、53-bit IID-grid、own-DF oracle・状態準備費用除外、CA/RPE政策を維持した。今回は新しいround評価・signal・shots算定をしていない。
- 新block政策は `max_support=4, guard=1, exact_clifford=true` のまま。数値結果を見たpartition変更・phase fit・精度変更はない。

SOURCE194ファイル・保存input11件、固定word/phase/record/旧primitive plan、依存環境、fresh条件を照合した。既存の `c_cx_structural_readiness`／`c_cx_structural_blocks` と既存局所primitive・SU2・Arb・Callsを無変更で再利用した。
`scripts/prepare_research_c_cx_structural_blocks.py --execute-block-pilot` を一度だけ起動した。

## 科学call・結果

Fresh出力は `artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01`。

| logical区分 | 実績 |
|---|---:|
| records_load | 1 COMPLETE |
| identity_gate | 1 COMPLETE |
| structural_plan | 1 COMPLETE |
| local_certificate | 1 COMPLETE |
| comparison | 1 COMPLETE |
| summary | 1 COMPLETE |
| 合計 | **6 COMPLETE、FAIL 0** |

| 保存された包囲項 | 結果（表示近似） |
|---|---:|
| Primitive translation上界 | 2.0462396409820367e-13 |
| 局所block合成上界 | 23.713886551286482 |
| 補償付きglobal phase上界 | 1.4142135623726069 |
| 合成後のnative unitary C/R上界 | **2（exact）** |
| 旧保存C/R上界 | 約1.585733058571e-11 |
| 比較判定 | **VALID_BUT_LOOSER** |

Exact rational endpointと全局所witnessの正本は [compiled_raw.json](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/compiled_raw.json)。上界2は未知項の補完ではない。全登録項を評価した後、両targetが厳密native unitaryであることによる一般上界2を使った結果である。Nonunitary literal targetには適用していない。

実行では175 unique primitive定義と245 unique block bodyを評価し、multiplicity付きの数値factor5,776件を合成した。そのうち内部blockは15件、最大support4／局所行列16次元。構造planは8 window、共通CX factor2,876、1q factor5,761を完全に覆う。全register列伝播・sector射影・dense fallbackは0で、local embeddingとunitary telescopeによるfull512 operator保証である。Controlled相対位相・exact補償scalarは残した。

これはoperator差が2であることや、C/Rが不一致であることを示さない。旧保存約1.586e-11の包囲は依然として有効であり、本結果で置換・弱化しない。
今回の主要な粗さは局所block合成項で観測され、global phase項も小さくなかった。構造的に全gateを分割できることは、個々のfactorが近いことや、factor間の相殺をtelescopeが保持することを保証しない。個別blockの原因診断・別の分割・追加Arbは今回実施していない。

## 資源と時間

実行前はaffinity32、available RAM約56.0GB、load約1.38。Cgroup CPU/memory quotaと他host/user/schedulerのjobは現在のtool範囲では未観測で、その制約がないとは推定しない。
実ディスク空き約5.8GBのため、主checkoutとは隔離済みのworktreeを新しい実行branchで利用し、数GBの重複checkoutを作らなかった。固定SOURCE branch/commitは保持した。

Primitiveとblockの独立算術taskは、既存resource planによりそれぞれ32-worker設定のprocess poolを使用した。BLAS/OMPは当該job内だけ1thread/processとし、共有venv/cache/OS/他jobは変更していない。CPU/RAM/worker/wallに新しい人工capは設けていない。最大support4は局所アルゴリズムの適用範囲である。

| 時間・資源 | 実測 |
|---|---:|
| 新plan生成 | .7211862020 s |
| 新certificate evaluator | 1.9266076135 s |
| 新plan＋certificate | 2.6477938155 s |
| Child起動から終了までのwall | 3.9301301176 s |
| Child＋回収された子孫のCPU | 12.857672 s |
| Child lifetime max RSS | 225,216 KiB |
| Exit code／launches | 0／1 |
| 旧保存C/R evaluator wall | 1173.0860055881 s |

RSSはprocess別lifetime high-waterであり、全worker合計peakではない。各logical callのwallはjournalにあり、上記内部計時とは測定範囲が異なる。
旧1173秒は保存された歴史的対照で、同時期・同資源でのfresh timing再実行はしていない。**計算時間は短くても上界の鋭さが大幅に悪化したため、実用的な高速化成功とは認定しない。**

## 証拠・保全・検証

- [結果](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/result.json)、[比較](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/comparison.json)、[完全構造plan](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/structural_plan.json)。
- [Journal](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/science_calls.jsonl)、[STOP](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/STOP.json)、[SOURCEコピー](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/SOURCE.json)、[契約コピー](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/contract.json)。
- [実行前gate](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot-execution-preflight.json)、[終了状態](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot-launch01/exit_status.json)、[worker log](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot-launch01/worker.log)、[実行後保全照合](../../artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot-verification.json)。

実行後もSOURCE194・input11件を照合し、SOURCE/契約コピーのbyte一致と6 attempts/6 COMPLETE/0 FAIL、各区分上限、composition・資格の保存状態を検査した。科学報告の編集前には全tracked fileが科学SOURCEと不変であることも確認した。新しいoperator計算を検証目的で再実行していない。
技術準備時の79人工PASSと既存回帰476PASS／旧Stage R fresh-only guard1FAIL／3除外は当時の記録として保全し、再度の科学testを行わず同じ凍結SOURCEを実行した。結果記録はJSON/参照・hash・mode・call整合性として確認する。

旧4/6/7-call結果・SOURCE/raw/契約/grant/ledger/marker/STOP、旧未使用2 units、Head/Stage Pの証拠、別Codex-DのDirect-round作業を変更していない。既存契約や科学SOURCEを結果commitへ付け替えず、結果commitは保存・閲覧用である。

## 未資格事項・次担当・STOP

今回成立したのは固定1-word native C/R pairの数学的operator上界であり、鋭さの要件は満たせなかった。全word/compiler成功・全law synthesis・期待bias/tail/variance・RPE shots・actual CA・Stage Cは未資格／未実行を維持する。
New fit、sampling、native build/compile、旧pilot retry/resume、inverse/shell/R→literal再計算、重要科学レビューは0。

次の提案は、Codexによる**保存plan/witnessとSOURCEの技術調査**で、factor境界と補償phaseを含む相殺の損失を切り分けること。今回その調査、実装変更、別partitionやArb検証を自動開始しない。ユーザー／研究Cが後続方針を判断し、新しい科学評価が必要なら新SOURCE・契約・明示許可で限定する。

結果・科学SOURCE・固定入力・公開tree/主要blobを照合後、**MANDATORY STOP**。
