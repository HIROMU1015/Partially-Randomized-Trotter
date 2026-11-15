# PR-Pauli LD5：LD3と共通の固定2配分・total/Q3

LD5のtotal viewを作成し、新規Q3 2件、LD5の両配分ともモデル適合。FULL_PF__B_PRICE=MODEL_BOUND_MEETS_ADOPTED_THRESHOLD、VP=MODEL_BOUND_MEETS_ADOPTED_THRESHOLD。停止理由`TWO_FIXED_ALLOCATIONS_COMPLETED`。LD3の2点は保存引用のみ。新規signal/sampling/MC/compile/価格再fit/新配分生成/retryは各0件。mandatory STOP。

## 4点比較

H4／1.00 Å／STO-3G／tol-only DF10／sector(2,2)。LD3/LD5それぞれ自身のhead/tail辞書、ideal finite-K raw、κ/scheduleを使用。共通Z1+T1 total decoder、保存full DF10 solver proxy −2.1663874486347634 Ha、全事象二乗loss、shots/X/Y/round IIDモデル。exact τ=(0.00159360010199040 Ha)²=0.00000253956128506381328204169216 Ha²。

| 構成 | 配分 | 条件付きモデル費用 RZ | MSE下界 Ha² | MSE上界 Ha² | 閾値余裕 τ−上界 Ha² | 判定 |
|---|---|---:|---:|---:|---:|---|
| LD3 | FULL_PF__B_PRICE | 541936295.9549784 | 0.000002529235675649838530158704039 | 0.000002529235675649838987499630579 | 1.032560941397429454206158107E-8 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| LD5 | FULL_PF__B_PRICE | 541743905.5714286 | 0.000002500259367355751624050728247 | 0.000002500259367355751628068186548 | 3.930191770806165397350561247E-8 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| LD3 | VP | 1051862454.812121 | 6.206210455307360129958626682E-7 | 6.206210455307820120226411551E-7 | 0.000001918940239533031270019051005 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| LD5 | VP | 872241518.1428571 | 6.157824848434372131846004112E-7 | 6.157824848434381067457376476E-7 | 0.000001923778800220375175295954512 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |

exact判定は原有理数を使用。余裕の正/負を明示し、UNKNOWN（gate不成立）とモデル不適合（有効下界>τ）、INCONCLUSIVE（有効区間がまたぐ）を区別する。上表と[SUMMARY](SUMMARY.json)に原APIの解決可能な固定commit/pathを保存した。

- FULL_PF__B_PRICE：X/Y各`[48,45,42,39,36,33,29,26,23,19,16,8,2]`。
- VP：X/Y各`[48,45,42,39,36,33,29,26,23,19,16,13,9]`。

固定順序で①を評価後、適合/不適合を理由に早期終了せず②を評価した。新しい配分は生成していない。これらの列を十分N、最小N、方式全体の優劣とは認定しない。

## total view・SOURCE・会計

元LD5 ID `4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb` → total ID `42309d00702fa11c43d62eb27a5ff928657dc1a4d9d8e9401547433af5c25403`、binding `59198e931cb97974dd8a5101f4a2982865ee0b831946170e250d00af551f343e`。元A packet ac46d100e、config39a1fc7df、辞書6255fbc42、数値SOURCE c67469ad4/原carrier1e3985e7eを保全。Aの8c8ce542e::b_intake_dictionary_identityと既存P1 jobを使い、同じbind_packet/C metadata/total binding経路をLD5の固定SOURCEへ最小拡張した。辞書hash・bit順・残余locator・13 physical ID・各round SOURCE・law・scalar onceを照合し、rawやnormalizationは変更/再適用していない。Aの元total_view=nullは変更せずBのviewとして保存した。

価格SOURCEはb8bcdf97b693fa7e99d3d35f44f5ce6d3a691201のPRICE_26AXIS.json。元rowsのNと積はVP用である。①は①自身のNで26軸積を作り直し、exact総額3792207339/7 RZ、②は6105690627/7 RZとなり、各A.comparison[name].T_LD5とexact一致した。新しい価格予測やfitはしていない。全26軸はモデル適用、E[C]/CI/価格資格は未取得。較正cellの直接receiptをround receiptへ付け替えていない。

実行SOURCE `b6d4dea49c39f364048af422bc2928ee7c169261` でcaller/INPUTS/total viewを実行前commit/push固定。原科学結果snapshot `892a5217b59d8ffc35e4323d84aef21407cb6618`。kernel710aa69b・高速backend9e294778・既存worker自身のAPI/partials/resources保存・修正済み観測contextを再利用。各配分のcount-onlyを確認し、LD5自身のρ1閉包・質量・残余・bounds/moments・principal guard・数値allowanceは各call内で検査した。数値判定と保存状態・CPU/RSS/wall/子exitcodeはSUMMARY/原記録で別々に保全した。人工/回帰testは[TESTS](TESTS.txt)。作成者自身の検査であり独立レビューではない。

保存solverは条件付きproxyで、ground認証falseを保持。ideal IIDから実samplerへの移送は未認証。binary64 allowanceは厳密certificateではない。物理的全RMSE、十分N、正式E[C]/G/winnerへ昇格せず、A δEは二重加算していない。価格差もモデル参照値の差である。元SOURCE/raw/旧STOPとLD3の保存2点は変更せず、追加評価へ進まない。

[C_COMPARISON_JOIN](C_COMPARISON_JOIN.json)に元/total configuration・allocation ID・26軸N/単価/積・exact総額・Q3判定・原API SOURCEを結合した。ロードマップ本体は更新せず[HANDOFF](HANDOFF.md)で差分を渡す。

2件の全体wallは約303.523秒。各worker wallは約143.312秒/154.873秒、CPU（子含む）は約233.948秒/255.527秒。観測process tree peak RSSは3,315,072/4,022,464 KiB。記録されたworker/子processの203終了codeは全0。詳細と測定範囲はSUMMARY/RESOURCE_FINAL/SUPERVISION。

FULL_PF__B_PRICEのLD5/LD3の参照費用差は約192,390 RZと小さいが、価格モデルにCI/期待価格資格がないため、実compiled期待費用の差や優位性の有意性は認定しない。VPを含め、今回評価した固定配分の条件付き比較に限定する。
